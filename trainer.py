# trainer.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from transformers import DebertaV2Tokenizer
from config import DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, MAX_SEQ_LENGTH, MODEL_NAME, USE_DATA_PARALLEL, EARLY_STOP_ACC
from multi_task_model import MultiTaskEmotionSentimentModel
from data_preprocessing import advanced_clean_text
import pandas as pd

CHECKPOINT_PATH = "./checkpoint_latest.pt"
HISTORY_CSV = "training_history.csv"

class EmotionDataset(Dataset):
    """
    Dataset expecting a DataFrame with columns: ["text", "label", "sentiment"].
    Uses advanced_clean_text() to preprocess text.
    """
    def __init__(self, df, tokenizer):
        self.data = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        raw_text = str(row["text"])
        cleaned_text = advanced_clean_text(raw_text)
        # Get emotion label (default 0) and sentiment intensity (default 0.0)
        emotion_label = row.get("label", 0)
        sentiment_value = row.get("sentiment", 0.0)
        encoding = self.tokenizer(
            cleaned_text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(emotion_label, dtype=torch.long),
            "sentiment": torch.tensor(sentiment_value, dtype=torch.float)
        }

def create_data_loader(df, tokenizer, batch_size=BATCH_SIZE, shuffle=False):
    dataset = EmotionDataset(df, tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,  # Reduce to 0 if you face issues.
        pin_memory=True
    )

def train_emotion_classifier(model, optimizer, scaler, train_loader, val_loader, start_epoch=0):
    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()
    history = []  # To record metrics for each epoch

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        batch_count = 0

        print(f"\n--- Starting Epoch {epoch+1}/{EPOCHS} ---")
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"):
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)
            sentiments = batch["sentiment"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with autocast("cuda"):
                logits, sentiment_output = model(input_ids, attention_mask)
                loss_classification = classification_criterion(logits, labels)
                loss_regression = regression_criterion(sentiment_output, sentiments)
                loss = loss_classification + loss_regression
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            batch_count += 1
            if batch_count % 100 == 0:
                print(f"Epoch {epoch+1} - Processed {batch_count} batches; Current Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time

        # Validation phase
        model.eval()
        y_true_cls, y_pred_cls = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
                attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
                labels = batch["labels"].cpu().numpy()
                with autocast("cuda"):
                    logits, _ = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_true_cls.extend(labels)
                y_pred_cls.extend(preds)

        accuracy = accuracy_score(y_true_cls, y_pred_cls)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true_cls, y_pred_cls, average="weighted")
        print(f"\n[Epoch {epoch+1} Completed] Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | "
              f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | "
              f"Epoch Time: {epoch_time/60:.2f} minutes")
        print("-------------------------------------------------------------")

        # Record metrics
        history.append({
            "epoch": epoch+1,
            "train_loss": avg_loss,
            "val_accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "epoch_time": epoch_time
        })

        # Save checkpoint after each epoch
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict()
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        print(f"Checkpoint saved at epoch {epoch+1} to {CHECKPOINT_PATH}")

        # Early stopping: if validation accuracy meets threshold, stop training.
        if accuracy >= EARLY_STOP_ACC:
            print(f"Early stopping triggered at epoch {epoch+1} as val accuracy {accuracy:.4f} >= {EARLY_STOP_ACC}")
            break

    # Save training history to CSV for later analysis.
    history_df = pd.DataFrame(history)
    history_df.to_csv(HISTORY_CSV, index=False)
    print(f"Training history saved to {HISTORY_CSV}")

def fit_emotion_model(df_train, df_val, base_model=MODEL_NAME):
    # Instantiate the multi-task model
    model = MultiTaskEmotionSentimentModel(base_model_name=base_model, num_emotions=6).to(DEVICE)
    if USE_DATA_PARALLEL and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    tokenizer = DebertaV2Tokenizer.from_pretrained(base_model, use_fast=False)
    train_loader = create_data_loader(df_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = create_data_loader(df_val, tokenizer, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(device="cuda")
    start_epoch = 0

    # Resume from checkpoint if it exists.
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming training from checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch+1}")

    train_emotion_classifier(model, optimizer, scaler, train_loader, val_loader, start_epoch=start_epoch)

    print("Saving final model and tokenizer to './multi_task_model' ...")
    model.save_pretrained("./multi_task_model")
    tokenizer.save_pretrained("./multi_task_model")
    print("Final model and tokenizer saved successfully.")
