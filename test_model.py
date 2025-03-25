#!/usr/bin/env python3
# test_model.py

import os
import re
import string
import time
import emoji
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    precision_recall_fscore_support
)
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import TweetTokenizer

# ----------------------------
# 1. Configuration
# ----------------------------
MODEL_DIR = "./multi_task_model"
BASE_MODEL_NAME = "microsoft/deberta-v3-base"
BATCH_SIZE = 64
MAX_SEQ_LENGTH = 128
NUM_EMOTIONS = 6
EMOTION_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}
TRAINING_HISTORY_FILE = "training_history.csv"

# spaCy model for lemmatization
nlp = spacy.load("en_core_web_sm")

# ----------------------------
# 2. GPU Selection Logic
# ----------------------------
def find_free_gpus(threshold_mb=1000):
    """
    Returns a list of GPU indices that have memory usage below 'threshold_mb'.
    Requires 'nvidia-smi' to be available in your system PATH.
    """
    if not torch.cuda.is_available():
        return []  # No CUDA at all
    
    try:
        cmd = "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"
        output = subprocess.check_output(cmd.split()).decode("utf-8").strip().split("\n")
        free_gpus = []
        for line in output:
            idx_str, mem_used_str = line.split(",")
            idx = int(idx_str.strip())
            mem_used = int(mem_used_str.strip())
            if mem_used < threshold_mb:
                free_gpus.append(idx)
        return free_gpus
    except Exception as e:
        print(f"Could not run nvidia-smi to detect free GPUs: {e}")
        # Fallback: just return all GPUs if torch sees them
        return list(range(torch.cuda.device_count()))

def setup_device():
    """
    Detects free GPUs using find_free_gpus. 
    - If multiple GPUs are free, returns a DataParallel model setup.
    - If one GPU is free, uses that single GPU.
    - If none, falls back to CPU.
    """
    free_gpus = find_free_gpus(threshold_mb=1000)  # Adjust threshold as needed
    if len(free_gpus) == 0:
        print("No free GPUs available (or no GPUs at all). Using CPU.")
        return torch.device("cpu"), None
    
    if len(free_gpus) == 1:
        print(f"Using a single GPU: {free_gpus[0]}")
        return torch.device(f"cuda:{free_gpus[0]}"), None
    
    # More than one GPU
    print(f"Using multiple GPUs: {free_gpus}")
    # We'll place the model on the first free GPU, then wrap in DataParallel
    primary_gpu = free_gpus[0]
    return torch.device(f"cuda:{primary_gpu}"), free_gpus

# ----------------------------
# 3. Advanced Text Preprocessing
# ----------------------------
def replace_emoji(token: str) -> str:
    """Convert emojis to text (e.g., 🙂 -> smiling_face)."""
    if emoji.is_emoji(token):
        return emoji.demojize(token, delimiters=(" ", " ")).replace(":", "")
    return token

def tokenize_text(text: str) -> list:
    tokenizer_obj = TweetTokenizer()
    return tokenizer_obj.tokenize(text)

def spacy_lemmatize(tokens: list) -> list:
    doc = nlp(" ".join(tokens))
    return [token.lemma_.strip() for token in doc if token.lemma_.strip()]

def advanced_clean_text(text: str) -> str:
    """
    Convert text to lowercase, remove URLs, mentions, numeric digits, punctuation,
    convert emojis, and lemmatize.
    """
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = tokenize_text(text)
    cleaned_tokens = []
    for token in tokens:
        token = replace_emoji(token)
        token = token.translate(str.maketrans("", "", string.punctuation))
        if token.strip():
            cleaned_tokens.append(token.strip())
    lemmatized_tokens = spacy_lemmatize(cleaned_tokens)
    final_tokens = [w for w in lemmatized_tokens if w.isalpha() and len(w) > 1]
    return " ".join(final_tokens)

# ----------------------------
# 4. Multi-Task Model Definition
# ----------------------------
class MultiTaskEmotionSentimentModel(nn.Module):
    """
    Multi-task model for emotion classification (6 classes)
    and sentiment intensity regression (single float).
    """
    def __init__(self, base_model_name=BASE_MODEL_NAME, num_emotions=NUM_EMOTIONS):
        super(MultiTaskEmotionSentimentModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.transformer.config.hidden_size
        # Classification head for emotion
        self.emotion_classifier = nn.Linear(hidden_size, num_emotions)
        # Regression head for sentiment intensity
        self.sentiment_regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        classification_logits = self.emotion_classifier(pooled_output)
        sentiment_output = self.sentiment_regressor(pooled_output).squeeze(-1)
        return classification_logits, sentiment_output

# ----------------------------
# 5. Dataset & DataLoader
# ----------------------------
class EmotionDataset(Dataset):
    """
    Expects a DataFrame with columns: ["text", "label", "sentiment"].
    Applies advanced_clean_text() and tokenization.
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

def create_data_loader(df, tokenizer, batch_size=BATCH_SIZE, shuffle=False, num_workers=0):
    """
    Creates a DataLoader with optional num_workers. 
    If you encounter multiprocessing issues, set num_workers=0.
    """
    dataset = EmotionDataset(df, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

# ----------------------------
# 6. Evaluation Functions
# ----------------------------
def evaluate_model_on_test(df_test, model_dir=MODEL_DIR, device=None, device_ids=None):
    """
    1. Loads the final model from model_dir (./multi_task_model).
    2. Evaluates emotion classification via classification report & confusion matrix.
    3. Evaluates sentiment regression (MAE).
    4. Plots classification metrics bar chart & confusion matrix.
    5. Returns predictions and ground truths.
    """
    print(f"\n=== Evaluating Model on Test Dataset ===")

    # If device wasn't provided, fallback to CPU
    if device is None:
        device = torch.device("cpu")

    # Load model
    print(f"Loading model from '{model_dir}' ...")
    base_model = MultiTaskEmotionSentimentModel()
    state_dict = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location=device)
    base_model.load_state_dict(state_dict)

    # If multiple GPUs are available, wrap in DataParallel
    if device_ids is not None and len(device_ids) > 1:
        print(f"Wrapping model in DataParallel on GPUs: {device_ids}")
        model = nn.DataParallel(base_model.to(device), device_ids=device_ids)
    else:
        model = base_model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    # Create DataLoader
    test_loader = create_data_loader(df_test, tokenizer, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Lists to store predictions and labels
    y_true_cls, y_pred_cls = [], []
    y_true_sent, y_pred_sent = [], []

    # Inference
    inference_start = time.time()
    for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()
        sentiments = batch["sentiment"].to(device)

        with torch.no_grad():
            classification_logits, sentiment_output = model(input_ids, attention_mask)

        preds_cls = torch.argmax(classification_logits, dim=1).cpu().numpy()
        preds_sent = sentiment_output.cpu().numpy()

        y_true_cls.extend(labels)
        y_pred_cls.extend(preds_cls)
        y_true_sent.extend(sentiments.cpu().numpy())
        y_pred_sent.extend(preds_sent)

    inference_time = time.time() - inference_start
    print(f"Inference completed in {inference_time:.2f} seconds.\n")

    # --- Classification Report ---
    print("=== Classification Report ===")
    cls_report = classification_report(y_true_cls, y_pred_cls, digits=4)
    print(cls_report)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true_cls, y_pred_cls)
    labels_order = [EMOTION_MAP[i] for i in sorted(EMOTION_MAP.keys())]
    disp = ConfusionMatrixDisplay(cm, display_labels=labels_order)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax_cm, xticks_rotation="vertical", values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close(fig_cm)
    print("Saved confusion matrix as 'confusion_matrix.png'")

    # --- Classification Metrics Bar Chart ---
    precisions, recalls, f1_scores, _ = precision_recall_fscore_support(y_true_cls, y_pred_cls, labels=sorted(EMOTION_MAP.keys()))
    x = np.arange(len(labels_order))
    width = 0.25
    fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
    ax_bar.bar(x - width, precisions, width, label="Precision")
    ax_bar.bar(x, recalls, width, label="Recall")
    ax_bar.bar(x + width, f1_scores, width, label="F1-Score")
    ax_bar.set_ylabel("Scores")
    ax_bar.set_title("Classification Metrics by Emotion")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels_order)
    ax_bar.legend()
    plt.tight_layout()
    plt.savefig("classification_metrics.png")
    plt.close(fig_bar)
    print("Saved classification metrics bar chart as 'classification_metrics.png'")

    # --- Sentiment Regression Metrics ---
    mae = mean_absolute_error(y_true_sent, y_pred_sent)
    print(f"\n=== Sentiment Regression ===")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    # Scatter Plot of Actual vs. Predicted Sentiment
    fig_scatter, ax_scatter = plt.subplots()
    ax_scatter.scatter(y_true_sent, y_pred_sent, alpha=0.3, color="blue")
    ax_scatter.set_xlabel("True Sentiment")
    ax_scatter.set_ylabel("Predicted Sentiment")
    ax_scatter.set_title("Actual vs. Predicted Sentiment")
    plt.savefig("sentiment_scatter.png")
    plt.close(fig_scatter)
    print("Saved sentiment scatter plot as 'sentiment_scatter.png'")

    # Error Distribution Histogram
    errors = np.array(y_true_sent) - np.array(y_pred_sent)
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(errors, bins=30, color="orange", edgecolor="black")
    ax_hist.set_xlabel("Error (True - Predicted)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Sentiment Error Distribution")
    plt.savefig("sentiment_error_distribution.png")
    plt.close(fig_hist)
    print("Saved sentiment error distribution histogram as 'sentiment_error_distribution.png'")

    return y_true_cls, y_pred_cls, y_true_sent, y_pred_sent

def track_sentiment_evolution(conversation_texts, model_dir=MODEL_DIR, device=None, device_ids=None):
    """
    1. Loads model from model_dir.
    2. Cleans and tokenizes each message in conversation_texts.
    3. Generates sentiment intensity for each.
    4. Plots the evolution of sentiment across the conversation.
    5. Prints a quick summary of the overall emotional trajectory.
    """
    print("\n=== Tracking Sentiment Evolution ===")
    start_time = time.time()

    if device is None:
        device = torch.device("cpu")

    # Load base model
    base_model = MultiTaskEmotionSentimentModel()
    state_dict = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location=device)
    base_model.load_state_dict(state_dict)

    # If multiple GPUs, wrap in DataParallel
    if device_ids is not None and len(device_ids) > 1:
        print(f"Wrapping sentiment model in DataParallel on GPUs: {device_ids}")
        model = nn.DataParallel(base_model.to(device), device_ids=device_ids)
    else:
        model = base_model.to(device)

    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    sentiments = []
    for i, text in enumerate(conversation_texts):
        cleaned_text = advanced_clean_text(text)
        encoding = tokenizer(
            cleaned_text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            _, sentiment_output = model(input_ids, attention_mask)

        sentiment_value = sentiment_output.item()
        sentiments.append(sentiment_value)

    # Plot the sentiment evolution
    fig, ax = plt.subplots()
    ax.plot(range(len(sentiments)), sentiments, marker="o", color="green")
    ax.set_xlabel("Message Index")
    ax.set_ylabel("Sentiment Intensity")
    ax.set_title("Sentiment Evolution Over Conversation")
    plt.tight_layout()
    plt.savefig("sentiment_evolution.png")
    plt.close(fig)
    print("Saved sentiment evolution plot as 'sentiment_evolution.png'")

    # Simple summary of the emotional trajectory
    min_sent = min(sentiments)
    max_sent = max(sentiments)
    avg_sent = sum(sentiments) / len(sentiments)
    print(f"Minimum Sentiment: {min_sent:.4f}")
    print(f"Maximum Sentiment: {max_sent:.4f}")
    print(f"Average Sentiment: {avg_sent:.4f}")
    if avg_sent > 0:
        overall_desc = "Positive"
    elif avg_sent < 0:
        overall_desc = "Negative"
    else:
        overall_desc = "Neutral"
    print(f"Overall Emotional Trajectory: {overall_desc}")

    end_time = time.time()
    print(f"Sentiment evolution analysis took {end_time - start_time:.2f} seconds.")
    print("=== Finished Tracking Sentiment Evolution ===\n")

def plot_training_history(history_file=TRAINING_HISTORY_FILE):
    """
    Plots training vs. validation metrics from a CSV file
    that includes columns: epoch, train_loss, val_accuracy, ...
    """
    print("\n=== Plotting Training History ===")
    if not os.path.exists(history_file):
        print(f"No {history_file} found. Skipping training history plot.")
        return
    try:
        history = pd.read_csv(history_file)
    except Exception as e:
        print(f"Could not load training history: {e}")
        return

    fig, ax1 = plt.subplots()
    ax1.plot(history['epoch'], history['train_loss'], label='Train Loss', color='blue', marker='o')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    if 'val_accuracy' in history.columns:
        ax2 = ax1.twinx()
        ax2.plot(history['epoch'], history['val_accuracy'], label='Val Accuracy', color='green', marker='x')
        ax2.set_ylabel("Validation Accuracy", color='green')
        ax2.tick_params(axis='y', labelcolor='green')

    plt.title("Training Loss and Validation Accuracy")
    fig.tight_layout()
    plt.savefig("training_history.png")
    plt.close(fig)
    print("Saved training history plot as 'training_history.png'")
    print("=== Finished Plotting Training History ===\n")

# ----------------------------
# 7. Main Function
# ----------------------------
def main():
    """
    1. Dynamically select free GPUs if available.
    2. Evaluate classification & regression on test.csv.
    3. Track sentiment evolution on a sample conversation.
    4. Plot training history (if available).
    """
    # Disable parallelism for tokenizers to avoid conflicts
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device, device_ids = setup_device()

    # Evaluate classification & regression on test.csv
    if not os.path.exists("test.csv"):
        print("No 'test.csv' found. Skipping classification/regression test.")
    else:
        df_test = pd.read_csv("test.csv")
        evaluate_model_on_test(df_test, model_dir=MODEL_DIR, device=device, device_ids=device_ids)

    # Track sentiment evolution on a sample conversation
    conversation_texts = [
        "I'm so excited about the new project!",
        "Now I'm feeling a bit anxious about the upcoming deadline.",
        "The client meeting didn't go as well as I hoped, I'm frustrated.",
        "But I'm determined to find a solution, let's see how it goes.",
        "At last, I've found a workaround, feeling more optimistic now!",
        "Overall, it seems we're heading toward a great outcome!"
    ]
    track_sentiment_evolution(conversation_texts, model_dir=MODEL_DIR, device=device, device_ids=device_ids)

    # Plot training history if available
    plot_training_history(TRAINING_HISTORY_FILE)

if __name__ == "__main__":
    main()
