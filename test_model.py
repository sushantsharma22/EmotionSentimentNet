import os
import re
import string
import time
import emoji
import spacy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
from transformers import AutoTokenizer, AutoModel, DebertaV2Tokenizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error
from tqdm import tqdm
import numpy as np

# ----------------------------
# 1. Configuration
# ----------------------------
MODEL_NAME = "microsoft/deberta-v3-base"
NUM_EMOTIONS = 6
MAX_SEQ_LENGTH = 128
# You can try increasing this to 300 if memory allows; adjust based on your GPU
BATCH_SIZE = 300  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTION_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# ----------------------------
# 2. Data Preprocessing
# ----------------------------
nlp = spacy.load("en_core_web_sm")

def replace_emoji(token: str) -> str:
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
    filtered_tokens = cleaned_tokens  # Optionally filter stopwords here.
    lemmatized_tokens = spacy_lemmatize(filtered_tokens)
    final_tokens = [w for w in lemmatized_tokens if w.isalpha() and len(w) > 1]
    return " ".join(final_tokens)

# ----------------------------
# 3. Model Definition
# ----------------------------
class MultiTaskEmotionSentimentModel(nn.Module):
    def __init__(self, base_model_name=MODEL_NAME, num_emotions=NUM_EMOTIONS):
        super(MultiTaskEmotionSentimentModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.transformer.config.hidden_size
        self.emotion_classifier = nn.Linear(hidden_size, num_emotions)
        self.sentiment_regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] representation
        classification_logits = self.emotion_classifier(pooled_output)
        sentiment_output = self.sentiment_regressor(pooled_output).squeeze(-1)
        return classification_logits, sentiment_output

# ----------------------------
# 4. Dataset & DataLoader
# ----------------------------
class EmotionDataset(Dataset):
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
    dataset = EmotionDataset(df, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

# ----------------------------
# 5. Evaluation Functions
# ----------------------------
def evaluate_model_on_test(test_df, model_dir="./multi_task_model"):
    print(f"\n=== Evaluating Model on Test Dataset ===")
    load_start = time.time()
    print("Loading model from '{}' ...".format(model_dir))
    
    # Load model without DataParallel first
    model = MultiTaskEmotionSentimentModel(base_model_name=MODEL_NAME, num_emotions=NUM_EMOTIONS).to(DEVICE)
    state_path = os.path.join(model_dir, "pytorch_model.bin")
    state_dict = torch.load(state_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    
    # Wrap with DataParallel AFTER loading if multiple GPUs available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs for inference.")
    
    model.eval()
    
    tokenizer_load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    tokenizer_load_end = time.time()
    print(f"Tokenizer loaded in {tokenizer_load_end - tokenizer_load_start:.2f} seconds.")
    
    load_end = time.time()
    print(f"Model loaded in {load_end - load_start:.2f} seconds.\n")
    
    print("Creating test DataLoader...")
    loader_start = time.time()
    test_loader = create_data_loader(test_df, tokenizer, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    loader_end = time.time()
    print(f"DataLoader created in {loader_end - loader_start:.2f} seconds.\n")
    
    print("Starting inference on test set...")
    inference_start = time.time()
    y_true_cls, y_pred_cls = [], []
    regression_true, regression_pred = [], []
    
    for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to("cpu").numpy()
        sentiments = batch["sentiment"].to(DEVICE)
        with torch.no_grad():
            classification_logits, sentiment_output = model(input_ids, attention_mask)
        preds = torch.argmax(classification_logits, dim=1).to("cpu").numpy()
        
        y_true_cls.extend(labels)
        y_pred_cls.extend(preds)
        regression_true.extend(sentiments.to("cpu").numpy())
        regression_pred.extend(sentiment_output.to("cpu").numpy())
    
    inference_end = time.time()
    print(f"Inference completed in {inference_end - inference_start:.2f} seconds.\n")
    
    # Classification report and confusion matrix
    print("=== Classification Report ===")
    report = classification_report(y_true_cls, y_pred_cls, digits=4, output_dict=True)
    print(classification_report(y_true_cls, y_pred_cls, digits=4))
    
    cm = confusion_matrix(y_true_cls, y_pred_cls)
    labels_order = [EMOTION_MAP[i] for i in sorted(EMOTION_MAP.keys())]
    disp = ConfusionMatrixDisplay(cm, display_labels=labels_order)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax_cm, xticks_rotation="vertical", values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close(fig_cm)
    print("Saved confusion matrix as 'confusion_matrix.png'")
    
    # Regression metric
    mae = mean_absolute_error(regression_true, regression_pred)
    print(f"\nRegression Mean Absolute Error (MAE): {mae:.4f}")
    
    # Save classification metrics as a bar chart
    save_classification_metrics(report)
    
    print("=== Evaluation Finished ===\n")
    return y_true_cls, y_pred_cls, regression_true, regression_pred

def save_classification_metrics(report):
    # Exclude overall metrics; plot per-class metrics only (keys that are digits or in EMOTION_MAP)
    classes = [EMOTION_MAP[int(k)] for k in report.keys() if k.isdigit()]
    precisions = [report[k]["precision"] for k in report.keys() if k.isdigit()]
    recalls = [report[k]["recall"] for k in report.keys() if k.isdigit()]
    f1s = [report[k]["f1-score"] for k in report.keys() if k.isdigit()]
    
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width, precisions, width, label="Precision")
    ax.bar(x, recalls, width, label="Recall")
    ax.bar(x + width, f1s, width, label="F1-Score")
    
    ax.set_ylabel("Scores")
    ax.set_title("Classification Metrics by Emotion")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("classification_metrics.png")
    plt.close(fig)
    print("Saved classification metrics bar chart as 'classification_metrics.png'")

def plot_training_history(history_file="training_history.csv"):
    print("\n=== Plotting Training History ===")
    if not os.path.exists(history_file):
        print(f"No {history_file} found. Skipping training history plot.")
        return
    try:
        history = pd.read_csv(history_file)
    except Exception as e:
        print(f"Could not load training history: {e}")
        return

    fig_hist, ax1 = plt.subplots()
    ax1.plot(history['epoch'], history['train_loss'], label='Train Loss', color='blue', marker='o')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color='blue')
    ax2 = ax1.twinx()
    ax2.plot(history['epoch'], history['val_accuracy'], label='Val Accuracy', color='green', marker='x')
    ax2.set_ylabel("Validation Accuracy", color='green')
    plt.title("Training Loss and Validation Accuracy")
    fig_hist.tight_layout()
    plt.savefig("training_history.png")
    plt.close(fig_hist)
    print("Saved training history plot as 'training_history.png'")
    print("=== Finished Plotting Training History ===\n")

def track_sentiment_evolution(conversation_texts, model_dir="./multi_task_model"):
    print("\n=== Tracking Sentiment Evolution ===")
    start_time_ = time.time()
    model = MultiTaskEmotionSentimentModel(base_model_name=MODEL_NAME, num_emotions=NUM_EMOTIONS).to(DEVICE)
    state_dict = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    load_end_ = time.time()
    print(f"Loaded model & tokenizer in {load_end_ - start_time_:.2f} seconds.")
    
    sentiments = []
    for text in conversation_texts:
        cleaned_text = advanced_clean_text(text)
        encoding = tokenizer(
            cleaned_text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)
        with torch.no_grad():
            _, sentiment = model(input_ids, attention_mask)
        sentiments.append(sentiment.item())
    
    fig_sent, ax_sent = plt.subplots()
    ax_sent.plot(sentiments, marker="o")
    ax_sent.set_xlabel("Message Index")
    ax_sent.set_ylabel("Sentiment Intensity")
    ax_sent.set_title("Sentiment Evolution Over Conversation")
    plt.tight_layout()
    plt.savefig("sentiment_evolution.png")
    plt.close(fig_sent)
    end_time_ = time.time()
    print(f"Sentiment evolution tracking took {end_time_ - load_end_:.2f} seconds.")
    print("Saved sentiment evolution plot as 'sentiment_evolution.png'")
    print("Sentiment evolution values:", sentiments)
    print("=== Finished Tracking Sentiment Evolution ===\n")
    return sentiments

# ----------------------------
# 6. Main Test Cases
# ----------------------------
def test_evaluation():
    print("=== TEST 1: Model Evaluation on Test Dataset ===")
    try:
        df_test = pd.read_csv("test.csv")
    except Exception as e:
        print("Error loading test.csv:", e)
        return
    evaluate_model_on_test(df_test)

def test_training_history():
    print("=== TEST 2: Training History Plot ===")
    plot_training_history("training_history.csv")

def test_sentiment_evolution():
    print("=== TEST 3: Sentiment Evolution Tracking ===")
    conversation_texts = [
        "I'm so happy today! Everything is amazing.",
        "Now I'm feeling a bit sad after some bad news.",
        "Things took an unexpected turn, and I'm confused.",
        "Finally, I'm feeling hopeful and excited!"
    ]
    track_sentiment_evolution(conversation_texts)

if __name__ == "__main__":
    # Disable tokenizer parallelism to avoid multiprocessing issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Run test cases
    test_evaluation()
    test_training_history()
    test_sentiment_evolution()
