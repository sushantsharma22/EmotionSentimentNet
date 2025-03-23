# analysis.py
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import mean_absolute_error
from transformers import AutoTokenizer
from config import DEVICE, MAX_SEQ_LENGTH, EMOTION_MAP, MODEL_NAME, NUM_EMOTIONS
from multi_task_model import MultiTaskEmotionSentimentModel
from trainer import create_data_loader
import pandas as pd


def evaluate_model_on_test(test_df, model_dir="./multi_task_model"):
    print(f"Loading model from '{model_dir}' ...")
    model = MultiTaskEmotionSentimentModel(base_model_name=MODEL_NAME, num_emotions=NUM_EMOTIONS).to(DEVICE)
    state_dict = torch.load(f"{model_dir}/pytorch_model.bin", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    test_loader = create_data_loader(test_df, tokenizer, batch_size=8, shuffle=False)

    y_true_cls, y_pred_cls = [], []
    regression_true, regression_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].cpu().numpy()
            sentiments = batch["sentiment"].to(DEVICE, non_blocking=True)
            classification_logits, sentiment_output = model(input_ids, attention_mask)
            preds = torch.argmax(classification_logits, dim=1).cpu().numpy()
            y_true_cls.extend(labels)
            y_pred_cls.extend(preds)
            regression_true.extend(sentiments.cpu().numpy())
            regression_pred.extend(sentiment_output.cpu().numpy())

    print("\n=== Classification Report ===")
    print(classification_report(y_true_cls, y_pred_cls, digits=4))
    cm = confusion_matrix(y_true_cls, y_pred_cls)
    labels_order = [EMOTION_MAP[i] for i in sorted(EMOTION_MAP.keys())]
    disp = ConfusionMatrixDisplay(cm, display_labels=labels_order)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, xticks_rotation="vertical", values_format="d")
    plt.title("Confusion Matrix")
    plt.show()

    mae = mean_absolute_error(regression_true, regression_pred)
    print(f"\nRegression Mean Absolute Error (MAE): {mae:.4f}")
    return y_true_cls, y_pred_cls, regression_true, regression_pred


def track_sentiment_evolution(conversation_texts, model_dir="./multi_task_model"):
    from data_preprocessing import advanced_clean_text
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = MultiTaskEmotionSentimentModel(base_model_name=MODEL_NAME, num_emotions=NUM_EMOTIONS).to(DEVICE)
    state_dict = torch.load(f"{model_dir}/pytorch_model.bin", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
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
    plt.plot(sentiments, marker="o")
    plt.xlabel("Message Index")
    plt.ylabel("Sentiment Intensity")
    plt.title("Sentiment Evolution Over Conversation")
    plt.show()
    return sentiments


def plot_training_history(history_file="training_history.csv"):
    try:
        history = pd.read_csv(history_file)
    except Exception as e:
        print(f"Could not load training history: {e}")
        return
    fig, ax1 = plt.subplots()
    ax1.plot(history['epoch'], history['train_loss'], label='Train Loss', color='blue', marker='o')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color='blue')
    ax2 = ax1.twinx()
    ax2.plot(history['epoch'], history['val_accuracy'], label='Val Accuracy', color='green', marker='x')
    ax2.set_ylabel("Validation Accuracy", color='green')
    plt.title("Training Loss and Validation Accuracy")
    fig.tight_layout()
    plt.show()
