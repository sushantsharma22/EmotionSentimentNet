# generate_sentiment_transformer_gpu.py

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Load the CSV
df = pd.read_csv("/home/sharm2s1/Downloads/text.csv")
texts = df["text"].astype(str).tolist()

# Load transformer-based sentiment model
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Move model to GPU (and use all GPUs if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"🚀 Using {torch.cuda.device_count()} GPUs with DataParallel!")
    model = torch.nn.DataParallel(model)
model = model.to(device)
model.eval()

# Process in batches
batch_size = 1024
sentiment_scores = []

for i in tqdm(range(0, len(texts), batch_size), desc="Computing Sentiment on GPU"):
    batch_texts = texts[i:i + batch_size]
    encoded = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=128)
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        probs = F.softmax(outputs.logits, dim=1)
        # Weighted average: [-1 * neg, 0 * neutral, 1 * pos]
        batch_scores = (-1 * probs[:, 0] + 0 * probs[:, 1] + 1 * probs[:, 2]).tolist()
        sentiment_scores.extend([round(s, 4) for s in batch_scores])

# Add to DataFrame and save
df["sentiment"] = sentiment_scores
df.to_csv("text_with_sentiment.csv", index=False)
print("✅ Done! Saved as text_with_sentiment.csv")

