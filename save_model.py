import torch
from config import MODEL_NAME, DEVICE
from transformers import DebertaV2Tokenizer
from multi_task_model import MultiTaskEmotionSentimentModel

# Load model definition
model = MultiTaskEmotionSentimentModel(base_model_name=MODEL_NAME, num_emotions=6)

# Load saved state dict from checkpoint
checkpoint = torch.load("checkpoint_latest.pt", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

# Save model
model.save_pretrained("./multi_task_model")

# Save tokenizer
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME, use_fast=False)
tokenizer.save_pretrained("./multi_task_model")

print("✅ Final model and tokenizer saved to ./multi_task_model")
