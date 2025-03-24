import torch
from config import MODEL_NAME
from transformers import DebertaV2Tokenizer
from multi_task_model import MultiTaskEmotionSentimentModel

# Force CPU to avoid GPU memory issues
DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# Load model architecture
model = MultiTaskEmotionSentimentModel(base_model_name=MODEL_NAME, num_emotions=6)

# Load checkpoint
checkpoint = torch.load("checkpoint_latest.pt", map_location=DEVICE)
state_dict = checkpoint["model_state_dict"]

# Remove "module." prefix from keys if saved using DataParallel
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("module.", "") if k.startswith("module.") else k
    new_state_dict[new_key] = v

# Load cleaned state dict into model
model.load_state_dict(new_state_dict)

# Save model and tokenizer
model.save_pretrained("./multi_task_model")
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME, use_fast=False)
tokenizer.save_pretrained("./multi_task_model")

print("✅ Model and tokenizer saved to './multi_task_model'")
