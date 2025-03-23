# config.py
import torch
from nltk.corpus import stopwords

# MODEL CONFIGURATION
MODEL_NAME = "microsoft/deberta-v3-base"  # Change if desired
USE_DATA_PARALLEL = True  # Set to True if you want to use multiple GPUs
NUM_EMOTIONS = 6

# Emotion mapping (adjust if needed):
# For example: 0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise
EMOTION_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}
LABEL2ID = {v: k for k, v in EMOTION_MAP.items()}

# TRAINING HYPERPARAMETERS
BATCH_SIZE = 305
EPOCHS = 10           # Set the number of epochs as needed
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 128

# Early stopping threshold (if validation accuracy reaches this value, training stops early)
EARLY_STOP_ACC = 0.93

# STOP WORDS
STOP_WORDS = set(stopwords.words("english"))

# DEVICE CONFIGURATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
