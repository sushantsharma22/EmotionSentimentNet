# multi_task_model.py
import torch
import torch.nn as nn
import os
from transformers import AutoModel


class MultiTaskEmotionSentimentModel(nn.Module):
    def __init__(self, base_model_name="microsoft/deberta-v3-base", num_emotions=6):
        super(MultiTaskEmotionSentimentModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.transformer.config.hidden_size
        # Classification head for emotion (six classes)
        self.emotion_classifier = nn.Linear(hidden_size, num_emotions)
        # Regression head for sentiment intensity (output a single float)
        self.sentiment_regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token representation
        classification_logits = self.emotion_classifier(pooled_output)
        sentiment_output = self.sentiment_regressor(pooled_output).squeeze(-1)
        return classification_logits, sentiment_output

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        self.transformer.save_pretrained(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
