# main.py
import os
import time
import pandas as pd
import torch
from trainer import fit_emotion_model
from analysis import evaluate_model_on_test, plot_training_history

def print_gpu_info():
    print("Checking GPU information...\n")
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}\n")
    for i in range(gpu_count):
        print(f"--- GPU {i} Info ---")
        print(f"Name: {torch.cuda.get_device_name(i)}")
        print(f"Capability: {torch.cuda.get_device_capability(i)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(i) / 1024**2:.2f} MB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
        print(f"Max Memory Reserved: {torch.cuda.max_memory_reserved(i) / 1024**2:.2f} MB")
        print()

def split_dataset(csv_path):
    """
    Reads the CSV file and splits it into train, validation, and test sets.
    Assumes the CSV has columns "text", "label", and "sentiment".
    Splits: 80% train, 10% val, 10% test.
    """
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_val_split = int(0.8 * len(df))
    val_test_split = int(0.9 * len(df))
    df_train = df.iloc[:train_val_split]
    df_val = df.iloc[train_val_split:val_test_split]
    df_test = df.iloc[val_test_split:]
    print(f"Dataset split into {len(df_train)} train, {len(df_val)} val, {len(df_test)} test samples.")
    df_train.to_csv("train.csv", index=False)
    df_val.to_csv("val.csv", index=False)
    df_test.to_csv("test.csv", index=False)
    return df_train, df_val, df_test

def main():
    print_gpu_info()
    csv_path = "/home/sharm2s1/PycharmProjects/Emotional analysis/text_with_sentiment.csv"
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}. Please check the path.")
        return
    print("\nSplitting dataset...")
    df_train, df_val, df_test = split_dataset(csv_path)
    print("Dataset split completed.\n")
    print("Starting training process...")
    start_time = time.time()
    fit_emotion_model(df_train, df_val)
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    print("\nEvaluating model on test set...")
    evaluate_model_on_test(df_test)
    print("\nPlotting training history...")
    plot_training_history()

if __name__ == "__main__":
    main()
