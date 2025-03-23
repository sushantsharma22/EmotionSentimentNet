# cuda.py
import torch

gpu_count = torch.cuda.device_count()
print(f"Number of GPUs available: {gpu_count}")

for i in range(gpu_count):
    print(f"\n--- GPU {i} Info ---")
    print(f"Name: {torch.cuda.get_device_name(i)}")
    print(f"Capability: {torch.cuda.get_device_capability(i)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
    print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(i) / 1024 ** 2:.2f} MB")
    print(f"Memory Reserved: {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
    print(f"Max Memory Reserved: {torch.cuda.max_memory_reserved(i) / 1024 ** 2:.2f} MB")
