import torch
import time

# Define the tensor
x = torch.randn(16, 1024, 75, 75, device='cuda')

# Measure the time for x.clone()
start_time = time.time()
for _ in range(100):  # Repeat to average the performance
    y_clone = x.clone()
clone_time = time.time() - start_time

# Measure the time for x * 1
start_time = time.time()
for _ in range(100):  # Repeat to average the performance
    y_mul = x * 1
mul_time = time.time() - start_time

print(f"x.clone() time: {clone_time:.6f} seconds")
print(f"x * 1 time: {mul_time:.6f} seconds")
