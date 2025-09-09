#!/usr/bin/env python3
"""
Check GPU memory usage and availability
"""

import torch
import subprocess
import os

def check_gpu():
    print("=" * 50)
    print("GPU Information")
    print("=" * 50)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Memory info
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            
            print(f"  Total Memory: {total_memory:.2f} GB")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
            print(f"  Free: {(total_memory - reserved):.2f} GB")
    else:
        print("CUDA is not available")
    
    print("\n" + "=" * 50)
    print("nvidia-smi output:")
    print("=" * 50)
    
    # Run nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
    except FileNotFoundError:
        print("nvidia-smi not found")

if __name__ == "__main__":
    check_gpu()