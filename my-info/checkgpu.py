import torch

# Basic CUDA check
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

# Test GPU tensor operations
if torch.cuda.is_available():
    # Create a test tensor on GPU
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print(f"Tensor device: {x.device}")
    
    # Try a simple operation
    y = x * 2
    print(f"Test calculation: {y}")
    
    # Get GPU info
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    print(f"Current GPU device: {torch.cuda.current_device()}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated()}")