import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Print which device is being used
print(f"Using device: {device}")

# More detailed CUDA diagnostics
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    try:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
        print(f"Current GPU device: {torch.cuda.current_device()}")
    except Exception as e:
        print(f"Error initializing CUDA: {e}")
