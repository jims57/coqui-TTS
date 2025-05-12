import torch
from TTS.api import TTS
# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    try:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
        print(f"Current GPU device: {torch.cuda.current_device()}")
        device = "cuda"
    except Exception as e:
        print(f"Error initializing CUDA: {e}")
        print("Falling back to CPU")
        device = "cpu"
else:
    device = "cpu"

print(f"Using device: {device}")

# Print information about available models
print("Available TTS models:")
print(TTS().list_models())



#【german】
tts = TTS("tts_models/de/thorsten/vits").to(device)
# Generate first sentence
tts.tts_to_file(text="Ich bin ein Berliner und genieße das schöne Wetter in Deutschland.", file_path="output_1.wav")

# Generate second sentence
tts.tts_to_file(text="Die deutsche Küche ist bekannt für ihre Vielfalt an Würsten und Bieren.", file_path="output_2.wav")

# Generate third sentence
tts.tts_to_file(text="Meine Freunde und ich planen einen Ausflug in den Schwarzwald nächstes Wochenende.", file_path="output_3.wav")

# Generate fourth sentence
tts.tts_to_file(text="Die Autobahnen in Deutschland haben keine Geschwindigkeitsbegrenzung in bestimmten Abschnitten.", file_path="output_4.wav")

# Generate fifth sentence
tts.tts_to_file(text="Heute habe ich in einem gemütlichen Café in der Altstadt einen leckeren Apfelstrudel gegessen.", file_path="output_5.wav")

# Check if DeepSpeed is available and being used
try:
    import deepspeed
    print(f"DeepSpeed version: {deepspeed.__version__}")
    print("DeepSpeed is available")
    
    # Check if DeepSpeed is initialized
    if hasattr(torch.distributed, 'is_initialized') and torch.distributed.is_initialized():
        print("DeepSpeed is currently being used for distributed training/inference")
    else:
        print("DeepSpeed is available but not currently initialized for this session")
except ImportError:
    print("DeepSpeed is not installed")

# Print additional acceleration information
print("\nAcceleration Information:")
if torch.cuda.is_available():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Current GPU memory usage: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    print(f"Max GPU memory allocated: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
    
    # Check for CUDA optimizations
    print(f"CUDA Arch List: {torch.cuda.get_arch_list() if hasattr(torch.cuda, 'get_arch_list') else 'Not available'}")
    print(f"CUDNN Enabled: {torch.backends.cudnn.enabled}")
    print(f"CUDNN Benchmark: {torch.backends.cudnn.benchmark}")
else:
    print("Running on CPU only - no GPU acceleration available")

print("\nPreparing to generate German TTS samples...")

