# Check if CUDA is being used and print device information
import torch
from TTS.api import TTS

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    # Get the current device
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    device_count = torch.cuda.device_count()
    
    print(f"Current CUDA device: {current_device}")
    print(f"CUDA device name: {device_name}")
    print(f"Number of CUDA devices: {device_count}")
    print(f"First TTS model is using: CUDA")
else:
    print(f"First TTS model is using: CPU (CUDA not available)")

# List all available models
print("\nAvailable TTS Models:")
print(TTS().list_models())

# Use a standard model that should be available in all installations
device = "cuda" if cuda_available else "cpu"
print(f"\nUsing device: {device}")
api = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC").to(device)

# Print the model info - without trying to access model.parameters()
print(f"\nModel loaded successfully")
print(f"Model name: {api.model_name}")

# Generate speech to file
print("\nGenerating speech...")
api.tts_to_file("This is a test.", file_path="output.wav")
print(f"Audio saved to output.wav")

# Example of other models you might try:
# api = TTS(model_name="tts_models/en/ljspeech/glow-tts")
# api = TTS(model_name="tts_models/en/ljspeech/vits")

# TTS with on the fly voice conversion example
# api = TTS("tts_models/en/ljspeech/vits")
# api.tts_with_vc_to_file(
#     "This is voice conversion test.",
#     speaker_wav="target/speaker.wav",
#     file_path="output_vc.wav"
# )