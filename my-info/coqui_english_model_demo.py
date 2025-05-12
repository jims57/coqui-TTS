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
tts = TTS("tts_models/en/ljspeech/fast_pitch").to(device)
# Generate first sentence
tts.tts_to_file(text="I am a Berliner and I enjoy the nice weather in Germany.", file_path="output_1.wav")

# Generate second sentence
tts.tts_to_file(text="German cuisine is known for its variety of sausages and beers.", file_path="output_2.wav")

# Generate third sentence
tts.tts_to_file(text="My friends and I are planning a trip to the Black Forest next weekend.", file_path="output_3.wav")

# Generate fourth sentence
tts.tts_to_file(text="The highways in Germany have no speed limit in certain sections.", file_path="output_4.wav")

# Generate fifth sentence
tts.tts_to_file(text="Today I ate a delicious apple strudel in a cozy cafe in the old town.", file_path="output_5.wav")

