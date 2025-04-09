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

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
#wav = tts.tts(text="Hello world!", speaker_wav="speaker_wavs/andy-liu-en-1.wav", language="en")


# Andy Liu
# tts.tts_to_file(text="The sun sets behind the mountains, casting long shadows across the valley.", speaker_wav="speaker_wavs/andy-liu-en-1.wav", language="en", file_path="output.wav")

# Jack Ma
tts.tts_to_file(text="The sun sets behind the mountains, casting long shadows across the valley.", speaker_wav="speaker_wavs/jack-mark-en-1.wav", language="en", file_path="output.wav")
