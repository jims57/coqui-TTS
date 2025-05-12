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
