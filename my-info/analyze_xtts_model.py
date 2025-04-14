import torch
from TTS.api import TTS
import os

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize the TTS model
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
tts = TTS(model_name).to(device)

def analyze_model_requirements():
    # Sample inputs
    sample_text = "This is a test sentence for analyzing model requirements."
    sample_speaker = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/speakers_xtts.pth"
    
    print("\nModel Architecture:")
    print("===================")
    print(tts.synthesizer.encoder)
    
    print("\nAnalyzing input requirements:")
    print("============================")
    
    # Get model inputs
    inputs = tts.synthesizer.preprocess(sample_text, sample_speaker, "en")
    
    print("\nInput shapes and types:")
    print("=====================")
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}:")
            print(f"  Shape: {value.shape}")
            print(f"  Type: {value.dtype}")
            print(f"  Device: {value.device}")
            print()

    # Try a test inference to see output shapes
    print("\nAnalyzing output shapes:")
    print("======================")
    try:
        with torch.no_grad():
            outputs = tts.synthesizer.encoder(inputs)
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}:")
                    print(f"  Shape: {value.shape}")
                    print(f"  Type: {value.dtype}")
                    print(f"  Device: {value.device}")
                    print()
    except Exception as e:
        print(f"Error during inference: {str(e)}")

if __name__ == "__main__":
    analyze_model_requirements() 