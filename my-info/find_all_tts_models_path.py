import os
from pathlib import Path
from TTS.utils.manage import ModelManager

def find_model_files():
    # Common TTS cache directories
    possible_paths = [
        Path.home() / ".local/share/tts",
        Path.home() / ".cache/huggingface/hub",
    ]
    
    for base_path in possible_paths:
        if base_path.exists():
            print(f"Searching in {base_path}")
            for root, dirs, files in os.walk(base_path):
                if "xtts_v2" in root:
                    print(f"\nFound model files in: {root}")
                    for file in files:
                        print(f"- {file}")

find_model_files()