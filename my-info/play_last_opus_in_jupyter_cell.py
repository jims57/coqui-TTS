import os
import glob
from IPython.display import Audio, display

def play_last_audio(directory):
    """Plays the last modified audio file (.opus or .wav) in the given directory using IPython Audio."""
    try:
        # Look for both opus and wav files
        audio_files = glob.glob(os.path.join(directory, "*.opus")) + glob.glob(os.path.join(directory, "*.wav"))
        if not audio_files:
            print(f"No audio files (.opus or .wav) found in {directory}")
            return None

        # Get the most recently modified file
        last_audio_file = max(audio_files, key=os.path.getmtime)
        print(f"Playing: {last_audio_file}")
        # Create Audio object but don't display it yet - this prevents duplicate playback
        audio = Audio(filename=last_audio_file, autoplay=True)
        # Return the audio object without displaying it
        return audio

    except FileNotFoundError:
        print(f"Error: Directory not found: {directory}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None

# Specify the directory
tts_output_dir = os.path.expanduser("~/TTS/outputs/")

# Play the last audio file automatically
# The function returns the Audio object, and Jupyter automatically displays the last expression
# No need to call display() as it causes duplicate audio playback
play_last_audio(tts_output_dir)