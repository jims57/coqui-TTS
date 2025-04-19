import os
import glob
from IPython.display import Audio, display

def play_last_opus(directory):
    """Plays the last modified .opus file in the given directory using IPython Audio."""
    try:
        opus_files = glob.glob(os.path.join(directory, "*.opus"))
        if not opus_files:
            print(f"No .opus files found in {directory}")
            return None

        last_opus_file = max(opus_files, key=os.path.getmtime)
        print(f"Playing: {last_opus_file}")
        # Create Audio object but don't display it yet - this prevents duplicate playback
        audio = Audio(filename=last_opus_file, autoplay=True)
        # Return the audio object without displaying it
        return audio

    except FileNotFoundError:
        print(f"Error: Directory not found: {directory}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None

# Specify the directory
tts_output_dir = os.path.expanduser("~/TTS/outputs/")

# Play the last opus file automatically
# The function returns the Audio object, and Jupyter automatically displays the last expression
# No need to call display() as it causes duplicate audio playback
play_last_opus(tts_output_dir)