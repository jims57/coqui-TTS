import os
import glob
import time
from typing import List
import asyncio

def clean_wav_files(output_dir: str = "outputs", total_files: int = 5) -> None:
    """
    Keep only the specified number of most recent audio files (WAV and Opus) in the output directory.
    
    Args:
        output_dir: Directory containing audio files (default: "outputs")
        total_files: Maximum number of audio files to keep (default: 5)
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' does not exist. Creating it...")
        os.makedirs(output_dir)
        return
    
    # Get all WAV and Opus files in the directory
    wav_files = glob.glob(os.path.join(output_dir, "*.wav"))
    opus_files = glob.glob(os.path.join(output_dir, "*.opus"))
    
    # Combine all audio files
    audio_files = wav_files + opus_files
    
    # If we have fewer files than the limit, no cleanup needed
    if len(audio_files) <= total_files:
        print(f"Only {len(audio_files)} audio files found. No cleanup needed.")
        return
    
    # Sort files by modification time (oldest first)
    audio_files.sort(key=os.path.getmtime)
    
    # Determine files to delete (all except the newest ones)
    files_to_delete = audio_files[:-total_files]
    
    # Delete the old files
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    print(f"Cleanup complete. Kept the {total_files} most recent audio files.")

async def async_clean_wav_files():
    """Clean WAV files in outputs directory asynchronously, keeping only the 5 most recent files."""
    try:
        print("Starting cleanup process...")
        keep_files = 5
        # Get all wav and opus files in the outputs directory
        files = []
        if os.path.exists("outputs"):
            for file in os.listdir("outputs"):
                if file.endswith(".wav") or file.endswith(".opus"):
                    file_path = os.path.join("outputs", file)
                    if os.path.isfile(file_path):
                        files.append(file_path)
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Keep the 5 most recent files, delete the rest
        for file_path in files[keep_files:]:
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except FileNotFoundError:
                # File might have been deleted by another process
                print(f"File already deleted: {file_path}")
            except Exception as e:
                # Log other errors but continue
                print(f"Error deleting {file_path}: {e}")
        
        print(f"Cleanup complete. Kept the {keep_files} most recent audio files.")
    except Exception as e:
        print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    clean_wav_files()
