import os
import glob
import time
from typing import List
import asyncio

def clean_wav_files(output_dir: str = "outputs", total_wav_numbers: int = 5) -> None:
    """
    Keep only the specified number of most recent WAV files in the output directory.
    
    Args:
        output_dir: Directory containing WAV files (default: "outputs")
        total_wav_numbers: Maximum number of WAV files to keep (default: 5)
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' does not exist. Creating it...")
        os.makedirs(output_dir)
        return
    
    # Get all WAV files in the directory
    wav_files = glob.glob(os.path.join(output_dir, "*.wav"))
    
    # If we have fewer files than the limit, no cleanup needed
    if len(wav_files) <= total_wav_numbers:
        print(f"Only {len(wav_files)} WAV files found. No cleanup needed.")
        return
    
    # Sort files by modification time (oldest first)
    wav_files.sort(key=os.path.getmtime)
    
    # Determine files to delete (all except the newest ones)
    files_to_delete = wav_files[:-total_wav_numbers]
    
    # Delete the old files
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    print(f"Cleanup complete. Kept the {total_wav_numbers} most recent WAV files.")

async def async_clean_wav_files(output_dir: str = "outputs", total_wav_numbers: int = 5) -> None:
    """
    Async version of clean_wav_files that runs the cleanup in a separate thread.
    
    Args:
        output_dir: Directory containing WAV files (default: "outputs")
        total_wav_numbers: Maximum number of WAV files to keep (default: 5)
    """
    # Run the synchronous cleanup function in a thread to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: clean_wav_files(output_dir, total_wav_numbers))

if __name__ == "__main__":
    clean_wav_files()
