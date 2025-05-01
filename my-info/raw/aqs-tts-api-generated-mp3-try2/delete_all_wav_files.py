import os
import glob

def delete_all_wav_files():
    """
    Deletes all .wav files in the current directory.
    """
    # Get all .wav files in the current directory
    wav_files = glob.glob("*.wav")
    
    if not wav_files:
        print("No .wav files found in the current directory.")
        return False
    
    print(f"Found {len(wav_files)} wav files to delete.")
    
    # Delete each wav file
    for file in wav_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    
    # Verify all files were deleted
    remaining_files = glob.glob("*.wav")
    if not remaining_files:
        print("All wav files successfully deleted.")
        return True
    else:
        print(f"Warning: {len(remaining_files)} wav files could not be deleted.")
        return False

def main():
    delete_all_wav_files()

if __name__ == "__main__":
    main()
