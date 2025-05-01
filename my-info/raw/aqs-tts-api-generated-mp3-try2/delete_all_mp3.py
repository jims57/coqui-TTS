import os
import glob

def delete_all_mp3_files():
    """
    Deletes all .mp3 files in the current directory.
    """
    # Get all .mp3 files in the current directory
    mp3_files = glob.glob("*.mp3")
    
    if not mp3_files:
        print("No .mp3 files found in the current directory.")
        return False
    
    print(f"Found {len(mp3_files)} mp3 files to delete.")
    
    # Delete each mp3 file
    for file in mp3_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    
    # Verify all files were deleted
    remaining_files = glob.glob("*.mp3")
    if not remaining_files:
        print("All mp3 files successfully deleted.")
        return True
    else:
        print(f"Warning: {len(remaining_files)} mp3 files could not be deleted.")
        return False

def main():
    delete_all_mp3_files()

if __name__ == "__main__":
    main()
