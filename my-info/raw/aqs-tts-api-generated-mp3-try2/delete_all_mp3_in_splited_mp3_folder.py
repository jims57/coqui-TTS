import os
import glob
import re

def natural_sort_key(s):
    """
    Sort strings with numbers in a natural way.
    For example: '1', '2', '10' instead of '1', '10', '2'
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def delete_all_mp3_files_in_splited_mp3():
    """
    Deletes all .mp3 files in the splited_mp3 directory without confirmation.
    """
    # Define the directory
    mp3_dir = "splited_mp3"
    
    # Check if directory exists
    if not os.path.exists(mp3_dir):
        print(f"Directory '{mp3_dir}' does not exist.")
        return False
    
    # Get all .mp3 files in the splited_mp3 directory
    mp3_files = glob.glob(os.path.join(mp3_dir, "*.mp3"))
    
    if not mp3_files:
        print(f"No .mp3 files found in the '{mp3_dir}' directory.")
        return False
    
    # Sort files naturally for display purposes
    mp3_files.sort(key=natural_sort_key)
    
    print(f"Found {len(mp3_files)} mp3 files to delete in '{mp3_dir}'.")
    print(f"Files will be deleted in this order: {mp3_files}")
    
    # Delete each mp3 file
    deleted_count = 0
    for file in mp3_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    
    # Verify all files were deleted
    remaining_files = glob.glob(os.path.join(mp3_dir, "*.mp3"))
    if not remaining_files:
        print("All mp3 files successfully deleted.")
    else:
        print(f"Warning: {len(remaining_files)} mp3 files could not be deleted.")
    
    print(f"Successfully deleted {deleted_count} out of {len(mp3_files)} mp3 files.")
    return deleted_count > 0

def main():
    print("Starting deletion of mp3 files in splited_mp3 folder...")
    if delete_all_mp3_files_in_splited_mp3():
        print("MP3 files deleted successfully!")
    else:
        print("No MP3 files were deleted.")

if __name__ == "__main__":
    main()
