import os
import glob
import re

def natural_sort_key(s):
    """
    Sort strings with numbers in a natural way.
    For example: '1', '2', '10' instead of '1', '10', '2'
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def delete_raw_files():
    """
    Deletes all .raw files in the current directory.
    """
    # Get all .raw files in the current directory
    raw_files = glob.glob("*.raw")
    
    if not raw_files:
        print("No .raw files found in the current directory.")
        return False
    
    # Sort files naturally for display purposes
    raw_files.sort(key=natural_sort_key)
    
    print(f"Found {len(raw_files)} raw files to delete.")
    print(f"Files will be deleted in this order: {raw_files}")
    
    # Delete each raw file
    deleted_count = 0
    for file in raw_files:
        try:
            os.remove(file)
            print(f"Deleted {file}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    
    print(f"Successfully deleted {deleted_count} out of {len(raw_files)} raw files.")
    return deleted_count > 0

def main():
    print("Starting raw file deletion process...")
    
    if delete_raw_files():
        print("Raw files deleted successfully!")
    else:
        print("No raw files were deleted.")

if __name__ == "__main__":
    main()
