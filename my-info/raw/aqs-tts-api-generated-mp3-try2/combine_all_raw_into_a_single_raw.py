import os
import glob
import re

def natural_sort_key(s):
    """
    Sort strings with numbers in a natural way.
    For example: '1', '2', '10' instead of '1', '10', '2'
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def combine_raw_files(output_file="combined.raw"):
    """
    Combines all .raw files in the current directory into a single raw file.
    Files are combined in numerical order based on their filenames.
    """
    # Get all .raw files in the current directory
    raw_files = glob.glob("*.raw")
    
    if not raw_files:
        print("No .raw files found in the current directory.")
        return False
    
    # Sort files naturally to ensure correct order (1, 2, 10 instead of 1, 10, 2)
    raw_files.sort(key=natural_sort_key)
    
    print(f"Found {len(raw_files)} raw files to combine.")
    print(f"Files will be combined in this order: {raw_files}")
    
    # Combine the raw files
    with open(output_file, 'wb') as outfile:
        for file in raw_files:
            print(f"Adding {file} to combined raw...")
            with open(file, 'rb') as infile:
                outfile.write(infile.read())
    
    # Verify the combined file exists and has content
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"Successfully combined {len(raw_files)} files into {output_file}")
        return True
    else:
        print(f"Failed to create combined file {output_file}")
        return False

def main():
    # Set output filename
    output_file = "1746055652775-full.raw"
    
    if combine_raw_files(output_file):
        print("Raw files combined successfully!")
    else:
        print("Failed to combine raw files.")

if __name__ == "__main__":
    main()
