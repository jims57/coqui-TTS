import os
import glob
import re
from pydub import AudioSegment

def natural_sort_key(s):
    """
    Sort strings with numbers in a natural way.
    For example: '1', '2', '10' instead of '1', '10', '2'
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def combine_mp3_files(output_file="combined.mp3"):
    """
    Combines all .mp3 files in the current directory into a single mp3 file.
    Files are combined in numerical order based on their filenames.
    """
    # Get all .mp3 files in the current directory
    mp3_files = glob.glob("*.mp3")
    
    if not mp3_files:
        print("No .mp3 files found in the current directory.")
        return False
    
    # Filter out files that match the segment pattern
    segment_files = [f for f in mp3_files if re.match(r'\d+-full_segment_\d+\.mp3', f)]
    
    if not segment_files:
        print("No segment mp3 files found in the current directory.")
        return False
    
    # Extract timestamp from the first file to use in output filename
    match = re.match(r'(\d+)-full_segment_\d+\.mp3', segment_files[0])
    if match:
        timestamp = match.group(1)
        output_file = f"{timestamp}-full.mp3"
    
    # Sort files naturally to ensure correct order (1, 2, 10 instead of 1, 10, 2)
    segment_files.sort(key=natural_sort_key)
    
    print(f"Found {len(segment_files)} mp3 files to combine.")
    print(f"Files will be combined in this order: {segment_files}")
    
    # Combine the mp3 files
    combined = AudioSegment.empty()
    for file in segment_files:
        print(f"Adding {file} to combined mp3...")
        audio = AudioSegment.from_mp3(file)
        combined += audio
    
    # Export the combined mp3
    combined.export(output_file, format="mp3")
    print(f"Successfully combined {len(segment_files)} files into {output_file}")
    return True

def main():
    if combine_mp3_files():
        print("MP3 files combined successfully!")
    else:
        print("Failed to combine MP3 files.")

if __name__ == "__main__":
    main()
