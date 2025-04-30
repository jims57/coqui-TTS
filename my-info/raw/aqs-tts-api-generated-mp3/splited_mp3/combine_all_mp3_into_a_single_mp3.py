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
    
    # Sort files naturally to ensure correct order (1, 2, 10 instead of 1, 10, 2)
    mp3_files.sort(key=natural_sort_key)
    
    print(f"Found {len(mp3_files)} mp3 files to combine.")
    print(f"Files will be combined in this order: {mp3_files}")
    
    # Combine the mp3 files
    combined = AudioSegment.empty()
    for file in mp3_files:
        print(f"Adding {file} to combined mp3...")
        audio = AudioSegment.from_mp3(file)
        combined += audio
    
    # Export the combined mp3
    combined.export(output_file, format="mp3", bitrate="128k", parameters=["-q:a", "2", "-joint_stereo", "0"])
    print(f"Successfully combined {len(mp3_files)} files into {output_file}")
    return True

def main():
    # Create output filename
    output_file = "combined_output.mp3"
    
    if combine_mp3_files(output_file):
        print("MP3 files combined successfully!")
    else:
        print("Failed to combine MP3 files.")

if __name__ == "__main__":
    main()
