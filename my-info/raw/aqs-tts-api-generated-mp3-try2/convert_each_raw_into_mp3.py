import os
import glob
import re
import subprocess
from pydub import AudioSegment

def natural_sort_key(s):
    """
    Sort strings with numbers in a natural way.
    For example: '1', '2', '10' instead of '1', '10', '2'
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def convert_raw_to_mp3(raw_file, sample_rate=24000, channels=1, bit_depth=16, bitrate="128k"):
    """
    Convert a raw audio file to MP3 format.
    
    Args:
        raw_file: Path to the raw audio file
        sample_rate: Sample rate of the raw audio (default: 24000)
        channels: Number of audio channels (default: 1)
        bit_depth: Bit depth of the raw audio (default: 16)
        bitrate: Bitrate for the output MP3 (default: "128k")
    
    Returns:
        Path to the created MP3 file
    """
    # Create output filename by replacing .raw extension with .mp3
    mp3_file = os.path.splitext(raw_file)[0] + ".mp3"
    
    # Load the raw audio data
    with open(raw_file, 'rb') as f:
        raw_data = f.read()
    
    # Create an audio segment from the raw data
    audio = AudioSegment(
        data=raw_data,
        sample_width=bit_depth // 8,
        frame_rate=sample_rate,
        channels=channels
    )
    
    # Export as MP3
    audio.export(mp3_file, format="mp3", bitrate=bitrate)
    
    print(f"Converted {raw_file} to {mp3_file}")
    return mp3_file

def convert_all_raw_files():
    """
    Converts all .raw files in the current directory to MP3 format.
    """
    # Get all .raw files in the current directory
    raw_files = glob.glob("*.raw")
    
    if not raw_files:
        print("No .raw files found in the current directory.")
        return False
    
    # Sort files naturally for processing
    raw_files.sort(key=natural_sort_key)
    
    print(f"Found {len(raw_files)} raw files to convert.")
    
    # Convert each raw file to MP3
    converted_count = 0
    for file in raw_files:
        try:
            convert_raw_to_mp3(file)
            converted_count += 1
        except Exception as e:
            print(f"Error converting {file}: {e}")
    
    print(f"Successfully converted {converted_count} out of {len(raw_files)} raw files to MP3.")
    return converted_count > 0

def main():
    print("Starting raw to MP3 conversion process...")
    
    if convert_all_raw_files():
        print("Raw files converted to MP3 successfully!")
    else:
        print("No raw files were converted.")

if __name__ == "__main__":
    main()
