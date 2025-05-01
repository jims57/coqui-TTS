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

def convert_raw_to_wav(raw_file, sample_rate=24000, channels=1, bit_depth=16):
    """Convert a raw audio file to WAV format."""
    print(f"Converting {raw_file} to WAV...")
    
    # Load the raw audio data
    with open(raw_file, 'rb') as f:
        raw_data = f.read()
    
    # Create a WAV file using pydub
    raw_audio = AudioSegment(
        data=raw_data,
        sample_width=bit_depth // 8,
        frame_rate=sample_rate,
        channels=channels
    )
    
    # Create output filename
    output_file = os.path.splitext(raw_file)[0] + ".wav"
    
    # Export as WAV
    raw_audio.export(output_file, format="wav")
    
    print(f"Successfully converted to {output_file}")
    return output_file

def find_full_raw_files():
    """Find all raw files with 'full' in their filename."""
    raw_files = glob.glob("*full*.raw")
    
    if not raw_files:
        print("No raw files with 'full' in their name found in the current directory.")
        return []
    
    # Sort files naturally
    raw_files.sort(key=natural_sort_key)
    return raw_files

def main():
    # Find all raw files with 'full' in their filename
    raw_files = find_full_raw_files()
    
    if not raw_files:
        print("No files to convert.")
        return
    
    print(f"Found {len(raw_files)} raw files to convert: {raw_files}")
    
    # Convert each raw file to WAV
    for raw_file in raw_files:
        convert_raw_to_wav(raw_file)
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()
