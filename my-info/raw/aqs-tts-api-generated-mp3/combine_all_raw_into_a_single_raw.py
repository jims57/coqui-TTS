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
    
    # Group files by their timestamp prefix
    file_groups = {}
    for file in raw_files:
        # Extract timestamp and chunk number using regex
        match = re.match(r'(\d+)-(\d+)\.raw', file)
        if match:
            timestamp, chunk_num = match.groups()
            if timestamp not in file_groups:
                file_groups[timestamp] = []
            file_groups[timestamp].append(file)
    
    if not file_groups:
        print("No properly formatted raw files found (expected format: timestamp-number.raw)")
        return False
    
    # Process the most recent timestamp group (or you can choose a specific one)
    if len(file_groups) > 1:
        print(f"Found {len(file_groups)} different timestamp groups.")
        for ts in file_groups:
            print(f"Timestamp: {ts}, Files: {len(file_groups[ts])}")
        
        # Sort timestamps and use the most recent one
        timestamps = sorted(file_groups.keys())
        selected_timestamp = timestamps[-1]
        print(f"Using the most recent timestamp: {selected_timestamp}")
    else:
        selected_timestamp = list(file_groups.keys())[0]
    
    # Sort files in natural order
    files_to_combine = sorted(file_groups[selected_timestamp], key=natural_sort_key)
    
    print(f"Found {len(files_to_combine)} .raw files to combine:")
    for file in files_to_combine:
        print(f"  {file}")
    
    # Combine the files
    with open(output_file, 'wb') as outfile:
        for file in files_to_combine:
            with open(file, 'rb') as infile:
                outfile.write(infile.read())
    
    print(f"Successfully combined {len(files_to_combine)} files into {output_file}")
    return True

def main():
    # Create output filename with timestamp from the input files
    output_file = "1745912332914-full.raw"
    
    # Combine the raw files
    if combine_raw_files(output_file):
        print(f"Combined raw file saved as: {output_file}")
        
        # Optional: Convert the combined raw file to WAV
        try:
            import subprocess
            wav_file = output_file.replace('.raw', '.wav')
            subprocess.run([
                "ffmpeg",
                "-f", "s16le",      # 16-bit PCM input
                "-ar", "24000",     # Sample rate - XTTS uses 24kHz
                "-ac", "1",         # Mono
                "-i", output_file,  # Input file
                wav_file,           # Output file
                "-y"                # Overwrite if exists
            ], check=True)
            print(f"Converted combined raw to WAV: {wav_file}")
        except Exception as e:
            print(f"Error converting to WAV: {e}")
            print("You may need to install ffmpeg to convert to WAV format.")

if __name__ == "__main__":
    main()
