import os
import re
from pydub import AudioSegment

def combine_aac_files(directory):
    # Get all .aac files in the directory
    aac_files = [f for f in os.listdir(directory) if f.endswith('.aac')]
    
    # Group files by their prefix
    file_groups = {}
    for file in aac_files:
        # Extract prefix and sequence number using regex
        match = re.match(r'(\d+)-(\d+)\.aac', file)
        if match:
            prefix, number = match.groups()
            if prefix not in file_groups:
                file_groups[prefix] = []
            file_groups[prefix].append((int(number), file))
    
    # Process each group
    for prefix, files in file_groups.items():
        # Sort files by sequence number
        files.sort(key=lambda x: x[0])
        
        if not files:
            continue
        
        # Create the combined audio
        combined = AudioSegment.empty()
        for _, filename in files:
            file_path = os.path.join(directory, filename)
            audio = AudioSegment.from_file(file_path, format="aac")
            combined += audio
        
        # Export the combined audio - use m4a container with aac codec
        output_file = os.path.join(directory, f"{prefix}-full.aac")
        combined.export(output_file, format="ipod", codec="aac")
        print(f"Created combined file: {output_file}")

if __name__ == "__main__":
    # Use the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    combine_aac_files(current_dir)
