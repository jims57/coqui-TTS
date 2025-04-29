import os
import subprocess
import glob

def convert_raw_to_aac(raw_file, aac_file):
    """Convert a raw audio file to AAC format using FFmpeg."""
    try:
        # Run FFmpeg command to convert raw to AAC
        subprocess.run(
            [
                "ffmpeg",
                "-f", "s16le",      # 16-bit PCM input
                "-ar", "24000",     # Sample rate - XTTS uses 24kHz
                "-ac", "1",         # Mono
                "-i", raw_file,     # Input file
                "-c:a", "aac",
                "-b:a", "128k",     # AAC bitrate
                "-q:a", "2",        # Quality setting - lower is better
                "-joint_stereo", "0", # Follow MP3 standard for better compatibility
                aac_file,           # Output file
                "-y"                # Overwrite if exists
            ],
            check=False,
            capture_output=True
        )
        return True
    except Exception as e:
        print(f"Error converting {raw_file} to AAC: {e}")
        return False

def main():
    # Get all .raw files in the current directory
    raw_files = glob.glob("*.raw")
    
    if not raw_files:
        print("No .raw files found in the current directory.")
        return
    
    print(f"Found {len(raw_files)} .raw files to convert:")
    
    successful = 0
    failed = 0
    
    for raw_file in raw_files:
        # Create output filename (replace .raw with .aac)
        aac_file = os.path.splitext(raw_file)[0] + ".aac"
        
        print(f"Converting {raw_file} to {aac_file}...")
        
        if convert_raw_to_aac(raw_file, aac_file):
            print(f"Successfully converted {raw_file} to {aac_file}")
            successful += 1
        else:
            print(f"Failed to convert {raw_file}")
            failed += 1
    
    print(f"\nConversion complete. Successfully converted {successful} files, failed {failed} files.")

if __name__ == "__main__":
    main()
