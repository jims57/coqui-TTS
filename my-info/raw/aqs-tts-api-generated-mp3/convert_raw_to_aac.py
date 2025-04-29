import os
import subprocess
import glob2

def convert_raw_to_mp3(raw_file, mp3_file):
    """Convert a raw audio file to MP3 format using FFmpeg."""
    try:
        # Run FFmpeg command to convert raw to MP3
        subprocess.run(
            [
                "ffmpeg",
                "-f", "s16le",      # 16-bit PCM input
                "-ar", "24000",     # Sample rate - XTTS uses 24kHz
                "-ac", "1",         # Mono
                "-i", raw_file,     # Input file
                "-c:a", "libmp3lame",
                "-b:a", "128k",     # MP3 bitrate
                "-q:a", "2",        # Quality setting - lower is better
                "-joint_stereo", "0", # Follow MP3 standard for better compatibility
                mp3_file,           # Output file
                "-y"                # Overwrite if exists
            ],
            check=False,
            capture_output=True
        )
        return True
    except Exception as e:
        print(f"Error converting {raw_file} to MP3: {e}")
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
        # Create output filename (replace .raw with .mp3)
        mp3_file = os.path.splitext(raw_file)[0] + ".mp3"
        
        print(f"Converting {raw_file} to {mp3_file}...")
        
        if convert_raw_to_mp3(raw_file, mp3_file):
            print(f"Successfully converted {raw_file} to {mp3_file}")
            successful += 1
        else:
            print(f"Failed to convert {raw_file}")
            failed += 1
    
    print(f"\nConversion complete. Successfully converted {successful} files, failed {failed} files.")

if __name__ == "__main__":
    main()
