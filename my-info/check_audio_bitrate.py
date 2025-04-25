from tinytag import TinyTag
import os
import argparse
import wave
import contextlib
import subprocess

def check_audio_bitrate(audio_path):
    # Expand user path if it contains ~
    audio_file_path = os.path.expanduser(audio_path)
    
    if os.path.exists(audio_file_path):
        try:
            # Get file extension
            file_ext = os.path.splitext(audio_file_path)[1].lower()
            
            # For WAV files, calculate bitrate from file properties
            if file_ext == '.wav':
                with contextlib.closing(wave.open(audio_file_path, 'r')) as wf:
                    channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                    frame_rate = wf.getframerate()
                    n_frames = wf.getnframes()
                    
                    # Calculate duration in seconds
                    duration = n_frames / float(frame_rate)
                    
                    # Calculate bitrate: bits per sample * sample rate * channels
                    bitrate = (sample_width * 8 * frame_rate * channels) / 1000
                    
                    print(f"The bitrate of '{audio_file_path}' is: {bitrate:.2f} kbps")
                    print(f"Format: WAV, Channels: {channels}, Sample Rate: {frame_rate} Hz, Bit Depth: {sample_width*8} bits")
            else:
                # Try using TinyTag first
                tag = TinyTag.get(audio_file_path)
                bitrate = tag.bitrate
                
                # If TinyTag couldn't determine the bitrate, use ffprobe as fallback
                if bitrate is None:
                    try:
                        # Use ffprobe to get bitrate information
                        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', 
                               '-show_entries', 'stream=bit_rate', '-of', 
                               'default=noprint_wrappers=1:nokey=1', audio_file_path]
                        
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        if result.stdout.strip():
                            # Convert from bits/s to kbps
                            bitrate = float(result.stdout.strip()) / 1000
                        else:
                            # If bitrate not in stream info, calculate from file size and duration
                            file_size = os.path.getsize(audio_file_path) * 8  # Size in bits
                            if tag.duration:
                                bitrate = file_size / (tag.duration * 1000)  # kbps
                            else:
                                bitrate = "Unknown"
                    except Exception as e:
                        print(f"Error using ffprobe: {e}")
                        bitrate = "Unknown"
                
                # Determine format from extension
                format_name = file_ext.strip('.').upper()
                
                print(f"The bitrate of '{audio_file_path}' is: {bitrate} kbps")
                print(f"Format: {format_name}, Duration: {tag.duration:.2f} seconds" if tag.duration else f"Format: {format_name}, Duration: Unknown")
                
            print("The file exists, and metadata was read successfully.")
            
        except Exception as e:
            print(f"Error reading metadata from '{audio_file_path}': {e}")
            print(f"Make sure the file is a valid audio file (WAV, AAC, or Opus).")
    else:
        print(f"Error: The file '{audio_file_path}' does not exist.")
        print("Please double-check the file path.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check audio file bitrate")
    parser.add_argument("--audioPath", type=str, default="~/TTS/outputs/1745587790532-full.aac",
                        help="Path to the audio file (supports WAV, AAC, Opus)")
    
    args = parser.parse_args()
    check_audio_bitrate(args.audioPath)