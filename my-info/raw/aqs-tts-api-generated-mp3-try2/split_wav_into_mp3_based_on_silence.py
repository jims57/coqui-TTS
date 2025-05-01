import os
import io
import subprocess
import numpy as np
import argparse
from pydub import AudioSegment
from pydub.silence import split_on_silence

def convert_raw_to_wav(raw_file, sample_rate=24000, channels=1, bit_depth=16):
    """Convert a raw audio file to WAV format for processing."""
    # Load the raw audio data
    with open(raw_file, 'rb') as f:
        raw_data = f.read()
    
    # Create an in-memory WAV file using pydub
    raw_audio = AudioSegment(
        data=raw_data,
        sample_width=bit_depth // 8,
        frame_rate=sample_rate,
        channels=channels
    )
    
    return raw_audio

def split_audio_on_silence(audio, min_silence_len=700, silence_thresh=-40, keep_silence=300):
    """Split audio into chunks based on silence detection."""
    print(f"Splitting audio with parameters: min_silence_len={min_silence_len}ms, silence_thresh={silence_thresh}dBFS, keep_silence={keep_silence}ms")
    
    # Split the audio on silence
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,    # Minimum silence duration in milliseconds
        silence_thresh=silence_thresh,      # Threshold in dBFS
        keep_silence=keep_silence           # Keep some silence at segment edges
    )
    
    return chunks

def export_chunks_to_mp3(chunks, output_dir, base_filename, sample_rate=24000, bitrate="128k"):
    """Export audio chunks to MP3 files."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Exporting {len(chunks)} audio segments to MP3...")
    
    for i, chunk in enumerate(chunks):
        # Create output filename
        output_file = os.path.join(output_dir, f"{base_filename}_segment_{i+1}.mp3")
        
        # Export as MP3
        chunk.export(
            output_file,
            format="mp3",
            bitrate=bitrate,
            parameters=["-ac", "1"]  # Ensure mono output
        )
        
        duration_sec = len(chunk) / 1000.0
        print(f"  Segment {i+1}: {output_file} ({duration_sec:.2f} seconds)")
    
    return len(chunks)

def main(wavPath=None):
    # Input raw file
    raw_file = "1745912332914-full.raw"
    
    # Update raw_file if wavPath is provided
    if wavPath:
        raw_file = wavPath
    
    # Check if file exists
    if not os.path.exists(raw_file):
        print(f"Error: File {raw_file} not found.")
        return
    
    # Output directory for MP3 segments
    output_dir = "splited_mp3"
    
    # Base filename for output segments (without extension)
    base_filename = os.path.splitext(os.path.basename(raw_file))[0]
    
    print(f"Processing {raw_file}...")
    
    # Convert raw to WAV for processing
    audio = convert_raw_to_wav(raw_file)
    print(f"Loaded audio: {len(audio)/1000:.2f} seconds, {audio.channels} channels, {audio.frame_rate}Hz")
    
    # Split audio on silence
    chunks = split_audio_on_silence(
        audio,
        min_silence_len=80,     # Detect phrase-level pauses
        silence_thresh=-28,     # Moderate sensitivity
        keep_silence=60         # Keep natural pauses
    )
    
    if not chunks:
        print("No segments were detected. Try adjusting the silence detection parameters.")
        return
    
    # Export chunks to MP3
    num_segments = export_chunks_to_mp3(chunks, output_dir, base_filename)
    
    print(f"\nProcessing complete. Split {raw_file} into {num_segments} MP3 segments in '{output_dir}' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split audio file into segments based on silence')
    parser.add_argument('--wavPath', type=str, help='Path to the raw audio file')
    args = parser.parse_args()
    
    main(args.wavPath)
