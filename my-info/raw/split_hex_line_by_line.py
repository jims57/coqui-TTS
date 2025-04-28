import os
import glob

def process_mp3_txt_files():
    # Find all files with mp3.txt suffix
    mp3_txt_files = glob.glob("*mp3.txt")
    
    if not mp3_txt_files:
        print("No .mp3.txt files found in the current directory.")
        return
    
    print(f"Found {len(mp3_txt_files)} .mp3.txt files to process:")
    
    processed = 0
    
    for file_path in mp3_txt_files:
        try:
            print(f"Processing {file_path}...")
            
            # Read the file content
            with open(file_path, 'r') as f:
                content = f.read().strip()
            
            # Split the content by spaces
            hex_values = content.split()
            
            # Join with newlines
            new_content = '\n'.join(hex_values)
            
            # Write back to the file
            with open(file_path, 'w') as f:
                f.write(new_content)
            
            print(f"Successfully processed {file_path}")
            processed += 1
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"\nProcessing complete. Successfully processed {processed} out of {len(mp3_txt_files)} files.")

if __name__ == "__main__":
    process_mp3_txt_files()