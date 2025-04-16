import torch
from TTS.api import TTS
import time
import os

# avaliable languages:
# ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi']

# Model list:
# tts_models/zh-CN/baker/tacotron2-DDC-GST

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    try:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
        print(f"Current GPU device: {torch.cuda.current_device()}")
        device = "cuda"
    except Exception as e:
        print(f"Error initializing CUDA: {e}")
        print("Falling back to CPU")
        device = "cpu"
else:
    device = "cpu"

print(f"Using device: {device}")	

# List available 🐸TTS models
print(TTS().list_models())

# ==== Example: [Multi-lingual] xtts_v2 ==
# Init TTS
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
tts = TTS(model_name).to(device)

# Enable Flash Attention via environmental variable if on CUDA 
if device == "cuda":
    # Set environment variables to enable memory-efficient attention
    os.environ["PYTORCH_ENABLE_MEM_EFFICIENT_ATTENTION"] = "1"
    os.environ["PYTORCH_ENABLE_FLASH_ATTENTION"] = "1"
    
    # These variables need to be set before the model is loaded
    print("Memory-efficient attention and Flash Attention enabled through environment variables")
    
    # Additionally, try to use mixed precision for faster inference
    if hasattr(torch.cuda, "amp") and torch.cuda.is_available():
        print("Using mixed precision for faster inference")
        # We'll use this context in our benchmark function
    
print(tts.languages)
print('=== speakers ===')
# print(tts.speakers)

# Function to benchmark TTS speed
def benchmark_tts(text, speaker_wav, language, output_file, num_runs=1, use_amp=False):
    print(f"\nBenchmarking TTS for language: {language}")
    
    total_time = 0
    for i in range(num_runs):
        start_time = time.time()
        
        # Run inference with optional mixed precision
        if use_amp and device == "cuda" and hasattr(torch.cuda, "amp"):
            with torch.cuda.amp.autocast():
                tts.tts_to_file(
                    text=text, 
                    speaker_wav=speaker_wav, 
                    language=language, 
                    file_path=output_file
                )
        else:
            tts.tts_to_file(
                text=text, 
                speaker_wav=speaker_wav, 
                language=language, 
                file_path=output_file
            )
        
        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time
        print(f"Run {i+1}/{num_runs}: {inference_time:.2f} seconds")
    
    avg_time = total_time / num_runs
    print(f"Average inference time: {avg_time:.2f} seconds")
    return avg_time

# Additional optimizations to try
def set_torch_optimizations():
    if device == "cuda":
        # Set torch deterministic to False for better performance
        torch.backends.cudnn.deterministic = False
        # Let cuDNN benchmark for optimal algorithms
        torch.backends.cudnn.benchmark = True
        print("CuDNN benchmarking enabled for optimal performance")

# Apply torch optimizations
set_torch_optimizations()

# Test samples in different languages
test_samples = [
    {
        "language": "en",
        "text": "The sun sets behind the mountains, casting long shadows across the valley.",
        "speaker_wav": "speaker_wavs/jack-mark-en-1.wav",
        "output_file": "output_en.wav"
    },
    {
        "language": "zh",
        "text": "他下午坐在窗边舒适的扶手椅上津津有味地读着一本引人入胜的小说。",
        "speaker_wav": "speaker_wavs/jack-mark-en-1.wav",
        "output_file": "output_zh.wav"
    }
]

# Run benchmarks
for sample in test_samples:
    # First run without AMP
    print("\nRunning without mixed precision:")
    benchmark_tts(
        text=sample["text"],
        speaker_wav=sample["speaker_wav"],
        language=sample["language"],
        output_file=sample["output_file"],
        num_runs=1,
        use_amp=False
    )
    
    # Then run with AMP if available
    if device == "cuda" and hasattr(torch.cuda, "amp"):
        print("\nRunning with mixed precision:")
        benchmark_tts(
            text=sample["text"],
            speaker_wav=sample["speaker_wav"],
            language=sample["language"],
            output_file=f"amp_{sample['output_file']}",
            num_runs=1,
            use_amp=True
        )

# # Original TTS call (for reference)
# tts.tts_to_file(text="他下午坐在窗边舒适的扶手椅上津津有味地读着一本引人入胜的小说。", speaker_wav="speaker_wavs/jack-mark-en-1.wav", language="zh", file_path="output.wav")
