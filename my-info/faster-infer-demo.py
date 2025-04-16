import torch
import time
import os
from TTS.api import TTS

# Get device - make sure we use the correct format for CUDA device
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Specifically use the first GPU
    # Configure PyTorch for maximum performance
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster computation
    torch.backends.cudnn.benchmark = True  # Use cuDNN benchmarking for faster convolutions
    torch.backends.cudnn.allow_tf32 = True  # Allow TF32 in cuDNN as well
    
    # Enable flash attention if available (for PyTorch 2.0+)
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
        print("Enabled Flash Attention")
    
    # Enable memory-efficient attention
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        print("Enabled memory-efficient attention")
else:
    device = torch.device("cpu")
    
print(f"PyTorch version: {torch.__version__}")
print(f"Using device: {device}")

# Create output directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Initialize TTS model
print("Loading XTTS v2 model...")
start_time = time.time()
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
load_time = time.time() - start_time
print(f"Model loading time: {load_time:.2f} seconds")

# Reference sample
speaker_wav = "speaker_wavs/jack-mark-en-1.wav"
if not os.path.exists(speaker_wav):
    print(f"Error: Speaker file {speaker_wav} not found!")
    exit(1)

chinese_text = "青石板上泛着水光，雨丝斜斜地织着帘子。我撑一把油纸伞，踩着湿润的石板路，听脚步声在巷子里轻轻回响。"
language = "zh"

# Get the model instance
model = tts.synthesizer.tts_model

# Helper function to ensure tensors stay on GPU
def ensure_gpu(model):
    """
    Ensures all model operations keep tensors on GPU whenever possible.
    """
    def force_cuda_output_hook(module, input, output):
        if isinstance(output, torch.Tensor) and not output.is_cuda:
            return output.to(device)
        return output
    
    # Apply recursively to all submodules
    for module in model.modules():
        module.register_forward_hook(force_cuda_output_hook)
    
    return model

# Apply GPU optimization to model
model = ensure_gpu(model)

# Run standard inference for comparison
print("\n--- Running standard inference ---")
standard_output_path = "outputs/output_standard.wav"
start_time = time.time()
tts.tts_to_file(
    text=chinese_text,
    file_path=standard_output_path,
    speaker_wav=speaker_wav,
    language=language
)
standard_time = time.time() - start_time
print(f"Standard inference time: {standard_time:.4f} seconds")

# Now run optimized approaches
print("\n--- Running optimized inference ---")

# First run might include compilation overhead, so we'll do a warmup
print("Warmup run...")
with torch.no_grad():
    _ = tts.tts(
        text="Hello, this is a warm-up run.",
        speaker_wav=speaker_wav,
        language="en"
    )

# Define the number of benchmark runs
num_runs = 3

# Approach 1: Using torch.compile for the vocoder
print("\nApproach 1: Using torch.compile for vocoder acceleration")
total_time_compiled = 0

if hasattr(torch, 'compile'):
    try:
        # Save the original vocoder
        original_decoder = model.hifigan_decoder
        
        # Compile the vocoder with optimized settings
        print("Compiling HiFiGAN decoder...")
        compiled_decoder = torch.compile(
            model.hifigan_decoder,
            backend="inductor",
            mode="reduce-overhead",
            fullgraph=False,
            dynamic=True
        )
        
        # Replace with compiled version
        model.hifigan_decoder = compiled_decoder
        print("Successfully compiled HiFiGAN decoder")
        
        # Run benchmark
        for i in range(num_runs):
            torch.cuda.empty_cache()
            start_time = time.time()
            with torch.no_grad():
                tts.tts_to_file(
                    text=chinese_text,
                    file_path=f"outputs/output_compiled_{i+1}.wav",
                    speaker_wav=speaker_wav,
                    language=language
                )
            run_time = time.time() - start_time
            total_time_compiled += run_time
            print(f"Run {i+1}: {run_time:.4f} seconds")
        
        # Restore original
        model.hifigan_decoder = original_decoder
    
    except Exception as e:
        print(f"Error with torch.compile approach: {e}")
        total_time_compiled = 0  # Mark as failed

avg_compiled_time = total_time_compiled / num_runs if total_time_compiled > 0 else 0
speedup_compiled = standard_time / avg_compiled_time if avg_compiled_time > 0 else 0

# Approach 2: Using Half-Precision (FP16) for the vocoder
print("\nApproach 2: Using FP16 for vocoder acceleration")
total_time_fp16 = 0

try:
    # Save original forward method
    original_forward = model.hifigan_decoder.forward
    
    # Create an FP16-accelerated version
    def fp16_forward(self, mel_tensor, g=None):
        with torch.cuda.amp.autocast():
            return original_forward(mel_tensor, g=g)
    
    # Apply the patch
    import types
    model.hifigan_decoder.forward = types.MethodType(fp16_forward, model.hifigan_decoder)
    print("Applied FP16 acceleration to HiFiGAN decoder")
    
    for i in range(num_runs):
        torch.cuda.empty_cache()
        start_time = time.time()
        with torch.no_grad():
            tts.tts_to_file(
                text=chinese_text,
                file_path=f"outputs/output_fp16_{i+1}.wav",
                speaker_wav=speaker_wav,
                language=language
            )
        run_time = time.time() - start_time
        total_time_fp16 += run_time
        print(f"Run {i+1}: {run_time:.4f} seconds")
    
    # Restore original method
    model.hifigan_decoder.forward = original_forward

except Exception as e:
    print(f"Error with FP16 approach: {e}")
    total_time_fp16 = 0  # Mark as failed

avg_fp16_time = total_time_fp16 / num_runs if total_time_fp16 > 0 else 0
speedup_fp16 = standard_time / avg_fp16_time if avg_fp16_time > 0 else 0

# Approach 3: Using TF32 for the vocoder (best in previous runs)
print("\nApproach 3: Using TF32 for vocoder acceleration")
total_time_tf32 = 0

try:
    # Save original forward method
    original_forward_tf32 = model.hifigan_decoder.forward
    
    # Create a TF32-accelerated version
    def tf32_forward(self, mel_tensor, g=None):
        # Explicitly enable TF32 just for this operation
        with torch.no_grad():
            old_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # Run with TF32 enabled
            result = original_forward_tf32(mel_tensor, g=g)
            
            # Restore previous setting
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
            return result
    
    # Apply the patch
    import types
    model.hifigan_decoder.forward = types.MethodType(tf32_forward, model.hifigan_decoder)
    print("Applied TF32 acceleration to HiFiGAN decoder")
    
    for i in range(num_runs):
        torch.cuda.empty_cache()
        start_time = time.time()
        with torch.no_grad():
            tts.tts_to_file(
                text=chinese_text,
                file_path=f"outputs/output_tf32_{i+1}.wav",
                speaker_wav=speaker_wav,
                language=language
            )
        run_time = time.time() - start_time
        total_time_tf32 += run_time
        print(f"Run {i+1}: {run_time:.4f} seconds")
    
    # Restore original method
    model.hifigan_decoder.forward = original_forward_tf32

except Exception as e:
    print(f"Error with TF32 approach: {e}")
    total_time_tf32 = 0  # Mark as failed

avg_tf32_time = total_time_tf32 / num_runs if total_time_tf32 > 0 else 0
speedup_tf32 = standard_time / avg_tf32_time if avg_tf32_time > 0 else 0

# Approach 4: Combined approach with both TF32 and FP16 
print("\nApproach 4: Combined TF32+FP16 for vocoder acceleration")
total_time_combined = 0

try:
    # Save original forward method
    original_forward_combined = model.hifigan_decoder.forward
    
    # Create a combined acceleration version
    def combined_forward(self, mel_tensor, g=None):
        # Use both TF32 and FP16 for maximum performance
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                result = original_forward_combined(mel_tensor, g=g)
        
        # Restore previous TF32 setting
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        return result
    
    # Apply the patch
    import types
    model.hifigan_decoder.forward = types.MethodType(combined_forward, model.hifigan_decoder)
    print("Applied combined TF32+FP16 acceleration to HiFiGAN decoder")
    
    for i in range(num_runs):
        torch.cuda.empty_cache()
        start_time = time.time()
        with torch.no_grad():
            tts.tts_to_file(
                text=chinese_text,
                file_path=f"outputs/output_combined_{i+1}.wav",
                speaker_wav=speaker_wav,
                language=language
            )
        run_time = time.time() - start_time
        total_time_combined += run_time
        print(f"Run {i+1}: {run_time:.4f} seconds")
    
    # Restore original method
    model.hifigan_decoder.forward = original_forward_combined

except Exception as e:
    print(f"Error with combined approach: {e}")
    total_time_combined = 0  # Mark as failed

avg_combined_time = total_time_combined / num_runs if total_time_combined > 0 else 0
speedup_combined = standard_time / avg_combined_time if avg_combined_time > 0 else 0

# Approach 5: GPU Optimizations approach (from faster-infer-demo-1.py)
print("\nApproach 5: GPU Optimizations (from original implementation)")
total_time_gpu_opt = 0

try:
    # This approach uses the same model instance but focuses on GPU-specific optimizations
    
    # First, reset any previous optimizations by clearing cache
    torch.cuda.empty_cache()
    
    # Set higher priority for our process
    if hasattr(torch.cuda, 'set_stream_priority'):
        try:
            torch.cuda.set_stream_priority(priority='high')
            print("Set CUDA stream priority to high")
        except Exception as e:
            print(f"Could not set stream priority: {e}")
    
    # Enable TF32 for Ampere GPUs (faster with minimal precision loss)
    if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Enabled TF32 precision for CUDA operations")
    
    # Run the benchmark
    for i in range(num_runs):
        torch.cuda.empty_cache()
        start_time = time.time()
        with torch.no_grad():
            tts.tts_to_file(
                text=chinese_text,
                file_path=f"outputs/output_gpu_opt_{i+1}.wav",
                speaker_wav=speaker_wav,
                language=language
            )
        run_time = time.time() - start_time
        total_time_gpu_opt += run_time
        print(f"Run {i+1}: {run_time:.4f} seconds")

except Exception as e:
    print(f"Error with GPU Optimizations approach: {e}")
    total_time_gpu_opt = 0  # Mark as failed

avg_gpu_opt_time = total_time_gpu_opt / num_runs if total_time_gpu_opt > 0 else 0
speedup_gpu_opt = standard_time / avg_gpu_opt_time if avg_gpu_opt_time > 0 else 0

# Determine the best approach
best_approach = "Standard"
best_time = standard_time
best_speedup = 1.0

approaches = [
    ("Standard Inference", standard_time, 1.0),
    ("Approach 1 (Compiled Vocoder)", avg_compiled_time, speedup_compiled),
    ("Approach 2 (FP16 Vocoder)", avg_fp16_time, speedup_fp16),
    ("Approach 3 (TF32 Vocoder)", avg_tf32_time, speedup_tf32),
    ("Approach 4 (Combined TF32+FP16 Vocoder)", avg_combined_time, speedup_combined),
    ("Approach 5 (GPU Optimizations)", avg_gpu_opt_time, speedup_gpu_opt)
]

for name, time_val, speedup in approaches:
    if time_val > 0 and time_val < best_time:
        best_approach = name
        best_time = time_val
        best_speedup = speedup

# Print performance summary
print("\n--- Performance Summary ---")
print(f"Chinese text: \"{chinese_text}\"")

for name, time_val, speedup in approaches:
    if time_val > 0:
        print(f"{name}: {time_val:.4f} seconds (Speedup: {speedup:.2f}x)")

print(f"\nBest approach: {best_approach} - {best_time:.4f} seconds (Speedup: {best_speedup:.2f}x)")
print(f"Standard output: {standard_output_path}")
print("Optimized outputs: outputs/output_*.wav")
print("\nVerify that the audio quality is similar between standard and optimized outputs")

# Add some optimization tips
print("\n--- Additional Optimization Tips ---")
print("1. For production usage, use the best performing approach from above")
print("2. You may further improve speed by using shorter speaker embeddings")
print("3. For batch processing, use batched inference rather than one-by-one processing")
print("4. Consider using a smaller model version if ultimate speed is required")
print("5. Ensure you're running on modern NVIDIA GPUs with tensor cores for best TF32 performance")