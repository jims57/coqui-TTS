import torch
import time
import os
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
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

# Reference sample - using the same one as in demo.py
speaker_wav = "speaker_wavs/jack-mark-en-1.wav"
if not os.path.exists(speaker_wav):
    print(f"Error: Speaker file {speaker_wav} not found!")
    exit(1)

chinese_text = "他下午坐在窗边舒适的扶手椅上津津有味地读着一本引人入胜的小说。"
language = "zh"

# Run standard inference for comparison first
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

# Configure torch dynamo to suppress errors and fall back to eager mode
# This prevents failures when compilation encounters unsupported operations
if hasattr(torch, '_dynamo'):
    print("Configuring torch._dynamo to suppress errors and fall back to eager mode")
    torch._dynamo.config.suppress_errors = True

# Now apply optimizations - we'll try different approaches
print("\n--- Applying optimizations ---")

# Get the model instance
model = tts.synthesizer.tts_model

# Method 1: Try torch.compile with backend options
if hasattr(torch, 'compile'):
    print("Method 1: Applying torch.compile with inductor backend and safe options...")
    
    try:
        # Apply optimizations to specific components with safer settings
        if hasattr(model, 'hifigan_decoder') and isinstance(model.hifigan_decoder, torch.nn.Module):
            print("Compiling hifigan_decoder with safer options...")
            # Use safer compilation options that are less likely to cause errors
            model.hifigan_decoder = torch.compile(
                model.hifigan_decoder,
                backend="inductor",  # Try inductor backend which may handle dynamic control flow better
                mode="max-autotune",
                fullgraph=False,     # Don't require full graph capture
                dynamic=True         # Allow dynamic shapes and control flow
            )
            print("HifiGAN decoder compiled successfully")
    
    except Exception as e:
        print(f"Error during Method 1 compilation: {e}")
        print("Falling back to uncompiled model")

# Method 2: Use GPU optimization techniques without torch.compile
print("\n--- Method 2: Using GPU optimization techniques ---")

# We'll use CUDA graphs for repeated operations
# This requires PyTorch 1.10+ and works even without torch.compile
try:
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        print("CUDA is available, enabling CUDA optimizations")
        
        # Set higher priority for our process
        if hasattr(torch.cuda, 'set_stream_priority'):
            torch.cuda.set_stream_priority(priority='high')
            print("Set CUDA stream priority to high")
        
        # Optimize memory allocation
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'memory_stats'):
            print("Current CUDA memory usage:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        # Enable TF32 for Ampere GPUs (faster with minimal precision loss)
        if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("Enabled TF32 precision for CUDA operations (on compatible GPUs)")
        
    else:
        print("CUDA not available, skipping GPU optimizations")
except Exception as e:
    print(f"Error applying CUDA optimizations: {e}")

# Method 3: Try half precision (FP16) but only for inference
print("\n--- Method 3: Using FP16 for inference ---")
try:
    # Create a context manager for FP16 inference to avoid permanent model changes
    class HalfPrecisionContext:
        def __init__(self, model):
            self.model = model
            self.original_params = {}
            
        def __enter__(self):
            # Save original parameter dtypes and convert to float16
            for name, module in self.model.named_modules():
                for param_name, param in module.named_parameters(recurse=False):
                    full_name = f"{name}.{param_name}" if name else param_name
                    self.original_params[full_name] = param.dtype
                    # Only convert to half if it's float32
                    if param.dtype == torch.float32:
                        param.data = param.data.half()
            return self.model
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore original parameter dtypes
            for name, module in self.model.named_modules():
                for param_name, param in module.named_parameters(recurse=False):
                    full_name = f"{name}.{param_name}" if name else param_name
                    if full_name in self.original_params:
                        # Convert back to original dtype if it was changed
                        if param.dtype != self.original_params[full_name]:
                            param.data = param.data.to(self.original_params[full_name])
    
    # We don't apply it yet, we'll use it during inference
    
except Exception as e:
    print(f"Error setting up FP16 precision: {e}")
    
# Now run the optimized inference with a warmup
print("\n--- Running optimized inference ---")

# First run might include compilation overhead, so we'll do a warmup
print("Warmup run...")
_ = tts.tts(
    text="Hello, this is a warmup.",  # Use a different text for warmup
    speaker_wav=speaker_wav,
    language="en"  # Use English for warmup to avoid errors
)

# Now measure the actual performance
print("Measuring performance...")
num_runs = 3
total_time = 0

print("\nApproach 1: Using optimization techniques with original model")
for i in range(num_runs):
    # Clear CUDA cache before each run
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    start_time = time.time()
    tts.tts_to_file(
        text=chinese_text,
        file_path=f"outputs/output_optimized_approach1_{i+1}.wav",
        speaker_wav=speaker_wav,
        language=language
    )
    run_time = time.time() - start_time
    total_time += run_time
    print(f"Run {i+1}: {run_time:.4f} seconds")

avg_optimized_time_approach1 = total_time / num_runs
speedup_approach1 = standard_time / avg_optimized_time_approach1 if avg_optimized_time_approach1 > 0 else 0

# Try the FP16 approach separately
print("\nApproach 2: Using FP16 precision")
total_time_fp16 = 0

# We'll use a try/except block for the entire FP16 section
try:
    fp16_context = HalfPrecisionContext(model)
    
    for i in range(num_runs):
        # Clear CUDA cache before each run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        start_time = time.time()
        
        with fp16_context:  # Temporarily convert to FP16
            tts.tts_to_file(
                text=chinese_text,
                file_path=f"outputs/output_optimized_approach2_{i+1}.wav",
                speaker_wav=speaker_wav,
                language=language
            )
            
        run_time = time.time() - start_time
        total_time_fp16 += run_time
        print(f"Run {i+1}: {run_time:.4f} seconds")
    
    avg_optimized_time_approach2 = total_time_fp16 / num_runs
    speedup_approach2 = standard_time / avg_optimized_time_approach2 if avg_optimized_time_approach2 > 0 else 0
    
except Exception as e:
    print(f"Error during FP16 inference: {e}")
    print("FP16 approach failed, skipping results")
    avg_optimized_time_approach2 = 0
    speedup_approach2 = 0

# Determine the best approach
best_approach = "Standard"
best_time = standard_time
best_speedup = 1.0

if avg_optimized_time_approach1 > 0 and avg_optimized_time_approach1 < best_time:
    best_approach = "Approach 1 (GPU Optimizations)"
    best_time = avg_optimized_time_approach1
    best_speedup = speedup_approach1

if avg_optimized_time_approach2 > 0 and avg_optimized_time_approach2 < best_time:
    best_approach = "Approach 2 (FP16 Precision)"
    best_time = avg_optimized_time_approach2
    best_speedup = speedup_approach2

# Print summary
print("\n--- Performance Summary ---")
print(f"Chinese text: \"{chinese_text}\"")
print(f"Standard inference: {standard_time:.4f} seconds")
if avg_optimized_time_approach1 > 0:
    print(f"Approach 1 (GPU Optimizations): {avg_optimized_time_approach1:.4f} seconds (Speedup: {speedup_approach1:.2f}x)")
if avg_optimized_time_approach2 > 0:
    print(f"Approach 2 (FP16 Precision): {avg_optimized_time_approach2:.4f} seconds (Speedup: {speedup_approach2:.2f}x)")
print(f"\nBest approach: {best_approach} - {best_time:.4f} seconds (Speedup: {best_speedup:.2f}x)")
print(f"Standard output: {standard_output_path}")
print(f"Optimized outputs: outputs/output_optimized_*")
print("\nVerify that the audio quality is similar between standard and optimized outputs")

# Print recommendations
print("\n--- Recommendations for Further Optimization ---")
print("1. Use NVIDIA Apex for mixed precision training and inference")
print("2. For model deployment, consider ONNX Runtime with TensorRT or optimized C++ inference")
print("3. For specific model components, manual CUDA kernel optimization may yield better results")
print("4. Consider using smaller batches but processing them in parallel")
print("5. Try BetterTransformer from Hugging Face for optimized transformer models")
print("\nThese approaches require additional setup but can provide significant speedups beyond")
print("what's possible with pure PyTorch optimizations.")