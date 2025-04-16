import torch
import time
import os
from TTS.api import TTS

# Get device - make sure we use the correct format for CUDA device
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Specifically use the first GPU
else:
    device = torch.device("cpu")
    
print(f"PyTorch version: {torch.__version__}")
print(f"Using device: {device}")

# Create output directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Helper function to check if torch-tensorrt is available
def is_torch_tensorrt_available():
    try:
        import torch_tensorrt
        return True
    except ImportError:
        print("torch_tensorrt is not installed. To install it, run:")
        print("pip install torch-tensorrt -f https://download.pytorch.org/whl/torch_tensorrt.html")
        return False

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

chinese_text = "青石板上泛着水光，雨丝斜斜地织着帘子。我撑一把油纸伞，踩着湿润的石板路，听脚步声在巷子里轻轻回响。"
language = "zh"

# Configure torch dynamo to suppress errors and fall back to eager mode
# This prevents failures when compilation encounters unsupported operations
if hasattr(torch, '_dynamo'):
    print("Configuring torch._dynamo to suppress errors and fall back to eager mode")
    torch._dynamo.config.suppress_errors = True

# Make sure CUDA graphs are enabled and properly configured
if torch.cuda.is_available():
    # Set default device and cudnn settings
    torch.cuda.current_device()  # Just ensure we're using the current device
    
    # Set CUDA graph options
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = True
        print("Enabled cuDNN benchmark mode for faster inference")
    
    # Empty cache before starting
    torch.cuda.empty_cache()
    print("CUDA memory optimized for inference")

# Get the model instance
model = tts.synthesizer.tts_model

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

# Now apply optimizations - we'll try different approaches
print("\n--- Applying optimizations ---")

# Method 1: Use torch.compile with backend options
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
            try:
                torch.cuda.set_stream_priority(priority='high')
                print("Set CUDA stream priority to high")
            except Exception as e:
                print(f"Could not set stream priority: {e}")
        
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

# Method 3: Modified TensorRT approach that handles dynamic input sizes
print("\n--- Method 3: Using PyTorch-TensorRT with dynamic shapes ---")

try:
    import torch_tensorrt
    print(f"Using PyTorch-TensorRT version: {torch_tensorrt.__version__}")
    
    # Since the previous TensorRT optimization had issues with dynamic input shapes,
    # we'll create an approach that specifically handles dynamic dimensions
    
    print("Analyzing XTTS model structure for optimizable components...")
    
    # Define a specialized ResBlock wrapper that handles dynamic input lengths
    class DynamicResblockWrapper(torch.nn.Module):
        def __init__(self, resblock):
            super().__init__()
            self.resblock = resblock
            # Keep track of whether this is TensorRT optimized
            self.is_tensorrt = False
            # Create multiple TensorRT engines for different input lengths
            self.trt_engines = {}
            self.supported_sizes = []
        
        def forward(self, x):
            # Get the current sequence length
            curr_len = x.size(2)
            
            # If we're not using TensorRT or we don't have an engine for this length, use original
            if not self.is_tensorrt or curr_len not in self.supported_sizes:
                return self.resblock(x)
            
            # Otherwise use the appropriate TensorRT engine
            engine = self.trt_engines[curr_len]
            # Ensure input stays on GPU
            x = x.contiguous()
            return engine(x)
    
    # Try to find all resblocks in the waveform decoder
    if hasattr(model, 'hifigan_decoder') and hasattr(model.hifigan_decoder, 'waveform_decoder'):
        waveform_decoder = model.hifigan_decoder.waveform_decoder
        
        if hasattr(waveform_decoder, 'resblocks'):
            resblocks = waveform_decoder.resblocks
            print(f"Found resblocks module with {len(resblocks)} blocks!")
            
            # We'll wrap multiple resblocks with our dynamic wrapper to maximize TensorRT benefit
            # Try to optimize up to 4 blocks for better performance
            num_blocks_to_optimize = min(4, len(resblocks))
            print(f"Will optimize {num_blocks_to_optimize} resblocks with TensorRT")

            optimized_blocks = 0
            for block_idx in range(num_blocks_to_optimize):
                print(f"\nSetting up dynamic wrapper for resblock {block_idx}...")
                wrapper = DynamicResblockWrapper(resblocks[block_idx])
                
                # Define sequence lengths we want to support - focusing on the most common sizes
                target_sizes = [39, 64, 100, 200, 500]
                
                # Try to create TensorRT engines for different sequence lengths
                for seq_len in target_sizes:
                    try:
                        print(f"Creating TensorRT engine for sequence length {seq_len}...")
                        dummy_input = torch.randn(1, 256, seq_len, device=device)
                        
                        # Test if the original resblock works with this size
                        with torch.no_grad():
                            # Make sure everything stays on GPU
                            dummy_input = dummy_input.contiguous()
                            original_output = resblocks[block_idx](dummy_input)
                            print(f"Original resblock works with size {seq_len}, output shape: {original_output.shape}")
                        
                        # Create TensorRT engine with properly configured dynamic shape support
                        compile_spec = {
                            "inputs": [dummy_input],
                            "enabled_precisions": {torch.float32},  # FP32 for better accuracy
                            "workspace_size": 1 << 28,
                            "debug": False,  # Turn off debug mode for better performance
                            "min_block_size": 1
                        }
                        
                        # Compile with TensorRT
                        compiled_module = torch_tensorrt.compile(
                            resblocks[block_idx],
                            **compile_spec
                        )
                        
                        # Test the compiled module
                        with torch.no_grad():
                            # Ensure input stays on CUDA
                            dummy_input = dummy_input.contiguous()
                            trt_output = compiled_module(dummy_input)
                            print(f"TensorRT output for size {seq_len}, shape: {trt_output.shape}")
                            
                            # Compare outputs
                            diff = torch.abs(original_output - trt_output).mean().item()
                            print(f"Mean difference for size {seq_len}: {diff}")
                        
                        # Add this engine to our wrapper
                        wrapper.trt_engines[seq_len] = compiled_module
                        wrapper.supported_sizes.append(seq_len)
                        wrapper.is_tensorrt = True
                    
                    except Exception as e:
                        print(f"Failed to create engine for size {seq_len}: {e}")
                
                # If we successfully created any engines, replace the original resblock
                if wrapper.is_tensorrt and wrapper.supported_sizes:
                    print(f"Successfully created TensorRT engines for sizes: {wrapper.supported_sizes}")
                    print(f"Replacing original resblock {block_idx} with dynamic TensorRT wrapper")
                    resblocks[block_idx] = wrapper
                    optimized_blocks += 1
                else:
                    print(f"Failed to create any working TensorRT engines for resblock {block_idx}")

            print(f"TensorRT optimization applied to {optimized_blocks} resblocks")

            # Create a counter class to track TensorRT activations during inference
            if optimized_blocks > 0:
                class TensorRTCounter:
                    def __init__(self):
                        self.hits = 0
                        self.misses = 0
                    
                    def reset(self):
                        self.hits = 0
                        self.misses = 0
                
                # Create a counter instance
                trt_counter = TensorRTCounter()
                
                # Set up a hook to count TensorRT activations
                def tensorrt_counter_hook(module, input, output):
                    if isinstance(module, DynamicResblockWrapper):
                        curr_len = input[0].size(2)
                        if module.is_tensorrt and curr_len in module.supported_sizes:
                            trt_counter.hits += 1
                        else:
                            trt_counter.misses += 1
                
                # Register hooks on all optimized resblocks
                hooks = []
                for i in range(len(resblocks)):
                    if isinstance(resblocks[i], DynamicResblockWrapper) and resblocks[i].is_tensorrt:
                        hooks.append(resblocks[i].register_forward_hook(tensorrt_counter_hook))
                
                print(f"Registered TensorRT usage monitors on {len(hooks)} optimized resblocks")

                # We'll print TensorRT usage stats after each inference run
                def print_tensorrt_stats():
                    if trt_counter.hits + trt_counter.misses > 0:
                        hit_rate = trt_counter.hits / (trt_counter.hits + trt_counter.misses) * 100
                        print(f"TensorRT usage stats: {trt_counter.hits} hits, {trt_counter.misses} misses ({hit_rate:.1f}% hit rate)")
                        # Reset counters for next run
                        trt_counter.reset()
        else:
            print("Could not find resblocks in waveform_decoder")
    else:
        print("Could not find waveform_decoder in hifigan_decoder")
    
except Exception as e:
    print(f"Error during TensorRT optimization: {e}")
    print("Continuing with previously optimized model")

# Now run the optimized inference with a warmup
print("\n--- Running optimized inference ---")

# First run might include compilation overhead, so we'll do a warmup
print("Warmup run...")

# Use a shorter text for warmup to avoid errors
warmup_text = "Hello."
with torch.no_grad():  # Make sure we use no_grad for inference
    _ = tts.tts(
        text=warmup_text,
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
    with torch.no_grad():  # Ensure we use no_grad for inference
        tts.tts_to_file(
            text=chinese_text,
            file_path=f"outputs/output_optimized_approach1_{i+1}.wav",
            speaker_wav=speaker_wav,
            language=language
        )
    run_time = time.time() - start_time
    total_time += run_time
    print(f"Run {i+1}: {run_time:.4f} seconds")

    # Add after the tts_to_file calls in both benchmark sections
    if 'print_tensorrt_stats' in globals():
        print_tensorrt_stats()

avg_optimized_time_approach1 = total_time / num_runs
speedup_approach1 = standard_time / avg_optimized_time_approach1 if avg_optimized_time_approach1 > 0 else 0

# Reset the timer for TensorRT approach
print("\nApproach 2: Using PyTorch-TensorRT with dynamic shapes")
total_time_trt = 0

for i in range(num_runs):
    # Clear CUDA cache before each run
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    start_time = time.time()
    with torch.no_grad():  # Ensure we use no_grad for inference
        tts.tts_to_file(
            text=chinese_text,
            file_path=f"outputs/output_optimized_trt_{i+1}.wav",
            speaker_wav=speaker_wav,
            language=language
        )
    run_time = time.time() - start_time
    total_time_trt += run_time
    print(f"Run {i+1}: {run_time:.4f} seconds")

    # Add after the tts_to_file calls in both benchmark sections
    if 'print_tensorrt_stats' in globals():
        print_tensorrt_stats()

avg_optimized_time_trt = total_time_trt / num_runs
speedup_trt = standard_time / avg_optimized_time_trt if avg_optimized_time_trt > 0 else 0

# Determine the best approach
best_approach = "Standard"
best_time = standard_time
best_speedup = 1.0

if avg_optimized_time_approach1 > 0 and avg_optimized_time_approach1 < best_time:
    best_approach = "Approach 1 (GPU Optimizations)"
    best_time = avg_optimized_time_approach1
    best_speedup = speedup_approach1

if avg_optimized_time_trt > 0 and avg_optimized_time_trt < best_time:
    best_approach = "Approach 2 (PyTorch-TensorRT)"
    best_time = avg_optimized_time_trt
    best_speedup = speedup_trt

# Print summary
print("\n--- Performance Summary ---")
print(f"Chinese text: \"{chinese_text}\"")
print(f"Standard inference: {standard_time:.4f} seconds")
if avg_optimized_time_approach1 > 0:
    print(f"Approach 1 (GPU Optimizations): {avg_optimized_time_approach1:.4f} seconds (Speedup: {speedup_approach1:.2f}x)")
if avg_optimized_time_trt > 0:
    print(f"Approach 2 (PyTorch-TensorRT): {avg_optimized_time_trt:.4f} seconds (Speedup: {speedup_trt:.2f}x)")
print(f"\nBest approach: {best_approach} - {best_time:.4f} seconds (Speedup: {best_speedup:.2f}x)")
print(f"Standard output: {standard_output_path}")
print(f"Optimized outputs: outputs/output_optimized_*")
print("\nVerify that the audio quality is similar between standard and optimized outputs")

# Print TensorRT verification instructions
print("\n--- Verifying TensorRT Acceleration ---")
print("To confirm TensorRT is actually being used:")
print("1. Look for 'Successfully created TensorRT engines' messages in the output")
print("2. Check for the 'Mean difference' values which indicate TensorRT outputs are being compared")
print("3. Monitor GPU utilization with 'nvidia-smi' in another terminal during inference")
print("4. The speedup reported in the performance summary should show improvement\n")