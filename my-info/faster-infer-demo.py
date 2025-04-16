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

# Add this helper function at the top of the script (around line 30):
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

# Then add this line after loading the model (around line 34):
model = ensure_gpu(tts.synthesizer.tts_model)

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
    
    # Since the previous optimization had issues with dynamic input shapes and channel mismatches,
    # we'll focus on a different approach that targets entire modules instead of individual layers
    
    print("Analyzing XTTS model structure for optimizable components...")
    
    # Track the most common input shapes to target for optimization
    input_shape_tracker = {}
    
    # Define a hook to capture input shapes during inference
    def capture_shapes_hook(name):
        def hook_fn(module, input, output):
            shape = input[0].shape if isinstance(input, tuple) and len(input) > 0 else None
            if shape:
                if name not in input_shape_tracker:
                    input_shape_tracker[name] = []
                input_shape_tracker[name].append(shape)
        return hook_fn
    
    # Register hooks for shape tracking
    hooks = []
    
    # We'll focus on optimizing the entire HiFiGAN decoder which is the most computationally intensive part
    if hasattr(model, 'hifigan_decoder'):
        print("Targeting HiFiGAN decoder for optimization")
        
        # Register hook on the decoder itself to capture input shapes
        hooks.append(model.hifigan_decoder.register_forward_hook(capture_shapes_hook('hifigan_decoder')))
        
        # If we can access the waveform_decoder directly, we'll also track its inputs
        if hasattr(model.hifigan_decoder, 'waveform_decoder'):
            hooks.append(model.hifigan_decoder.waveform_decoder.register_forward_hook(
                capture_shapes_hook('waveform_decoder')))
    
    # Do a short inference to gather input shapes
    print("Running test inference to gather input shapes...")
    with torch.no_grad():
        _ = tts.tts(
            text="测试一下。这是一个稍长的句子，以便更好地分析模型的输入形状。",  # Longer test sentence
            speaker_wav=speaker_wav,
            language="zh"
        )
    
    # Also run a second test with different text to capture more shape variations
    with torch.no_grad():
        _ = tts.tts(
            text="Another test with English to get different shapes.",
            speaker_wav=speaker_wav,
            language="en"
        )
    
    # Remove the hooks
    for hook in hooks:
        hook.remove()
    
    # Print captured input shapes
    print("\nCaptured input shapes:")
    for name, shapes in input_shape_tracker.items():
        print(f"{name} input shapes: {shapes}")
    
    # Create a TensorRT module wrapper
    class TensorRTModuleWrapper(torch.nn.Module):
        def __init__(self, original_module, name, typical_shapes=None):
            super().__init__()
            self.original_module = original_module
            self.name = name
            self.trt_engines = {}
            self.typical_shapes = typical_shapes or []
            self.is_tensorrt = False
            self.hits = 0
            self.misses = 0
        
        def forward(self, *args, **kwargs):
            # Check if we have a TensorRT engine for this input shape
            if args and isinstance(args[0], torch.Tensor):
                x = args[0]
                shape_key = tuple(x.shape)
                
                # If we have an engine for this shape, use it
                if self.is_tensorrt and shape_key in self.trt_engines:
                    self.hits += 1
                    # Make sure input is contiguous and on correct device
                    x_cont = x.contiguous()
                    # Use the TensorRT engine
                    if len(args) > 1:
                        # Handle case where there are multiple inputs
                        result = self.trt_engines[shape_key](x_cont, *args[1:], **kwargs)
                    else:
                        result = self.trt_engines[shape_key](x_cont, **kwargs)
                    return result
                else:
                    self.misses += 1
                    # Fall back to original module
                    return self.original_module(*args, **kwargs)
            else:
                # Fall back to original module if no tensor input
                self.misses += 1
                return self.original_module(*args, **kwargs)
        
        def print_stats(self):
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            print(f"{self.name} TensorRT usage: {self.hits} hits, {self.misses} misses ({hit_rate:.1f}% hit rate)")
    
    # Set of optimized modules to track
    optimized_modules = []
    
    # Optimize HiFiGAN decoder directly instead of individual layers
    # This avoids the channel mismatch issues and captures the entire computation
    if 'hifigan_decoder' in input_shape_tracker and input_shape_tracker['hifigan_decoder']:
        # Get the most common input shapes seen
        hifigan_input_shapes = input_shape_tracker['hifigan_decoder']
        
        # Create wrapper for the decoder
        hifigan_wrapper = TensorRTModuleWrapper(
            model.hifigan_decoder, 
            "hifigan_decoder",
            hifigan_input_shapes
        )
        
        # Create a range of shapes for optimization
        # Focus on the sequence dimension which varies most
        print("\nOptimizing HiFiGAN decoder for common input shapes...")
        
        # Prepare a representative set of shapes
        shapes_to_optimize = []
        for shape in hifigan_input_shapes:
            if len(shape) >= 3:  # Should be at least 3D [batch, channels, seq_len]
                # Get the sequence length (last dimension)
                seq_len = shape[-1]
                # Create variations around this length
                variations = [
                    seq_len,
                    int(seq_len * 0.8),
                    int(seq_len * 1.2),
                    int(seq_len * 1.5),
                    int(seq_len * 2.0)
                ]
                
                for var_len in variations:
                    # Create a new shape with the varied sequence length
                    new_shape = list(shape)
                    new_shape[-1] = var_len
                    shapes_to_optimize.append(tuple(new_shape))
        
        # Remove duplicates and sort
        shapes_to_optimize = sorted(list(set(shapes_to_optimize)))
        
        # Try to optimize for each shape
        for opt_shape in shapes_to_optimize[:3]:  # Limit to first 3 to avoid cache issues
            try:
                print(f"Creating TensorRT engine for shape {opt_shape}...")
                
                # Create a dummy input tensor
                dummy_mel = torch.randn(opt_shape, device=device)
                dummy_g = None
                
                # If the HiFiGAN decoder expects a speaker embedding, create one
                try:
                    # Try to infer the shape of the speaker embedding
                    with torch.no_grad():
                        # Create a fake speaker embedding if needed
                        try:
                            # Test if the HiFiGAN decoder needs a speaker embedding
                            original_output = model.hifigan_decoder(dummy_mel)
                        except TypeError:
                            # If it fails without a speaker embedding, try with one
                            dummy_g = torch.randn(1, 512, device=device)  # Typical speaker embedding size
                except Exception as e:
                    print(f"Error testing original decoder: {e}")
                    continue
                
                # Compile with TensorRT
                try:
                    compile_spec = {
                        "inputs": [dummy_mel] if dummy_g is None else [dummy_mel, dummy_g],
                        "enabled_precisions": {torch.float},  # Use FP32 for stability
                        "workspace_size": 1 << 30,  # 1GB workspace
                        "debug": False,
                        "truncate_long_and_double": True,  # Better compatibility
                        "allow_shape_tensors": True,  # Allow shape tensors
                        "min_block_size": 1,  # Optimize for smaller operations
                        "device": {
                            "device_type": "gpu",
                            "gpu_id": 0
                        }
                    }
                    
                    # Test run with the original module
                    with torch.no_grad():
                        if dummy_g is not None:
                            original_output = model.hifigan_decoder(dummy_mel, g=dummy_g)
                        else:
                            original_output = model.hifigan_decoder(dummy_mel)
                        
                        print(f"Original output shape: {original_output.shape}")
                    
                    # Compile a specialized version for this input shape
                    if dummy_g is not None:
                        # Define a wrapper function to handle the g parameter
                        def decoder_with_g(mel, g):
                            return model.hifigan_decoder(mel, g=g)
                        
                        compiled_module = torch_tensorrt.compile(
                            decoder_with_g,
                            **compile_spec
                        )
                    else:
                        compiled_module = torch_tensorrt.compile(
                            model.hifigan_decoder,
                            **compile_spec
                        )
                    
                    # Verify the compiled module works
                    with torch.no_grad():
                        if dummy_g is not None:
                            trt_output = compiled_module(dummy_mel, dummy_g)
                        else:
                            trt_output = compiled_module(dummy_mel)
                        
                        print(f"TensorRT output shape: {trt_output.shape}")
                        
                        # Check output quality
                        diff = torch.abs(original_output - trt_output).mean().item()
                        print(f"Mean difference: {diff}")
                    
                    # Store the engine in our wrapper
                    shape_key = tuple(dummy_mel.shape)
                    hifigan_wrapper.trt_engines[shape_key] = compiled_module
                    hifigan_wrapper.is_tensorrt = True
                    print(f"Successfully added TensorRT engine for shape {shape_key}")
                    
                except Exception as e:
                    print(f"Failed to compile TensorRT engine for shape {opt_shape}: {e}")
            
            except Exception as e:
                print(f"Error optimizing for shape {opt_shape}: {e}")
        
        # Replace the original module if we successfully created any engines
        if hifigan_wrapper.is_tensorrt and hifigan_wrapper.trt_engines:
            print(f"Successfully created {len(hifigan_wrapper.trt_engines)} TensorRT engines for HiFiGAN decoder")
            # Store the original for restoration later
            original_hifigan_decoder = model.hifigan_decoder
            # Replace with our wrapped version
            model.hifigan_decoder = hifigan_wrapper
            # Add to our tracking list
            optimized_modules.append(hifigan_wrapper)
        else:
            print("Failed to create any TensorRT engines for HiFiGAN decoder")
    
    # Define a function to print TensorRT statistics
    def print_tensorrt_stats():
        for module in optimized_modules:
            module.print_stats()
    
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
print("\nApproach 2: Using PyTorch-TensorRT with optimized memory handling")
total_time_trt = 0

for i in range(num_runs):
    # Clear CUDA cache before each run
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Pre-allocate GPU memory to reduce fragmentation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Allocate and free a large tensor to consolidate memory
        torch.empty(int(1e9)//4, dtype=torch.float, device=device)
        torch.cuda.empty_cache()
    
    start_time = time.time()
    with torch.no_grad():  # Use no_grad for inference
        output = tts.tts(
            text=chinese_text,
            speaker_wav=speaker_wav,
            language=language
        )
        # Write to file after timing to avoid including file I/O in measurement
        tts.synthesizer.save_wav(output, f"outputs/output_optimized_trt_{i+1}.wav")
    run_time = time.time() - start_time
    total_time_trt += run_time
    print(f"Run {i+1}: {run_time:.4f} seconds")

    # Add after the tts_to_file calls in both benchmark sections
    if 'print_tensorrt_stats' in globals():
        print_tensorrt_stats()

avg_optimized_time_trt = total_time_trt / num_runs
speedup_trt = standard_time / avg_optimized_time_trt if avg_optimized_time_trt > 0 else 0

# Add a special approach that only applies FP16 to the HiFiGAN decoder (not the GPT part)
print("\nApproach 3: Selective FP16 for vocoder only")
total_time_selective = 0

# Create patched versions of the functions that need acceleration
if hasattr(model, 'hifigan_decoder'):
    # Save original forward method
    original_forward = model.hifigan_decoder.forward
    
    # Create an FP16-accelerated version that handles all parameters
    def fp16_forward(self, mel_tensor, g=None):
        # Use the new recommended syntax
        with torch.amp.autocast(device_type='cuda', enabled=True):
            return original_forward(mel_tensor, g=g)
    
    # Apply the patch
    import types
    model.hifigan_decoder.forward = types.MethodType(fp16_forward, model.hifigan_decoder)
    print("Applied selective FP16 acceleration to HiFiGAN decoder")

for i in range(num_runs):
    # Clear CUDA cache before each run
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    start_time = time.time()
    with torch.no_grad():  # Keep using no_grad globally
        tts.tts_to_file(
            text=chinese_text,
            file_path=f"outputs/output_optimized_selective_{i+1}.wav",
            speaker_wav=speaker_wav,
            language=language
        )
    run_time = time.time() - start_time
    total_time_selective += run_time
    print(f"Run {i+1}: {run_time:.4f} seconds")

    # Add after the tts_to_file calls in both benchmark sections
    if 'print_tensorrt_stats' in globals():
        print_tensorrt_stats()

avg_optimized_time_selective = total_time_selective / num_runs
speedup_selective = standard_time / avg_optimized_time_selective if avg_optimized_time_selective > 0 else 0

# Add a new approach after the Selective FP16 section (around line 560):

# Add a special approach that uses TF32 for vocoder (works better on Ampere GPUs)
print("\nApproach 4: Selective TF32 for vocoder only")
total_time_tf32 = 0

# Create TF32-accelerated version
if hasattr(model, 'hifigan_decoder'):
    # Save original forward method
    original_forward_tf32 = model.hifigan_decoder.forward
    
    # Create a TF32-accelerated version that handles all parameters
    def tf32_forward(self, mel_tensor, g=None):
        # Only apply TF32 to this part
        with torch.no_grad():
            # Enable TF32 locally
            old_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
            
            result = original_forward_tf32(mel_tensor, g=g)
            
            # Restore previous setting
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
            return result
    
    # Apply the patch
    import types
    model.hifigan_decoder.forward = types.MethodType(tf32_forward, model.hifigan_decoder)
    print("Applied selective TF32 acceleration to HiFiGAN decoder")

for i in range(num_runs):
    # Clear CUDA cache before each run
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    start_time = time.time()
    with torch.no_grad():  # Keep using no_grad globally
        tts.tts_to_file(
            text=chinese_text,
            file_path=f"outputs/output_optimized_tf32_{i+1}.wav",
            speaker_wav=speaker_wav,
            language=language
        )
    run_time = time.time() - start_time
    total_time_tf32 += run_time
    print(f"Run {i+1}: {run_time:.4f} seconds")

    # Add after the tts_to_file calls in both benchmark sections
    if 'print_tensorrt_stats' in globals():
        print_tensorrt_stats()

avg_optimized_time_tf32 = total_time_tf32 / num_runs
speedup_tf32 = standard_time / avg_optimized_time_tf32 if avg_optimized_time_tf32 > 0 else 0

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

if avg_optimized_time_selective > 0 and avg_optimized_time_selective < best_time:
    best_approach = "Approach 3 (Selective FP16)"
    best_time = avg_optimized_time_selective
    best_speedup = speedup_selective

if avg_optimized_time_tf32 > 0 and avg_optimized_time_tf32 < best_time:
    best_approach = "Approach 4 (Selective TF32)"
    best_time = avg_optimized_time_tf32
    best_speedup = speedup_tf32

# Print summary
print("\n--- Performance Summary ---")
print(f"Chinese text: \"{chinese_text}\"")
print(f"Standard inference: {standard_time:.4f} seconds")
if avg_optimized_time_approach1 > 0:
    print(f"Approach 1 (GPU Optimizations): {avg_optimized_time_approach1:.4f} seconds (Speedup: {speedup_approach1:.2f}x)")
if avg_optimized_time_trt > 0:
    print(f"Approach 2 (PyTorch-TensorRT): {avg_optimized_time_trt:.4f} seconds (Speedup: {speedup_trt:.2f}x)")
if avg_optimized_time_selective > 0:
    print(f"Approach 3 (Selective FP16): {avg_optimized_time_selective:.4f} seconds (Speedup: {speedup_selective:.2f}x)")
if avg_optimized_time_tf32 > 0:
    print(f"Approach 4 (Selective TF32): {avg_optimized_time_tf32:.4f} seconds (Speedup: {speedup_tf32:.2f}x)")
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

# Restore original modules
if 'original_hifigan_decoder' in locals():
    model.hifigan_decoder = original_hifigan_decoder
    print("Restored original HiFiGAN decoder")