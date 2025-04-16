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
    
    # Since the previous TensorRT optimization had issues with dynamic input shapes,
    # we'll create an approach that specifically handles dynamic dimensions
    
    print("Analyzing XTTS model structure for optimizable components...")
    
    # First identify the actual bottlenecks in the model
    if hasattr(model, 'hifigan_decoder'):
        print("Analyzing HiFiGAN decoder structure...")
        # Create a counter to track layer execution during a test run
        layer_calls = {}
        
        # Define hooks to monitor the model
        def layer_hook(name):
            def hook(module, input, output):
                if name not in layer_calls:
                    layer_calls[name] = 1
                else:
                    layer_calls[name] += 1
                if hasattr(input[0], 'shape'):
                    print(f"Layer {name} received input shape: {input[0].shape}")
            return hook
        
        # Register hooks on key components
        hooks = []
        
        # Check if we can directly optimize the entire decoder
        if hasattr(model.hifigan_decoder, 'waveform_decoder'):
            dec = model.hifigan_decoder.waveform_decoder
            
            # Look at the upsample layers - these are often computationally intensive
            if hasattr(dec, 'ups'):
                print(f"Found {len(dec.ups)} upsample layers")
                for i, layer in enumerate(dec.ups):
                    hooks.append(layer.register_forward_hook(layer_hook(f"upsample_{i}")))
            
            # Look at the main components
            for name, module in dec.named_children():
                if name not in ['ups', 'resblocks']:  # Already added these
                    hooks.append(module.register_forward_hook(layer_hook(name)))
            
        # Do a short test inference to gather information
        print("Running test inference to identify optimization targets...")
        with torch.no_grad():
            _ = tts.tts(
                text="测试一下。",  # Short Chinese test
                speaker_wav=speaker_wav,
                language="zh"
            )
        
        # Remove the hooks
        for hook in hooks:
            hook.remove()
        
        # Print layer execution stats
        print("\nLayer execution stats from test run:")
        for name, count in layer_calls.items():
            print(f"Layer {name}: called {count} times")
        
        # Create a specialized wrapper for multi-resolution upsampling layers
        class TRTUpsampleWrapper(torch.nn.Module):
            def __init__(self, original_module):
                super().__init__()
                self.original = original_module
                self.trt_engines = {}
                self.is_tensorrt = False
                
            def forward(self, x):
                in_shape = tuple(x.shape)
                
                # Use TensorRT if available for this shape
                if self.is_tensorrt and in_shape in self.trt_engines:
                    return self.trt_engines[in_shape](x.contiguous())
                else:
                    return self.original(x)
        
        # Identify and optimize the upsample layers - these are often bottlenecks
        # in waveform generation and have fixed weights
        if hasattr(model.hifigan_decoder, 'waveform_decoder') and hasattr(model.hifigan_decoder.waveform_decoder, 'ups'):
            ups = model.hifigan_decoder.waveform_decoder.ups
            print(f"\nFound {len(ups)} upsample layers to optimize")
            
            # Optimize each upsample layer
            optimized_ups = 0
            for i, up_layer in enumerate(ups):
                print(f"\nOptimizing upsample layer {i}...")
                wrapper = TRTUpsampleWrapper(up_layer)
                
                # Extract the actual sizes we need from the logs
                if i == 2:  # For upsample layer 2 which successfully compiles
                    test_sizes = [
                        (1, 128, 24768),  # Seen in logs
                        (1, 128, 25600),  # Seen in logs
                        (1, 128, 27264),  # Seen in logs
                        (1, 128, 28416),  # Seen in logs
                        (1, 128, 30336),  # Seen in logs
                        (1, 128, 31744),  # Seen in logs
                        (1, 128, 35904),  # Seen in logs
                        (1, 128, 38144),  # Seen in logs
                        (1, 128, 39552),  # Seen in logs
                        (1, 128, 40064),  # Seen in logs
                        (1, 128, 40384),  # Seen in logs
                        (1, 128, 42880),  # Seen in logs
                        (1, 128, 44800),  # Seen in logs
                        (1, 128, 45120),  # Seen in logs
                    ]
                else:
                    test_sizes = [
                        (1, 128, 20000),  # Base size
                        (1, 128, 30000),  # Medium size
                        (1, 128, 40000),  # Large size
                    ]
                
                for in_size in test_sizes:
                    try:
                        print(f"Creating TensorRT engine for input shape: {in_size}...")
                        dummy_input = torch.randn(in_size, device=device)
                        
                        # Test the original layer
                        with torch.no_grad():
                            dummy_input = dummy_input.contiguous()
                            original_output = up_layer(dummy_input)
                            print(f"Original output shape: {original_output.shape}")
                        
                        # Compile with TensorRT with dynamic shape handling
                        compile_spec = {
                            "inputs": [dummy_input],
                            "enabled_precisions": {torch.float32},  # FP32 only for stability
                            "workspace_size": 1 << 30,  # 1GB workspace
                            "debug": False,
                            # Allow for more variation in sequence length
                            "input_shapes": {
                                "x": {
                                    "min": [in_size[0], in_size[1], max(1, int(in_size[2] * 0.7))], 
                                    "opt": [in_size[0], in_size[1], in_size[2]],
                                    "max": [in_size[0], in_size[1], int(in_size[2] * 1.3)]
                                }
                            }
                        }
                        
                        compiled_module = torch_tensorrt.compile(
                            up_layer, 
                            **compile_spec
                        )
                        
                        # Test the compiled module
                        with torch.no_grad():
                            trt_output = compiled_module(dummy_input)
                            print(f"TensorRT output shape: {trt_output.shape}")
                            
                            # Compare the outputs
                            diff = torch.abs(original_output - trt_output).mean().item()
                            print(f"Mean difference: {diff}")
                        
                        # Store the TensorRT engine
                        wrapper.trt_engines[in_size] = compiled_module
                        wrapper.is_tensorrt = True
                    except Exception as e:
                        print(f"Failed to create engine for size {in_size}: {e}")
                
                # Replace the original layer if optimization was successful
                if wrapper.is_tensorrt:
                    print(f"Successfully created TensorRT engines for upsample layer {i}")
                    ups[i] = wrapper
                    optimized_ups += 1
                else:
                    print(f"Failed to optimize upsample layer {i}")
            
            print(f"Successfully optimized {optimized_ups} of {len(ups)} upsample layers")
            
            # Now try to optimize the conv_post layer
            # This is usually a significant bottleneck as it generates the final waveform
            if hasattr(model.hifigan_decoder.waveform_decoder, 'conv_post'):
                print("\nOptimizing conv_post layer...")
                conv_post = model.hifigan_decoder.waveform_decoder.conv_post
                wrapper = TRTUpsampleWrapper(conv_post)
                
                # Define test sizes based on typical output sizes of upsample layers
                # These would be much larger because of the upsampling
                test_sizes = [
                    (1, 128, 30000),
                    (1, 128, 60000),
                    (1, 128, 120000)
                ]
                
                for in_size in test_sizes:
                    try:
                        print(f"Creating TensorRT engine for conv_post with input shape: {in_size}...")
                        dummy_input = torch.randn(in_size, device=device)
                        
                        # Test the original layer
                        with torch.no_grad():
                            dummy_input = dummy_input.contiguous()
                            original_output = conv_post(dummy_input)
                            print(f"Original output shape: {original_output.shape}")
                        
                        # Compile with TensorRT - allow dynamic sequence length
                        compile_spec = {
                            "inputs": [dummy_input],
                            "enabled_precisions": {torch.float32},  # FP32 only for stability
                            "workspace_size": 1 << 30,  # 1GB workspace
                            "debug": False,
                            # Allow varied lengths
                            "input_shapes": {
                                "x": {
                                    "min": [in_size[0], in_size[1], max(1, int(in_size[2] * 0.8))], 
                                    "opt": [in_size[0], in_size[1], in_size[2]],
                                    "max": [in_size[0], in_size[1], int(in_size[2] * 1.2)]
                                }
                            }
                        }
                        
                        compiled_module = torch_tensorrt.compile(
                            conv_post, 
                            **compile_spec
                        )
                        
                        # Test the compiled module
                        with torch.no_grad():
                            trt_output = compiled_module(dummy_input)
                            print(f"TensorRT output shape: {trt_output.shape}")
                            
                            # Compare the outputs
                            diff = torch.abs(original_output - trt_output).mean().item()
                            print(f"Mean difference: {diff}")
                        
                        # Store the TensorRT engine
                        wrapper.trt_engines[in_size] = compiled_module
                        wrapper.is_tensorrt = True
                    except Exception as e:
                        print(f"Failed to create engine for size {in_size}: {e}")
                
                # Replace the original layer if optimization was successful
                if wrapper.is_tensorrt:
                    print("Successfully created TensorRT engines for conv_post layer")
                    model.hifigan_decoder.waveform_decoder.conv_post = wrapper
                    optimized_ups += 1
                else:
                    print("Failed to optimize conv_post layer")
                    
            # Create a counter to track TensorRT activations
            class TensorRTCounter:
                def __init__(self):
                    self.hits = 0
                    self.misses = 0
                    self.last_shapes = []
                
                def reset(self):
                    self.hits = 0
                    self.misses = 0
                    self.last_shapes = []
            
            # Create counter instance
            trt_counter = TensorRTCounter()
            
            # Set up a hook to count TensorRT usage and track shapes
            def tensorrt_counter_hook(name):
                def hook_fn(module, input, output):
                    in_shape = tuple(input[0].shape)
                    trt_counter.last_shapes.append((name, in_shape))
                    
                    if hasattr(module, 'is_tensorrt') and module.is_tensorrt:
                        if in_shape in module.trt_engines:
                            trt_counter.hits += 1
                        else:
                            trt_counter.misses += 1
                            # Print the shape that wasn't found in engines
                            print(f"TensorRT miss for {name} with shape {in_shape}")
                
                return hook_fn
            
            # Register hooks on all optimized modules
            hooks = []
            if hasattr(model.hifigan_decoder, 'waveform_decoder'):
                dec = model.hifigan_decoder.waveform_decoder
                if hasattr(dec, 'ups'):
                    for i, layer in enumerate(dec.ups):
                        if hasattr(layer, 'is_tensorrt') and layer.is_tensorrt:
                            hooks.append(layer.register_forward_hook(tensorrt_counter_hook(f"upsample_{i}")))
                
                if hasattr(dec, 'conv_post') and hasattr(dec.conv_post, 'is_tensorrt') and dec.conv_post.is_tensorrt:
                    hooks.append(dec.conv_post.register_forward_hook(tensorrt_counter_hook("conv_post")))
            
            # Print stats function
            def print_tensorrt_stats():
                print(f"TensorRT usage stats: {trt_counter.hits} hits, {trt_counter.misses} misses")
                if trt_counter.hits + trt_counter.misses > 0:
                    hit_rate = trt_counter.hits / (trt_counter.hits + trt_counter.misses) * 100
                    print(f"Hit rate: {hit_rate:.1f}%")
                
                # Reset counters
                trt_counter.reset()

        print("\nBased on TensorRT misses during test inference, we need to create engines for:")
        for shape in trt_counter.last_shapes:
            print(f"- {shape[0]} with shape {shape[1]}")
        print("Consider updating the test_sizes to include these shapes for better hit rate.")

    else:
        print("Could not find HiFiGAN decoder structure")
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
    with torch.no_grad():  # Use no_grad but NOT autocast for the whole pipeline
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

# Restore original methods
if hasattr(model, 'hifigan_decoder') and 'original_forward' in locals():
    import types
    model.hifigan_decoder.forward = types.MethodType(original_forward, model.hifigan_decoder)
    print("Restored original HiFiGAN decoder forward method")