import torch
import sys

# Print Python version and PyTorch version
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

# Check if torch_tensorrt is installed and functional
try:
    import torch_tensorrt
    print(f"PyTorch-TensorRT version: {torch_tensorrt.__version__}")
    
    # Create a simple test model
    if torch.cuda.is_available():
        device = "cuda"
        model = torch.nn.Linear(10, 10).eval().to(device)
        dummy_input = torch.randn(1, 10, device=device)
        
        # Try compiling the model
        try:
            compiled_model = torch_tensorrt.compile(
                model,
                inputs=[dummy_input],
                enabled_precisions={torch.float32}
            )
            
            # Run inference
            with torch.no_grad():
                _ = compiled_model(dummy_input)
                
            print("✅ PyTorch-TensorRT is FUNCTIONAL - successfully compiled and ran a test model!")
            print("Method 4 should work in your script.")
        except Exception as e:
            print(f"❌ PyTorch-TensorRT is installed but NOT FUNCTIONAL: {e}")
            print("Method 4 will likely fail with similar errors.")
    else:
        print("❌ Cannot test PyTorch-TensorRT functionality - CUDA not available")
        print("Method 4 requires a CUDA-capable GPU.")
        
except ImportError:
    print("❌ PyTorch-TensorRT is NOT INSTALLED")
    print("Method 4 will not run until torch_tensorrt is installed.")

# Check TensorRT installation
try:
    import tensorrt as trt
    print(f"TensorRT version: {trt.__version__}")
    
    # Check TensorRT components
    import pkg_resources
    for package in ['tensorrt', 'tensorrt-cu12', 'tensorrt-cu12-bindings', 'tensorrt-cu12-libs']:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"  {package}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"  {package}: Not found")
    
except ImportError:
    print("❌ Base TensorRT (tensorrt) is NOT INSTALLED")