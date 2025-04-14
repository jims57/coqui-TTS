import torch
from TTS.api import TTS
import os
import inspect

def convert_xtts_components_to_onnx():
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    model = tts.synthesizer.tts_model

    # Create output directory
    os.makedirs("onnx_model", exist_ok=True)

    # Export GPT component
    try:
        print("\nExporting GPT component...")
        gpt_model = model.gpt
        
        # Create a wrapper to handle the required inputs
        class GPTWrapper(torch.nn.Module):
            def __init__(self, gpt_model):
                super().__init__()
                self.gpt = gpt_model
            
            def forward(self, text_inputs, text_lengths):
                return self.gpt(text_inputs, text_lengths)
        
        wrapped_gpt = GPTWrapper(gpt_model)
        
        # Create dummy inputs for GPT
        batch_size = 1
        seq_len = 402
        
        # Create the inputs GPT expects
        dummy_text = torch.randint(0, 6681, (batch_size, seq_len), device=device)
        dummy_lengths = torch.tensor([seq_len], device=device)
        
        # Verify the forward pass works
        print("Testing GPT forward pass...")
        with torch.no_grad():
            try:
                output = wrapped_gpt(dummy_text, dummy_lengths)
                print("GPT forward pass successful!")
                if isinstance(output, torch.Tensor):
                    print(f"Output shape: {output.shape}")
                else:
                    print(f"Output type: {type(output)}")
            except Exception as e:
                print(f"GPT forward pass error: {e}")
        
        # Export GPT model
        torch.onnx.export(
            wrapped_gpt,
            (dummy_text, dummy_lengths),
            "onnx_model/gpt_component.onnx",
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['text_inputs', 'text_lengths'],
            output_names=['gpt_output'],
            dynamic_axes={
                'text_inputs': {0: 'batch_size', 1: 'sequence_length'},
                'text_lengths': {0: 'batch_size'},
                'gpt_output': {0: 'batch_size', 1: 'output_length'}
            }
        )
        print("GPT component exported successfully!")
        
        # Verify the ONNX model
        import onnx
        onnx_model = onnx.load("onnx_model/gpt_component.onnx")
        onnx.checker.check_model(onnx_model)
        print("ONNX GPT model validated!")
        
    except Exception as e:
        print(f"Error during GPT export: {e}")
        import traceback
        traceback.print_exc()
    
    # Export HifiGAN decoder component
    try:
        print("\nExporting HifiGAN decoder component...")
        decoder_model = model.hifigan_decoder
        
        # Create a wrapper for the decoder
        class DecoderWrapper(torch.nn.Module):
            def __init__(self, decoder):
                super().__init__()
                self.decoder = decoder
            
            def forward(self, gpt_output):
                return self.decoder(gpt_output)
        
        wrapped_decoder = DecoderWrapper(decoder_model)
        
        # Create dummy input for decoder (shape depends on GPT output)
        # Typical shape is [batch_size, sequence_length, hidden_size]
        dummy_gpt_output = torch.randn(1, 100, 1024, device=device)  # Adjust size as needed
        
        # Verify decoder forward pass
        print("Testing decoder forward pass...")
        with torch.no_grad():
            try:
                output = wrapped_decoder(dummy_gpt_output)
                print("Decoder forward pass successful!")
                if isinstance(output, torch.Tensor):
                    print(f"Output shape: {output.shape}")
                else:
                    print(f"Output type: {type(output)}")
            except Exception as e:
                print(f"Decoder forward pass error: {e}")
                # Try with different input shape if first attempt fails
                try:
                    # Try alternative shape if first one failed
                    dummy_gpt_output = torch.randn(1, 1024, device=device)
                    output = wrapped_decoder(dummy_gpt_output)
                    print("Decoder forward pass successful with alternative shape!")
                    print(f"Output shape: {output.shape if isinstance(output, torch.Tensor) else type(output)}")
                except Exception as e2:
                    print(f"Alternative decoder forward pass error: {e2}")
        
        # Export decoder model (if forward pass worked)
        torch.onnx.export(
            wrapped_decoder,
            (dummy_gpt_output,),
            "onnx_model/hifigan_decoder.onnx",
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['gpt_output'],
            output_names=['audio_output'],
            dynamic_axes={
                'gpt_output': {0: 'batch_size', 1: 'sequence_length'},
                'audio_output': {0: 'batch_size', 1: 'audio_length'}
            }
        )
        print("HifiGAN decoder exported successfully!")
        
        # Verify the ONNX model
        onnx_model = onnx.load("onnx_model/hifigan_decoder.onnx")
        onnx.checker.check_model(onnx_model)
        print("ONNX decoder model validated!")
        
    except Exception as e:
        print(f"Error during decoder export: {e}")
        import traceback
        traceback.print_exc()
    
    # Analyze synthesizer inference process
    try:
        print("\nAnalyzing synthesizer inference process...")
        
        # Create a hook to capture intermediate tensors
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output
            return hook
        
        # Register hooks on key components
        if hasattr(model.gpt, 'register_forward_hook'):
            model.gpt.register_forward_hook(get_activation('gpt_output'))
        if hasattr(model.hifigan_decoder, 'register_forward_hook'):
            model.hifigan_decoder.register_forward_hook(get_activation('decoder_output'))
        
        # Run an inference to capture activations
        text = "This is a test"
        speaker_wav = "speaker_wavs/jack-mark-en-1.wav"
        language = "en"
        
        with torch.no_grad():
            wav = tts.tts(text=text, speaker_wav=speaker_wav, language=language)
        
        # Print captured activations shapes
        print("\nCaptured activation shapes:")
        for name, tensor in activations.items():
            if isinstance(tensor, torch.Tensor):
                print(f"{name}: {tensor.shape}")
            else:
                print(f"{name}: {type(tensor)}")
                
    except Exception as e:
        print(f"Error during activation tracing: {e}")

def verify_onnx_models():
    try:
        import onnxruntime
    except ImportError:
        print("onnxruntime not installed. Please install it with:")
        print("pip install onnxruntime-gpu  # for GPU")
        print("pip install onnxruntime      # for CPU")
        return

    import numpy as np
    
    # Verify GPT model
    try:
        print("\nVerifying GPT ONNX model...")
        if os.path.exists("onnx_model/gpt_component.onnx"):
            session = onnxruntime.InferenceSession("onnx_model/gpt_component.onnx")
            
            # Prepare inputs
            batch_size = 1
            seq_len = 402
            
            inputs = {
                'text_inputs': np.random.randint(0, 6681, (batch_size, seq_len)).astype(np.int64),
                'text_lengths': np.array([seq_len]).astype(np.int64)
            }
            
            # Run inference
            outputs = session.run(None, inputs)
            print("GPT ONNX model verification successful!")
            print(f"Output shape: {outputs[0].shape}")
        else:
            print("GPT ONNX model file not found")
    except Exception as e:
        print(f"Error during GPT ONNX verification: {e}")
    
    # Verify decoder model
    try:
        print("\nVerifying HifiGAN decoder ONNX model...")
        if os.path.exists("onnx_model/hifigan_decoder.onnx"):
            session = onnxruntime.InferenceSession("onnx_model/hifigan_decoder.onnx")
            
            # Prepare inputs (shape from GPT output)
            gpt_output = np.random.randn(1, 100, 1024).astype(np.float32)
            
            # Run inference
            outputs = session.run(None, {'gpt_output': gpt_output})
            print("HifiGAN decoder ONNX model verification successful!")
            print(f"Output shape: {outputs[0].shape}")
        else:
            print("HifiGAN decoder ONNX model file not found")
    except Exception as e:
        print(f"Error during HifiGAN decoder ONNX verification: {e}")

if __name__ == "__main__":
    convert_xtts_components_to_onnx()
    verify_onnx_models()