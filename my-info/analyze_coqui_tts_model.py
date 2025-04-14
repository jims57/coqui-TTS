from TTS.api import TTS
import torch

def analyze_xtts_model_inputs():
    # Initialize model exactly as in your demo.py
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    model = tts.synthesizer.tts_model

    print("\n=== Model Device ===")
    print(f"Using device: {device}")

    # Sample inputs matching your demo.py
    text = "他下午坐在窗边舒适的扶手椅上津津有味地读着一本引人入胜的小说。"
    speaker_wav = "speaker_wavs/jack-mark-en-1.wav"
    language = "zh"

    print("\n=== Model Information ===")
    print(f"Model type: {type(model).__name__}")
    print(f"Available languages: {tts.languages}")

    # Add forward hook to capture input shapes
    def input_shape_hook(module, input, output):
        print("\n=== Input Shapes from Forward Pass ===")
        for idx, inp in enumerate(input):
            if isinstance(inp, torch.Tensor):
                print(f"Input {idx}:")
                print(f"  Shape: {inp.shape}")
                print(f"  Type: {inp.dtype}")
                print(f"  Device: {inp.device}")

    # Register the hook
    hook_handle = model.register_forward_hook(input_shape_hook)

    try:
        print("\n=== Attempting Inference ===")
        # Use the same parameters as your working demo
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            file_path="output_test.wav"
        )
    except Exception as e:
        print(f"Error during inference: {e}")
    finally:
        # Remove the hook
        hook_handle.remove()

    # Print model configuration
    print("\n=== Model Configuration ===")
    config = tts.synthesizer.tts_config
    if hasattr(config, 'model_args'):
        print("Model Arguments:", config.model_args)
    if hasattr(config, 'audio'):
        print("\nAudio Configuration:")
        print(f"Sample Rate: {config.audio.sample_rate}")
        print(f"Hop Length: {config.audio.hop_length if hasattr(config.audio, 'hop_length') else 'Not specified'}")
        print(f"Win Length: {config.audio.win_length if hasattr(config.audio, 'win_length') else 'Not specified'}")

if __name__ == "__main__":
    analyze_xtts_model_inputs()