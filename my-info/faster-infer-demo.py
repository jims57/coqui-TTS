import torch
from TTS.api import TTS

# Initialize TTS model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
model = tts.synthesizer.tts_model

# Apply FP16 quantization to the model
model.half()  # Convert to FP16

# Apply torch.compile to key components (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    # Compile with specific optimizations for TTS models
    model.text_encoder = torch.compile(
        model.text_encoder, 
        mode="reduce-overhead",
        fullgraph=True
    )
    
    model.gpt = torch.compile(
        model.gpt,
        mode="reduce-overhead", 
        fullgraph=True
    )
    
    if hasattr(model, 'vocoder'):
        model.vocoder = torch.compile(
            model.vocoder,
            mode="reduce-overhead",
            fullgraph=True
        )

# Now use the accelerated model
tts.tts_to_file(text="Hello world", file_path="output.wav", speaker_wav="speaker.wav")