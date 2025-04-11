import torch
from TTS.api import TTS

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
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

print(tts.languages)
print('=== speakers ===')
# print(tts.speakers)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
#wav = tts.tts(text="Hello world!", speaker_wav="speaker_wavs/andy-liu-en-1.wav", language="en")

# Andy Liu
# tts.tts_to_file(text="The sun sets behind the mountains, casting long shadows across the valley.", speaker_wav="speaker_wavs/andy-liu-en-1.wav", language="en", file_path="output.wav")

# (work)Jack Ma(en)
# tts.tts_to_file(text="The sun sets behind the mountains, casting long shadows across the valley.", speaker_wav="speaker_wavs/jack-mark-en-1.wav", language="en", file_path="output.wav")

# (work)Jack Ma(zh)
# tts.tts_to_file(text="他下午坐在窗边舒适的扶手椅上津津有味地读着一本引人入胜的小说。", speaker_wav="speaker_wavs/jack-mark-en-1.wav", language="zh", file_path="output.wav")

# (work)Jack Ma(zh-cn)
# tts.tts_to_file(text="他下午坐在窗边舒适的扶手椅上津津有味地读着一本引人入胜的小说。", speaker_wav="speaker_wavs/jack-mark-en-1.wav", language="zh-cn", file_path="output.wav")

# (work)Jack Ma(ja)
# [install]:  pip install cutlet
tts.tts_to_file(text="彼は午後、窓際の心地よい肘掛け椅子に座って、面白い小説を夢中で読んでいた。", speaker_wav="speaker_wavs/jack-mark-en-1.wav", language="ja", file_path="output.wav")


# ==== Example: (work)Use Chinese model(tacotron2-DDC-GST) ====
# tts = TTS("tts_models/zh-CN/baker/tacotron2-DDC-GST").to(device)
# tts.tts_to_file(text="他下午坐在窗边舒适的扶手椅上津津有味地读着一本引人入胜的小说。", file_path="output.wav")
