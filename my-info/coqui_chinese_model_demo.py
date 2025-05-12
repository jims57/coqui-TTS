import torch
from TTS.api import TTS
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

# Print information about available models
print("Available TTS models:")
print(TTS().list_models())



#【german】
tts = TTS("tts_models/zh-CN/baker/tacotron2-DDC-GST").to(device)
# Generate first sentence
tts.tts_to_file(text="我是柏林人，我很享受德国的好天气。", file_path="output_1.wav")

# Generate second sentence
tts.tts_to_file(text="德国菜以其各种香肠和啤酒而闻名。", file_path="output_2.wav")

# Generate third sentence
tts.tts_to_file(text="我和我的朋友们计划下周末去黑森林旅行。", file_path="output_3.wav")

# Generate fourth sentence
tts.tts_to_file(text="德国的高速公路在某些路段没有限速。", file_path="output_4.wav")

# Generate fifth sentence
tts.tts_to_file(text="今天我在老城区的一家舒适的咖啡馆里吃了美味的苹果派。", file_path="output_5.wav")

