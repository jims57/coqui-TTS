import os
import time
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

use_deepspeed = True  # You can change this to False to disable DeepSpeed

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

print("Loading model...")
config = XttsConfig()
config.load_json("/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/", use_deepspeed=use_deepspeed)
model.cuda()

# Add log message about DeepSpeed usage - modified version
print(f"Using DeepSpeed: {use_deepspeed}")

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["reference_samples/andy-liu-en-1.wav"])

print("Inference...")
t0 = time.time()
chunks = model.inference_stream(
    # "昨天我在书店发现了一本很有趣的小说，立刻就买下来了。",
    # "有时候我觉得生活就像一场冒险，我们都在不断探索新的可能性。",
    # "青石板上泛着水光，雨丝斜斜地织着帘子。我撑一把油纸伞，踩着湿润的石板路，听脚步声在巷子里轻轻回响。",
    "世界是一幅由无数不同色彩的丝线编织而成的挂毯，每一根丝线都为存在的整体复杂性和美丽贡献着力量。从最小的亚原子粒子到浩瀚的星系，一切都以微妙的因果之舞相互连接。生命以其无数种形式在全球繁荣生长，适应着不同的环境，并不断进化以应对不断变化的条件。人类的经验是这宏伟设计中一个独特的方面，其特点是卓越的思考、情感和创造力。",
    "zh-cn",
    gpt_cond_latent,
    speaker_embedding,
    stream_chunk_size=20,
    overlap_wav_len=1024,
    temperature=0.1,
    length_penalty=1.0,
    repetition_penalty=10.0,
    top_k=50,
    speed=1,
    enable_text_splitting=True
)

wav_chuncks = []
timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds

for i, chunk in enumerate(chunks):
    chunk_time = (time.time() - t0) * 1000  # Time in milliseconds since inference began
    if i == 0:
        print(f"Time to first chunck: {chunk_time:.2f} ms")
    print(f"Received chunk {i} of audio length {chunk.shape[-1]} at {chunk_time:.2f} ms")
    wav_chuncks.append(chunk)
    
    # Save each chunk as opus file with timestamp-based naming
    chunk_filename = f"outputs/{timestamp}-{i+1}.opus"
    # Convert to CPU and ensure right format before saving
    chunk_audio = chunk.squeeze().unsqueeze(0).cpu()
    torchaudio.save(chunk_filename, chunk_audio, 24000, format="opus")

# Still concatenate for reference but save as opus instead of wav
wav = torch.cat(wav_chuncks, dim=0)
torchaudio.save(f"outputs/{timestamp}-full.opus", wav.squeeze().unsqueeze(0).cpu(), 24000, format="opus")