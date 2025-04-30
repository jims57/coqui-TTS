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

# Load xtts_v2 model
config.load_json("/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/", checkpoint_path="/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/model.pth", use_deepspeed=use_deepspeed)

# Load xtts_v1 model
# config.load_json("/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v1.1/config.json")
# model = Xtts.init_from_config(config)
# model.load_checkpoint(config, checkpoint_dir="/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v1.1", checkpoint_path="/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v1.1/model.pth", use_deepspeed=use_deepspeed)


# Load ljspeech/fast_pitch model
# config.load_json("/root/.local/share/tts/tts_models--en--ljspeech--fast_pitch/config.json")
# model = Xtts.init_from_config(config)
# model.load_checkpoint(config, checkpoint_dir="/root/.local/share/tts/tts_models--en--ljspeech--fast_pitch/", checkpoint_path="/root/.local/share/tts/tts_models--en--ljspeech--fast_pitch/model_file.pth", use_deepspeed=use_deepspeed)


# Load tacotron2-DDC-GST model #[Not work]
# config.load_json("/root/.local/share/tts/tts_models--zh-CN--baker--tacotron2-DDC-GST/config.json")
# model = Xtts.init_from_config(config)
# model.load_checkpoint(config, checkpoint_dir="/root/.local/share/tts/tts_models--zh-CN--baker--tacotron2-DDC-GST/", checkpoint_path="/root/.local/share/tts/tts_models--zh-CN--baker--tacotron2-DDC-GST/model_file.pth", use_deepspeed=use_deepspeed)

model.cuda()

# Add log message about DeepSpeed usage - modified version
print(f"Using DeepSpeed: {use_deepspeed}")

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["reference_samples/andy-liu-en-1.wav"])

# Problem:
# 1."世界是一幅由无数不同色彩的丝线编织而成的挂毯，每一根丝线都为存在的整体复杂性和美丽贡献着力量。从最小的亚原子粒子到浩瀚的星系，。一切都以微妙的因果之舞相互连接"
# Problem: 一切都以微妙的因果之舞相互连接 is repeated
#
# 1. "一切都以微妙的因果之舞" 这句话会卡住（重复）


print("Inference...")
t0 = time.time()
chunks = model.inference_stream(
    #【Chinese usage】
    # "昨天我在书店发现了一本很有趣的小说，立刻就买下来了。",
    # "有时候我觉得生活就像一场冒险，我们都在不断探索新的可能性。", #[ok]
    "青石板上泛着水光，雨丝斜斜地织着帘子。我撑一把油纸伞，踩着湿润的石板路，听脚步声在巷子里轻轻回响。",#[err: 轻轻回响, jitter]
    # "从最小的亚原子粒子到浩瀚的星系，一切都以微妙的因果之舞相互连接。", #[ok]
    # "世界是一幅由无数不同色彩的丝线编织而成的挂毯，每一根丝线都为存在的整体复杂性和美丽贡献着力量。从最小的亚原子粒子到浩瀚的星系，一切都以微妙的因果之舞相互连接。",
    # "世界是一幅由无数不同色彩的丝线编织而成的挂毯，每一根丝线都为存在的整体复杂性和美丽贡献着力量。从最小的亚原子粒子到浩瀚的星系，一切都以微妙的因果之舞相互连接。生命以其无数种形式在全球繁荣生长，适应着不同的环境，并不断进化以应对不断变化的条件。人类的经验是这宏伟设计中一个独特的方面，其特点是卓越的思考、情感和创造力。",
    # "世界是一幅由无数不同色彩的丝线编织而成的挂毯，每一根丝线都为存在的整体复杂性和美丽贡献着力量。从最小的亚原子粒子到浩瀚的星系，一切都以微妙的因果之舞相互连接。生命以其无数种形式在全球繁荣生长，适应着不同的环境，并不断进化以应对不断变化的条件。人类的经验是这宏伟设计中一个独特的方面，其特点是卓越的思考、情感和创造力。我们努力理解我们在宇宙中的位置，解开宇宙的奥秘，并赋予我们的生活意义。我们的历史证明了我们的成就和失败，是一个持续进步和倒退、创新和毁灭的叙述。我们建立文明，发展复杂的社会结构，并通过复杂的语言进行交流，但我们也面临着冲突、不平等以及我们自身死亡的根本问题。对知识的追求驱使我们探索科学和哲学的边界，拓展我们已知和理解的极限。艺术以其各种形式，使我们能够表达内心世界的深度，捕捉转瞬即逝的美丽瞬间，并与他人分享我们的观点。音乐唤起超越语言的情感，在我们心中描绘生动的景象，并在原始的层面上将我们联系起来。文学使我们能够设身处地地为他人着想，体验不同的现实，并反思人类的状况。自然世界以其壮丽的景色、多样的生态系统和复杂的生命网络，为我们提供 sustenance 和灵感。高耸的山脉、广阔的海洋、茂密的森林和干旱的沙漠都拥有其独特的魅力，并为地球丰富的生物多样性做出贡献。我们越来越意识到我们对这种微妙平衡的影响，认识到为了子孙后代保护环境的必要性和责任。我们作为一个全球社区所面临的挑战，如气候变化、贫困和疾病，需要协作的解决方案和对更可持续和公平未来的共同承诺。技术进步促进了我们的相互联系，使得信息和思想能够迅速交流，这既带来了机遇也带来了挑战。数字时代改变了我们交流、工作和学习的方式，为连接和协作提供了新的可能性，同时也引发了对隐私、安全和虚假信息传播的担忧。当我们向前迈进，驾驭 21 世纪的复杂性时，培养批判性思维、同理心和全球公民意识至关重要。理解不同的文化、观点和价值观对于建立桥梁和促进和平共处至关重要。人类文明的旅程是一个持续不断的过程，充满了不确定性和潜力。我们从过去学习、适应现在和展望更美好未来的能力最终将塑造我们物种和我们所居住星球的命运。对意义的探索、对幸福的追求以及渴望对世界产生积极影响是基本的人类愿望，这些愿望继续驱动着我们的行动并塑造着我们的集体叙事。",
    "zh-cn",

    # 【English usage】
    # "Children played happily in the park while their parents watched from nearby benches.",
    # "The sun was setting over the tranquil lake, casting golden reflections across the water's surface. In the distance, mountains stood silhouetted against the fading light, their peaks reaching toward the first stars appearing in the twilight sky. Families were packing up their picnic baskets after a day of relaxation, while a few dedicated fishermen remained at the shoreline, hoping for one last catch before darkness fell completely. The gentle breeze carried the scent of pine trees and wildflowers, creating a perfect end to a beautiful summer day.",
    # "en",

    # 【Japanese usage】
    # "日本の四季は美しく変化に富んでいます。春には桜が咲き誇り、人々は花見を楽しみます。夏には蝉の声が響き、花火大会や祭りが各地で開催されます。秋になると紅葉が山々を彩り、冬には雪景色が広がります。日本の伝統文化も四季と深く結びついており、季節ごとの行事や食べ物があります。和食は世界遺産にも登録され、その繊細な味わいと美しい盛り付けは多くの外国人を魅了しています。日本の技術力も世界的に有名で、精密機器や自動車などの分野で革新を続けています。また、アニメやマンガなどのポップカルチャーも国際的に高い評価を受けており、多くの愛好家がいます。日本の歴史は古く、伝統と現代が共存する独特の文化を形成しています。",
    # "ja",

    gpt_cond_latent,
    speaker_embedding,
    stream_chunk_size=10,
    overlap_wav_len=1024,
    temperature=0.1,
    length_penalty=1.0,
    repetition_penalty=90.0,
    top_k=50,
    speed=1.0,
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
    
    # Save each chunk as mp3 file with timestamp-based naming
    chunk_filename = f"outputs/{timestamp}-{i+1}.mp3"
    # Convert to CPU and ensure right format before saving
    chunk_audio = chunk.squeeze().unsqueeze(0).cpu()
    torchaudio.save(chunk_filename, chunk_audio, 24000, format="mp3")

# Still concatenate for reference but save as mp3 instead of wav
wav = torch.cat(wav_chuncks, dim=0)
torchaudio.save(f"outputs/{timestamp}-full.mp3", wav.squeeze().unsqueeze(0).cpu(), 24000, format="mp3")