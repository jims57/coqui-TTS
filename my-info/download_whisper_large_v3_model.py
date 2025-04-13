from huggingface_hub import hf_hub_download

# Download the Whisper large-v3 model
hf_hub_download(repo_id="openai/whisper-large-v3", filename="pytorch_model.bin", local_dir="/Users/mac/jims57/模型备份/ASR/Whisper/whisper_large_v3")