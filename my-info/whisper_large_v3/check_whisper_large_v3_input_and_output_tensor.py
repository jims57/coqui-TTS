from transformers import AutoModel

# Load your model
model = AutoModel.from_pretrained("openai/whisper-large-v3")

# Print model's input specifications
print(model.config)
# For more detailed info about inputs
print(model.forward.__doc__)