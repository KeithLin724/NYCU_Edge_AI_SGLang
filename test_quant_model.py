from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

# Choose a model from Hugging Face Hub
model_name = "./Llama-3.2-3B-Instruct-W8A8-Dynamic-Per-Token"
model_name_tokenizer = "meta-llama/Llama-3.2-3B-Instruct"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare a dummy input
dummy_text = "Hello, this is a dummy input."
inputs = tokenizer(dummy_text, return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

print("Model output:", outputs)
