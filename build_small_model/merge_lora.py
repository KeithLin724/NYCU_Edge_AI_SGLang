import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base quantized model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Load PEFT adapters
model = PeftModel.from_pretrained(base_model, "<YOUR_OUTPUT_DIR>/epoch_0")
model = model.merge_and_unload()

tokenizer.save_pretrained("<YOUR_MERGED_MODEL_PATH>")
model.save_pretrained("<YOUR_MERGED_MODEL_PATH>")