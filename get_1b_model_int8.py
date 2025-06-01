from transformers import AutoTokenizer, AutoModelForCausalLM


model_name = "KYLiN724/llama-3.2-1b-KD-V1-W8A8-Dynamic-Per-Token"
save_path = "llama-3.2-1b-KD-V1-W8A8-Dynamic-Per-Token"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
