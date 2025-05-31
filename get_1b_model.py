from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("hlhsiao/llama-3.2-1b-KD-V1")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

model.save_pretrained("llama-3.2-1b-KD-V1")
tokenizer.save_pretrained("llama-3.2-1b-KD-V1")
