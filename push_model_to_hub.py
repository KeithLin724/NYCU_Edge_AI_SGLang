# Upload to Hugging Face Hub
from huggingface_hub import login
from SGLangModel import SgLangModel


# 首先登入 (只需要做一次)
login()  # 這會提示你輸入 token，或者你可以傳入 token 參數

sg_lang_model = SgLangModel("llama-3.2-1b-KD-V1-W8A8-Dynamic-Per-Token-V2")

model, tokenizer = sg_lang_model.build_raw_model()

# 設定 Hub 上的模型名稱
HUB_MODEL_NAME = "KYLiN724/llama-3.2-1b-KD-V1-W8A8-Dynamic-Per-Token"

# 上傳 tokenizer
tokenizer.push_to_hub(HUB_MODEL_NAME)

# 上傳模型
model.push_to_hub(HUB_MODEL_NAME)

print(f"Model uploaded to: https://huggingface.co/{HUB_MODEL_NAME}")
