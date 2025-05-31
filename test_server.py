# %%
import openai
import time

# from sglang.utils import wait_for_server, print_highlight

port = 30000

client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")


model_name = "meta-llama/Llama-3.2-3B-Instruct"

# %%


start_time = time.time()
response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=256,
)
end_time = time.time()
print(f"Run time: {end_time - start_time:.4f} seconds")
# %%
print(response.choices[0].message.content)
tokens = response.usage.total_tokens
print(f"Total tokens: {tokens}")
# %%
import requests
import json

port = 30000
# 使用 SGLang 的原生 API
url = f"http://127.0.0.1:{port}/generate"
data = {
    "text": "List 3 countries and their capitals.",
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": 64,
        "return_logprob": True,  # 返回 logprobs
        "logprob_start_len": 0,  # 從開始位置返回 logprobs
        "top_logprobs_num": 5,  # 返回 top-k logprobs
    },
}

response = requests.post(url, json=data)
# %%
print(response)

# %%
result = response.json()

# 訪問 logprobs
if "meta_info" in result and "prompt_logprobs" in result["meta_info"]:
    logprobs = result["meta_info"]["prompt_logprobs"]
    print("Logprobs:", logprobs)

# %%
import requests
from rich import print

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
    },
)
print(response.json())

# %%

from SGLangModel import SgLangModel

model = SgLangModel(model_name="Llama-3.2-3B-Instruct-W8A8-Dynamic-Per-Token-V2")

response = model.openai_request(
    prompt="List 3 countries and their capitals.",
    temperature=0.0,
    max_tokens=256,
)
print(response.choices[0].message.content)
# %%
tokens = response.usage.total_tokens
print(f"Total tokens: {tokens}")

response = model.python_request(
    json_data={
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
    }
)
print(response.json())
# %%
print(response.status_code)
