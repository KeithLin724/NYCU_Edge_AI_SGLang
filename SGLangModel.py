import openai
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM


class SgLangModel:

    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 30000,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    ):
        self._ip = ip
        self._port = port
        self._model_name = model_name
        self._heard_url = f"http://{self._ip}:{self._port}"
        self._client = openai.Client(base_url=f"{self._heard_url}/v1", api_key="None")

        return

    def openai_request(
        self, prompt: str, temperature: float = 0.0, max_tokens: int = 256
    ):
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response

    def python_request(self, json_data: str):
        url = f"{self._heard_url}/generate"
        response = requests.post(url, json=json_data)
        return response

    def build_raw_model(self, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(self._model_name, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        return model, tokenizer
