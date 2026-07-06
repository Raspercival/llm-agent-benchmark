"""
本地模型推理 — llama.cpp 封装，用于 Router 的快路径。
"""

import os
from typing import Optional


class LocalLLM:
    """llama.cpp 本地模型推理。用于简单任务的快速响应。"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 2048,
        n_threads: int = 8,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Run: pip install llama-cpp-python"
            )

        if model_path is None:
            model_path = os.path.join("models", "qwen2.5-1.5b-instruct-q4_k_m.gguf")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Download a GGUF model and place it in models/ directory."
            )

        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本回复。"""
        output = self.model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return output["choices"][0]["message"]["content"]


def is_local_model_available() -> bool:
    """检查本地模型是否可用。"""
    model_path = os.path.join("models", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
    return os.path.exists(model_path)
