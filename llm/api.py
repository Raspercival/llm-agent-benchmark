"""
LLM API 调用层 — 统一封装 DeepSeek / OpenAI 兼容接口。
"""

import os
import time
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMAPI:
    """DeepSeek API 调用封装，兼容 OpenAI SDK。"""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_retries: int = 3,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not set. Create a .env file or set environment variable.")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, messages: list[dict], **kwargs) -> str:
        """发送对话，返回文本回复。自动重试。"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)
        raise RuntimeError(f"API call failed after {self.max_retries} retries: {last_error}")

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """单轮生成：system prompt + user prompt。"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)
