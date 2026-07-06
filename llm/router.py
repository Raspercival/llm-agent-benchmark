"""
Router LLM — 快慢双系统分流。
简单任务 → 本地小模型（快/省钱），复杂任务 → API 大模型（准/贵）。
"""

from typing import Optional

from .api import LLMAPI
from .local import LocalLLM, is_local_model_available


class LLMRouter:
    """根据任务复杂度自动选择本地模型或 API。

    判断依据：
    - 动作候选数：候选多 → 决策空间大 → 用大模型
    - 历史步数：步数多 → 容易混乱 → 用大模型
    - 循环检测：连续重复动作 → 陷入循环 → 用大模型突破
    """

    # 阈值
    ACTION_COUNT_THRESHOLD = 5        # 可选动作超过此数 → API
    HISTORY_LENGTH_THRESHOLD = 3      # 历史步数超过此数 → API
    REPEAT_ACTION_THRESHOLD = 2       # 连续重复动作次数 → API

    def __init__(
        self,
        api_model: Optional[LLMAPI] = None,
        local_model: Optional[LocalLLM] = None,
    ):
        self.api = api_model or LLMAPI()
        self.local = local_model
        if self.local is None and is_local_model_available():
            self.local = LocalLLM()

    @property
    def has_local(self) -> bool:
        return self.local is not None

    def decide(self, num_actions: int, history: list[dict]) -> str:
        """判断用哪个模型，返回 "api" 或 "local"。"""
        # 无本地模型 → 一律 API
        if not self.has_local:
            return "api"

        # 规则 1：动作太多
        if num_actions > self.ACTION_COUNT_THRESHOLD:
            return "api"

        # 规则 2：历史太长
        if len(history) > self.HISTORY_LENGTH_THRESHOLD:
            return "api"

        # 规则 3：陷入循环（连续相同 action）
        if self._is_stuck(history):
            return "api"

        return "local"

    def generate(
        self,
        prompt: str,
        num_actions: int,
        history: list[dict],
        system: Optional[str] = None,
    ) -> str:
        """自动选择模型并生成回复。本地失败时自动 fallback 到 API。"""
        backend = self.decide(num_actions, history)

        if backend == "local" and self.local:
            try:
                return self.local.generate(prompt)
            except Exception as e:
                print(f"  [Router] Local model failed ({e}), falling back to API")

        return self.api.generate(prompt, system=system)

    def _is_stuck(self, history: list[dict]) -> bool:
        """检测是否重复同一动作（陷入循环）。"""
        if len(history) < self.REPEAT_ACTION_THRESHOLD:
            return False

        recent_actions = [
            h.get("action", "") for h in history[-self.REPEAT_ACTION_THRESHOLD:]
        ]
        return len(set(recent_actions)) == 1
