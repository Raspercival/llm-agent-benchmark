"""
Reflexion Agent — ReAct + Router LLM + 经验库。

核心改进：
1. 每步前检索相似失败经验，注入 prompt
2. Router 自动分流：简单步 → 本地小模型，复杂步 → API
3. 失败后 LLM 自省生成反思，存回经验库
"""

from typing import Optional

from llm.api import LLMAPI
from llm.router import LLMRouter
from memory.experience import (
    ExperienceBank,
    generate_reflection,
    inject_experiences,
)
from agents.react import (
    SYSTEM_PROMPT,
    ALFWORLD_FEWSHOT,
    WEBSHOP_FEWSHOT,
)


class ReflexionAgent:
    """ReAct + Router + Reflexion 改进版 Agent。"""

    def __init__(
        self,
        api_llm: Optional[LLMAPI] = None,
        router: Optional[LLMRouter] = None,
        experience_bank: Optional[ExperienceBank] = None,
        max_steps: int = 50,
        verbose: bool = False,
        use_reflexion: bool = True,
        use_router: bool = True,
    ):
        self.api = api_llm or LLMAPI()
        self.router = router or LLMRouter(api_model=self.api)
        self.bank = experience_bank or ExperienceBank()
        self.max_steps = max_steps
        self.verbose = verbose
        self.use_reflexion = use_reflexion
        self.use_router = use_router
        self.history: list[dict] = []

    def run(
        self,
        env,
        task_id: int | None = None,
        goal: str | None = None,
    ) -> dict:
        """运行 agent。

        Args:
            env: 环境实例（需提供 reset / step 接口）
            task_id: AlfWorld 任务 ID
            goal: WebShop 任务目标
            use_reflexion: 是否启用经验检索
            use_router: 是否启用 Router 分流

        Returns:
            结果字典
        """
        self.history = []

        # 重置环境
        if task_id is not None:
            obs = env.reset(task_id)
            env_type = "alfworld"
            task_desc = env.task["description"]
            task_type_tag = env.task["type"]
        else:
            obs = env.reset(goal)
            env_type = "webshop"
            task_desc = goal or ""
            task_type_tag = "webshop"

        fewshot = ALFWORLD_FEWSHOT if env_type == "alfworld" else WEBSHOP_FEWSHOT

        if self.verbose:
            print(f"[Reflexion] {env_type} task: {task_desc}")
            print(f"[Reflexion] Router={'on' if self.use_router else 'off'}, "
                  f"Reflexion={'on' if self.use_reflexion else 'off'}")
            print(obs)
            print("---")

        total_reward = 0.0
        scenario = f"Task: {task_desc}\n{obs}"

        for step in range(self.max_steps):
            # Step A: 检索相关经验
            experiences = []
            if self.use_reflexion and not self.bank.is_empty():
                experiences = self.bank.query_failures(
                    f"{task_desc} {obs}", k=3, task_type=task_type_tag
                )

            # Step B: 构建 prompt
            prompt = self._build_prompt(obs, fewshot, experiences)

            # Step C: 选择 LLM 并生成
            if self.use_router and self.router.has_local:
                # 估算动作数：AlfWorld 用 get_valid_actions，WebShop 固定 3 类
                num_actions = self._estimate_actions(env)
                response = self.router.generate(
                    prompt, num_actions, self.history, system=SYSTEM_PROMPT
                )
            else:
                response = self.api.generate(prompt, system=SYSTEM_PROMPT)

            # Step D: 解析
            thought, action = self._parse_response(response)

            self.history.append({
                "step": step,
                "observation": obs,
                "thought": thought,
                "action": action,
                "experiences_used": len(experiences),
            })

            if self.verbose:
                if experiences:
                    print(f"[Memory] {len(experiences)} related experiences found")
                print(f"Thought: {thought}")
                print(f"Action: {action}")

            # 停止
            if action == "done" or action is None:
                break

            # Step E: 执行
            obs, reward, done = env.step(action)
            total_reward += reward

            if self.verbose:
                print(f"Obs: {obs[:150]}...")
                print("---")

            if done:
                break

        success = total_reward > 0

        # Step F: 失败则反思
        if not success and self.use_reflexion:
            reflection = generate_reflection(self.api, task_desc, self.history)
            self.bank.add(
                scenario=scenario,
                reflection=reflection,
                task_type=task_type_tag,
                success=False,
            )
            if self.verbose:
                print(f"[Reflexion] Stored reflection: {reflection[:120]}...")
        elif success and self.use_reflexion:
            # 也存成功经验（带正样本）
            self.bank.add(
                scenario=scenario,
                reflection="",
                task_type=task_type_tag,
                success=True,
            )

        return {
            "task_id": task_id,
            "goal": goal,
            "env_type": env_type,
            "steps_taken": len(self.history),
            "total_reward": total_reward,
            "success": success,
            "history": self.history,
        }

    # ── 内部方法 ──

    def _build_prompt(
        self, obs: str, fewshot: str, experiences: list[dict]
    ) -> str:
        lines = []

        # 注入经验
        if experiences:
            lines.append("## Past experiences (learn from these):")
            for exp in experiences:
                lines.append(f"- Lesson: {exp['reflection']}")
            lines.append("")

        lines.append(fewshot)
        lines.append("")

        # 历史（最近 5 轮）
        for h in self.history[-5:]:
            lines.append(f"Observation: {h['observation']}")
            lines.append(f"Thought: {h['thought']}")
            lines.append(f"Action: {h['action']}")
            lines.append("")

        lines.append("Current observation:")
        lines.append(obs)
        lines.append("")
        lines.append("What is your next thought and action?")

        return "\n".join(lines)

    def _parse_response(self, response: str) -> tuple[str, Optional[str]]:
        thought = ""
        action = ""
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.lower().startswith("thought:"):
                thought = line.split(":", 1)[1].strip()
            elif line.lower().startswith("action:"):
                action = line.split(":", 1)[1].strip()
        if not action and response.strip():
            action = response.strip()
        return thought, action if action else None

    def _estimate_actions(self, env) -> int:
        """估算当前可用的动作数（给 Router 做复杂度判断）。"""
        if hasattr(env, "get_valid_actions"):
            try:
                return len(env.get_valid_actions())
            except Exception:
                pass
        # WebShop 默认 3 种动作类型
        return 3
