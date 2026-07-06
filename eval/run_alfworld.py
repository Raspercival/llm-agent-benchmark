"""
AlfWorld 评测 — 在 36 个任务上对比 ReAct vs Reflexion。
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.api import LLMAPI
from agents.react import ReActAgent
from agents.reflexion import ReflexionAgent
from envs.alfworld_simple import AlfWorldEnv, TASKS
from eval.runner import EvalRunner


def make_env():
    return AlfWorldEnv()


def make_react():
    return ReActAgent(LLMAPI(temperature=0.0), max_steps=20)


def make_reflexion():
    return ReflexionAgent(api_llm=LLMAPI(temperature=0.0), max_steps=20)


def run_all(config: str = "react", max_tasks: int | None = None):
    """跑 AlfWorld 全量评测。

    Args:
        config: "react" | "reflexion" | "compare"
        max_tasks: 限制任务数（None = 全部 36 个）
    """
    runner = EvalRunner()
    specs = [{"task_id": t["id"]} for t in TASKS]

    if config == "compare":
        runner.compare(
            task_specs=specs,
            make_env=make_env,
            configs=[
                {"name": "alfworld_react", "make_agent": make_react},
                {"name": "alfworld_reflexion", "make_agent": make_reflexion},
            ],
            max_tasks=max_tasks,
        )
    elif config == "react":
        runner.run_benchmark(
            name="alfworld_react",
            make_env=make_env,
            make_agent=make_react,
            task_specs=specs,
            max_tasks=max_tasks,
        )
    elif config == "reflexion":
        runner.run_benchmark(
            name="alfworld_reflexion",
            make_env=make_env,
            make_agent=make_reflexion,
            task_specs=specs,
            max_tasks=max_tasks,
        )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="react", choices=["react", "reflexion", "compare"])
    p.add_argument("--max", type=int, default=None, help="max tasks to run")
    args = p.parse_args()
    run_all(args.config, args.max)
