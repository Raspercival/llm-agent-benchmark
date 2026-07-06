"""
WebShop 评测 — 从实际商品生成目标，保证可达成。
"""

import os
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.api import LLMAPI
from agents.react import ReActAgent
from agents.reflexion import ReflexionAgent
from envs.webshop_simple import WebShopEnv, PRODUCTS
from envs.webshop_simple.env import _generate_goal_from_product
from eval.runner import EvalRunner


random.seed(123)


def make_env():
    return WebShopEnv()


def make_react():
    return ReActAgent(LLMAPI(temperature=0.0), max_steps=10)


def make_reflexion():
    return ReflexionAgent(api_llm=LLMAPI(temperature=0.0), max_steps=10)


def run_all(config: str = "react", max_tasks: int | None = None):
    runner = EvalRunner()
    # 从实际商品反推目标，保证每个目标都能买到
    specs = [{"goal": _generate_goal_from_product(random.choice(PRODUCTS))}
             for _ in range(25)]

    if config == "compare":
        runner.compare(
            task_specs=specs,
            make_env=make_env,
            configs=[
                {"name": "webshop_react", "make_agent": make_react},
                {"name": "webshop_reflexion", "make_agent": make_reflexion},
            ],
            max_tasks=max_tasks,
        )
    elif config == "react":
        runner.run_benchmark(
            name="webshop_react",
            make_env=make_env,
            make_agent=make_react,
            task_specs=specs,
            max_tasks=max_tasks,
        )
    elif config == "reflexion":
        runner.run_benchmark(
            name="webshop_reflexion",
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
