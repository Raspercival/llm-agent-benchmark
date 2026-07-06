"""
评测运行器 — 批量跑 benchmark，支持多种 agent 配置。
"""

import csv
import os
import time
from datetime import datetime


class EvalRunner:
    """通用评测运行器。"""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def run_benchmark(
        self,
        name: str,
        make_env,
        make_agent,
        task_specs: list[dict],
        max_tasks: int | None = None,
    ) -> list[dict]:
        """运行评测并保存结果。

        Args:
            name: 实验名（如 "react_baseline_alfworld"）
            make_env: () → env 工厂函数
            make_agent: () → agent 工厂函数
            task_specs: 任务规格列表 [{"task_id": 1} | {"goal": "..."}]
            max_tasks: 上限（None = 全部）

        Returns:
            结果列表
        """
        results = []
        specs = task_specs[:max_tasks] if max_tasks else task_specs
        total = len(specs)

        print(f"\n{'='*60}")
        print(f"Experiment: {name}")
        print(f"Tasks: {total}")
        print(f"{'='*60}")

        for i, spec in enumerate(specs):
            env = make_env()
            agent = make_agent()

            task_id = spec.get("task_id")
            goal = spec.get("goal")
            label = f"#{task_id}" if task_id else goal[:40]

            start = time.time()
            try:
                r = agent.run(env, task_id=task_id, goal=goal)
            except Exception as e:
                r = {"success": False, "total_reward": 0, "steps_taken": 0,
                     "history": [], "error": str(e)}

            elapsed = time.time() - start

            record = {
                "experiment": name,
                "task_id": task_id,
                "goal": goal,
                "success": r.get("success", False),
                "reward": r.get("total_reward", 0),
                "steps": r.get("steps_taken", 0),
                "time": round(elapsed, 1),
                "error": r.get("error", ""),
            }
            results.append(record)

            status = "OK" if record["success"] else "FAIL"
            print(f"  [{i+1}/{total}] {label} | {status} | "
                  f"steps={record['steps']} reward={record['reward']:.2f} "
                  f"time={elapsed:.1f}s")

        self._save_csv(name, results)
        self._print_summary(name, results)
        return results

    def compare(
        self,
        task_specs: list[dict],
        make_env,
        configs: list[dict],
        max_tasks: int | None = None,
    ) -> dict:
        """对比多种 agent 配置。

        Args:
            configs: [{"name": "ReAct", "make_agent": fn, "reflexion": False, "router": False}, ...]

        Returns:
            {"ReAct": [results...], "Reflexion": [results...]}
        """
        all_results = {}
        specs = task_specs[:max_tasks] if max_tasks else task_specs

        for cfg in configs:
            results = self.run_benchmark(
                name=cfg["name"],
                make_env=make_env,
                make_agent=cfg["make_agent"],
                task_specs=specs,
            )
            all_results[cfg["name"]] = results

        self._print_comparison(configs, all_results)
        return all_results

    def _save_csv(self, name: str, results: list[dict]):
        path = os.path.join(self.results_dir, f"{name}.csv")
        if not results:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)

    def _print_summary(self, name: str, results: list[dict]):
        if not results:
            return
        n = len(results)
        ok = sum(1 for r in results if r["success"])
        avg_reward = sum(r["reward"] for r in results) / n
        avg_steps = sum(r["steps"] for r in results) / n
        avg_time = sum(r["time"] for r in results) / n

        print(f"\n--- {name} Summary ---")
        print(f"  Success: {ok}/{n} ({ok/n*100:.1f}%)")
        print(f"  Avg reward: {avg_reward:.3f}")
        print(f"  Avg steps: {avg_steps:.1f}")
        print(f"  Avg time: {avg_time:.1f}s")

    def _print_comparison(self, configs: list[dict], all_results: dict):
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"{'Method':<25} {'Success':>8} {'Avg Reward':>12} {'Avg Steps':>10} {'Avg Time':>9}")
        print("-" * 65)
        for cfg in configs:
            results = all_results[cfg["name"]]
            n = len(results)
            ok = sum(1 for r in results if r["success"])
            avg_r = sum(r["reward"] for r in results) / n
            avg_s = sum(r["steps"] for r in results) / n
            avg_t = sum(r["time"] for r in results) / n
            print(f"{cfg['name']:<25} {ok:>3}/{n:<4} {avg_r:>12.3f} {avg_s:>10.1f} {avg_t:>8.1f}s")
