"""
Reflexion 经验库 — 存储失败反思并支持相似检索。

流程：
1. Agent 失败 → LLM 生成反思（为什么失败 + 改进策略）
2. 将 (场景描述, 反思) 存入经验库
3. 新任务开始时检索相似经验，注入 prompt

嵌入方案：sklearn TfidfVectorizer + 余弦相似度。
无需网络下载，纯本地运行，适合几百条规模的轻量检索。
"""

from typing import Optional


class ExperienceBank:
    """基于 TF-IDF 的经验存储与检索。"""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        self.cosine_similarity = cosine_similarity
        self.np = np

        self.experiences: list[dict] = []      # 经验列表
        self._texts: list[str] = []            # 对应文本（用于重新向量化）
        self._matrix = None                    # TF-IDF 矩阵，每次添加后重建

    def add(self, scenario: str, reflection: str, task_type: str = "",
            success: bool = False) -> int:
        """添加一条经验。"""
        exp = {
            "id": len(self.experiences),
            "scenario": scenario[:500],
            "reflection": reflection,
            "task_type": task_type,
            "success": success,
        }
        self.experiences.append(exp)
        # 用场景+反思作为索引文本
        self._texts.append(f"{scenario} {reflection}")
        self._rebuild_index()
        return len(self.experiences)

    def _rebuild_index(self):
        """重建 TF-IDF 矩阵（少量数据下重建很快）。"""
        if self._texts:
            self._matrix = self.vectorizer.fit_transform(self._texts)

    def query(self, scenario: str, k: int = 3,
              task_type: Optional[str] = None) -> list[dict]:
        """检索与当前场景最相似的 k 条经验。"""
        if self.is_empty() or self._matrix is None:
            return []

        query_vec = self.vectorizer.transform([scenario])
        sims = self.cosine_similarity(query_vec, self._matrix)[0]

        indices = self.np.argsort(sims)[::-1]
        results = []
        for i in indices:
            exp = self.experiences[int(i)]
            if task_type and exp["task_type"] != task_type:
                continue
            exp["similarity"] = float(sims[i])
            results.append(exp)
            if len(results) >= k:
                break

        return results

    def query_failures(self, scenario: str, k: int = 3,
                       task_type: Optional[str] = None) -> list[dict]:
        """只检索失败经验。"""
        all_results = self.query(scenario, k=k * 2, task_type=task_type)
        return [r for r in all_results if not r["success"]][:k]

    def is_empty(self) -> bool:
        return len(self.experiences) == 0

    def stats(self) -> dict:
        return {
            "total": len(self.experiences),
            "failures": sum(1 for e in self.experiences if not e["success"]),
            "successes": sum(1 for e in self.experiences if e["success"]),
            "task_types": list(set(e["task_type"] for e in self.experiences)),
        }


# ── 反思生成器 ──

REFLECTION_PROMPT = """You are an AI agent that failed to complete a task. Analyze the failure and write a reflection.

Task: {task_description}

What you did:
{trajectory}

Please write a brief reflection (2-3 sentences) covering:
1. What went wrong and why
2. What you should have done differently
3. A specific strategy to avoid this mistake in the future

Reflection:"""


def generate_reflection(
    llm,
    task_description: str,
    trajectory: list[dict],
) -> str:
    """让 LLM 分析失败轨迹，生成反思。

    Args:
        llm: LLM 实例（需有 generate 方法）
        task_description: 任务描述
        trajectory: 历史步列表 [{"thought": ..., "action": ..., "observation": ...}, ...]

    Returns:
        反思文本
    """
    traj_text = ""
    for h in trajectory[-10:]:  # 最后 10 步
        traj_text += f"Thought: {h.get('thought', '')}\n"
        traj_text += f"Action: {h.get('action', '')}\n"
        traj_text += f"Observation: {h.get('observation', '')[:200]}\n\n"

    prompt = REFLECTION_PROMPT.format(
        task_description=task_description,
        trajectory=traj_text,
    )

    return llm.generate(prompt)


# ── 经验注入 Prompt 构建 ──

def inject_experiences(
    experiences: list[dict],
    current_prompt: str,
) -> str:
    """将检索到的经验注入 prompt 开头。

    Args:
        experiences: query() 返回的经验列表
        current_prompt: 当前 prompt

    Returns:
        注入经验后的 prompt
    """
    if not experiences:
        return current_prompt

    lines = ["## Past experiences (learn from these):"]
    for exp in experiences:
        lines.append(f"- Situation: {exp['scenario'][:200]}")
        lines.append(f"  Lesson: {exp['reflection']}")
    lines.append("")
    lines.append(current_prompt)

    return "\n".join(lines)
