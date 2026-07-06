# LLM Agent Benchmark

> 大一暑期大作业 — 方向四：大模型智能体

基于 ReAct + Reflexion 的 LLM Agent，在 WebShop 和 AlfWorld 两个 benchmark 上评测。

## 创新点

1. **Router LLM（快慢双系统）**：简单任务本地跑 Qwen2.5-1.5B，复杂任务调 DeepSeek-V3 API
2. **Reflexion 自省机制**：失败后自动反思并存入经验库，后续任务检索相似经验避免重蹈覆辙

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env 填入你的 DeepSeek API Key

# 3. 安装 benchmark 环境（见下方说明）

# 4. 运行实验
python eval/run_webshop.py
python eval/run_alfworld.py
```

## 项目结构

```
├── agents/          # Agent 实现（ReAct / Reflexion）
├── llm/             # LLM 调用层（Router / API / Local）
├── memory/          # Reflexion 经验库
├── eval/            # 评测脚本
├── results/         # 实验数据
└── report.md        # 课程报告
```

## License

MIT
