"""
ReAct Agent — 标准 ReAct 循环。

Thought → Action → Observation → Thought → ...
与 benchmark 无关，通过统一的 env 接口运行。
"""

from typing import Optional


# ── 通用 System Prompt ──

SYSTEM_PROMPT = """You are an AI agent that completes tasks by reasoning and acting.

For each step, respond in exactly this format:

Thought: <your reasoning about the current situation and what to do next>
Action: <the exact action to take>

Available actions depend on the environment. Use the exact action format provided.

Only output ONE Thought+Action pair per turn. Do NOT output Observation — the environment will provide that.

When you have completed the task or cannot proceed further, output:
Thought: I have completed the task.
Action: done"""


# ── AlfWorld Few-shot 示例 ──

ALFWORLD_FEWSHOT = """Examples of completing tasks in a household environment.

## Example 1: Clean + Place

Task: put a clean plate on the desk

Observation:
You are in the kitchen. A cozy cooking space with tiled floors and the smell of garlic.
On the floor: apple, knife, plate.
  fridge (closed) — contains: milk, lettuce.
  microwave (open) — empty.
  sink (open) — contains: bowl.
In your hand: nothing.
Exits: living_room.

Thought: I need to put a clean plate on the desk. The plate is on the floor. I'll pick it up first.
Action: take plate

Observation: You take the plate.

Thought: Now I need to clean the plate. I'm in the kitchen which has a sink, so I can clean it here.
Action: clean plate

Observation: You clean the plate in the sink. It is now clean.

Thought: The plate is clean. Now I need to take it to the desk, which is in the bedroom.
Action: go to living_room

Observation:
You are in the living_room. On the floor: book, remote, vase, newspaper.
  cabinet (closed) — contains: photo, candle.
In your hand: plate.
Exits: kitchen, bedroom.

Thought: I'm in the living room. The bedroom is next.
Action: go to bedroom

Observation:
You are in the bedroom. On the floor: pillow, laptop, pen, mug.
  desk (open) — contains: paper, notebook.
In your hand: plate.
Exits: living_room.

Thought: I'm in the bedroom with the clean plate. The desk is open. I'll put the plate in the desk.
Action: put plate in desk

Observation: You put the plate in the desk. Task complete!

## Example 2: Drop on floor

Task: move the pillow to the living room

Observation:
You are in the bedroom. On the floor: pillow, laptop, pen.
In your hand: nothing.
Exits: living_room.

Thought: I need to move the pillow to the living room. It's on the floor here. I'll pick it up.
Action: take pillow

Observation: You take the pillow.

Thought: Now I'll go to the living room and drop it on the floor.
Action: go to living_room

Observation: You are in the living_room. On the floor: book, remote.
In your hand: pillow.
Exits: kitchen, bedroom.

Thought: I'm in the living room. I'll drop the pillow here.
Action: drop pillow

Observation: You drop the pillow on the floor. Task complete!

## Example 3: Cool (make something cold)

Task: put a cold apple on the sofa

Observation:
You are in the kitchen. On the floor: apple, knife.
  fridge (open) — contains: milk.
In your hand: nothing.
Exits: living_room.

Thought: I need to put a cold apple on the sofa. The apple is on the floor. I'll take it first, then cool it using the cool action. Cooling is done while holding the object in the kitchen.
Action: take apple

Observation: You take the apple.

Thought: I have the apple. Now I need to make it cold. I'm in the kitchen so I can use the cool action.
Action: cool apple

Observation: You cool the apple. It is now cold.

Thought: The apple is now cold. I need to take it to the sofa in the living room.
Action: go to living_room

Observation: You are in the living_room. On the floor: book, remote.
  sofa (open) — contains: cushion.
In your hand: apple.
Exits: kitchen, bedroom.

Thought: I'm in the living room. I need to put the cold apple on the sofa. I'll put it in the sofa.
Action: put apple in sofa

Observation: You put the apple in the sofa. Task complete!

## Example 4: Heat (make something hot)

Task: put a hot knife in the drawer

Observation:
You are in the kitchen. On the floor: apple, knife.
In your hand: nothing.
Exits: living_room.

Thought: I need to make the knife hot and put it in the drawer. I'll take the knife first.
Action: take knife

Observation: You take the knife.

Thought: Now I'll heat it. I'm in the kitchen so I can use the heat action.
Action: heat knife

Observation: You heat the knife. It is now hot.

Thought: The knife is hot. The drawer is in the bedroom. I need to go there.
Action: go to living_room

Observation: You are in the living_room. Exits: kitchen, bedroom.

Thought: Now to the bedroom.
Action: go to bedroom

Observation: You are in the bedroom. On the floor: pillow, laptop.
  drawer (closed) — contains: key, watch.
In your hand: knife.
Exits: living_room.

Thought: The drawer is closed. I need to open it first.
Action: open drawer

Observation: You open the drawer.

Thought: Now I can put the hot knife in the drawer.
Action: put knife in drawer

Observation: You put the knife in the drawer. Task complete!

## Example 5: Two objects

Task: put a knife and a plate in the fridge

Observation:
You are in the kitchen. On the floor: apple, knife, plate.
  fridge (closed) — contains: milk.
In your hand: nothing.
Exits: living_room.

Thought: I need to put both a knife and a plate in the fridge. I'll take the knife first, put it in, then come back for the plate.
Action: take knife

Observation: You take the knife.

Thought: The fridge is closed. I need to open it first.
Action: open fridge

Observation: You open the fridge.

Thought: Now put the knife in.
Action: put knife in fridge

Observation: You put the knife in the fridge.

Thought: Now I need to take the plate and put it in the fridge too.
Action: take plate

Observation: You take the plate.

Thought: The fridge is open. I'll put the plate in.
Action: put plate in fridge

Observation: You put the plate in the fridge. Task complete!

## Example 6: Examine (look and report state)

Task: look at the apple in the fridge and tell me its state

Observation:
You are in the living_room. On the floor: book, remote.
Exits: kitchen, bedroom.

Thought: I need to check the apple in the fridge. The fridge is in the kitchen. I'll go there.
Action: go to kitchen

Observation:
You are in the kitchen. On the floor: knife, plate.
  fridge (closed) — contains: apple, milk.
In your hand: nothing.
Exits: living_room.

Thought: The apple is in the fridge, but the fridge is closed. I need to open it to check.
Action: open fridge

Observation: You open the fridge.

Thought: I can see the apple in the fridge. I need to report its state. Since I haven't changed it, it should be in its default normal state. I'll reply with the state.
Action: apple is normal

Observation: Correct!

## Example 7: Two objects from different rooms

Task: put the newspaper and the pillow in the fridge

Observation:
You are in the kitchen. On the floor: apple, knife.
  fridge (closed) — contains: milk.
In your hand: nothing.
Exits: living_room.

Thought: I need to put both the newspaper and the pillow in the fridge. They are in different rooms. I'll complete one full cycle (take → transport → put) before starting the next. Never drop an object in a random room just to free my hand. Let me get the newspaper from the living room first.
Action: go to living_room

Observation:
You are in the living_room. On the floor: book, remote, newspaper.
Exits: kitchen, bedroom.

Thought: I see the newspaper. I'll take it to the fridge.
Action: take newspaper

Observation: You take the newspaper.

Thought: Back to the kitchen to put it in the fridge.
Action: go to kitchen

Observation:
You are in the kitchen. On the floor: apple, knife.
  fridge (closed) — contains: milk.
In your hand: newspaper.

Thought: Need to open the fridge first.
Action: open fridge

Observation: You open the fridge.

Thought: Put the newspaper in.
Action: put newspaper in fridge

Observation: You put the newspaper in the fridge.

Thought: First object done. Now for the pillow in the bedroom.
Action: go to living_room

Observation: You are in the living_room. Exits: kitchen, bedroom.

Thought: On to the bedroom.
Action: go to bedroom

Observation:
You are in the bedroom. On the floor: pillow, laptop, pen.
Exits: living_room.

Thought: Take the pillow.
Action: take pillow

Observation: You take the pillow.

Thought: Back to the kitchen.
Action: go to living_room

Observation: You are in the living_room. Exits: kitchen, bedroom.

Thought: On to the kitchen.
Action: go to kitchen

Observation:
You are in the kitchen. On the floor: apple, knife.
  fridge (open) — contains: milk, newspaper.
In your hand: pillow.

Thought: Fridge is still open. Put the pillow in.
Action: put pillow in fridge

Observation: You put the pillow in the fridge. Task complete!

--- End of examples ---

CRITICAL RULES:
- Objects sitting in the fridge are NOT automatically cold. You MUST explicitly use the "cool" action while holding the object in the kitchen to make it cold.
- Objects are NOT automatically hot. You MUST explicitly use the "heat" action in the kitchen.
- For two-object tasks: complete the FULL cycle (take → transport → put) for one object before starting the next. NEVER drop an object in a random room just to free your hand — you'll forget it there.
- Use "drop <obj>" only when the goal explicitly says to move something to a room floor.

Now complete the actual task. Follow the same Thought/Action pattern. Available actions: take <obj>, put <obj> in <container>, drop <obj>, go to <room>, open <container>, close <container>, clean <obj>, heat <obj>, cool <obj>, look.

For examine tasks: after finding the object, reply with "<object> is <state>" (e.g. "apple is normal")."""


# ── WebShop Few-shot 示例 ──

WEBSHOP_FEWSHOT = """
Example of shopping on a website:

Task: I need a pair of blue running shoes under 60 dollars, size 9, by Nike

Observation:
Goal: I need a pair of blue running shoes under 60 dollars, size 9, by Nike
Steps taken: 0/20
You are on the WebShop homepage.
Available actions: search[<keywords>]

Thought: The customer wants blue Nike running shoes under $60 in size 9. I should search for these items.
Action: search[blue nike running shoes]

Observation:
Search results (10 found):

  [42] running blue Nike sneakers — $49.99 | Rating: 4.5/5 | blue, Nike, 9
  [58] basketball blue Nike trainers — $89.99 | Rating: 4.2/5 | blue, Nike, 10
  [71] casual blue Adidas sneakers — $55.00 | Rating: 4.1/5 | blue, Adidas, 8
  [105] trail blue New Balance shoes — $72.50 | Rating: 4.3/5 | blue, New Balance, 9
...

Thought: I can see item [42] "running blue Nike sneakers" at $49.99, which is blue, Nike, size 9, and under $60. This matches all criteria. Let me view its details.
Action: click[42]

Observation:
Product #42: running blue Nike sneakers
Category: shoes
Price: $49.99
Rating: 4.5/5
Color: blue
Size: 9
Brand: Nike
Description: running blue Nike sneakers. High quality shoes product. Customers love it — 4.5 stars. Priced at $49.99. In stock.

Thought: This matches perfectly — blue Nike running shoes, size 9, $49.99 (under $60). I should buy it for the customer.
Action: buy[42]

Observation: You purchased: running blue Nike sneakers ($49.99). Score: 0.92. Task complete!

--- End of example ---

Now complete the actual task. Follow the same Thought/Action pattern.
"""


class ReActAgent:
    """标准 ReAct Agent。Thought → Action → Observation 循环。"""

    def __init__(self, llm, max_steps: int = 50, verbose: bool = False):
        self.llm = llm          # 需要有 generate(prompt, system=...) 方法
        self.max_steps = max_steps
        self.verbose = verbose
        self.history: list[dict] = []

    def run(
        self,
        env,
        task_id: int | None = None,
        goal: str | None = None,
    ) -> dict:
        """在指定环境中运行 agent，返回结果字典。

        env 必须提供: reset(...) → obs, step(action) → (obs, reward, done)
        """
        self.history = []

        # 重置环境
        if task_id is not None:
            obs = env.reset(task_id)
            env_type = "alfworld"
        else:
            obs = env.reset(goal)
            env_type = "webshop"

        fewshot = ALFWORLD_FEWSHOT if env_type == "alfworld" else WEBSHOP_FEWSHOT

        if self.verbose:
            print(f"[ReAct] Starting {env_type} task...")
            print(obs)
            print("---")

        total_reward = 0.0

        for step in range(self.max_steps):
            # 构建 prompt
            prompt = self._build_prompt(obs, fewshot)

            # 调用 LLM
            response = self.llm.generate(prompt, system=SYSTEM_PROMPT)

            # 解析 LLM 输出
            thought, action = self._parse_response(response)

            self.history.append({
                "step": step,
                "observation": obs,
                "thought": thought,
                "action": action,
            })

            if self.verbose:
                print(f"Thought: {thought}")
                print(f"Action: {action}")

            # 停止条件
            if action == "done" or action is None:
                break

            # 执行动作
            obs, reward, done = env.step(action)
            total_reward += reward

            if self.verbose:
                print(f"Observation: {obs[:200]}...")
                print("---")

            if done:
                break

        return {
            "task_id": task_id,
            "goal": goal,
            "env_type": env_type,
            "steps_taken": len(self.history),
            "total_reward": total_reward,
            "success": total_reward > 0,
            "history": self.history,
        }

    def _build_prompt(self, obs: str, fewshot: str) -> str:
        """构建当前步的 prompt：few-shot 示例 + 历史 + 当前观察。"""
        lines = [fewshot, ""]

        # 历史（最近 5 轮，防止 context 溢出）
        for h in self.history[-5:]:
            lines.append(f"Observation: {h['observation']}")
            lines.append(f"Thought: {h['thought']}")
            lines.append(f"Action: {h['action']}")
            lines.append("")

        # 当前观察
        lines.append(f"Current observation:")
        lines.append(obs)
        lines.append("")
        lines.append("What is your next thought and action?")

        return "\n".join(lines)

    def _parse_response(self, response: str) -> tuple[str, Optional[str]]:
        """解析 LLM 输出 → (thought, action)。"""
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
