"""
AlfWorld 手搓版 — Step 3: 环境交互。

标准接口：reset(task) → obs | step(action) → (obs, reward, done)
"""

import re
from .world import WorldState, build_world, ObjState
from .tasks import get_task_by_id, TASKS


class AlfWorldEnv:
    """AlfWorld 文本冒险环境。"""

    MAX_STEPS = 50

    # ── TODO: 定义所有可用的动作模板 ──
    # 提示：8 种动作，每种有固定格式
    ACTIONS = [
        "go to <room>",
        "take <object>",
        "put <object> in <container>",
        "drop <object>",
        "open <container>",
        "close <container>",
        "clean <object>",
        "heat <object>",
        "cool <object>",
        "look",
    ]

    def __init__(self):
        self.world: WorldState = None
        self.task: dict = None
        self.steps_taken: int = 0
        self.done: bool = False
        self.last_action: str = ""           # 用于 examine 类型的额外判断

    def reset(self, task_id: int) -> str:
        self.world = build_world()
        self.task = get_task_by_id(task_id)
        if self.task is None:
            raise ValueError(f"Task {task_id} not found. Valid: {[t['id'] for t in TASKS]}")
        self.steps_taken = 0
        self.done = False
        self.last_action = ""

        obs = self.world.look()
        hint = ""
        if self.task["type"] == "examine":
            hint = ("\n(Hint: find the object, then reply with its state. "
                    "Format: '<object> is <state>'. States: normal, clean, dirty, hot, cold)")
        return f"Task: {self.task['description']}{hint}\n\n{obs}"

    def step(self, action: str) -> tuple[str, float, bool]:
        action = action.strip()
        self.last_action = action

        # 允许 examine 任务的 agent 直接回复状态（格式："<obj> is <state>"）
        if self.task["type"] == "examine" and self._is_state_reply(action):
            self.done = True
            correct = self._check_examine_reply(action)
            if correct:
                return ("Correct!", 1.0, True)
            else:
                return ("Wrong answer.", 0.0, True)

        if self.done:
            return ("Task already finished.", 0.0, True)

        self.steps_taken += 1
        if self.steps_taken >= self.MAX_STEPS:
            self.done = True
            return ("Too many steps. Task failed.", 0.0, True)

        act_type, params = self._parse_action(action)
        msg = self._execute(act_type, params)

        if self._check_goal():
            self.done = True
            return (f"{msg}\n\nTask complete!", 1.0, True)

        return (msg, 0.0, False)

    def get_valid_actions(self) -> list[str]:
        room = self.world._room()
        actions = ["look"]

        for exit_name in room.exits:
            actions.append(f"go to {exit_name}")

        if not self.world.holding:
            for obj in room.objects_on_floor:
                actions.append(f"take {obj}")
        else:
            actions.append(f"drop {self.world.holding}")
            for name, cont in room.containers.items():
                if cont["state"] == "open":
                    actions.append(f"put {self.world.holding} in {name}")
            if self.world.current_room == "kitchen":
                actions.append(f"clean {self.world.holding}")
                actions.append(f"heat {self.world.holding}")
                actions.append(f"cool {self.world.holding}")

        for name, cont in room.containers.items():
            if cont["state"] == "open":
                actions.append(f"close {name}")
            else:
                actions.append(f"open {name}")

        # 打开其他房间的容器需要先过去
        # 这里只列当前房间的，简化处理

        return actions

    # ── 内部方法 ──

    def _parse_action(self, action: str) -> tuple[str, dict]:
        action = action.strip().lower()

        if m := re.match(r"go to (.+)", action):
            return ("go_to", {"room": m.group(1)})

        if m := re.match(r"put (.+) in (.+)", action):
            return ("put_in", {"object": m.group(1), "container": m.group(2)})

        if m := re.match(r"drop (.+)", action):
            return ("drop", {"object": m.group(1)})

        if m := re.match(r"take (.+)", action):
            return ("take", {"object": m.group(1)})

        if m := re.match(r"open (.+)", action):
            return ("open", {"container": m.group(1)})

        if m := re.match(r"close (.+)", action):
            return ("close", {"container": m.group(1)})

        if m := re.match(r"clean (.+)", action):
            return ("clean", {"object": m.group(1)})

        if m := re.match(r"heat (.+)", action):
            return ("heat", {"object": m.group(1)})

        if m := re.match(r"cool (.+)", action):
            return ("cool", {"object": m.group(1)})

        if action in ("look", "look around"):
            return ("look", {})

        return ("unknown", {})

    def _execute(self, act_type: str, params: dict) -> str:
        w = self.world

        if act_type == "go_to":
            return w.walk_to(params["room"])
        elif act_type == "take":
            return w.take(params["object"])
        elif act_type == "put_in":
            return w.put_in(params["object"], params["container"])
        elif act_type == "drop":
            return w.drop(params["object"])
        elif act_type == "open":
            return w.open_container(params["container"])
        elif act_type == "close":
            return w.close_container(params["container"])
        elif act_type == "clean":
            return w.clean(params["object"])
        elif act_type == "heat":
            return w.heat(params["object"])
        elif act_type == "cool":
            return w.cool(params["object"])
        elif act_type == "look":
            return w.look()
        else:
            return f"Unknown action. Try: take <obj>, go to <room>, open <container>, etc."

    def _check_goal(self) -> bool:
        w = self.world
        t = self.task
        gtype = t["type"]

        if gtype == "examine":
            return False  # examine 走特殊通道

        gs = t["goal_state"]
        for obj, want in gs.items():
            room, container = w.find_obj_location(obj)

            # 检查位置
            loc = want.get("location")
            if loc:
                actual = container if container else room
                if actual != loc:
                    return False

            # 检查物d体状态
            state = want.get("state")
            if state:
                actual_state = w.obj_states.get(obj, ObjState.NORMAL)
                if actual_state.value != state:
                    return False

        return True

    def _is_state_reply(self, action: str) -> bool:
        """判断是否为 examine 任务的状态回复（如 'apple is normal'）。"""
        return bool(re.match(r".+ is (normal|clean|dirty|hot|cold)", action.lower()))

    def _check_examine_reply(self, action: str) -> bool:
        """验证 examine 回复是否正确。"""
        t = self.task
        w = self.world
        obj = t["target_obj"]
        state = w.obj_states.get(obj, ObjState.NORMAL).value
        return state in action.lower() and obj in action.lower()
