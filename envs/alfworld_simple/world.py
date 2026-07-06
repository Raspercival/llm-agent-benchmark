"""
AlfWorld 手搓版 — Step 1: 世界定义。

数据结构：
- rooms: dict[str, Room]，每个房间有物体、容器、出口
- 物体状态: clean/dirty, hot/cold, 位置（房间地面/容器内/手上）
- 容器状态: open/closed
"""

from dataclasses import dataclass, field
from enum import Enum


class ObjState(Enum):
    NORMAL = "normal"
    CLEAN = "clean"
    DIRTY = "dirty"
    HOT = "hot"
    COLD = "cold"


class ContainerState(Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class Room:
    name: str
    description: str
    objects_on_floor: list[str]          # 地上的物体
    containers: dict[str, dict] = field(default_factory=dict)
    # containers: {"fridge": {"state": "closed", "contains": ["apple"]}}
    exits: list[str] = field(default_factory=list)


@dataclass
class WorldState:
    """世界当前状态。"""
    rooms: dict[str, Room]
    current_room: str
    holding: str | None = None                    # 手上拿的东西
    obj_states: dict[str, ObjState] = field(default_factory=dict)   # 每个物体的状态

    # ── 内部工具方法 ──

    def _room(self) -> Room:
        return self.rooms[self.current_room]

    def _find_obj(self, obj: str) -> str | None:
        """找物体在当前房间的位置。返回 None | "floor" | "<container>"。"""
        room = self._room()
        if obj in room.objects_on_floor:
            return "floor"
        for name, cont in room.containers.items():
            if obj in cont["contains"]:
                return name
        return None

    def find_obj_location(self, obj: str) -> tuple[str | None, str | None]:
        """全局查找物体。返回 (房间名, 容器名或None=地面)。"""
        if self.holding == obj:
            return (self.current_room, "hand")
        for room_name, room in self.rooms.items():
            if obj in room.objects_on_floor:
                return (room_name, None)
            for cont_name, cont in room.containers.items():
                if obj in cont["contains"]:
                    return (room_name, cont_name)
        return (None, None)

    # ── 玩家动作 ──

    def look(self) -> str:
        room = self._room()

        lines = [f"You are in the {room.name}. {room.description}"]

        # 地上物体
        if room.objects_on_floor:
            lines.append(f"On the floor: {', '.join(room.objects_on_floor)}.")
        else:
            lines.append("On the floor: nothing.")

        # 容器
        for name, cont in room.containers.items():
            if cont["contains"]:
                items = ", ".join(cont["contains"])
                lines.append(f"  {name} ({cont['state']}) — contains: {items}.")
            else:
                lines.append(f"  {name} ({cont['state']}) — empty.")

        # 手上
        if self.holding:
            lines.append(f"In your hand: {self.holding}.")
        else:
            lines.append("In your hand: nothing.")

        # 出口
        lines.append(f"Exits: {', '.join(room.exits)}.")

        return "\n".join(lines)

    def walk_to(self, target_room: str) -> str:
        room = self._room()
        if target_room not in room.exits:
            return f"You can't go to {target_room} from here. Exits: {', '.join(room.exits)}."
        if target_room not in self.rooms:
            return f"No room named '{target_room}'."
        self.current_room = target_room
        return self.look()

    def take(self, obj: str) -> str:
        if self.holding:
            return f"You're already holding {self.holding}. Put it down first."
        room = self._room()

        # 先检查地上
        if obj in room.objects_on_floor:
            room.objects_on_floor.remove(obj)
            self.holding = obj
            return f"You take the {obj}."

        # 再检查开着容器里
        for name, cont in room.containers.items():
            if obj in cont["contains"]:
                if cont["state"] != "open":
                    return f"The {obj} is in the {name}, but it's closed. Open it first."
                cont["contains"].remove(obj)
                self.holding = obj
                return f"You take the {obj} from the {name}."

        return f"No {obj} here."

    def put_in(self, obj: str, container: str) -> str:
        if self.holding != obj:
            return f"You're not holding {obj}."
        room = self._room()
        if container not in room.containers:
            return f"No {container} here."
        if room.containers[container]["state"] != "open":
            return f"The {container} is closed. Open it first."
        self.holding = None
        room.containers[container]["contains"].append(obj)
        return f"You put the {obj} in the {container}."

    def drop(self, obj: str) -> str:
        """把手中的物品放在当前房间地面上。"""
        if self.holding != obj:
            return f"You're not holding {obj}."
        room = self._room()
        self.holding = None
        room.objects_on_floor.append(obj)
        return f"You drop the {obj} on the floor."

    def open_container(self, container: str) -> str:
        room = self._room()
        if container not in room.containers:
            return f"No {container} here."
        if room.containers[container]["state"] == "open":
            return f"The {container} is already open."
        room.containers[container]["state"] = "open"
        return f"You open the {container}."

    def close_container(self, container: str) -> str:
        room = self._room()
        if container not in room.containers:
            return f"No {container} here."
        if room.containers[container]["state"] == "closed":
            return f"The {container} is already closed."
        room.containers[container]["state"] = "closed"
        return f"You close the {container}."

    def clean(self, obj: str) -> str:
        if self.holding != obj:
            return f"You're not holding {obj}. Pick it up first."
        if self.current_room != "kitchen":
            return "You need a sink to clean things. Go to the kitchen."
        self.obj_states[obj] = ObjState.CLEAN
        return f"You clean the {obj} in the sink. It is now clean."

    def heat(self, obj: str) -> str:
        if self.holding != obj:
            return f"You're not holding {obj}. Pick it up first."
        if self.current_room != "kitchen":
            return "You need a microwave or stove to heat things. Go to the kitchen."
        self.obj_states[obj] = ObjState.HOT
        return f"You heat the {obj}. It is now hot."

    def cool(self, obj: str) -> str:
        if self.holding != obj:
            return f"You're not holding {obj}. Pick it up first."
        if self.current_room != "kitchen":
            return "You need a fridge to cool things. Go to the kitchen."
        self.obj_states[obj] = ObjState.COLD
        return f"You cool the {obj}. It is now cold."


# ── TODO: 定义至少 3 个房间 ──
# 提示：厨房(living_room出口)、客厅(kitchen,bedroom出口)、卧室(living_room出口)
# 厨房：apple, knife, plate | fridge(closed,内含milk), microwave(open,空)
# 客厅：book, remote, vase | cabinet(closed,内含photo)
# 卧室：pillow, laptop, pen | desk(open,内含paper)

def build_world() -> WorldState:
    """构建游戏世界，返回初始状态。"""
    rooms = {
        "kitchen": Room(
            name="kitchen",
            description="A cozy cooking space with tiled floors and the smell of garlic.",
            objects_on_floor=["apple", "knife", "plate"],
            containers={
                "fridge":    {"state": "closed", "contains": ["milk", "lettuce"]},
                "microwave": {"state": "open",   "contains": []},
                "sink":      {"state": "open",   "contains": ["bowl"]},
            },
            exits=["living_room"],
        ),
        "living_room": Room(
            name="living_room",
            description="A warm living room with a soft sofa and a wooden coffee table.",
            objects_on_floor=["book", "remote", "vase", "newspaper"],
            containers={
                "cabinet": {"state": "closed", "contains": ["photo", "candle"]},
                "sofa":    {"state": "open",   "contains": ["cushion"]},
            },
            exits=["kitchen", "bedroom"],
        ),
        "bedroom": Room(
            name="bedroom",
            description="A quiet bedroom with a large desk by the window.",
            objects_on_floor=["pillow", "laptop", "pen", "mug"],
            containers={
                "desk":   {"state": "open",   "contains": ["paper", "notebook"]},
                "drawer": {"state": "closed", "contains": ["key", "watch"]},
            },
            exits=["living_room"],
        ),
    }

    all_objects = [
        "apple", "knife", "plate", "milk", "lettuce", "bowl",
        "book", "remote", "vase", "newspaper", "photo", "candle", "cushion",
        "pillow", "laptop", "pen", "mug", "paper", "notebook", "key", "watch",
    ]
    obj_states = {obj: ObjState.NORMAL for obj in all_objects}

    return WorldState(
        rooms=rooms,
        current_room="kitchen",
        obj_states=obj_states,
    )
