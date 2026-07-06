"""
AlfWorld 手搓版 — Step 2: 任务定义。

6 类任务：
1. pick_and_place      — 把 X 放到 Y
2. pick_clean_place    — 把 X 洗干净放到 Y
3. pick_heat_place     — 把 X 加热放到 Y
4. pick_cool_place     — 把 X 冷却放到 Y
5. pick_two_obj        — 把 X 和 Z 放到 Y
6. examine             — 在某个位置查看 X 的状态
"""

TASKS = [
    # === pick_and_place (把 X 放到 Y 的位置) ===
    # goal_state: {"<object>": {"location": "<room>"|"<container>"}}
    # location 是容器名表示放进容器，是房间名表示放在该房间地面
    {"id": 1, "type": "pick_and_place", "description": "put a knife on the desk",
     "goal_state": {"knife": {"location": "desk"}}},
    {"id": 2, "type": "pick_and_place", "description": "put a book in the fridge",
     "goal_state": {"book": {"location": "fridge"}}},
    {"id": 3, "type": "pick_and_place", "description": "put the remote on the desk",
     "goal_state": {"remote": {"location": "desk"}}},
    {"id": 4, "type": "pick_and_place", "description": "move the pillow to the living room",
     "goal_state": {"pillow": {"location": "living_room"}}},
    {"id": 5, "type": "pick_and_place", "description": "put the vase in the cabinet",
     "goal_state": {"vase": {"location": "cabinet"}}},
    {"id": 6, "type": "pick_and_place", "description": "put the pen in the drawer",
     "goal_state": {"pen": {"location": "drawer"}}},

    # === pick_clean_place ===
    {"id": 7, "type": "pick_clean_place", "description": "put a clean plate on the desk",
     "goal_state": {"plate": {"location": "desk", "state": "clean"}}},
    {"id": 8, "type": "pick_clean_place", "description": "put a clean mug in the cabinet",
     "goal_state": {"mug": {"location": "cabinet", "state": "clean"}}},
    {"id": 9, "type": "pick_clean_place", "description": "put a clean knife on the sofa",
     "goal_state": {"knife": {"location": "sofa", "state": "clean"}}},
    {"id": 10, "type": "pick_clean_place", "description": "put a clean bowl in the drawer",
     "goal_state": {"bowl": {"location": "drawer", "state": "clean"}}},
    {"id": 11, "type": "pick_clean_place", "description": "put a clean apple on the desk",
     "goal_state": {"apple": {"location": "desk", "state": "clean"}}},
    {"id": 12, "type": "pick_clean_place", "description": "put a clean vase in the fridge",
     "goal_state": {"vase": {"location": "fridge", "state": "clean"}}},

    # === pick_heat_place ===
    {"id": 13, "type": "pick_heat_place", "description": "put a hot apple in the fridge",
     "goal_state": {"apple": {"location": "fridge", "state": "hot"}}},
    {"id": 14, "type": "pick_heat_place", "description": "put a hot plate on the sofa",
     "goal_state": {"plate": {"location": "sofa", "state": "hot"}}},
    {"id": 15, "type": "pick_heat_place", "description": "put hot milk on the desk",
     "goal_state": {"milk": {"location": "desk", "state": "hot"}}},
    {"id": 16, "type": "pick_heat_place", "description": "put a hot bowl in the cabinet",
     "goal_state": {"bowl": {"location": "cabinet", "state": "hot"}}},
    {"id": 17, "type": "pick_heat_place", "description": "put a hot knife in the drawer",
     "goal_state": {"knife": {"location": "drawer", "state": "hot"}}},
    {"id": 18, "type": "pick_heat_place", "description": "put a hot mug on the desk",
     "goal_state": {"mug": {"location": "desk", "state": "hot"}}},

    # === pick_cool_place ===
    {"id": 19, "type": "pick_cool_place", "description": "put a cold milk in the sofa",
     "goal_state": {"milk": {"location": "sofa", "state": "cold"}}},
    {"id": 20, "type": "pick_cool_place", "description": "put a cold apple in the cabinet",
     "goal_state": {"apple": {"location": "cabinet", "state": "cold"}}},
    {"id": 21, "type": "pick_cool_place", "description": "put cold lettuce on the desk",
     "goal_state": {"lettuce": {"location": "desk", "state": "cold"}}},
    {"id": 22, "type": "pick_cool_place", "description": "put a cold plate in the drawer",
     "goal_state": {"plate": {"location": "drawer", "state": "cold"}}},
    {"id": 23, "type": "pick_cool_place", "description": "put a cold knife on the desk",
     "goal_state": {"knife": {"location": "desk", "state": "cold"}}},
    {"id": 24, "type": "pick_cool_place", "description": "put a cold mug in the fridge",
     "goal_state": {"mug": {"location": "fridge", "state": "cold"}}},

    # === pick_two_obj ===
    {"id": 25, "type": "pick_two_obj", "description": "put a knife and a plate in the fridge",
     "goal_state": {"knife": {"location": "fridge"}, "plate": {"location": "fridge"}}},
    {"id": 26, "type": "pick_two_obj", "description": "put the book and the remote on the desk",
     "goal_state": {"book": {"location": "desk"}, "remote": {"location": "desk"}}},
    {"id": 27, "type": "pick_two_obj", "description": "put the pen and the mug in the cabinet",
     "goal_state": {"pen": {"location": "cabinet"}, "mug": {"location": "cabinet"}}},
    {"id": 28, "type": "pick_two_obj", "description": "put the apple and the bowl in the drawer",
     "goal_state": {"apple": {"location": "drawer"}, "bowl": {"location": "drawer"}}},
    {"id": 29, "type": "pick_two_obj", "description": "put the vase and the candle on the desk",
     "goal_state": {"vase": {"location": "desk"}, "candle": {"location": "desk"}}},
    {"id": 30, "type": "pick_two_obj", "description": "put the newspaper and the pillow in the fridge",
     "goal_state": {"newspaper": {"location": "fridge"}, "pillow": {"location": "fridge"}}},

    # === examine ===
    {"id": 31, "type": "examine", "description": "look at the apple in the fridge and tell me its state",
     "target_obj": "apple", "target_location": "fridge"},
    {"id": 32, "type": "examine", "description": "look at the photo in the cabinet and tell me its state",
     "target_obj": "photo", "target_location": "cabinet"},
    {"id": 33, "type": "examine", "description": "look at the key in the drawer and tell me its state",
     "target_obj": "key", "target_location": "drawer"},
    {"id": 34, "type": "examine", "description": "look at the notebook on the desk and tell me its state",
     "target_obj": "notebook", "target_location": "desk"},
    {"id": 35, "type": "examine", "description": "look at the milk in the fridge and tell me its state",
     "target_obj": "milk", "target_location": "fridge"},
    {"id": 36, "type": "examine", "description": "look at the candle in the cabinet and tell me its state",
     "target_obj": "candle", "target_location": "cabinet"},
]


def get_tasks_by_type(task_type: str) -> list[dict]:
    """筛选某类任务。"""
    return [t for t in TASKS if t["type"] == task_type]


def get_task_by_id(task_id: int) -> dict | None:
    for t in TASKS:
        if t["id"] == task_id:
            return t
    return None
