"""
WebShop 手搓版 — Step 3: 环境交互。

状态机：browsing → viewing → purchased
动作：search[keywords], click[item_id], buy[item_id]
"""

import random
from .products import SEARCH_ENGINE, PRODUCTS, get_product
from .scorer import score_purchase


def _generate_goal_from_product(p: dict) -> str:
    """从实际商品反推可达成目标——直接用名字中的关键词。"""
    cat = p["category"]
    budget = int(p["price"] + random.randint(3, 15))
    name_words = p["name"].lower().split()
    color = p.get("color", "")
    brand = p.get("brand", "")
    size = p.get("size", "")

    # 从名字中提取品类名词
    category_nouns = {
        "shoes": ["sneakers", "shoes", "trainers", "loafers", "boots", "sandals", "flats", "heels"],
        "clothing": ["t-shirt", "shirt", "hoodie", "jacket", "jeans", "shorts", "sweater", "dress", "polo", "tank", "top"],
        "electronics": ["headphones", "speaker", "mouse", "keyboard", "charger", "cable", "webcam", "earbuds", "power", "bank", "usb", "hub"],
        "kitchen": ["frying", "pan", "saucepan", "knife", "set", "cutting", "board", "blender", "toaster", "coffee", "maker", "mixing", "bowl", "measuring", "cup", "spatula"],
        "books": ["novel", "cookbook", "biography", "guide", "textbook", "thriller", "mystery", "history", "fantasy", "sci-fi"],
    }
    nouns = category_nouns.get(cat, ["item"])
    matched_noun = next((w for w in name_words if w in nouns), cat.rstrip("s"))

    parts = []
    if color:
        parts.append(color)
    parts.append(matched_noun)

    if cat == "books":
        if p.get("author"):
            parts.append(f"by {p['author']}")
        return f"I'm looking for a {' '.join(parts)} under {budget} dollars"

    if brand:
        parts.append(f"by {brand}")
    if size:
        parts.append(f"size {size}")
    parts.append(f"under {budget} dollars")

    return f"I need {' '.join(parts)}"


GOALS = [
    "I need a pair of blue running shoes under 60 dollars, size 9, by Nike",
    "Looking for black wireless headphones under 100 dollars by Sony",
    "I want a red kitchen blender under 50 dollars",
    "Find me a grey hoodie size M under 40 dollars by Adidas",
    "I need a white Logitech mouse under 30 dollars",
    "Searching for a paperback sci-fi novel under 15 dollars",
    "I want a brown leather boot size 10 under 80 dollars",
    "Find a black Samsung speaker under 80 dollars",
    "I need a stainless steel frying pan under 40 dollars",
    "Looking for a navy blue jacket size L under 70 dollars by Patagonia",
    "Find me a green t-shirt size S under 25 dollars",
    "I need a non-stick saucepan under 30 dollars",
    "Searching for silver Apple earbuds under 150 dollars",
    "I want a cast iron skillet under 50 dollars by Lodge",
    "Find a white Nike sneaker size 11 under 90 dollars",
    "I need a hardcover history book under 25 dollars",
    "Looking for a pink dress size M under 45 dollars",
    "Find me black running shoes size 8 under 55 dollars by Puma",
    "I need a USB-C charger under 20 dollars by Anker",
    "Searching for a graphic t-shirt size XL under 20 dollars",
    "I want a portable bluetooth speaker under 50 dollars by JBL",
    "Find a grey New Balance trainer size 7 under 70 dollars",
    "I need a ceramic mixing bowl set under 25 dollars",
    "Looking for a black mechanical keyboard under 80 dollars",
    "Find me a fantasy novel paperback under 12 dollars",
]


class WebShopEnv:
    """WebShop 购物环境。"""

    MAX_STEPS = 20

    def __init__(self):
        self.goal: str = ""
        self.search_results: list[dict] = []     # 当前搜索结果
        self.viewing: dict | None = None          # 正在查看的商品
        self.purchased: dict | None = None        # 已购买的商品
        self.steps_taken: int = 0
        self.done: bool = False

    def reset(self, goal: str | None = None) -> str:
        """重置环境。不传 goal 则从实际商品自动生成可达成目标。"""
        if goal is None:
            p = random.choice(PRODUCTS)
            goal = _generate_goal_from_product(p)
        self.goal = goal
        self.search_results = []
        self.viewing = None
        self.purchased = None
        self.steps_taken = 0
        self.done = False
        return self._make_observation()

    def step(self, action: str) -> tuple[str, float, bool]:
        """执行动作，返回 (observation, reward, done)。"""
        action = action.strip()

        if self.done:
            return ("Task already finished.", 0.0, True)

        self.steps_taken += 1
        if self.steps_taken >= self.MAX_STEPS:
            self.done = True
            reward, _ = score_purchase(self.purchased, self.goal)
            return (f"Too many steps.\n\n{self._result_summary(reward)}", reward, True)

        act_type, params = self._parse_action(action)

        if act_type == "search":
            self.search_results = SEARCH_ENGINE.search(params["query"])
            self.viewing = None
            obs = self._format_search_results()

        elif act_type == "click":
            pid = params["item_id"]
            product = get_product(pid)
            if product is None:
                obs = f"No product with id {pid}."
            else:
                self.viewing = product
                obs = self._format_product_detail(product)

        elif act_type == "buy":
            pid = params["item_id"]
            product = get_product(pid)
            if product is None:
                obs = f"No product with id {pid}."
            else:
                self.purchased = product
                self.done = True
                reward, detail = score_purchase(product, self.goal)
                obs = self._format_purchase_result(product, reward, detail)
                return (obs, reward, True)

        else:
            obs = f"Unknown action: {action}. Try: search[keywords], click[123], buy[456]"

        return (obs, 0.0, False)

    def get_valid_actions(self) -> list[str]:
        """返回当前可选动作模板。"""
        actions = ["search[<keywords>]"]
        if self.search_results:
            for p in self.search_results[:10]:
                actions.append(f"click[{p['id']}]")
        if self.viewing:
            actions.append(f"buy[{self.viewing['id']}]")
        return actions

    # ── 内部方法 ──

    def _parse_action(self, action: str) -> tuple[str, dict]:
        """解析动作：search[keywords] | click[id] | buy[id]"""
        import re
        action = action.strip()

        if m := re.match(r"search\[(.+)\]", action):
            return ("search", {"query": m.group(1)})

        if m := re.match(r"click\[(\d+)\]", action):
            return ("click", {"item_id": int(m.group(1))})

        if m := re.match(r"buy\[(\d+)\]", action):
            return ("buy", {"item_id": int(m.group(1))})

        return ("unknown", {})

    def _make_observation(self) -> str:
        """构建当前状态的文字描述。"""
        obs = f"Goal: {self.goal}\n\n"
        obs += f"Steps taken: {self.steps_taken}/{self.MAX_STEPS}\n\n"

        if self.viewing:
            obs += self._format_product_detail(self.viewing)
        elif self.search_results:
            obs += self._format_search_results()
        else:
            obs += "You are on the WebShop homepage.\n"
            obs += f"Available actions: search[<keywords>]\n"
            obs += "Try searching for what the customer needs!"
        return obs

    def _format_search_results(self) -> str:
        if not self.search_results:
            return "No results found. Try a different search."

        lines = [f"Search results ({len(self.search_results)} found):\n"]
        for i, p in enumerate(self.search_results[:10]):
            color = p.get("color", "")
            brand = p.get("brand", "")
            size = p.get("size", "")
            author = p.get("author", "")
            extras = ", ".join(x for x in [color, brand, size, author] if x)
            lines.append(
                f"  [{p['id']}] {p['name']} — ${p['price']:.2f} | "
                f"Rating: {p['rating']}/5"
            )
            if extras:
                lines[-1] += f" | {extras}"
        lines.append(f"\nClick on an item to view details, or search again.")
        return "\n".join(lines)

    def _format_product_detail(self, p: dict) -> str:
        lines = [
            f"Product #{p['id']}: {p['name']}",
            f"Category: {p['category']}",
            f"Price: ${p['price']:.2f}",
            f"Rating: {p['rating']}/5",
        ]
        for key in ["color", "size", "brand", "author"]:
            if key in p:
                lines.append(f"{key.title()}: {p[key]}")
        lines.append(f"\nDescription: {p['description']}")
        lines.append(f"\nYou can buy[{p['id']}] to purchase this item, or search for something else.")
        return "\n".join(lines)

    def _format_purchase_result(self, product: dict, reward: float, detail: dict) -> str:
        lines = [
            f"You purchased: {product['name']} (${product['price']:.2f})",
            f"Score: {reward:.2f}",
            f"Breakdown: {detail}",
        ]
        return "\n".join(lines)

    def _result_summary(self, reward: float) -> str:
        if self.purchased:
            return f"Purchased: {self.purchased['name']} | Score: {reward:.2f}"
        return f"No purchase made. Score: {reward:.2f}"
