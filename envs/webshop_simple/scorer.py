"""
WebShop 手搓版 — Step 2: 评分器。

比较 agent 购买的商品与任务目标的匹配度。
"""

import re


def parse_constraints(goal: str) -> dict:
    """从任务目标字符串中提取约束条件。

    示例：
    "I need a pair of blue basketball shoes under 50 dollars, size 10, by Nike"
    → {"category": "shoes", "color": "blue", "brand": "Nike", "size": "10", "max_price": 50.0}
    """
    goal_lower = goal.lower()
    constraints = {}

    # 品类
    cat_keywords = {
        "shoes": ["shoe", "sneaker", "boot", "sandal", "loafer", "heel", "flat", "trainer", "running", "basketball"],
        "clothing": ["shirt", "t-shirt", "hoodie", "jacket", "jean", "short", "sweater", "dress", "polo", "tank"],
        "electronics": ["headphone", "speaker", "mouse", "keyboard", "charger", "cable", "webcam", "earbud", "power bank", "usb hub", "electronic"],
        "kitchen": ["pan", "knife", "cutting board", "blender", "toaster", "coffee maker", "bowl", "cup", "spatula", "kitchen"],
        "books": ["book", "novel", "cookbook", "biography", "guide", "textbook", "thriller", "mystery"],
    }
    for cat, kws in cat_keywords.items():
        if any(kw in goal_lower for kw in kws):
            constraints["category"] = cat
            break

    # 颜色
    colors = ["black", "white", "blue", "red", "grey", "gray", "green", "pink", "yellow", "navy", "brown", "silver", "purple", "orange"]
    for c in colors:
        if c in goal_lower:
            constraints["color"] = c
            if c == "gray":
                constraints["color"] = "grey"
            break

    # 品牌
    brands = ["nike", "adidas", "puma", "new balance", "skechers", "vans", "converse", "reebok",
              "h&m", "zara", "uniqlo", "gap", "levi", "patagonia",
              "sony", "samsung", "apple", "bose", "jbl", "logitech", "anker", "dell",
              "kitchenaid", "oxo", "cuisinart", "lodge", "pyrex", "instant pot", "ninja", "hamilton beach"]
    for b in brands:
        if b in goal_lower:
            constraints["brand"] = b.title()
            break

    # 尺码
    size_match = re.search(r'size\s*(\d{1,2})', goal_lower)
    if not size_match:
        size_match = re.search(r'\b(xs|s|m|l|xl|xxl)\b', goal_lower, re.IGNORECASE)
    if size_match:
        constraints["size"] = size_match.group(1).upper()

    # 价格上限
    price_match = re.search(r'(?:under|below|less than|at most|max|<=|<)\s*\$?(\d+)', goal_lower)
    if not price_match:
        price_match = re.search(r'\$?(\d+)\s*(?:dollars|bucks|usd)?\s*(?:or less|or under|max)?', goal_lower)
    if price_match:
        constraints["max_price"] = float(price_match.group(1))

    return constraints


def score_purchase(product: dict | None, goal: str) -> tuple[float, dict]:
    """评分：0.0 ~ 1.0。

    返回 (总分, 各项得分明细)。
    """
    if product is None:
        return 0.0, {"error": "No product purchased"}

    constraints = parse_constraints(goal)
    if not constraints:
        return 0.5, {"warning": "No constraints parsed"}

    scores = {}
    total = 0.0
    weight = 0

    # 品类匹配（权重 3）
    if "category" in constraints:
        weight += 3
        if product["category"] == constraints["category"]:
            scores["category"] = 3.0
        else:
            scores["category"] = 0.0
        total += scores["category"]

    # 颜色匹配（权重 2）
    if "color" in constraints:
        weight += 2
        if product.get("color", "").lower() == constraints["color"]:
            scores["color"] = 2.0
        else:
            scores["color"] = 0.0
        total += scores["color"]

    # 品牌匹配（权重 2）
    if "brand" in constraints:
        weight += 2
        if product.get("brand", "").lower() == constraints["brand"].lower():
            scores["brand"] = 2.0
        else:
            scores["brand"] = 0.0
        total += scores["brand"]

    # 尺码匹配（权重 1）
    if "size" in constraints:
        weight += 1
        if product.get("size", "").upper() == constraints["size"]:
            scores["size"] = 1.0
        else:
            scores["size"] = 0.0
        total += scores["size"]

    # 价格（权重 3）
    if "max_price" in constraints:
        weight += 3
        if product["price"] <= constraints["max_price"]:
            scores["price"] = 3.0
        else:
            # 超过预算按比例扣分
            overage = (product["price"] - constraints["max_price"]) / constraints["max_price"]
            scores["price"] = max(0.0, 3.0 - overage * 3.0)
        total += scores["price"]

    # 评分加成（权重 1）
    weight += 1
    scores["rating"] = product["rating"] / 5.0
    total += scores["rating"]

    return total / weight, scores
