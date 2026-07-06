"""
WebShop 手搓版 — Step 1: 商品数据库。

生成 200 个商品，覆盖 5 个品类。
先定属性再起名，确保名与属性一致。
"""

import random

random.seed(42)

# ── 品类属性空间 ──

CATEGORY_CONFIG = {
    "shoes": {
        "brands": ["Nike", "Adidas", "Puma", "New Balance", "Skechers", "Vans", "Converse", "Reebok"],
        "colors": ["black", "white", "blue", "red", "grey", "green", "navy", "brown"],
        "sizes": ["6", "7", "8", "9", "10", "11", "12"],
        "price_range": (29, 180),
        "rating_range": (3.0, 4.9),
        "adjectives": ["running", "casual", "basketball", "trail", "walking", "training", "slip-on", "lace-up"],
    },
    "clothing": {
        "brands": ["H&M", "Zara", "Uniqlo", "Gap", "Levi's", "Nike", "Adidas", "Patagonia"],
        "colors": ["black", "white", "blue", "red", "grey", "green", "pink", "yellow", "navy"],
        "sizes": ["XS", "S", "M", "L", "XL", "XXL"],
        "price_range": (9, 120),
        "rating_range": (3.0, 4.9),
        "adjectives": ["cotton", "slim-fit", "oversized", "striped", "plain", "graphic", "formal", "casual"],
    },
    "electronics": {
        "brands": ["Sony", "Samsung", "Apple", "Bose", "JBL", "Logitech", "Anker", "Dell"],
        "colors": ["black", "white", "silver", "blue", "grey"],
        "price_range": (19, 500),
        "rating_range": (3.5, 5.0),
        "adjectives": ["wireless", "bluetooth", "USB-C", "portable", "noise-cancelling", "ergonomic", "compact", "high-speed"],
    },
    "kitchen": {
        "brands": ["KitchenAid", "OXO", "Cuisinart", "Lodge", "Pyrex", "Instant Pot", "Ninja", "Hamilton Beach"],
        "colors": ["black", "white", "silver", "red", "grey", "stainless steel"],
        "price_range": (5, 200),
        "rating_range": (3.5, 4.9),
        "adjectives": ["non-stick", "cast iron", "stainless steel", "glass", "silicone", "ceramic", "BPA-free", "dishwasher safe"],
    },
    "books": {
        "brands": [],
        "colors": [],
        "price_range": (3, 45),
        "rating_range": (3.5, 5.0),
        "adjectives": ["hardcover", "paperback", "illustrated", "revised", "collector's", "pocket"],
    },
}

CATEGORY_NOUNS = {
    "shoes": ["sneakers", "shoes", "trainers", "loafers", "boots", "sandals", "flats", "heels"],
    "clothing": ["t-shirt", "shirt", "hoodie", "jacket", "jeans", "shorts", "sweater", "dress", "polo", "tank top"],
    "electronics": ["headphones", "speaker", "mouse", "keyboard", "charger", "cable", "webcam", "earbuds", "power bank", "USB hub"],
    "kitchen": ["frying pan", "saucepan", "knife set", "cutting board", "blender", "toaster", "coffee maker", "mixing bowl", "measuring cup", "spatula"],
    "books": ["mystery novel", "sci-fi novel", "cookbook", "biography", "self-help book", "history book", "fantasy novel", "travel guide", "textbook", "thriller"],
}


def _make_name(cat: str, attrs: dict) -> str:
    """根据实际属性生成商品名——确保名与属性一致。"""
    config = CATEGORY_CONFIG[cat]
    adj = random.choice(config["adjectives"])
    noun = random.choice(CATEGORY_NOUNS[cat])
    if cat in ("shoes", "clothing", "electronics", "kitchen"):
        return f"{adj} {attrs['color']} {attrs['brand']} {noun}"
    else:
        return f"{adj} {noun}"


def _make_description(cat: str, name: str, price: float, rating: float) -> str:
    templates = [
        "{name}. A great choice for everyday use. Rated {rating:.1f}/5 by verified buyers. ${price:.2f}. Free shipping on orders over $50.",
        "{name}. High quality {cat} product. Customers love it — {rating:.1f} stars. Priced at ${price:.2f}. In stock.",
        "{name}. Best-selling {cat} item. {rating:.1f} average rating from hundreds of reviews. Only ${price:.2f}. Limited stock!",
        "{name}. Top-rated {cat} with {rating:.1f}/5 stars. Affordable at ${price:.2f}. Ships within 1-2 business days.",
    ]
    return random.choice(templates).format(name=name, cat=cat, rating=rating, price=price)


def generate_products(n_per_category: int = 40) -> list[dict]:
    """先定属性，再用属性生成商品名——确保名与属性一致。"""
    products = []
    pid = 1

    for cat in CATEGORY_CONFIG:
        for _ in range(n_per_category):
            config = CATEGORY_CONFIG[cat]
            price = round(random.uniform(*config["price_range"]), 2)
            rating = round(random.uniform(*config["rating_range"]), 1)

            attrs = {}

            if cat in ("shoes", "clothing"):
                attrs["color"] = random.choice(config["colors"])
                attrs["size"] = random.choice(config["sizes"])
                attrs["brand"] = random.choice(config["brands"])
            elif cat in ("electronics", "kitchen"):
                attrs["color"] = random.choice(config["colors"])
                attrs["brand"] = random.choice(config["brands"])
            elif cat == "books":
                attrs["author"] = random.choice([
                    "J.K. Rowling", "George Orwell", "Yuval Harari", "Malcolm Gladwell",
                    "Stephen King", "Agatha Christie", "Haruki Murakami", "J.R.R. Tolkien",
                ])

            product = {
                "id": pid,
                "name": _make_name(cat, attrs),
                "category": cat,
                "price": price,
                "rating": rating,
                **attrs,
            }
            product["description"] = _make_description(cat, product["name"], price, rating)
            products.append(product)
            pid += 1

    return products


# ── 简易搜索引擎 ──

class SearchEngine:
    """基于词频匹配的轻量搜索引擎。"""

    def __init__(self, products: list[dict]):
        self.products = products

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        keywords = query.lower().split()
        scored = []
        for p in self.products:
            text = (
                f"{p['name']} {p['category']} {p.get('brand', '')} "
                f"{p.get('color', '')} {p.get('author', '')} {p['description']}"
            ).lower()
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_k]]


# ── 全局数据 ──

PRODUCTS = generate_products(40)
SEARCH_ENGINE = SearchEngine(PRODUCTS)


def get_product(pid: int) -> dict | None:
    for p in PRODUCTS:
        if p["id"] == pid:
            return p
    return None
