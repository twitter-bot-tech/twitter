#!/usr/bin/env python3
"""
Rand — dev.to 自动发布脚本
使用官方 REST API，无需浏览器自动化
"""
import os
import re
import json
import requests
import logging
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

API_KEY   = os.getenv("DEVTO_API_KEY")
BASE_URL  = "https://dev.to/api"
LOG_DIR   = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
PUB_FILE  = LOG_DIR / "published_articles.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "devto_publisher.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── 文章队列（按顺序发布）──────────────────────────────────────────────────────
ARTICLES = [
    {
        "path":    "2026-03-05_SEO文章_polymarket-alternative.md",
        "title":   "The Best Polymarket Alternatives in 2026 (Including One That Tracks Smart Money)",
        "tags":    ["predictionmarkets", "crypto", "polymarket", "defi"],
        "canonical": "https://medium.com/@ppmworker/the-best-polymarket-alternatives-in-2026-including-one-that-tracks-smart-money-4d9dc2abd970",
    },
    {
        "path":    "2026-03-05_SEO文章_prediction-market-smart-money.md",
        "title":   "How Prediction Market Smart Money Moves Before the Rest of the World Notices",
        "tags":    ["predictionmarkets", "crypto", "trading", "polymarket"],
        "canonical": "https://medium.com/@ppmworker/how-prediction-market-smart-money-moves-before-the-rest-of-the-world-notices-97b4cd444df9",
    },
]

OUTBOX = Path(__file__).parent.parent / "outbox"


def load_published() -> list:
    if PUB_FILE.exists():
        return json.loads(PUB_FILE.read_text())
    return []


def save_published(records: list):
    PUB_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False))


def extract_body(path: Path) -> str:
    """跳过元数据注释，从 # 标题行开始提取正文"""
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    body, in_body = [], False
    for line in lines:
        if not in_body:
            if line.startswith("# ") and not line.startswith("# SEO Article"):
                in_body = True
        if in_body:
            body.append(line)
    return "\n".join(body).strip()


def already_published(records: list, title: str) -> bool:
    return any(r.get("title") == title and "dev.to" in r.get("platform", "") for r in records)


def publish_article(article: dict):
    path = OUTBOX / article["path"]
    if not path.exists():
        logger.error(f"文章文件不存在: {path}")
        return None

    body = extract_body(path)
    if not body:
        logger.error(f"正文提取失败: {path}")
        return None

    payload = {
        "article": {
            "title":          article["title"],
            "body_markdown":  body,
            "published":      True,
            "tags":           article["tags"],
            "canonical_url":  article.get("canonical", ""),
        }
    }

    resp = requests.post(
        f"{BASE_URL}/articles",
        json=payload,
        headers={
            "api-key":      API_KEY,
            "Content-Type": "application/json",
        },
        timeout=15,
    )

    if resp.status_code in (200, 201):
        data = resp.json()
        url = data.get("url", "")
        logger.info(f"✅ 发布成功: {url}")
        return {
            "title":        article["title"],
            "url":          url,
            "platform":     "dev.to",
            "target_keyword": article.get("tags", []),
            "canonical":    article.get("canonical", ""),
            "published_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        }
    else:
        logger.error(f"发布失败 {resp.status_code}: {resp.text[:200]}")
        return None


def run():
    logger.info("=== dev.to 自动发布 ===")
    records = load_published()

    for article in ARTICLES:
        if already_published(records, article["title"]):
            logger.info(f"已发布，跳过: {article['title'][:50]}")
            continue

        logger.info(f"发布: {article['title'][:60]}")
        result = publish_article(article)
        if result:
            records.append(result)
            save_published(records)

    logger.info("=== 完成 ===")


if __name__ == "__main__":
    run()
