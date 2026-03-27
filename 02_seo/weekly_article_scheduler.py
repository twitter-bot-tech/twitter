#!/usr/bin/env python3
"""
Rand — 每周文章生成 + 发布调度
周一、周四 10:00 BJT 自动写文章并发布到 dev.to
"""
import os
import json
import logging
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from typing import Optional
from claude_cli import Anthropic

load_dotenv(Path(__file__).parent.parent / ".env")


def _call_claude(**kwargs):
    return Anthropic().messages.create(**kwargs)

BJT     = ZoneInfo("Asia/Shanghai")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
OUTBOX  = Path(__file__).parent.parent / "outbox"
OUTBOX.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "weekly_articles.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

DEVTO_API_KEY = os.getenv("DEVTO_API_KEY")
MOONX_URL     = "https://www.bydfi.com/en/moonx/markets/trending"

# ── 文章选题队列（按顺序轮发）──────────────────────────────────────────────────
ARTICLE_QUEUE = [
    {"keyword": "kalshi vs polymarket",          "tags": ["predictionmarkets","crypto","kalshi","polymarket"]},
    {"keyword": "crypto prediction market tracker","tags": ["predictionmarkets","crypto","trading","defi"]},
    {"keyword": "smart money crypto signals",     "tags": ["crypto","smartmoney","trading","defi"]},
    {"keyword": "meme coin smart money",          "tags": ["memecoin","crypto","smartmoney","solana"]},
    {"keyword": "how to use prediction markets",  "tags": ["predictionmarkets","crypto","beginner","guide"]},
    {"keyword": "polymarket whale tracker",       "tags": ["polymarket","predictionmarkets","crypto","whale"]},
    {"keyword": "prediction market arbitrage",    "tags": ["predictionmarkets","trading","crypto","arbitrage"]},
    {"keyword": "kalshi prediction market review","tags": ["kalshi","predictionmarkets","review","crypto"]},
]

QUEUE_FILE = Path(__file__).parent / "logs" / "article_queue_index.json"
PUB_FILE   = Path(__file__).parent / "logs" / "published_articles.json"


def get_next_article() -> dict:
    idx = 0
    if QUEUE_FILE.exists():
        idx = json.loads(QUEUE_FILE.read_text()).get("index", 0)
    item = ARTICLE_QUEUE[idx % len(ARTICLE_QUEUE)]
    QUEUE_FILE.write_text(json.dumps({"index": idx + 1}))
    return item


def generate_article(keyword: str) -> tuple[str, str]:
    """用 Claude 生成文章，返回 (title, body_markdown)"""
    prompt = f"""Write a 900-1000 word SEO article for dev.to.

Primary keyword: "{keyword}"
Target audience: crypto traders interested in prediction markets

Requirements:
- Title must prominently include the keyword
- Hook with a specific data point or surprising fact in paragraph 1
- 3-4 H2 subheadings
- Include the keyword naturally 4-5 times
- Include exactly 2 links to {MOONX_URL} (vary anchor text)
- Reference real examples (Polymarket, Kalshi, market events)
- End with clear call to action
- Output ONLY the article starting from # Title — no meta headers

Tone: sharp trader sharing alpha, data-driven, not a marketing piece"""

    resp = _call_claude(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    full = resp.content[0].text.strip()
    # 提取标题
    lines = full.split("\n")
    title = lines[0].lstrip("# ").strip()
    return title, full


def publish_to_devto(title: str, body: str, tags: list) -> Optional[str]:
    resp = requests.post(
        "https://dev.to/api/articles",
        json={"article": {"title": title, "body_markdown": body, "published": True, "tags": tags}},
        headers={"api-key": DEVTO_API_KEY, "Content-Type": "application/json"},
        timeout=15,
    )
    if resp.status_code in (200, 201):
        return resp.json().get("url")
    logger.error(f"dev.to 发布失败 {resp.status_code}: {resp.text[:200]}")
    return None


def load_published() -> list:
    return json.loads(PUB_FILE.read_text()) if PUB_FILE.exists() else []


def run():
    now = datetime.now(BJT)
    logger.info(f"=== 每周文章调度 — {now.strftime('%Y-%m-%d %H:%M BJT')} ===")

    item    = get_next_article()
    keyword = item["keyword"]
    tags    = item["tags"]
    today   = now.strftime("%Y-%m-%d")

    logger.info(f"本次选题: {keyword}")

    # 生成文章
    logger.info("Claude 生成中...")
    title, body = generate_article(keyword)
    logger.info(f"标题: {title}")

    # 保存到 outbox
    slug     = keyword.replace(" ", "-")
    filename = f"{today}_SEO文章_{slug}.md"
    outpath  = OUTBOX / filename
    outpath.write_text(
        f"# SEO Article — {title}\n\n"
        f"**Target Keyword:** {keyword}\n\n---\n\n{body}",
        encoding="utf-8",
    )
    logger.info(f"文章已保存: {filename}")

    # 发布到 dev.to
    logger.info("发布到 dev.to...")
    url = publish_to_devto(title, body, tags)
    if url:
        logger.info(f"✅ 发布成功: {url}")
        records = load_published()
        records.append({
            "title":        title,
            "url":          url,
            "platform":     "dev.to",
            "keyword":      keyword,
            "published_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        })
        PUB_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False))

        # ── 多平台分发 ───────────────────────────────────────────────────────
        # 1. Hashnode 同步
        try:
            import hashnode_publisher
            hashnode_publisher.run()
        except Exception as e:
            logger.warning(f"Hashnode 分发跳过: {e}")

        # 2. Reddit 分发
        try:
            import reddit_poster
            reddit_poster.run([{"title": title, "url": url, "body": body}])
        except Exception as e:
            logger.warning(f"Reddit 分发跳过: {e}")

        # 3. Google/IndexNow 收录提交
        try:
            import gsc_submit
            gsc_submit.run()
        except Exception as e:
            logger.warning(f"收录提交跳过: {e}")

    else:
        logger.error("发布失败，文章已保存到 outbox 待手动处理")

    logger.info("=== 完成 ===")


if __name__ == "__main__":
    import sys as _sys; _sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.pid_lock import acquire_lock; acquire_lock()
    run()
