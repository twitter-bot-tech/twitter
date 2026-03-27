#!/usr/bin/env python3
"""
Reddit 自动分发 — 把 SEO 文章发到相关板块
目标：r/predictionmarkets, r/cryptomarkets, r/ethfinance
"""
import os
import json
import logging
import praw
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
HISTORY_FILE = LOG_DIR / "reddit_posts.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "reddit_poster.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# 目标板块：每篇文章轮流发到不同 subreddit（避免重复）
SUBREDDITS = [
    "predictionmarkets",
    "cryptomarkets",
    "ethfinance",
]

# 文章摘要模板（从 body 提取前 3 段作为正文）
POST_TEMPLATE = """{excerpt}

---
Full article: {url}

*Tracking smart money on prediction markets? Check out [MoonX](https://www.bydfi.com/en/moonx/markets/trending) — real-time aggregator for Polymarket, Kalshi and more.*"""


def load_history() -> list:
    return json.loads(HISTORY_FILE.read_text()) if HISTORY_FILE.exists() else []


def save_history(records: list):
    HISTORY_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False))


def already_posted(history: list, title: str) -> bool:
    return any(r.get("title") == title for r in history)


def get_reddit() -> praw.Reddit:
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        username=os.getenv("REDDIT_USERNAME"),
        password=os.getenv("REDDIT_PASSWORD"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "MoonX-SEO-Bot/1.0"),
    )


def extract_excerpt(body: str, max_chars: int = 800) -> str:
    """取文章前几段作为 Reddit 正文（跳过标题行）"""
    lines = body.split("\n")
    paragraphs = []
    char_count = 0
    for line in lines:
        if line.startswith("#"):
            continue
        if not line.strip():
            continue
        char_count += len(line)
        paragraphs.append(line)
        if char_count >= max_chars:
            break
    return "\n\n".join(paragraphs[:4])


def post_article(reddit, article: dict, subreddit_name: str) -> dict:
    subreddit = reddit.subreddit(subreddit_name)
    excerpt = extract_excerpt(article.get("body", ""))
    selftext = POST_TEMPLATE.format(excerpt=excerpt, url=article["url"])

    submission = subreddit.submit(
        title=article["title"],
        selftext=selftext,
        flair_id=None,
    )
    url = f"https://www.reddit.com{submission.permalink}"
    logger.info(f"✅ r/{subreddit_name}: {url}")
    return {
        "title":      article["title"],
        "subreddit":  subreddit_name,
        "reddit_url": url,
        "article_url": article["url"],
        "posted_at":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }


def run(articles: list = None):
    """
    articles: [{"title": str, "url": str, "body": str}, ...]
    若不传，则从 published_articles.json 读取最新 dev.to 文章
    """
    logger.info("=== Reddit 分发 ===")
    history = load_history()

    if articles is None:
        pub_file = LOG_DIR / "published_articles.json"
        if not pub_file.exists():
            logger.error("published_articles.json 不存在")
            return
        records = json.loads(pub_file.read_text())
        # 只取 dev.to 文章
        articles = [r for r in records if "dev.to" in r.get("platform", "") and r.get("url")]

    if not articles:
        logger.info("没有需要分发的文章")
        return

    reddit = get_reddit()
    new_records = []

    for i, article in enumerate(articles):
        title = article["title"]
        if already_posted(history, title):
            logger.info(f"已发过，跳过: {title[:50]}")
            continue

        # 轮流选 subreddit（按发布顺序）
        sub = SUBREDDITS[i % len(SUBREDDITS)]
        logger.info(f"发布到 r/{sub}: {title[:60]}")

        try:
            record = post_article(reddit, article, sub)
            new_records.append(record)
        except Exception as e:
            logger.error(f"发布失败: {e}")

    if new_records:
        history.extend(new_records)
        save_history(history)

    logger.info(f"=== 完成，新发布 {len(new_records)} 篇 ===")


if __name__ == "__main__":
    run()
