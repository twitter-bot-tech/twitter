#!/usr/bin/env python3
"""
Gary — Auto Comment Bot
Searches for high-engagement prediction market tweets and posts insightful comments
to grow account visibility. Max 5 comments/day, 90-200s intervals.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import tweepy
from claude_cli import Anthropic
from dotenv import load_dotenv

# ── Setup ──────────────────────────────────────────────────────────────────────
script_dir = Path(__file__).parent
load_dotenv(script_dir / ".env")

log_file = script_dir / "logs" / "comment.log"
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file)],
)
logger = logging.getLogger(__name__)

COMMENT_HISTORY_FILE = script_dir / "comment_history.json"
MAX_COMMENTS_PER_DAY = 5
MIN_WAIT_SEC = 90
MAX_WAIT_SEC = 200

# Known high-value accounts to monitor (real analysts/traders, NOT official brand accounts)
# Brand accounts (Polymarket, KalshiHQ) restrict replies to prior engagers — skip them
TARGET_ACCOUNTS = [
    "iamjasonlevin",     # Prediction markets analyst
    "domahhhh",          # Polymarket trader
    "natesilver538",     # Nate Silver (odds/elections)
    "kelxyz_",           # Prediction markets/crypto
    "byteofangelo",      # Crypto/prediction markets
    "RationalistJohn",   # Prediction markets
    "KyleSamani",        # Multicoin / crypto macro
    "TarunChitra",       # Crypto research
    "VelissariouD",      # Prediction markets analyst
    "hasufl",            # Crypto research/DeFi
    "ercwl",             # Crypto/markets
    "smyyguy",           # Prediction markets trader
    "PTLCapital",        # Prediction markets
    "ACXMarkets",        # Manifold/prediction markets
]

# Minimum engagement thresholds
MIN_LIKES = 3
MIN_AUTHOR_FOLLOWERS = 500

COMMENT_SYSTEM_PROMPT = """You are Gary, a sharp prediction markets analyst who adds genuine value to conversations on crypto Twitter.

Your comments are:
- Insightful and specific — always reference the actual market, odds, or mechanism being discussed
- Short: 1-3 sentences max, no fluff
- Never promotional — do NOT mention BYDFi, MoonX, or any product
- Written as a knowledgeable peer, not a fan
- Occasionally contrarian or ask a sharp follow-up question
- Use numbers/stats when relevant

Tone: like a smart trader who reads everything, has opinions, and doesn't waste words.

NEVER:
- Start with "Great tweet!", "Love this!", "Interesting point"
- Use hashtags (look spammy)
- Mention any product or URL
- Be generic ("prediction markets are fascinating")
- Agree blindly — push back or add a dimension they missed

Your comments should make people want to click your profile."""


def load_history() -> dict:
    if COMMENT_HISTORY_FILE.exists():
        with open(COMMENT_HISTORY_FILE) as f:
            return json.load(f)
    return {"daily": {}, "commented_authors": {}, "commented_tweet_ids": []}


def save_history(history: dict):
    with open(COMMENT_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def get_today_count(history: dict) -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return history["daily"].get(today, 0)


def increment_today_count(history: dict):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    history["daily"][today] = history["daily"].get(today, 0) + 1


def author_commented_recently(history: dict, author_id: str) -> bool:
    """Returns True if we commented on this author's tweet within 24h."""
    ts = history["commented_authors"].get(str(author_id))
    if not ts:
        return False
    last = datetime.fromisoformat(ts)
    return (datetime.now(timezone.utc) - last) < timedelta(hours=24)


def mark_author_commented(history: dict, author_id: str):
    history["commented_authors"][str(author_id)] = datetime.now(timezone.utc).isoformat()


def purge_old_entries(history: dict):
    """Remove author entries older than 48h to keep file small."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
    history["commented_authors"] = {
        k: v for k, v in history["commented_authors"].items()
        if datetime.fromisoformat(v) > cutoff
    }
    # Keep only last 200 tweet IDs
    history["commented_tweet_ids"] = history["commented_tweet_ids"][-200:]


def init_twitter() -> tweepy.Client:
    # 不传 bearer_token，只用 OAuth 1.0a — 与 tweet_bot.py 保持一致
    return tweepy.Client(
        consumer_key=os.getenv("TWITTER_API_KEY"),
        consumer_secret=os.getenv("TWITTER_API_SECRET"),
        access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
        access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
        wait_on_rate_limit=True,
    )


def fetch_user_id(client: tweepy.Client, username: str) -> str | None:
    try:
        resp = client.get_user(username=username, user_fields=["public_metrics"])
        return str(resp.data.id) if resp.data else None
    except Exception:
        return None


def search_target_tweets(client: tweepy.Client) -> list[dict]:
    """Search recent tweets by keyword, filter by engagement and open reply settings."""
    # 搜索用 bearer_token（app-only auth，有更高搜索配额）
    search_client = tweepy.Client(bearer_token=os.getenv("TWITTER_BEARER_TOKEN"))
    candidates = []
    since = datetime.now(timezone.utc) - timedelta(hours=24)

    KEYWORD_GROUPS = [
        "polymarket OR kalshi OR \"prediction market\" -is:retweet -is:reply lang:en",
    ]

    # 依次尝试每个关键词组，直到找到足够结果
    for query in KEYWORD_GROUPS:
        if len(candidates) >= 10:
            break
        logger.info(f"Keyword search: {query}")
        try:
            resp = search_client.search_recent_tweets(
                query=query,
                max_results=20,
                tweet_fields=["public_metrics", "text", "reply_settings", "created_at", "author_id"],
                expansions=["author_id"],
                user_fields=["public_metrics"],
                start_time=since,
            )
        except tweepy.errors.TweepyException as e:
            logger.warning(f"Search failed ({query}): {e}")
            continue

        users = {str(u.id): u.public_metrics.get("followers_count", 0)
                 for u in (resp.includes.get("users") or [])}

        seen_ids = {c["tweet_id"] for c in candidates}
        for tweet in (resp.data or []):
            if str(tweet.id) in seen_ids:
                continue
            if tweet.text.startswith("RT @") or tweet.text.startswith("@"):
                continue
            reply_settings = getattr(tweet, "reply_settings", "everyone") or "everyone"
            if reply_settings != "everyone":
                continue
            metrics = tweet.public_metrics or {}
            likes = metrics.get("like_count", 0)
            followers = users.get(str(tweet.author_id), 0)
            if likes >= MIN_LIKES and followers >= MIN_AUTHOR_FOLLOWERS:
                candidates.append({
                    "tweet_id": str(tweet.id),
                    "author_id": str(tweet.author_id),
                    "author_username": str(tweet.author_id),
                    "text": tweet.text,
                    "likes": likes,
                    "followers": followers,
                })

    logger.info(f"Found {len(candidates)} qualifying tweets (≥{MIN_LIKES} likes, ≥{MIN_AUTHOR_FOLLOWERS} followers)")
    candidates.sort(key=lambda x: x["likes"], reverse=True)
    return candidates


def generate_comment(tweet_text: str) -> str | None:
    """Use Claude to generate an insightful comment on the tweet."""
    client = Anthropic()
    prompt = f"""Tweet to comment on:
\"\"\"{tweet_text}\"\"\"

Write a single comment (1-3 sentences). No hashtags, no self-promotion. Add a specific insight, stat, or sharp question that makes the original author and readers want to click your profile."""

    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            system=COMMENT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        comment = resp.content[0].text.strip()
        # Strip quotes if Claude wrapped it
        comment = re.sub(r'^["\']|["\']$', '', comment)
        return comment
    except Exception as e:
        logger.error(f"Claude error: {e}")
        return None


def post_comment(client: tweepy.Client, tweet_id: str, comment: str) -> str | None:
    """Reply to the target tweet with insightful commentary."""
    if len(comment) > 270:
        comment = comment[:267] + "..."
    try:
        resp = client.create_tweet(
            text=comment,
            in_reply_to_tweet_id=tweet_id,
        )
        return str(resp.data["id"])
    except tweepy.errors.Forbidden as e:
        logger.error(f"Forbidden: {e}")
        return None
    except tweepy.errors.TweepyException as e:
        logger.error(f"Tweet post failed: {e}")
        return None


def send_to_lark(text: str):
    """发送到 Lark 社媒群"""
    import urllib.request
    load_dotenv(script_dir.parent / ".env", override=True)
    webhook = os.getenv("LARK_SOCIAL")
    if not webhook:
        logger.warning("LARK_SOCIAL webhook not set")
        return
    payload = json.dumps({"msg_type": "text", "content": {"text": text}}).encode()
    req = urllib.request.Request(webhook, data=payload, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        logger.warning(f"Lark send failed: {e}")


def run():
    logger.info("=== Comment Suggestion Bot Starting ===")
    bjt_now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")

    history = load_history()
    purge_old_entries(history)

    twitter = init_twitter()
    candidates = search_target_tweets(twitter)

    if not candidates:
        logger.info("No candidates found. Exiting.")
        return

    suggestions = []
    count = 0

    for tweet in candidates:
        if count >= 3:  # 每次推送3条建议
            break

        tweet_id = tweet["tweet_id"]
        author_id = tweet["author_id"]

        if tweet_id in history["commented_tweet_ids"]:
            continue
        if author_commented_recently(history, author_id):
            continue

        logger.info(f"Generating suggestion for: {tweet['text'][:80]}...")
        comment = generate_comment(tweet["text"])
        if not comment:
            continue

        tweet_url = f"https://twitter.com/i/web/status/{tweet_id}"
        suggestions.append({
            "tweet_id": tweet_id,
            "author_id": author_id,
            "url": tweet_url,
            "original": tweet["text"][:100],
            "comment": comment,
            "likes": tweet["likes"],
            "followers": tweet["followers"],
        })
        history["commented_tweet_ids"].append(tweet_id)
        mark_author_commented(history, author_id)
        count += 1

    if not suggestions:
        logger.info("No suggestions generated.")
        save_history(history)
        return

    # 格式化发 Lark
    lines = [f"💬 今日推文互动建议 {bjt_now} BJT\n手动回复/引用即可，预计每条30秒\n"]
    for i, s in enumerate(suggestions, 1):
        lines.append(
            f"【{i}】@{s['author_id']} · {s['likes']}赞 · {s['followers']:,}粉\n"
            f"原推：{s['original']}...\n"
            f"建议回复：{s['comment']}\n"
            f"链接：{s['url']}\n"
        )
    lark_msg = "\n".join(lines)

    send_to_lark(lark_msg)
    save_history(history)
    logger.info(f"=== Done — sent {len(suggestions)} suggestion(s) to Lark ===")


if __name__ == "__main__":
    import sys as _sys; _sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.pid_lock import acquire_lock; acquire_lock()
    run()
