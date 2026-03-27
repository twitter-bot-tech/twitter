#!/usr/bin/env python3
"""
竞品 Twitter 数据追踪 — 每周分析竞品内容表现
输出：竞品内容情报 + MoonX 本周发推建议
"""
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from claude_cli import Anthropic
import twitter_scraper

load_dotenv(Path(__file__).parent.parent / ".env")


def _call_claude(**kwargs):
    return Anthropic().messages.create(**kwargs)

LOG_DIR  = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
OUTBOX   = Path(__file__).parent.parent / "outbox"
OUTBOX.mkdir(exist_ok=True)
HISTORY  = LOG_DIR / "competitor_data.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "competitor_tracker.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── 追踪的竞品账号 ────────────────────────────────────────────────────────────
COMPETITORS = [
    {"handle": "Polymarket",      "type": "platform"},
    {"handle": "Kalshi",          "type": "platform"},
    {"handle": "ManifoldMarkets", "type": "platform"},
    {"handle": "natesilver538",   "type": "kol"},
    {"handle": "iamjasonlevin",   "type": "kol"},
    {"handle": "kelxyz_",         "type": "kol"},
    {"handle": "ACXMarkets",      "type": "kol"},
    {"handle": "domahhhh",        "type": "kol"},
]

MOONX_URL = "https://www.bydfi.com/en/moonx/markets/trending"


def fetch_user_tweets(handle: str, max_results: int = 20) -> tuple:
    """拉取账号最近推文及互动数据"""
    try:
        user = twitter_scraper.get_user(handle)
        if not user:
            return [], 0
        followers = user.get("followers_count", 0)

        raw_tweets = twitter_scraper.get_user_tweets(user["id"], limit=max_results)
        tweets = []
        for t in raw_tweets:
            likes = t.get("likes", 0)
            rts = t.get("retweets", 0)
            tweets.append({
                "id":         t.get("id", ""),
                "text":       t.get("text", ""),
                "created_at": t["created_at"].isoformat() if t.get("created_at") else "",
                "likes":      likes,
                "retweets":   rts,
                "replies":    t.get("replies", 0),
                "engagement": likes + rts * 3,
            })
        return tweets, followers
    except Exception as e:
        logger.error(f"拉取 @{handle} 失败: {e}")
        return [], 0


def analyze_with_claude(competitor_data: list) -> str:
    """用 Claude 分析竞品数据，生成本周内容建议"""

    # 整理数据摘要
    summary = []
    for comp in competitor_data:
        handle = comp["handle"]
        followers = comp.get("followers", 0)
        tweets = comp.get("tweets", [])
        if not tweets:
            continue
        top3 = sorted(tweets, key=lambda x: x["engagement"], reverse=True)[:3]
        avg_eng = sum(t["engagement"] for t in tweets) / len(tweets) if tweets else 0
        summary.append(
            f"@{handle} ({followers:,} followers, avg engagement: {avg_eng:.0f})\n"
            + "\n".join(f"  [{t['likes']}❤ {t['retweets']}RT] {t['text'][:100]}" for t in top3)
        )

    prompt = f"""You are a crypto Twitter growth strategist. Analyze this week's competitor data and give actionable advice.

COMPETITOR TOP TWEETS THIS WEEK:
{chr(10).join(summary)}

MoonX product: {MOONX_URL} — prediction market aggregator tracking smart money on Polymarket, Kalshi, and crypto markets.

Provide:
1. **Top 3 content themes** working for competitors this week (with examples)
2. **Best posting format** (thread vs single tweet, with/without image)
3. **3 specific tweet ideas for MoonX** to post this week, ready to copy-paste
4. **One thing competitors are NOT doing** that MoonX could own

Be specific and data-driven. Output in Chinese for the team."""

    resp = _call_claude(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def load_history() -> list:
    return json.loads(HISTORY.read_text()) if HISTORY.exists() else []


def run():
    now = datetime.now(timezone.utc)
    logger.info(f"=== 竞品追踪 — {now.strftime('%Y-%m-%d')} ===")

    competitor_data = []

    for comp in COMPETITORS:
        handle = comp["handle"]
        logger.info(f"拉取 @{handle}...")
        tweets, followers = fetch_user_tweets(handle)
        competitor_data.append({
            "handle":    handle,
            "type":      comp["type"],
            "followers": followers,
            "tweets":    tweets,
            "fetched_at": now.isoformat(),
        })
        logger.info(f"  @{handle}: {len(tweets)} 条推文, {followers:,} 粉丝")

    # 保存原始数据
    history = load_history()
    history.append({
        "date": now.strftime("%Y-%m-%d"),
        "data": competitor_data,
    })
    HISTORY.write_text(json.dumps(history, indent=2, ensure_ascii=False))

    # Claude 分析
    logger.info("Claude 分析中...")
    analysis = analyze_with_claude(competitor_data)

    # 生成报告
    date_str  = now.strftime("%Y-%m-%d")
    report    = f"# 竞品 Twitter 情报 — {date_str}\n\n{analysis}\n"
    out_path  = OUTBOX / f"{date_str}_竞品Twitter情报.md"
    out_path.write_text(report, encoding="utf-8")
    logger.info(f"报告已保存: {out_path.name}")

    # 打印到终端
    print("\n" + "="*60)
    print(report)
    print("="*60)

    logger.info("=== 完成 ===")


if __name__ == "__main__":
    run()
