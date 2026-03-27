#!/usr/bin/env python3
"""
每日自动发推 — 拉 Polymarket 实时数据 + Claude 生成推文 + 自动发布
每天按模板轮换：周一/四 市场数据，周二/五 叙事，周三 观点
"""
import os
import json
import logging
import requests
import tweepy
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from claude_cli import Anthropic

load_dotenv(Path(__file__).parent.parent / ".env")


def _call_claude(**kwargs):
    return Anthropic().messages.create(**kwargs)

BJT     = ZoneInfo("Asia/Shanghai")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
HISTORY = LOG_DIR / "tweet_history.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "daily_tweet_poster.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

MOONX_URL = "https://www.bydfi.com/en/moonx/markets/trending"

# 每周发推类型轮换
TWEET_TYPE_BY_WEEKDAY = {
    0: "market_data",   # 周一
    1: "narrative",     # 周二
    2: "opinion",       # 周三
    3: "market_data",   # 周四
    4: "narrative",     # 周五
    5: "market_data",   # 周六
    6: "opinion",       # 周日
}


def get_twitter_client():
    return tweepy.Client(
        consumer_key=os.getenv("TWITTER_API_KEY"),
        consumer_secret=os.getenv("TWITTER_API_SECRET"),
        access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
        access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
    )


def fetch_polymarket_hot() -> list:
    """拉 Polymarket 24h 成交量最高的市场"""
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"limit": 30, "active": "true", "order": "volume24hr", "ascending": "false"},
            timeout=15,
        )
        markets = []
        for m in resp.json():
            try:
                prices = m.get("outcomePrices", "[]")
                p = json.loads(prices) if isinstance(prices, str) else prices
                yes = float(p[0]) if p else 0
                vol = float(m.get("volume24hr", 0))
                if vol > 100000:  # 只取 10万+ 成交量
                    markets.append({
                        "question": m.get("question", "")[:100],
                        "yes_pct":  round(yes * 100),
                        "volume_24h": vol,
                    })
            except:
                continue
        return sorted(markets, key=lambda x: x["volume_24h"], reverse=True)[:5]
    except Exception as e:
        logger.error(f"Polymarket API 失败: {e}")
        return []


def generate_tweet(tweet_type: str, market_data: list) -> str:

    market_summary = "\n".join(
        f"- {m['question']} → {m['yes_pct']}% YES | ${m['volume_24h']:,.0f} 24h vol"
        for m in market_data
    ) if market_data else "No market data available"

    prompts = {
        "market_data": f"""Write a Twitter post about today's hottest Polymarket markets.

Today's top markets by 24h volume:
{market_summary}

Requirements:
- Start with a punchy hook like "Smart money on Polymarket today:" or "Where the money is moving on Polymarket right now:"
- Show 2-3 specific markets with real numbers (volume, odds %)
- End with insight about what smart money signals
- Last line: Track more markets: {MOONX_URL}
- Under 280 characters total
- In English
- NO hashtags""",

        "narrative": f"""Write a Twitter post about prediction market arbitrage or smart money insight.

Context - today's Polymarket top markets:
{market_summary}

Requirements:
- Tell a short story about an insight or opportunity (real or plausible)
- No direct promotion, feel like sharing alpha
- Sharp trader voice
- End naturally with {MOONX_URL}
- Under 280 characters
- In English
- NO hashtags""",

        "opinion": f"""Write a Twitter opinion post about the prediction market industry.

Context - today's Polymarket top markets:
{market_summary}

Requirements:
- Bold take about Polymarket vs Kalshi vs crypto prediction markets
- Position MoonX as the aggregator solving fragmentation
- Conversational, sharp trader voice
- End with {MOONX_URL}
- Under 280 characters
- In English
- NO hashtags""",
    }

    resp = _call_claude(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": prompts[tweet_type]}],
    )
    return resp.content[0].text.strip()


def already_posted_today(history: list) -> bool:
    today = datetime.now(BJT).strftime("%Y-%m-%d")
    return any(r.get("date") == today for r in history)


def load_history() -> list:
    return json.loads(HISTORY.read_text()) if HISTORY.exists() else []


def run():
    now = datetime.now(BJT)
    logger.info(f"=== 每日发推 — {now.strftime('%Y-%m-%d %H:%M BJT')} ===")

    history = load_history()
    if already_posted_today(history):
        logger.info("今天已发过推文，跳过")
        return

    # 确定今天的推文类型
    weekday    = now.weekday()
    tweet_type = TWEET_TYPE_BY_WEEKDAY.get(weekday, "market_data")
    logger.info(f"今日类型: {tweet_type}")

    # 拉 Polymarket 数据
    logger.info("拉取 Polymarket 数据...")
    market_data = fetch_polymarket_hot()
    logger.info(f"获取到 {len(market_data)} 个热门市场")

    # Claude 生成推文
    logger.info("Claude 生成推文...")
    tweet_text = generate_tweet(tweet_type, market_data)
    logger.info(f"推文内容:\n{tweet_text}")

    # 发布到 Twitter
    logger.info("发布到 Twitter...")
    try:
        client = get_twitter_client()
        resp = client.create_tweet(text=tweet_text)
        tweet_id = resp.data["id"]
        tweet_url = f"https://twitter.com/i/web/status/{tweet_id}"
        logger.info(f"✅ 发布成功: {tweet_url}")

        history.append({
            "date":       now.strftime("%Y-%m-%d"),
            "type":       tweet_type,
            "text":       tweet_text,
            "tweet_id":   tweet_id,
            "url":        tweet_url,
            "posted_at":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        })
        HISTORY.write_text(json.dumps(history, indent=2, ensure_ascii=False))

    except Exception as e:
        logger.error(f"Twitter 发布失败: {e}")
        # 保存草稿到 outbox
        draft_path = Path(__file__).parent.parent / "outbox" / f"{now.strftime('%Y-%m-%d')}_推文草稿.txt"
        draft_path.write_text(tweet_text, encoding="utf-8")
        logger.info(f"已保存草稿: {draft_path.name}")

    logger.info("=== 完成 ===")


if __name__ == "__main__":
    import sys as _sys; _sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.pid_lock import acquire_lock; acquire_lock()
    run()
