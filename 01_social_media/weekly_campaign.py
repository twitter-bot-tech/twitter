#!/usr/bin/env python3
"""
Sean — 每周 USDT 竞猜活动
周一 08:00 BJT 发起 → 周五 20:00 BJT 公布结果
奖励：$10 USDT，Twitter + Telegram 同步
"""
import os
import json
import random
import logging
import requests
import tweepy
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from claude_cli import Anthropic

load_dotenv(Path(__file__).parent.parent / ".env")


def _call_claude(**kwargs):
    return Anthropic().messages.create(**kwargs)
load_dotenv(Path(__file__).parent.parent / ".env.outreach")

BJT      = ZoneInfo("Asia/Shanghai")
LOG_DIR  = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
OUTBOX   = Path(__file__).parent.parent / "outbox"
CAMPAIGN_FILE = LOG_DIR / "campaigns.json"

TG_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = "-5133903231"
MOONX_URL = "https://www.bydfi.com/en/moonx/markets/trending"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "weekly_campaign.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ── Polymarket 拉热门市场 ─────────────────────────────────────────────────────

def fetch_best_market() -> dict:
    """挑选最适合做竞猜的市场（赔率40-60%，成交量高）"""
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"limit": 50, "active": "true", "order": "volume24hr", "ascending": "false"},
            timeout=15,
        )
        for m in resp.json():
            try:
                prices = m.get("outcomePrices", "[]")
                p = json.loads(prices) if isinstance(prices, str) else prices
                yes = float(p[0]) if p else 0
                vol = float(m.get("volume24hr", 0))
                # 优先选赔率接近50/50、成交量高的市场
                if 0.35 < yes < 0.65 and vol > 500000:
                    return {
                        "question": m.get("question", ""),
                        "yes_pct":  round(yes * 100),
                        "no_pct":   round((1 - yes) * 100),
                        "volume":   vol,
                        "end_date": m.get("endDate", ""),
                    }
            except:
                continue
        # 没有50/50的，取成交量第一
        m = resp.json()[0]
        prices = m.get("outcomePrices", "[0.5]")
        p = json.loads(prices) if isinstance(prices, str) else prices
        yes = float(p[0]) if p else 0.5
        return {
            "question": m.get("question", ""),
            "yes_pct":  round(yes * 100),
            "no_pct":   round((1 - yes) * 100),
            "volume":   float(m.get("volume24hr", 0)),
            "end_date": m.get("endDate", ""),
        }
    except Exception as e:
        logger.error(f"Polymarket fetch 失败: {e}")
        return {}


# ── 内容生成 ──────────────────────────────────────────────────────────────────

def generate_campaign_posts(market: dict) -> dict:
    prompt = f"""Create a weekly prediction contest for a crypto community. Prize: $10 USDT.

Market: {market['question']}
Current odds: YES {market['yes_pct']}% | NO {market['no_pct']}%
24h Volume: ${market['volume']:,.0f}

Generate TWO versions:

1. TELEGRAM POST (200-250 words):
- Fun opening with emojis
- Explain the market clearly
- Contest rules: reply YES or NO + follow @moonx_bydfi on Twitter
- Prize: $10 USDT to 1 random correct predictor
- Deadline: Friday this week
- Mention MoonX for tracking: {MOONX_URL}
- Engaging community question at the end

2. TWITTER POST (under 260 chars):
- Hook + market question
- "Reply YES or NO to win $10 USDT"
- "Winner announced Friday"
- {MOONX_URL}

Output as JSON:
{{"telegram": "...", "twitter": "..."}}"""

    resp = _call_claude(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    # 提取 JSON
    import re
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {"telegram": text, "twitter": ""}


def generate_winner_posts(campaign: dict) -> dict:
    prompt = f"""Create winner announcement posts for a prediction contest.

Market: {campaign['market']['question']}
Correct answer: (to be filled manually — use placeholder [CORRECT_ANSWER])
Winner: (random participant — use placeholder @[WINNER])
Prize: $10 USDT

Generate TWO versions:

1. TELEGRAM (100-150 words): Fun announcement, congratulate winner, tease next week's contest, mention MoonX: {MOONX_URL}
2. TWITTER (under 260 chars): Announce winner, correct answer, tease next contest

Output as JSON: {{"telegram": "...", "twitter": "..."}}"""

    resp = _call_claude(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    import re
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {"telegram": text, "twitter": ""}


# ── 发送 ──────────────────────────────────────────────────────────────────────

def send_telegram(text: str) -> bool:
    try:
        data = urllib.parse.urlencode({
            "chat_id":    CHAT_ID,
            "text":       text,
            "parse_mode": "HTML",
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", data=data
        )
        res = json.loads(urllib.request.urlopen(req, timeout=10).read())
        if res.get("ok"):
            logger.info("✅ TG 发送成功")
            return True
        logger.error(f"TG 发送失败: {res}")
        return False
    except Exception as e:
        logger.error(f"TG 错误: {e}")
        return False


def send_twitter(text: str) -> str:
    try:
        client = tweepy.Client(
            consumer_key=os.getenv("TWITTER_API_KEY"),
            consumer_secret=os.getenv("TWITTER_API_SECRET"),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
            access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
        )
        resp = client.create_tweet(text=text)
        tweet_id = resp.data["id"]
        url = f"https://twitter.com/i/web/status/{tweet_id}"
        logger.info(f"✅ Twitter 发送成功: {url}")
        return url
    except Exception as e:
        logger.error(f"Twitter 错误: {e}")
        return ""


# ── 活动记录 ──────────────────────────────────────────────────────────────────

def load_campaigns() -> list:
    return json.loads(CAMPAIGN_FILE.read_text()) if CAMPAIGN_FILE.exists() else []


def save_campaigns(data: list):
    CAMPAIGN_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def get_current_week() -> str:
    return datetime.now(BJT).strftime("%Y-W%W")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def launch_campaign():
    """周一：发起本周竞猜"""
    logger.info("=== 发起每周竞猜活动 ===")
    week = get_current_week()
    campaigns = load_campaigns()

    if any(c["week"] == week for c in campaigns):
        logger.info(f"本周活动已发起: {week}")
        return

    market = fetch_best_market()
    if not market:
        logger.error("无法获取市场数据")
        return

    logger.info(f"选定市场: {market['question']}")
    posts = generate_campaign_posts(market)

    tg_ok  = send_telegram(posts.get("telegram", ""))
    tw_url = send_twitter(posts.get("twitter", ""))

    campaigns.append({
        "week":       week,
        "market":     market,
        "posts":      posts,
        "twitter_url": tw_url,
        "launched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "status":     "active",
    })
    save_campaigns(campaigns)
    logger.info("=== 活动发起完成 ===")


def announce_winner():
    """周五：公布获奖者（文案生成 + 发出，获奖者需手动填入）"""
    logger.info("=== 公布竞猜结果 ===")
    week = get_current_week()
    campaigns = load_campaigns()

    current = next((c for c in campaigns if c["week"] == week), None)
    if not current:
        logger.error(f"找不到本周活动: {week}")
        return

    if current.get("status") == "closed":
        logger.info("本周结果已公布")
        return

    posts = generate_winner_posts(current)

    # 保存草稿到 outbox（因为需要手动填获奖者）
    draft = (
        f"# 本周竞猜结果 — {week}\n\n"
        f"**市场**: {current['market']['question']}\n\n"
        f"**TG文案**:\n{posts.get('telegram','')}\n\n"
        f"**Twitter文案**:\n{posts.get('twitter','')}\n\n"
        f"---\n⚠️ 请手动填入 [CORRECT_ANSWER] 和 @[WINNER]，然后发布"
    )
    draft_path = OUTBOX / f"{datetime.now(BJT).strftime('%Y-%m-%d')}_竞猜结果草稿.md"
    draft_path.write_text(draft, encoding="utf-8")
    logger.info(f"结果草稿已保存: {draft_path.name}")

    current["status"]      = "closed"
    current["result_draft"] = str(draft_path)
    save_campaigns(campaigns)
    logger.info("=== 完成，请手动填获奖者后发布 ===")


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    import sys as _sys; _sys.path.insert(0, str(_Path(__file__).parent.parent))
    from scripts.pid_lock import acquire_lock; acquire_lock()
    if len(sys.argv) > 1 and sys.argv[1] == "winner":
        announce_winner()
    else:
        launch_campaign()
