#!/usr/bin/env python3
"""
BYDFi MoonX — Telegram 每日自动发帖脚本
群：Meme&polymarket trede
每天自动发送 3 条内容：
  08:00 — Meme 币早报（热门趋势）
  14:00 — Polymarket 预测市场动态
  20:00 — MoonX 产品推广 + 晚间总结
"""

import os, json, time, random, logging, requests, urllib.request, urllib.parse
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# ── 配置 ──
script_dir = Path(__file__).parent
load_dotenv(script_dir / ".env")
load_dotenv(script_dir / ".env.outreach")

GEMINI_KEY   = os.getenv("GOOGLE_API_KEY")
TG_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID      = "-5133903231"   # Meme&polymarket trede
MOONX_URL    = "https://www.bydfi.com/en/moonx/markets/trending"
BJT          = ZoneInfo("Asia/Shanghai")

# ── 日志 ──
log_file = script_dir / "logs" / "tg_poster.log"
log_file.parent.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ── Gemini ──
client = genai.Client(api_key=GEMINI_KEY)

# ── Polymarket API ──
POLY_API = "https://gamma-api.polymarket.com/markets"

def fetch_trending_markets():
    try:
        params = {"limit": 5, "order": "volume24hr", "ascending": "false", "active": "true"}
        r = requests.get(POLY_API, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("Polymarket fetch error: %s", e)
        return []

def fetch_closing_soon():
    try:
        soon = (datetime.now(timezone.utc) + timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")
        params = {"limit": 5, "order": "volume24hr", "ascending": "false",
                  "active": "true", "end_date_max": soon}
        r = requests.get(POLY_API, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("Polymarket closing_soon error: %s", e)
        return []

# ── 内容生成 ──

def gen_meme_morning():
    prompt = f"""You are a meme coin trader sharing your morning market scan in a Telegram group.

Write an engaging morning update (150-200 words) for a meme coin trading community. Include:
- A catchy opening line with an emoji
- 2-3 trending meme coin narratives or themes to watch today (Solana, Base, or ETH chain)
- Smart money / whale activity insight
- A call to action to check MoonX for real-time data: {MOONX_URL}
- End with a motivational trading line

Tone: energetic, insider, like a fellow degen trader. Use emojis naturally.
Do NOT use hashtags. Write in English."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        logger.error("Gemini morning error: %s", e)
        return None

def gen_polymarket_update(markets):
    if not markets:
        return None
    lines = []
    for m in markets[:5]:
        q = m.get('question', '?')
        vol = float(m.get('volume24hr', 0))
        try:
            yes = round(float(json.loads(m.get('outcomePrices', '[0.5]'))[0]) * 100, 1)
        except Exception:
            yes = 50
        lines.append(f"- {q} | Volume 24h: ${vol:,.0f} | YES: {yes}%")
    market_info = "\n".join(lines)
    prompt = f"""You are a prediction market analyst sharing insights in a Telegram trading group.

Based on these top Polymarket markets today:
{market_info}

Write an engaging update (150-200 words) that:
- Opens with a hook about where smart money is betting
- Highlights 2 most interesting markets and their current odds
- Gives a brief insight on what the odds mean
- Mentions that BYDFi MoonX aggregates smart money signals: {MOONX_URL}
- Ends with an engaging question for the community

Tone: analytical but accessible, like a sharp trader sharing alpha. Use emojis naturally.
Do NOT use hashtags. Write in English."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        logger.error("Gemini polymarket error: %s", e)
        return None

def gen_evening_recap(markets):
    if markets:
        closing_lines = "\n".join([f"- {m.get('question','?')} closing soon" for m in markets[:3]])
    else:
        closing_lines = "- Election outcome markets closing soon\n- Crypto price prediction markets closing this week"

    prompt = (
        "You are a crypto trader doing an evening recap in a Telegram meme coin community.\n\n"
        "Write an evening summary post (150-200 words) that:\n"
        "- Opens with an evening greeting and energy check 🌙\n"
        "- Gives a brief meme coin market mood for the day (bullish/bearish/choppy)\n"
        "- Mentions 1-2 Polymarket markets closing soon:\n"
        f"{closing_lines}\n"
        f"- Promotes MoonX as the tool to catch tomorrow's early moves: {MOONX_URL}\n"
        '- Ends with a community engagement question (e.g. "What\'s your top meme pick for tomorrow?")\n\n'
        "Tone: friendly, community-driven, like a group admin wrapping up the day. Use emojis.\n"
        "Do NOT use hashtags. Write in English."
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        logger.error("Gemini evening error: %s", e)
        return None

# ── Telegram 发送 ──

def send_tg(text):
    try:
        data = urllib.parse.urlencode({
            "chat_id": CHAT_ID,
            "text": text,
            "parse_mode": "HTML"
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data=data
        )
        res = urllib.request.urlopen(req, timeout=10)
        result = json.loads(res.read())
        if result.get("ok"):
            logger.info("✓ 消息发送成功")
            return True
        else:
            logger.error("发送失败: %s", result)
            return False
    except Exception as e:
        logger.error("Telegram error: %s", e)
        return False

# ── 定时逻辑 ──

def wait_until(hour, minute=0):
    now = datetime.now(BJT)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    wait_sec = (target - now).total_seconds()
    logger.info(f"⏰ 等待至 {hour:02d}:{minute:02d} BJT，还剩 {int(wait_sec//3600)}h {int((wait_sec%3600)//60)}m")
    time.sleep(wait_sec)

def run_once(post_type):
    """单次执行指定类型的发帖"""
    logger.info(f"▶ 开始发帖: {post_type}")

    if post_type == "morning":
        content = gen_meme_morning()
        if content:
            send_tg(content)

    elif post_type == "afternoon":
        markets = fetch_trending_markets()
        content = gen_polymarket_update(markets)
        if content:
            send_tg(content)

    elif post_type == "evening":
        closing = fetch_closing_soon()
        content = gen_evening_recap(closing)
        if content:
            send_tg(content)

def run_scheduler():
    """持续运行，按北京时间 08:00 / 14:00 / 20:00 发帖"""
    logger.info("🚀 Telegram 每日发帖调度器已启动")
    logger.info(f"   群组: Meme&polymarket trede ({CHAT_ID})")
    logger.info(f"   发帖时间: 08:00 / 14:00 / 20:00 (北京时间)")

    schedule = [
        (8,  0,  "morning"),
        (14, 0,  "afternoon"),
        (20, 0,  "evening"),
    ]

    while True:
        now = datetime.now(BJT)
        next_post = None
        min_wait = None

        for hour, minute, ptype in schedule:
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            wait = (target - now).total_seconds()
            if min_wait is None or wait < min_wait:
                min_wait = wait
                next_post = (hour, minute, ptype)

        h, m, ptype = next_post
        logger.info(f"📅 下一条: {ptype} @ {h:02d}:{m:02d} BJT（{int(min_wait//60)} 分钟后）")
        time.sleep(min_wait)
        run_once(ptype)
        time.sleep(5)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ("morning", "afternoon", "evening"):
        # 手动触发单条测试
        run_once(sys.argv[1])
    else:
        run_scheduler()
