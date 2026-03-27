#!/usr/bin/env python3
"""
预测市场垂类 KOL — 个性化 DM 发送脚本
执行日期：2026-03-05（周三）
策略：伊朗地缘政治热点切入 + MoonX 聪明钱追踪差异化
每日上限：≤10 条，间隔 90～200 秒
"""

import os
import sys
import time
import random
import logging
from pathlib import Path
from dotenv import load_dotenv
import tweepy

load_dotenv(Path(__file__).parent.parent / ".env")
load_dotenv(Path(__file__).parent.parent / ".env.outreach")

# ── Twitter 认证 ──
client = tweepy.Client(
    bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
    consumer_key=os.getenv("TWITTER_API_KEY"),
    consumer_secret=os.getenv("TWITTER_API_SECRET"),
    access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
    access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
)

# ── 日志 ──
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "prediction_market_dm.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

MOONX_URL  = "https://www.bydfi.com/en/moonx/markets/trending"
SENDER_TG  = "@BDkelly"

# ── 个性化 DM — 预测市场垂类（伊朗热点版）──
DM_LIST = [
    # ⚠️ @PolymarketWhales / @PMAlphaXYZ / @PolyTraderXYZ — 账号不存在，已移除
    # 真实存在的预测市场 KOL
    {
        "handle": "NightHawkX",
        "dm": (
            "Hey 👋\n\n"
            "Your Polymarket alpha threads are consistently ahead of the market — "
            "keep finding the gems.\n\n"
            "I'm Kelly from BYDFi. We built MoonX ({url}) — "
            "tracks smart money flows across prediction markets (Poly + Kalshi) "
            "AND on-chain meme coins in one terminal.\n\n"
            "We're building our first partner group:\n"
            "✅ Early access before public launch\n"
            "✅ Revenue share on referrals\n"
            "✅ Custom data feeds for your content\n\n"
            "Interested? TG: {tg} 👀"
        ),
    },
    # Meme 中腰部（3/5 同批发送）
    {
        "handle": "CryptoWendyO",
        "dm": (
            "Hey Wendy —\n\n"
            "Your on-chain calls this cycle have been consistently early — "
            "your community clearly follows your moves.\n\n"
            "I'm Kelly from BYDFi. We launched MoonX ({url}) — "
            "real-time meme discovery + smart money/whale tracking in one terminal. "
            "Think GMGN but with cross-market intelligence including prediction markets.\n\n"
            "For launch partners:\n"
            "✅ Revenue share on every trader you bring\n"
            "✅ Exclusive smart money data for your content\n"
            "✅ Early access + referral dashboard\n\n"
            "Quick chat? TG: {tg}"
        ),
    },
    {
        "handle": "zachxbt",
        "dm": (
            "Hey —\n\n"
            "Your on-chain investigations are the gold standard for wallet tracking — "
            "respect the work.\n\n"
            "I'm Kelly from BYDFi. We built MoonX ({url}) — "
            "smart money aggregation across meme coins + prediction markets. "
            "The cross-chain wallet tracking layer would resonate with your audience.\n\n"
            "Onboarding early partners:\n"
            "✅ Revenue share on referrals\n"
            "✅ Exclusive on-chain data dashboard\n"
            "✅ Co-marketing support\n\n"
            "Curious to chat? TG: {tg}"
        ),
    },
]

DAILY_LIMIT = 10


def send_dm(handle: str, dm_text: str, dry_run: bool = True) -> bool:
    try:
        user = client.get_user(username=handle)
        if not user.data:
            logger.warning("  ✗ 用户不存在: @%s", handle)
            return False
        user_id = user.data.id
        full_text = dm_text.format(url=MOONX_URL, tg=SENDER_TG)

        if dry_run:
            logger.info("  [DRY RUN] @%s (ID:%s)", handle, user_id)
            logger.info("  内容预览: %s...", full_text[:120])
            return True

        client.create_direct_message(participant_id=user_id, text=full_text)
        logger.info("  ✅ DM 已发送 → @%s", handle)
        return True

    except tweepy.errors.Forbidden as e:
        logger.error("  ✗ @%s DM 已关闭: %s", handle, e)
        return False
    except tweepy.errors.TooManyRequests:
        logger.warning("  ⚠ Twitter 限速，等待 15 分钟...")
        time.sleep(900)
        return False
    except Exception as e:
        logger.error("  ✗ 失败 @%s: %s", handle, e)
        return False


def run(dry_run: bool = True, batch: int = 1):
    """batch=1 发第一批（3/5），batch=2 发第二批（3/6）"""
    if batch == 1:
        targets = DM_LIST[:3]   # @NightHawkX, @CryptoWendyO, @zachxbt
    elif batch == 2:
        targets = DM_LIST[:1]   # @NightHawkX（若第一批未发）
    else:
        targets = DM_LIST

    logger.info("="*50)
    logger.info("🚀 预测市场 KOL DM — 批次 %d（%d 个目标）", batch, len(targets))
    logger.info("模式: %s", "DRY RUN" if dry_run else "真实发送")
    logger.info("="*50)

    sent = 0
    for item in targets:
        if sent >= DAILY_LIMIT:
            logger.warning("已达每日上限 %d 条", DAILY_LIMIT)
            break
        handle = item["handle"]
        logger.info("▶ 发送给: @%s", handle)
        success = send_dm(handle, item["dm"], dry_run=dry_run)
        if success:
            sent += 1
            if not dry_run and sent < len(targets):
                wait = random.randint(90, 200)
                logger.info("  ⏱ 等待 %d 秒...", wait)
                time.sleep(wait)

    logger.info("\n完成！发送 %d 条 DM", sent)


if __name__ == "__main__":
    dry_run = "--send" not in sys.argv
    batch   = 2 if "--batch2" in sys.argv else 1

    if dry_run:
        logger.info("⚠️  测试模式 — 不会真实发送 DM")
        logger.info("真实发送（第一批）：python3 send_prediction_market_dms.py --send")
        logger.info("真实发送（第二批）：python3 send_prediction_market_dms.py --send --batch2")

    run(dry_run=dry_run, batch=batch)
