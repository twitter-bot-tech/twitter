#!/usr/bin/env python3
"""
MoonX Predict 功能上线 — KOL 外联 DM 脚本
活动：6万 USDT 奖池，4月7日上线，活动至4月30日
目标：预测市场垂类 KOL（含 Kalshi 流失创作者）
每日上限：≤10 条，间隔 90～200 秒
用法：
  测试模式（不发送）：python3 send_predict_launch_dms.py
  真实发送（Tier1）： python3 send_predict_launch_dms.py --send
  真实发送（Tier2）： python3 send_predict_launch_dms.py --send --tier2
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
        logging.FileHandler(log_dir / "predict_launch_dm.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

MOONX_URL = "https://www.bydfi.com/en/moonx/markets/trending"
SENDER_TG = "@BDkelly"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DM 模版
# EN = 实际发送；ZH = 仅供 Kelly 参考，不发出去
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Kalshi 流失 KOL 专用钩子：Kalshi 2月砍掉了创作者合作
DM_TEMPLATE_EN_KALSHI = """\
Hey {name} —

Saw Kalshi quietly dropped their creator program in February. Rough timing for people doing consistent prediction market content.

We're launching MoonX Predict on April 7th — a live crypto prediction market with a $60,000 USDT prize pool running through April 30th.

We're looking for a small group of creators to participate authentically. Make real predictions, share your results. No scripted promo.

Offering {fee} USDT for a post/video during launch week, plus you keep any winnings from the prize pool.

Interested? TG: {tg}\
"""

DM_TEMPLATE_ZH_KALSHI = """\
嗨 {name} ——

看到 Kalshi 在2月悄悄砍掉了创作者合作项目，对于一直在做预测市场内容的人来说时机挺糟的。

我们将于4月7日上线 MoonX Predict——一个实时加密预测市场，配套 6万 USDT 奖池，活动至4月30日。

我们在寻找一小部分创作者以真实参与的方式合作。自己做预测，把结果分享给受众。不需要照稿宣传。

上线周内发一条推文/视频提供 {fee} USDT 报酬，另外奖池里赢的归你自己。

有兴趣的话：TG: {tg}\
"""

# 通用预测市场 KOL（无 Kalshi 背景）
DM_TEMPLATE_EN_GENERAL = """\
Hey {name} —

Your prediction market content is consistently sharp — the way you break down market probabilities is exactly the kind of analysis our audience is looking for.

We're launching MoonX Predict on April 7th — a live crypto prediction market with a $60,000 USDT prize pool running through April 30th.

Looking for creators who actually use prediction markets to cover this authentically. Make real predictions, share your results with your audience.

Offering {fee} USDT for a post/video during launch week. You keep any winnings too.

Interested? TG: {tg}\
"""

DM_TEMPLATE_ZH_GENERAL = """\
嗨 {name} ——

你的预测市场内容一直很到位——你拆解市场概率的方式正是我们受众在寻找的那种分析。

我们将于4月7日上线 MoonX Predict——实时加密预测市场，配套 6万 USDT 奖池，活动至4月30日。

我们在找真正使用预测市场的创作者，以真实方式来做内容。自己下注，把结果分享给受众。

上线周内发一条推文/视频提供 {fee} USDT 报酬，奖池里赢的也归你。

有兴趣：TG: {tg}\
"""

# YouTube 实操测评专用（偏交易教程方向）
DM_TEMPLATE_EN_YOUTUBE = """\
Hey {name} —

Been following your trading content — your walkthroughs are genuinely useful, not just hype.

We're launching MoonX Predict on April 7th — a live crypto prediction market with a $60,000 USDT prize pool running through April 30th. Think Polymarket but built for crypto traders.

We're looking for 2-3 YouTube creators to do an honest walkthrough video. Use the product for real, show your actual predictions and results.

Offering {fee} USDT for a video during the launch window (April 7-30). You keep any prize pool winnings too.

Happy to send a brief with more details. TG: {tg}\
"""

DM_TEMPLATE_ZH_YOUTUBE = """\
嗨 {name} ——

一直在看你的交易内容——你的实操讲解是真的有用，不只是炒作。

我们将于4月7日上线 MoonX Predict——实时加密预测市场，配套 6万 USDT 奖池，活动至4月30日。可以理解为面向加密交易者的 Polymarket。

我们在找 2-3 位 YouTube 创作者做真实的实操测评视频。真实使用产品，展示你的预测过程和结果。

上线窗口期（4月7日-30日）内发视频提供 {fee} USDT 报酬，奖池里赢的也归你。

可以发更多详情给你。TG: {tg}\
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tier 1 目标（3000 USDT 预算，上线日引爆）
# ⚠️ Twitter handle 需人工确认后填入
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIER1_LIST = [
    {
        "handle": "cbinthegame",          # ⚠️ 待确认 Twitter handle（YouTube: @cbinthegame）
        "name": "CB",
        "fee": "1,200",
        "template": DM_TEMPLATE_EN_KALSHI,
        "note": "112k YouTube，预测市场+加密，US，Kalshi 前合作者",
    },
    {
        "handle": "cryptoblood_",           # Twitter: @cryptoblood_（YouTube: @cryptoblood）
        "name": "Crypto Blood",
        "fee": "800",
        "template": DM_TEMPLATE_EN_YOUTUBE,
        "note": "24k YouTube，Bloodalytics 深度交易分析，US",
    },
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tier 2 目标（2000 USDT 预算，活动中后期用）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIER2_LIST = [
    {
        "handle": "predmkttrader",        # ⚠️ 待确认 Twitter handle（YouTube: @predictionmarkettrader）
        "name": "Prediction Market Trader",
        "fee": "300",
        "template": DM_TEMPLATE_EN_KALSHI,
        "note": "3.6k YouTube，全职预测市场交易员，受众精准",
    },
    {
        "handle": "GingerGirlNYC",        # ⚠️ 待确认 Twitter handle（YouTube: @nycgingergirl）
        "name": "GingerGirl",
        "fee": "200",
        "template": DM_TEMPLATE_EN_GENERAL,
        "note": "1.1k YouTube，自称 Polymarket baddie，深度预测市场用户",
    },
    {
        "handle": "cameronpredicts",      # ⚠️ 待确认 Twitter handle（YouTube: @cameronpredicts）
        "name": "Cameron",
        "fee": "200",
        "template": DM_TEMPLATE_EN_GENERAL,
        "note": "1.1k YouTube，全职预测市场交易员",
    },
    {
        "handle": "colepicks",            # ⚠️ 待确认 Twitter handle（YouTube: @colepredicts）
        "name": "Cole",
        "fee": "200",
        "template": DM_TEMPLATE_EN_GENERAL,
        "note": "1k YouTube，预测市场，有机器人策略内容",
    },
]

DAILY_LIMIT = 10


def send_dm(handle: str, name: str, fee: str, template: str, dry_run: bool = True) -> bool:
    try:
        user = client.get_user(username=handle)
        if not user.data:
            logger.warning("  ✗ 用户不存在: @%s", handle)
            return False
        user_id = user.data.id
        full_text = template.format(name=name, fee=fee, url=MOONX_URL, tg=SENDER_TG)

        if dry_run:
            logger.info("  [DRY RUN] @%s (ID:%s)", handle, user_id)
            logger.info("  内容预览:\n%s\n", full_text)
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


def run(dry_run: bool = True, tier: int = 1):
    targets = TIER1_LIST if tier == 1 else TIER2_LIST
    tier_label = f"Tier {tier}"

    logger.info("=" * 50)
    logger.info("MoonX Predict 上线 DM — %s（%d 个目标）", tier_label, len(targets))
    logger.info("模式: %s", "DRY RUN" if dry_run else "真实发送")
    logger.info("=" * 50)

    sent = 0
    for item in targets:
        if sent >= DAILY_LIMIT:
            logger.warning("已达每日上限 %d 条", DAILY_LIMIT)
            break
        handle = item["handle"]
        logger.info("▶ 发送给: @%s (%s)", handle, item.get("note", ""))
        success = send_dm(
            handle=handle,
            name=item["name"],
            fee=item["fee"],
            template=item["template"],
            dry_run=dry_run,
        )
        if success:
            sent += 1
            if not dry_run and sent < len(targets):
                wait = random.randint(90, 200)
                logger.info("  ⏱ 等待 %d 秒...", wait)
                time.sleep(wait)

    logger.info("\n完成！发送 %d 条 DM", sent)


if __name__ == "__main__":
    dry_run = "--send" not in sys.argv
    tier = 2 if "--tier2" in sys.argv else 1

    if dry_run:
        logger.info("⚠️  测试模式 — 不会真实发送 DM")
        logger.info("真实发送（Tier1）：python3 send_predict_launch_dms.py --send")
        logger.info("真实发送（Tier2）：python3 send_predict_launch_dms.py --send --tier2")

    run(dry_run=dry_run, tier=tier)
