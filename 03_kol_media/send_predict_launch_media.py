#!/usr/bin/env python3
"""
MoonX Predict 功能上线 — 媒体外联邮件脚本
活动：6万 USDT 奖池，4月7日上线，活动至4月30日
目标：加密垂媒（Bitcoin Magazine / The Defiant / CoinDesk / CoinTelegraph）
用法：
  预览所有邮件：python3 send_predict_launch_media.py
  真实发送：    python3 send_predict_launch_media.py --send
"""

import os
import sys
import time
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
load_dotenv(Path(__file__).parent.parent / ".env.outreach")

# ── 日志 ──
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "predict_launch_media.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

SENDER_EMAIL = os.getenv("BYDFI_EMAIL", "kelly@bydfi.com")
SENDER_NAME  = "Kelly | MoonX"
MOONX_URL    = "https://www.bydfi.com/en/moonx/markets/trending"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 邮件模版
# EN = 实际发送；ZH = 仅供 Kelly 参考，不发出去
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EMAIL_SUBJECT = "MoonX Predict Launch — $60,000 USDT Prize Pool | Sponsored Coverage Opportunity"

# ── 通用主体（各媒体第一段个性化，其余通用）──
EMAIL_BODY_EN = """\
Hi {contact_name},

{personalized_opener}

We're launching MoonX Predict on April 7th — a live crypto prediction market built for active traders, with a $60,000 USDT prize pool running through April 30th.

We're looking for sponsored editorial coverage ahead of the launch (ideally publishing April 3-6) to reach your audience before the prize pool opens.

What we're proposing:
- Sponsored article or native ad placement
- Angle: "New prediction market enters the space with $60K USDT live prize pool"
- Timeline: publish April 3-6 (before launch day)
- Budget: flexible, open to your rate card

MoonX is built by the BYDFi team — we have 3M+ registered users across our exchange products. This is our prediction market vertical launch.

Could you share your rate card or connect us with the right person on your partnerships team?

Best,
Kelly
MoonX Marketing Lead | BYDFi
{url}
"""

EMAIL_BODY_ZH = """\
你好 {contact_name}，

{personalized_opener_zh}

我们将于4月7日上线 MoonX Predict——面向活跃交易者的实时加密预测市场，配套 6万 USDT 奖池，活动至4月30日。

我们正在寻找上线前的赞助报道合作（理想发布时间：4月3-6日），在奖池开放前触达你们的受众。

合作形式：
- 赞助文章或原生广告位
- 角度：「新预测市场入局，6万 USDT 实时奖池引爆」
- 时间：4月3-6日发布（上线日之前）
- 预算：灵活，以你们的报价为准

MoonX 由 BYDFi 团队出品，我们的交易所产品注册用户超过300万。这次是我们预测市场垂直赛道的首发。

方便发一下你们的报价单，或者帮我对接负责合作的同事吗？

Kelly
MoonX 营销负责人 | BYDFi
{url}
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 媒体目标列表
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MEDIA_LIST = [
    {
        "name": "The Defiant",
        "email": "editorial@thedefiant.io",
        "contact_name": "The Defiant Team",
        "priority": "A",
        "note": "DeFi 原生媒体，受众精准，流量 B，预算内首选",
        "personalized_opener": (
            "The Defiant has been one of the most credible voices covering DeFi and on-chain products — "
            "your audience is exactly who we want to reach."
        ),
        "personalized_opener_zh": (
            "The Defiant 一直是 DeFi 和链上产品领域最有公信力的声音之一——"
            "你们的受众正是我们想触达的群体。"
        ),
    },
    {
        "name": "Bitcoin Magazine",
        "email": "editori@bitcoinmagazine.com",
        "contact_name": "Bitcoin Magazine Team",
        "priority": "A",
        "note": "加密原生读者，流量 B，预算内可操作",
        "personalized_opener": (
            "Bitcoin Magazine has the most engaged crypto-native readership in the space — "
            "readers who actually participate in markets, not just read about them."
        ),
        "personalized_opener_zh": (
            "Bitcoin Magazine 拥有业内参与度最高的加密原生读者群——"
            "这些读者真的在参与市场，而不只是读读新闻。"
        ),
    },
    {
        "name": "CoinDesk",
        "email": "advertising@coindesk.com",
        "contact_name": "CoinDesk Advertising Team",
        "priority": "A",
        "note": "流量 S，预算可能偏紧，先询价",
        "personalized_opener": (
            "CoinDesk reaches the broadest crypto audience in the industry — "
            "we'd love to explore what a launch-window placement could look like."
        ),
        "personalized_opener_zh": (
            "CoinDesk 覆盖了业内最广泛的加密受众——"
            "我们希望探讨一下上线窗口期的广告位合作可能性。"
        ),
    },
    {
        "name": "CoinTelegraph",
        "email": "advertise@cointelegraph.com",
        "contact_name": "CoinTelegraph Advertising Team",
        "priority": "A",
        "note": "流量 S，预算可能偏紧，先询价",
        "personalized_opener": (
            "CoinTelegraph is the go-to destination for crypto news globally — "
            "a launch announcement in your pages would give MoonX Predict significant visibility."
        ),
        "personalized_opener_zh": (
            "CoinTelegraph 是全球加密新闻的首选媒体——"
            "在你们平台上做上线公告，MoonX Predict 的曝光度会非常可观。"
        ),
    },
]


def send_email(media: dict, dry_run: bool = True) -> bool:
    body_en = EMAIL_BODY_EN.format(
        contact_name=media["contact_name"],
        personalized_opener=media["personalized_opener"],
        url=MOONX_URL,
    )

    if dry_run:
        logger.info("\n" + "─" * 50)
        logger.info("媒体: %s (%s)", media["name"], media["note"])
        logger.info("收件人: %s", media["email"])
        logger.info("主题: %s", EMAIL_SUBJECT)
        logger.info("内容预览:\n%s", body_en)
        return True

    try:
        smtp_host = os.getenv("BYDFI_SMTP_HOST", "smtp.zoho.com")
        smtp_port = int(os.getenv("BYDFI_SMTP_PORT", "465"))
        smtp_user = os.getenv("BYDFI_EMAIL", SENDER_EMAIL)
        smtp_pass = os.getenv("BYDFI_EMAIL_PASSWORD", "")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = EMAIL_SUBJECT
        msg["From"]    = f"{SENDER_NAME} <{smtp_user}>"
        msg["To"]      = media["email"]
        msg.attach(MIMEText(body_en, "plain"))

        # Zoho port 465 使用 SSL
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, media["email"], msg.as_string())

        logger.info("  ✅ 邮件已发送 → %s (%s)", media["name"], media["email"])
        return True

    except Exception as e:
        logger.error("  ✗ 发送失败 %s: %s", media["name"], e)
        return False


def run(dry_run: bool = True):
    logger.info("=" * 50)
    logger.info("MoonX Predict 媒体外联邮件（%d 家）", len(MEDIA_LIST))
    logger.info("模式: %s", "DRY RUN" if dry_run else "真实发送")
    logger.info("=" * 50)

    sent = 0
    for media in MEDIA_LIST:
        success = send_email(media, dry_run=dry_run)
        if success:
            sent += 1
            if not dry_run:
                time.sleep(5)  # 邮件间隔 5 秒

    logger.info("\n完成！处理 %d 家媒体", sent)

    if dry_run:
        logger.info("\n⚠️  以上为预览，未真实发送")
        logger.info("真实发送：python3 send_predict_launch_media.py --send")
        logger.info("\n📌 中文版模版（仅供参考）：")
        for media in MEDIA_LIST:
            body_zh = EMAIL_BODY_ZH.format(
                contact_name=media["contact_name"],
                personalized_opener_zh=media["personalized_opener_zh"],
                url=MOONX_URL,
            )
            logger.info("\n--- %s ---\n%s", media["name"], body_zh)


if __name__ == "__main__":
    dry_run = "--send" not in sys.argv
    run(dry_run=dry_run)
