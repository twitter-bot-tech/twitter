#!/usr/bin/env python3
"""
今日外联邮件 — 2026-03-04
收件人：Zvi Mowshowitz + Altcoin Daily
角度：伊朗地缘政治预测市场热点切入
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env.outreach")

GMAIL        = os.getenv("GMAIL_ADDRESS")
APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

EMAILS = [
    {
        "to": "thezvi@gmail.com",
        "name": "Zvi",
        "subject": "Iran Markets, Smart Money Flows, and MoonX — Thought You'd Find This Interesting",
        "body": """Hi Zvi,

Prediction markets are having a moment today — Iran geopolitical markets on Polymarket just crossed $7.5M+ in single-day volume. What's interesting from a smart money perspective is that the whale positioning shifted visibly before the news cycle caught up. Exactly the kind of calibration signal your readers care about.

I'm Kelly from BYDFi. We built MoonX (https://www.bydfi.com/en/moonx/markets/trending) — a platform that aggregates smart money flows across Polymarket, Kalshi, and on-chain prediction markets. The core idea: surface where sophisticated capital is moving before it's reflected in the odds.

Given your writing on forecasting, calibration, and prediction market efficiency, I thought you might find it genuinely interesting — both as a research tool and as a content angle for your readers who trade these markets.

We're onboarding a small group of early partners:
• Exclusive early access to our smart money dashboard
• Revenue share on referrals from your community
• Custom analytics for your writing (we can pull flow data on any market you're covering)

No pressure — happy to share a quick demo or pull the Iran market whale data as a sample if you're curious.

Best,
Kelly | Head of Marketing | BYDFi MoonX
https://www.bydfi.com/en/moonx/markets/trending
ppmworker@gmail.com | TG: @BDkelly""",
    },
    {
        "to": "team@altcoindaily.co",
        "name": "Altcoin Daily Team",
        "subject": "Partnership Opportunity: MoonX × Altcoin Daily — Smart Money Meme Coin Platform",
        "body": """Hi Altcoin Daily Team,

Love the consistent coverage you've brought to the meme coin cycle — your audience genuinely trusts your signal.

I'm Kelly, Head of Marketing at BYDFi. We just launched MoonX (https://www.bydfi.com/en/moonx/markets/trending) — a real-time smart money terminal for meme coin traders that aggregates whale movements across pump.fun, DEX Screener, and on-chain data into one feed.

What makes MoonX different:
• Real-time meme coin discovery — catch pumps before they go viral
• Smart money / whale tracking aggregated from multiple sources
• One-click trading directly from the trend feed

We'd love Altcoin Daily as a launch partner:
✅ Revenue share on every trader you refer (exclusive tracking dashboard)
✅ Co-marketing — we'll amplify your MoonX content to our audience
✅ Custom smart money data for your videos and analysis
✅ Early exclusive access before broader public rollout

Would love to jump on a 15-min call or exchange a few DMs on Twitter/TG.

Kelly | Head of Marketing | BYDFi MoonX
https://www.bydfi.com/en/moonx/markets/trending
Email: ppmworker@gmail.com | TG: @BDkelly""",
    },
]


def send_email(to, name, subject, body):
    msg = MIMEMultipart("alternative")
    msg["From"]    = f"Kelly <{GMAIL}>"
    msg["To"]      = to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(GMAIL, APP_PASSWORD)
        smtp.sendmail(GMAIL, to, msg.as_string())
    print(f"  ✅ 已发送 → {name} <{to}>")


if __name__ == "__main__":
    print("=" * 50)
    print("📧 BYDFi MoonX — 今日外联邮件发送")
    print("=" * 50)

    if not GMAIL or not APP_PASSWORD:
        print("❌ 缺少邮件凭据，检查 .env.outreach")
        exit(1)

    for e in EMAILS:
        try:
            send_email(e["to"], e["name"], e["subject"], e["body"])
        except Exception as err:
            print(f"  ❌ 发送失败 {e['to']}: {err}")

    print("\n完成。")
