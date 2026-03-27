#!/usr/bin/env python3
"""
BYDFi MoonX — Meme 币 KOL 外联 + 社群推广脚本
"""

import smtplib, os, time, random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env.outreach")

GMAIL        = os.getenv("GMAIL_ADDRESS")
APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
SENDER_NAME  = os.getenv("SENDER_NAME", "Kelly")
SENDER_TITLE = os.getenv("SENDER_TITLE", "Head of Marketing")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_TG    = os.getenv("SENDER_TG", "@BDkelly")
PLATFORM     = "BYDFi MoonX"
MOONX_URL    = "https://www.bydfi.com/en/moonx/markets/trending"

# ══════════════════════════════════════════
# KOL 邮件模板
# ══════════════════════════════════════════

def tpl_kol_meme(kol_name):
    subject = "Collab Opportunity: MoonX — The Fastest Meme Coin Trading Platform"
    body = f"""Hey {kol_name} 👋

Love your meme coin content — you're one of the sharpest voices in the space.

I'm Kelly, Head of Marketing at BYDFi. We just launched MoonX ({MOONX_URL}), and I think your audience would love it.

What makes MoonX different:
• Real-time meme coin discovery — catch pumps before they go viral
• Smart money tracking — see where whales are moving before you miss the move
• One-click trading directly from the trend feed — no more switching tabs
• Aggregated from Polymarket, pump.fun, DEX Screener and more

We're building our first wave of KOL partners and would love to have you on board:
✅ Exclusive early access + referral dashboard
✅ Revenue share on every trader you bring in
✅ Custom meme coin data feeds for your content
✅ Co-marketing support (we'll amplify your posts)

Link to share with your audience: {MOONX_URL}

Interested? Let's jump on a quick call or chat on Telegram.

Cheers,
{SENDER_NAME}
{SENDER_TITLE} | {PLATFORM}
{MOONX_URL}
Email: {SENDER_EMAIL}
Telegram: {SENDER_TG}"""
    return subject, body


# ══════════════════════════════════════════
# 社群推广消息模板（Telegram / Discord）
# ══════════════════════════════════════════

COMMUNITY_MSG_EN = f"""👀 Meme coin traders — check this out

Just found MoonX by BYDFi and it's actually solid for finding early meme plays:

🔥 Real-time trending meme coins
🐋 Smart money / whale tracking
⚡ One-click trade straight from the feed
📊 Data from pump.fun, DEX Screener & more

Worth bookmarking if you're hunting early pumps 👇
{MOONX_URL}

(Not financial advice — always DYOR)"""

COMMUNITY_MSG_SOLANA = f"""Solana meme hunters 🚨

If you're not using MoonX yet you're probably missing moves.

Shows you what's trending in real-time + where smart money is going before it pumps. Built for speed — one-click trade from the discovery feed.

Try it: {MOONX_URL}

Drop your thoughts below 👇"""

COMMUNITY_MSG_GMGN = f"""For everyone who uses GMGN to track meme coins —

MoonX ({MOONX_URL}) is worth a look as a companion tool:

→ Aggregates meme coin signals across chains
→ Smart money alerts
→ Integrated trading (no wallet switching)

Free to use. Link: {MOONX_URL}"""


# ══════════════════════════════════════════
# 发送函数
# ══════════════════════════════════════════

def send_email(to_email, to_name, dry_run=True):
    subject, body = tpl_kol_meme(to_name)
    if dry_run:
        print(f"\n{'='*60}")
        print(f"[DRY RUN] 收件人: {to_name} <{to_email}>")
        print(f"主题: {subject}")
        print(f"内容:\n{body}")
        print(f"{'='*60}")
        return True
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"{SENDER_NAME} <{GMAIL}>"
        msg["To"]      = to_email
        msg.attach(MIMEText(body, "plain", "utf-8"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL, APP_PASSWORD)
            server.sendmail(GMAIL, to_email, msg.as_string())
        print(f"  ✓ 已发送 → {to_name} <{to_email}>")
        return True
    except Exception as e:
        print(f"  ✗ 失败 → {to_email}：{e}")
        return False


if __name__ == "__main__":
    import sys
    dry_run = "--send" not in sys.argv

    if dry_run:
        print("⚠️  测试模式 — 不会真实发送")
        print("   真实发送：python3 meme_outreach.py --send\n")

    # 打印社群消息供手动发布
    print("="*60)
    print("【社群推广消息 — 通用版】")
    print("="*60)
    print(COMMUNITY_MSG_EN)
    print("\n" + "="*60)
    print("【社群推广消息 — Solana 社群版】")
    print("="*60)
    print(COMMUNITY_MSG_SOLANA)
    print("\n" + "="*60)
    print("【社群推广消息 — GMGN 用户版】")
    print("="*60)
    print(COMMUNITY_MSG_GMGN)
