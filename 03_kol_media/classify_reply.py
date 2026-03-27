#!/usr/bin/env python3
"""
BYDFi MoonX — KOL 回复意图分类 + 自动跟进
每天 09:15 BJT 运行（check_replies.py 之后）
1. 读取 replies 表中 intent='unknown' 的记录
2. 用 Claude 分类意图：interested / rejected / ooo / inquiry / unknown
3. 根据意图自动触发：
   - interested → 发 TG 邀请邮件 + Lark 通知 Kelly
   - rejected   → 更新状态，记录
   - ooo        → 记录，不做其他操作（等待对方回来）
   - inquiry    → Lark 通知 Kelly 手动跟进
"""

import os
import re
import sys
import time
import smtplib
import logging
import urllib.request
import json
from datetime import datetime
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env.outreach", override=True)
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

_ON_FLY  = bool(os.getenv("FLY_APP_NAME"))
DATA_DIR = Path("/data") if _ON_FLY else Path(__file__).parent
LOG_DIR  = DATA_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "classify_reply.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

SMTP_HOST    = os.getenv("BYDFI_SMTP_HOST", "smtp.zoho.com")
SMTP_PORT    = int(os.getenv("BYDFI_SMTP_PORT", "465"))
SMTP_USER    = os.getenv("BYDFI_EMAIL", "kelly@bydfi.com")
SMTP_PASS    = os.getenv("BYDFI_EMAIL_PASSWORD")
SENDER_NAME  = os.getenv("SENDER_NAME", "Kelly")
SENDER_TITLE = os.getenv("SENDER_TITLE", "Head of Marketing")
SENDER_TG    = os.getenv("SENDER_TG", "@BDkelly")
LARK_KOL     = os.getenv("LARK_KOL")


# ─────────────────────────────────────────────────────────────────────────────
# Claude 意图分类
# ─────────────────────────────────────────────────────────────────────────────

def classify_intent(kol_name: str, body_snippet: str, subject: str = "") -> str:
    """
    用 Claude 分析邮件正文，返回意图标签：
    interested / rejected / ooo / inquiry / unknown
    """
    if not body_snippet or len(body_snippet.strip()) < 10:
        return "unknown"

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import claude_cli as anthropic
        client = anthropic.Anthropic()

        prompt = f"""You are analyzing a reply email from a YouTube creator (KOL) to a partnership offer.

Sender name: {kol_name}
Email subject: {subject or "(no subject)"}
Email body (first 500 chars):
---
{body_snippet}
---

Classify the sender's intent as exactly ONE of:
- interested: they express positive interest, want to know more, ask about terms, or say yes
- rejected: they decline, not interested, already have a deal elsewhere, or explicitly say no
- ooo: auto-reply or out-of-office message
- inquiry: they ask a specific question about rates, platform details, or requirements
- unknown: cannot determine intent from the text

Reply with ONLY the single lowercase label, nothing else."""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        result = response.content[0].text.strip().lower()
        valid = {"interested", "rejected", "ooo", "inquiry", "unknown"}
        return result if result in valid else "unknown"

    except Exception as e:
        logger.error(f"Claude 分类失败: {e}")
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# 邮件发送
# ─────────────────────────────────────────────────────────────────────────────

def send_tg_invite(to_email: str, kol_name: str, original_subject: str = "") -> bool:
    """向回复的 KOL 发送 TG 邀请邮件"""
    subject = f"Re: {original_subject}" if original_subject else "Let's connect on Telegram"
    body = (
        f"Hi {kol_name},\n\n"
        f"Thanks for getting back to me!\n\n"
        f"Let's continue on Telegram for faster communication — "
        f"I'm {SENDER_TG}.\n\n"
        f"Feel free to ping me there and I'll share more details about the partnership "
        f"terms and how other creators are using MoonX for their content.\n\n"
        f"Looking forward to chatting!\n\n"
        f"{SENDER_NAME}\n"
        f"{SENDER_TITLE}, BYDFi MoonX"
    )
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"{SENDER_NAME} <{SMTP_USER}>"
        msg["To"]      = to_email
        msg.attach(MIMEText(body, "plain", "utf-8"))

        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30) as server:
                server.login(SMTP_USER, SMTP_PASS)
                server.sendmail(SMTP_USER, to_email, msg.as_string())
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASS)
                server.sendmail(SMTP_USER, to_email, msg.as_string())

        logger.info(f"  ✓ TG 邀请邮件已发送 → {kol_name} <{to_email}>")
        return True
    except Exception as e:
        logger.error(f"  ✗ TG 邀请发送失败 → {to_email}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Lark 通知
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_LABELS = {
    "interested": "感兴趣",
    "rejected":   "已拒绝",
    "ooo":        "不在线(OOO)",
    "inquiry":    "询价/问细节",
    "unknown":    "未识别",
}

_INTENT_ACTIONS = {
    "interested": "已自动发送 TG 邀请邮件，等待添加 TG",
    "rejected":   "已记录，状态更新为已拒绝",
    "ooo":        "已记录，对方不在线，等待其回来",
    "inquiry":    "需要 Kelly 手动回复询价",
    "unknown":    "无法识别意图，建议手动查看",
}


def lark_notify(kol_name: str, email_from: str, intent: str, subject: str = ""):
    if not LARK_KOL:
        return
    label  = _INTENT_LABELS.get(intent, intent)
    action = _INTENT_ACTIONS.get(intent, "")
    _on_fly = bool(os.getenv("FLY_APP_NAME"))
    crm_url = "https://moonx-lark-server.fly.dev/crm/pending" if _on_fly else "http://localhost:8090/crm/pending"
    link_line = f"\n点此处理：{crm_url}" if intent in ("interested", "inquiry") else ""
    text   = (
        f"KOL 回复检测\n\n"
        f"KOL：{kol_name}\n"
        f"邮箱：{email_from}\n"
        f"主题：{subject or '(无主题)'}\n"
        f"意图：{label}\n"
        f"处理：{action}{link_line}"
    )
    payload = json.dumps({
        "msg_type": "text",
        "content":  {"text": text},
    }).encode()
    try:
        req = urllib.request.Request(
            LARK_KOL, data=payload,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        urllib.request.urlopen(req, timeout=8)
        logger.info(f"  Lark 通知已发送 (intent={intent})")
    except Exception as e:
        logger.warning(f"  Lark 通知失败: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    from kol_db import (get_unclassified_replies, set_reply_intent,
                        mark_auto_response_sent, create_pending_action)

    today = datetime.now().strftime("%Y-%m-%d")
    logger.info("=" * 60)
    logger.info(f"KOL 回复意图分类开始 — {today}")
    logger.info("=" * 60)

    replies = get_unclassified_replies()
    logger.info(f"待分类回复: {len(replies)} 条")

    if not replies:
        logger.info("无待分类回复，退出")
        return

    for r in replies:
        reply_id    = r["id"]
        kol_name    = r["name"]
        email_from  = r["email_from"] or r["email"]
        subject     = r["subject"] or ""
        body        = r["body_snippet"] or ""

        logger.info(f"\n[{reply_id}] {kol_name} <{email_from}>")
        logger.info(f"  主题: {subject[:60]}")

        # 1. 分类意图
        intent = classify_intent(kol_name, body, subject)
        logger.info(f"  意图: {intent}")
        set_reply_intent(reply_id, intent)

        # 2. 按意图执行动作
        if intent == "interested":
            if SMTP_PASS and email_from:
                ok = send_tg_invite(email_from, kol_name, subject)
                if ok:
                    mark_auto_response_sent(reply_id)
                    time.sleep(30)  # 防止发送过快
            # 推送待处理事项：报价待确认
            kol_id = r.get("kol_id")
            if kol_id:
                create_pending_action(kol_id, "quote_needed", {
                    "reply_snippet": body[:200],
                    "reply_subject": subject,
                })
                logger.info(f"  ✓ 已创建待处理事项 → /crm/pending (quote_needed)")
            lark_notify(kol_name, email_from, intent, subject)

        elif intent in ("inquiry", "ooo", "rejected", "unknown"):
            # inquiry = KOL 在询价，也推报价待确认
            if intent == "inquiry":
                kol_id = r.get("kol_id")
                if kol_id:
                    create_pending_action(kol_id, "quote_needed", {
                        "reply_snippet": body[:200],
                        "reply_subject": subject,
                    })
                    logger.info(f"  ✓ 已创建待处理事项 → /crm/pending (inquiry→quote_needed)")
            lark_notify(kol_name, email_from, intent, subject)

    logger.info(f"\n完成！处理 {len(replies)} 条回复")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
