#!/usr/bin/env python3
"""
MoonX KOL — 发布截止提醒 + 催稿
每日 09:45 BJT 运行

逻辑：
1. 已签约 KOL 发布截止前 48h → Lark 提醒 Kelly
2. 超过截止日期且状态仍为「已签约」→ 自动发催稿邮件给 KOL（仅发1次）
3. 已发布但状态未更新的提醒 Kelly 手动确认
"""

import os
import sys
import smtplib
import logging
import json
import urllib.request
from datetime import datetime, timedelta
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
        logging.FileHandler(LOG_DIR / "deadline_reminder.log", encoding="utf-8"),
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


def lark_notify(text: str):
    if not LARK_KOL:
        return
    payload = json.dumps({"msg_type": "text", "content": {"text": text}}).encode()
    try:
        req = urllib.request.Request(
            LARK_KOL, data=payload,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        urllib.request.urlopen(req, timeout=8)
    except Exception as e:
        logger.warning(f"Lark 通知失败: {e}")


def send_nudge_email(to_email: str, kol_name: str, deadline: str, days_overdue: int) -> bool:
    subject = "Quick check-in — MoonX content"
    body = (
        f"Hi {kol_name},\n\n"
        f"Just checking in on our MoonX partnership content.\n\n"
        f"We had agreed on a publish date around {deadline}, "
        f"and wanted to make sure everything is on track on your end.\n\n"
        f"If you need any additional assets, data, or support from our side, "
        f"please let me know — happy to help make the content as strong as possible.\n\n"
        f"You can reach me quickly on Telegram: {SENDER_TG}\n\n"
        f"Looking forward to your content!\n\n"
        f"{SENDER_NAME}\n"
        f"{SENDER_TITLE} | MoonX by BYDFi"
    )
    if not SMTP_PASS:
        logger.error("BYDFI_EMAIL_PASSWORD 未设置")
        return False
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
        return True
    except Exception as e:
        logger.error(f"催稿邮件发送失败 → {to_email}: {e}")
        return False


def main():
    sys.path.insert(0, str(Path(__file__).parent))
    import kol_db

    db_path = Path("/data/kol_crm.db") if _ON_FLY else Path(__file__).parent / "kol_crm.db"
    today   = datetime.now().strftime("%Y-%m-%d")
    in_48h  = (datetime.now() + timedelta(hours=48)).strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info(f"截止提醒检查开始 — {today}")
    logger.info("=" * 60)

    with kol_db.get_db(db_path) as conn:
        # 解析 contracts.notes 里存的 publish_deadline:YYYY-MM-DD
        contracts = conn.execute("""
            SELECT k.id, k.name, k.email, k.status,
                   c.id as contract_id, c.notes, c.total_value_usd
            FROM kols k
            JOIN contracts c ON c.kol_id = k.id
            WHERE k.status IN ('已签约', '审核中')
              AND c.notes LIKE '%publish_deadline:%'
        """).fetchall()

    reminders = 0
    nudges    = 0

    for row in contracts:
        # 提取截止日期
        notes = row["notes"] or ""
        deadline = ""
        for part in notes.split(";"):
            part = part.strip()
            if part.startswith("publish_deadline:"):
                deadline = part.split(":", 1)[1].strip()
                break
        if not deadline:
            continue

        kol_id   = row["id"]
        name     = row["name"]
        email    = row["email"] or ""
        days_left = (datetime.strptime(deadline, "%Y-%m-%d") - datetime.now()).days

        logger.info(f"[{kol_id}] {name} | 截止 {deadline}（{days_left}天）")

        # 截止前 48h 提醒 Kelly
        if 0 <= days_left <= 2:
            logger.info(f"  → 截止临近，Lark 提醒 Kelly")
            lark_notify(
                f"KOL 发布截止提醒\n\n"
                f"KOL：{name}\n"
                f"截止日：{deadline}（还剩 {days_left} 天）\n"
                f"合同金额：${row['total_value_usd'] or 'TBD'}\n"
                f"状态：{row['status']}\n\n"
                f"如内容已发布，请在 Lark 发：\n"
                f"kol 发布 {name} [发布链接]"
            )
            reminders += 1

        # 已超过截止日，发催稿邮件（仅发1次，检查 activities 表）
        elif days_left < 0 and email and "@" in email:
            with kol_db.get_db(db_path) as conn:
                already_nudged = conn.execute("""
                    SELECT id FROM activities
                    WHERE kol_id=? AND type='nudge_sent'
                """, (kol_id,)).fetchone()

            if not already_nudged:
                days_overdue = abs(days_left)
                logger.info(f"  → 逾期 {days_overdue} 天，发催稿邮件")
                ok = send_nudge_email(email, name, deadline, days_overdue)
                if ok:
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with kol_db.get_db(db_path) as conn:
                        conn.execute("""
                            INSERT INTO activities (kol_id, type, content, operator, created_at)
                            VALUES (?, 'nudge_sent', ?, 'auto', ?)
                        """, (kol_id, f"催稿邮件已发，逾期{days_overdue}天", now_str))
                    lark_notify(
                        f"KOL 催稿提醒\n\n"
                        f"KOL：{name}\n"
                        f"截止：{deadline}（已逾期{days_overdue}天）\n"
                        f"已自动发催稿邮件\n\n"
                        f"若超期7天无响应，建议标记「冷却」"
                    )
                    nudges += 1

    logger.info(f"\n完成！截止提醒 {reminders} 条 | 催稿邮件 {nudges} 封")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
