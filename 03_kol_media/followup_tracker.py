#!/usr/bin/env python3
"""
MoonX KOL — TG 跟进追踪器
每日 09:30 BJT 运行（classify_reply.py 之后）

逻辑：
1. TG 邀请已发 > 48h，KOL 仍未加 TG（tg_handle 为空）
   → 发1封 follow-up 邮件（仅触发1次），sequence_step → 2
2. Follow-up 发出 > 72h，KOL 仍无回应
   → 标记「冷却」，安排30天后重新激活提醒
"""

import os
import sys
import time
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
        logging.FileHandler(LOG_DIR / "followup_tracker.log", encoding="utf-8"),
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

TG_INVITE_HOURS  = 48   # TG 邀请发出后等待时间
FOLLOWUP_HOURS   = 72   # Follow-up 发出后等待时间
COOLING_DAYS     = 30   # 冷却期天数


# ─────────────────────────────────────────────────────────────────────────────
# 邮件模板
# ─────────────────────────────────────────────────────────────────────────────

def tpl_tg_followup(kol_name: str, original_subject: str = "") -> tuple[str, str]:
    subject = f"Re: {original_subject}" if original_subject else "Quick follow-up — MoonX partnership"
    body = (
        f"Hi {kol_name},\n\n"
        f"Just wanted to follow up on my previous message about a MoonX partnership.\n\n"
        f"I know your inbox gets busy — if you're interested in learning more, "
        f"the easiest way is to ping me on Telegram: {SENDER_TG}\n\n"
        f"Happy to share the full partnership details and answer any questions there.\n\n"
        f"If the timing isn't right, no worries at all — just let me know.\n\n"
        f"Best,\n"
        f"{SENDER_NAME}\n"
        f"{SENDER_TITLE} | MoonX by BYDFi"
    )
    return subject, body


def tpl_reactivation(kol_name: str) -> tuple[str, str]:
    """30天后复活邮件"""
    subject = "MoonX — checking back in"
    body = (
        f"Hi {kol_name},\n\n"
        f"We spoke a little while back about a potential MoonX partnership. "
        f"I wanted to reach out again — we've grown significantly since then and "
        f"have some exciting new collaboration formats.\n\n"
        f"If you have 5 minutes, I'd love to reconnect. "
        f"You can find me on Telegram at {SENDER_TG} or just reply here.\n\n"
        f"Best,\n"
        f"{SENDER_NAME}\n"
        f"{SENDER_TITLE} | MoonX by BYDFi"
    )
    return subject, body


# ─────────────────────────────────────────────────────────────────────────────
# 发送邮件
# ─────────────────────────────────────────────────────────────────────────────

def send_email(to_email: str, subject: str, body: str) -> bool:
    if not SMTP_PASS:
        logger.error("BYDFI_EMAIL_PASSWORD 未设置，跳过发送")
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
        logger.error(f"  发送失败 → {to_email}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Lark 通知
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    sys.path.insert(0, str(Path(__file__).parent))
    import kol_db

    db_path = Path("/data/kol_crm.db") if _ON_FLY else Path(__file__).parent / "kol_crm.db"
    now_bjt = datetime.now()
    today   = now_bjt.strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info(f"KOL 跟进追踪器开始 — {today}")
    logger.info("=" * 60)

    # ── 1. 检查需要发 follow-up 的 KOL ──────────────────────────────────────
    # 条件：status='TG接触' AND sequence_step=1 AND tg_handle 为空
    #        AND last_contact_at < now - 48h
    threshold_48h = (now_bjt - timedelta(hours=TG_INVITE_HOURS)).strftime("%Y-%m-%d %H:%M:%S")

    with kol_db.get_db(db_path) as conn:
        need_followup = conn.execute("""
            SELECT k.id, k.name, k.email, k.sequence_step,
                   k.last_contact_at, k.tg_handle,
                   r.subject as original_subject
            FROM kols k
            LEFT JOIN (
                SELECT kol_id, subject
                FROM replies
                WHERE auto_response_sent=1
                ORDER BY auto_response_at DESC
            ) r ON r.kol_id = k.id
            WHERE k.status='TG接触'
              AND (k.tg_handle IS NULL OR k.tg_handle='')
              AND k.sequence_step = 1
              AND k.last_contact_at < ?
        """, (threshold_48h,)).fetchall()

    logger.info(f"需要 Follow-up（TG邀请 >48h 未响应）: {len(need_followup)} 个")

    followup_sent = 0
    for row in need_followup:
        kol_id   = row["id"]
        name     = row["name"]
        email    = row["email"] or ""
        orig_subj = row["original_subject"] or ""

        logger.info(f"\n[{kol_id}] {name} <{email}>")

        if not email or "@" not in email:
            logger.warning("  无有效邮箱，跳过")
            continue

        subject, body = tpl_tg_followup(name, orig_subj)
        ok = send_email(email, subject, body)

        if ok:
            with kol_db.get_db(db_path) as conn:
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conn.execute(
                    "UPDATE kols SET sequence_step=2, last_contact_at=?, updated_at=? WHERE id=?",
                    (now_str, now_str, kol_id)
                )
                conn.execute(
                    "INSERT INTO contacts (kol_id, sent_at, subject, template, status) VALUES (?,?,?,?,'sent')",
                    (kol_id, now_str, subject, "tg_followup")
                )
                conn.execute(
                    "INSERT INTO activities (kol_id, type, content, operator, created_at) VALUES (?,?,?,?,?)",
                    (kol_id, "followup_sent", f"TG跟进邮件已发 | 主题: {subject}", "auto", now_str)
                )
            logger.info(f"  ✓ Follow-up 邮件已发 → {name}")
            followup_sent += 1
            time.sleep(30)

    # ── 2. 检查需要标记冷却的 KOL ────────────────────────────────────────────
    # 条件：status='TG接触' AND sequence_step=2 AND tg_handle 为空
    #        AND last_contact_at < now - 72h
    threshold_72h = (now_bjt - timedelta(hours=FOLLOWUP_HOURS)).strftime("%Y-%m-%d %H:%M:%S")

    with kol_db.get_db(db_path) as conn:
        need_cooling = conn.execute("""
            SELECT id, name, email, last_contact_at
            FROM kols
            WHERE status='TG接触'
              AND (tg_handle IS NULL OR tg_handle='')
              AND sequence_step = 2
              AND last_contact_at < ?
        """, (threshold_72h,)).fetchall()

    logger.info(f"\n需要标记冷却（Follow-up >72h 无响应）: {len(need_cooling)} 个")

    cooling_count = 0
    for row in need_cooling:
        kol_id = row["id"]
        name   = row["name"]
        logger.info(f"\n[{kol_id}] {name} → 标记冷却")

        kol_db.change_kol_status(kol_id, "冷却", operator="auto", path=db_path)
        kol_db.schedule_followup(kol_id, days=COOLING_DAYS, path=db_path)

        lark_notify(
            f"KOL 跟进追踪\n\n"
            f"KOL：{name}\n"
            f"状态：已标记冷却（TG邀请+跟进均无响应）\n"
            f"计划：{COOLING_DAYS}天后自动重新激活"
        )
        cooling_count += 1

    # ── 3. 检查30天冷却到期，自动发重激活邮件 ────────────────────────────────
    with kol_db.get_db(db_path) as conn:
        need_reactivate = conn.execute("""
            SELECT k.id, k.name, k.email
            FROM kols k
            JOIN followups f ON f.kol_id = k.id
            WHERE k.status='冷却'
              AND f.status='pending'
              AND f.scheduled_at <= ?
              AND (k.email IS NOT NULL AND k.email != '')
        """, (today,)).fetchall()

    logger.info(f"\n冷却到期需要重激活: {len(need_reactivate)} 个")

    reactivated = 0
    for row in need_reactivate:
        kol_id = row["id"]
        name   = row["name"]
        email  = row["email"]

        logger.info(f"\n[{kol_id}] {name} → 发重激活邮件")
        subject, body = tpl_reactivation(name)
        ok = send_email(email, subject, body)

        if ok:
            with kol_db.get_db(db_path) as conn:
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conn.execute(
                    "UPDATE kols SET status='已发送', sequence_step=1, last_contact_at=?, updated_at=? WHERE id=?",
                    (now_str, now_str, kol_id)
                )
                conn.execute(
                    "UPDATE followups SET status='reactivated' WHERE kol_id=?", (kol_id,)
                )
                conn.execute(
                    "INSERT INTO contacts (kol_id, sent_at, subject, template, status) VALUES (?,?,?,?,'sent')",
                    (kol_id, now_str, subject, "reactivation")
                )
                conn.execute(
                    "INSERT INTO activities (kol_id, type, content, operator, created_at) VALUES (?,?,?,?,?)",
                    (kol_id, "reactivated", f"30天冷却期满，重激活邮件已发", "auto", now_str)
                )
            logger.info(f"  ✓ 重激活邮件已发 → {name}")
            reactivated += 1
            time.sleep(30)

    # ── 汇报 ─────────────────────────────────────────────────────────────────
    logger.info(f"\n完成！Follow-up {followup_sent} 封 | 冷却 {cooling_count} 个 | 重激活 {reactivated} 封")

    if followup_sent + cooling_count + reactivated > 0:
        lark_notify(
            f"KOL 跟进追踪报告\n\n"
            f"Follow-up 邮件：{followup_sent} 封\n"
            f"标记冷却：{cooling_count} 个\n"
            f"重激活邮件：{reactivated} 封"
        )

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
