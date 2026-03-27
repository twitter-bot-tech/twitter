#!/usr/bin/env python3
"""
MoonX KOL — ROI 追踪 + 付款提醒 + 30天复购
每日 10:15 BJT 运行

逻辑：
1. 已发布 KOL → 检查 performance 表是否有数据，无则推 Lark 提醒 Kelly 填入
2. 付款到期前 3 天 → Lark 提醒 Kelly 打款
3. 合作完成 30 天后 → 自动发复购邮件（高 ROI KOL 优先）
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
        logging.FileHandler(LOG_DIR / "roi_tracker.log", encoding="utf-8"),
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

ROI_THRESHOLD_HIGH = 3.0   # ROI > 3x 视为高价值 KOL
REPURCHASE_DAYS    = 30    # 合作完成后多少天触发复购邮件


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


def send_repurchase_email(to_email: str, kol_name: str, roi_label: str) -> bool:
    subject = "New MoonX opportunities — let's collaborate again"
    body = (
        f"Hi {kol_name},\n\n"
        f"It's been about a month since our last collaboration — we hope your audience "
        f"found the MoonX content useful!\n\n"
        f"We've shipped some exciting new features since then:\n"
        f"• Enhanced prediction market feed (Polymarket + Kalshi + Manifold)\n"
        f"• Improved meme coin early signal detection\n"
        f"• New portfolio tracker for smart wallet copy-trading\n\n"
        f"Given how well our last collaboration went{roi_label}, "
        f"we'd love to work together again — and we're happy to discuss priority partner terms.\n\n"
        f"If you're open to it, ping me on TG: {SENDER_TG}\n\n"
        f"Best,\n"
        f"{SENDER_NAME}\n"
        f"{SENDER_TITLE} | MoonX by BYDFi"
    )
    if not SMTP_PASS:
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
        logger.error(f"复购邮件发送失败 → {to_email}: {e}")
        return False


def main():
    sys.path.insert(0, str(Path(__file__).parent))
    import kol_db

    db_path = Path("/data/kol_crm.db") if _ON_FLY else Path(__file__).parent / "kol_crm.db"
    today   = datetime.now().strftime("%Y-%m-%d")
    in_3d   = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
    days_ago_30 = (datetime.now() - timedelta(days=REPURCHASE_DAYS)).strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info(f"ROI 追踪 + 付款提醒 + 复购检查 — {today}")
    logger.info("=" * 60)

    # ── 1. 已发布但无 performance 数据 → 提醒 Kelly 填入 ────────────────────
    with kol_db.get_db(db_path) as conn:
        no_roi = conn.execute("""
            SELECT k.id, k.name, ct.published_at, ct.published_url, k.utm_code
            FROM kols k
            JOIN content ct ON ct.kol_id = k.id
            WHERE k.status = '已发布'
              AND ct.published_at IS NOT NULL
              AND ct.published_at <= ?
              AND NOT EXISTS (
                  SELECT 1 FROM performance p WHERE p.kol_id = k.id
              )
        """, ((datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),)).fetchall()

    if no_roi:
        lines = [f"KOL ROI 数据待填写（发布已超48h）："]
        for row in no_roi:
            lines.append(f"  • {row['name']} | 发布: {row['published_at']}")
            lines.append(f"    UTM码: {row['utm_code'] or '—'}")
            url = row['published_url'] or '—'
            lines.append(f"    链接: {url}")
        lines.append(f"\n填写方式（发到 Lark）：")
        lines.append(f"kol roi [名字] 点击:100 注册:20 交易:5")
        lark_notify("\n".join(lines))
        logger.info(f"ROI 待填写提醒: {len(no_roi)} 个")

    # ── 2. 付款到期前 3 天 → Lark 提醒 Kelly ────────────────────────────────
    with kol_db.get_db(db_path) as conn:
        due_payments = conn.execute("""
            SELECT k.id, k.name, p.id as pay_id,
                   p.amount_usd, p.due_date, p.currency
            FROM payments p
            JOIN kols k ON k.id = p.kol_id
            WHERE p.status = 'pending'
              AND p.due_date <= ?
              AND p.due_date >= ?
            ORDER BY p.due_date ASC
        """, (in_3d, today)).fetchall()

    if due_payments:
        lines = ["KOL 付款提醒（3天内到期）："]
        total = 0.0
        for row in due_payments:
            lines.append(f"  • {row['name']} — ${row['amount_usd']} 到期: {row['due_date']}")
            total += row["amount_usd"] or 0
        lines.append(f"\n合计待付：${total:.0f}")
        lines.append(f"\n付款完成后更新状态：kol 付款确认 [名字]")
        lark_notify("\n".join(lines))
        logger.info(f"付款提醒: {len(due_payments)} 笔，合计 ${total:.0f}")

    # ── 3. 合作完成 30 天 → 自动发复购邮件 ──────────────────────────────────
    with kol_db.get_db(db_path) as conn:
        repurchase_candidates = conn.execute("""
            SELECT k.id, k.name, k.email, k.utm_code,
                   ct.published_at,
                   COALESCE(p.revenue_usd, 0) as revenue,
                   COALESCE(c.total_value_usd, 0) as cost
            FROM kols k
            JOIN content ct ON ct.kol_id = k.id
            LEFT JOIN performance p ON p.kol_id = k.id
            LEFT JOIN contracts c ON c.kol_id = k.id
            WHERE k.status = '已完成'
              AND ct.published_at <= ?
              AND NOT EXISTS (
                  SELECT 1 FROM activities a
                  WHERE a.kol_id = k.id AND a.type = 'repurchase_sent'
              )
              AND k.email IS NOT NULL AND k.email != ''
        """, (days_ago_30,)).fetchall()

    repurchased = 0
    for row in repurchase_candidates:
        kol_id  = row["id"]
        name    = row["name"]
        email   = row["email"]
        revenue = row["revenue"] or 0
        cost    = row["cost"] or 0

        # 计算 ROI
        roi = revenue / cost if cost > 0 else 0
        if roi >= ROI_THRESHOLD_HIGH:
            roi_label = f" (ROI {roi:.1f}x — excellent)"
            is_high_roi = True
        elif roi > 0:
            roi_label = f" (ROI {roi:.1f}x)"
            is_high_roi = False
        else:
            roi_label = ""
            is_high_roi = False

        logger.info(f"[{kol_id}] {name} | ROI={roi:.1f}x | 发复购邮件")
        ok = send_repurchase_email(email, name, roi_label)

        if ok:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with kol_db.get_db(db_path) as conn:
                conn.execute("""
                    INSERT INTO activities (kol_id, type, content, operator, created_at)
                    VALUES (?, 'repurchase_sent', ?, 'auto', ?)
                """, (kol_id, f"复购邮件已发 | ROI={roi:.1f}x", now_str))
                # 高 ROI KOL 打上战略标签
                if is_high_roi:
                    conn.execute(
                        "UPDATE kols SET notes=COALESCE(notes||'; ','')|| '战略KOL-高ROI', score=? WHERE id=?",
                        (min(int(roi * 20), 100), kol_id)
                    )
            repurchased += 1

    if repurchased > 0:
        lark_notify(f"复购邮件已自动发送：{repurchased} 封")

    logger.info(f"\n完成！ROI提醒 {len(no_roi)} | 付款提醒 {len(due_payments)} | 复购 {repurchased}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
