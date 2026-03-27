#!/usr/bin/env python3
"""
MoonX KOL — Content Brief 自动生成器
每日 10:00 BJT 运行，检查昨日新签约 KOL，自动：
1. 用 Claude 生成个性化 Content Brief
2. 发邮件给 KOL
3. Lark 通知 Kelly 确认
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
        logging.FileHandler(LOG_DIR / "content_brief.log", encoding="utf-8"),
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

MOONX_URL   = "https://www.bydfi.com/en/moonx/markets/trending"
REBATE_URL  = "https://www.bydfi.com/zh/moonx/account/my-rebate?type=my-rebate"
PUBLISH_DEADLINE_DAYS = 14  # 签约后多少天内需发布


# ─────────────────────────────────────────────────────────────────────────────
# Claude 生成 Content Brief
# ─────────────────────────────────────────────────────────────────────────────

def generate_brief(kol_name: str, platform: str, subscribers: int,
                   content_type: str, deliverables: str, total_value: float,
                   utm_code: str, publish_deadline: str) -> str:
    """用 Claude 生成个性化 Content Brief 正文"""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import claude_cli as anthropic
        client = anthropic.Anthropic()

        utm_url = (
            f"https://www.bydfi.com/en/moonx/markets/trending"
            f"?utm_source=kol&utm_medium={platform.lower()}&utm_campaign=collab&utm_content={utm_code}"
            if utm_code else MOONX_URL
        )

        prompt = f"""Write a professional Content Brief email for a KOL partnership.

KOL Details:
- Name: {kol_name}
- Platform: {platform}
- Subscribers/followers: {subscribers:,}
- Content type agreed: {content_type or 'video review / mention'}
- Deliverables: {deliverables or '1 dedicated video or 2 mentions'}
- Partnership value: ${total_value or 'TBD'}
- Publish deadline: {publish_deadline}
- Their tracking link: {utm_url}
- Their rebate dashboard: {REBATE_URL}

Write the email in English. Tone: professional, friendly, clear. Structure:
1. Opening: thank them for partnering, confirm excitement
2. What we need: specific content requirements (mention MoonX features: smart wallet tracking, Polymarket/Kalshi aggregation, meme coin signals)
3. Key messages to include: 3-4 bullet points they must convey
4. Their tracking link (must be prominently placed)
5. Rebate dashboard link so they can track earnings
6. Publish deadline reminder
7. Review process: they can send draft to this email before publishing
8. Contact: {SENDER_TG} on TG for quick questions

Keep it under 400 words. Do NOT add a subject line. Start directly with "Hi {kol_name},"."""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    except Exception as e:
        logger.error(f"Claude 生成 Brief 失败: {e}")
        # 降级到固定模板
        return _fallback_brief(kol_name, content_type, deliverables,
                               total_value, utm_code, publish_deadline)


def _fallback_brief(kol_name: str, content_type: str, deliverables: str,
                    total_value: float, utm_code: str, publish_deadline: str) -> str:
    utm_url = (
        f"{MOONX_URL}?utm_source=kol&utm_medium=email&utm_campaign=collab&utm_content={utm_code}"
        if utm_code else MOONX_URL
    )
    return (
        f"Hi {kol_name},\n\n"
        f"Thank you for partnering with MoonX! We're excited to work with you.\n\n"
        f"Here's your Content Brief:\n\n"
        f"Content Type: {content_type or 'Video review / mention'}\n"
        f"Deliverables: {deliverables or '1 dedicated video or 2 mentions'}\n"
        f"Partnership Value: ${total_value or 'TBD'}\n"
        f"Publish By: {publish_deadline}\n\n"
        f"Key points to cover:\n"
        f"• MoonX aggregates real-time smart money flows from Polymarket, Kalshi, and on-chain markets\n"
        f"• Tracks 500K+ wallets, filters 97% noise — only high-conviction signals\n"
        f"• One-click copy trading for any wallet\n"
        f"• Free to use, with a generous rev-share program\n\n"
        f"Your tracking link (use this in your video description / bio):\n"
        f"{utm_url}\n\n"
        f"Track your earnings here:\n"
        f"{REBATE_URL}\n\n"
        f"Please send your draft/script to this email before publishing so we can give feedback.\n\n"
        f"Questions? Ping me on TG: {SENDER_TG}\n\n"
        f"Looking forward to your content!\n\n"
        f"{SENDER_NAME}\n"
        f"{SENDER_TITLE} | MoonX by BYDFi"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 发送邮件
# ─────────────────────────────────────────────────────────────────────────────

def send_brief_email(to_email: str, kol_name: str, body: str) -> bool:
    subject = f"MoonX Partnership — Content Brief for {kol_name}"
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
        logger.error(f"发送失败 → {to_email}: {e}")
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
    today   = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info(f"Content Brief 生成器开始 — {today}")
    logger.info("=" * 60)

    # 查找昨日以来新签约且 brief 尚未发送的 KOL
    with kol_db.get_db(db_path) as conn:
        rows = conn.execute("""
            SELECT k.id, k.name, k.email, k.platform, k.subscribers,
                   k.utm_code, k.tg_handle,
                   c.id as contract_id, c.total_value_usd,
                   c.revenue_share_rate, c.deliverables, c.contract_date,
                   n.content_type, n.price_usd as nego_price
            FROM kols k
            JOIN contracts c ON c.kol_id = k.id
            LEFT JOIN negotiations n ON n.kol_id = k.id
            WHERE k.status = '已签约'
              AND c.contract_date >= ?
              AND NOT EXISTS (
                  SELECT 1 FROM content ct
                  WHERE ct.kol_id = k.id AND ct.brief_sent_at IS NOT NULL
              )
            ORDER BY c.contract_date DESC
        """, (yesterday,)).fetchall()

    logger.info(f"待发 Content Brief: {len(rows)} 个")

    sent_count = 0
    for row in rows:
        kol_id      = row["id"]
        name        = row["name"]
        email       = row["email"] or ""
        platform    = row["platform"] or "YouTube"
        subscribers = row["subscribers"] or 0
        utm_code    = row["utm_code"] or ""
        contract_id = row["contract_id"]
        total_value = row["total_value_usd"] or row["nego_price"] or 0
        deliverables = row["deliverables"] or ""
        content_type = row["content_type"] or ""
        publish_deadline = (
            datetime.now() + timedelta(days=PUBLISH_DEADLINE_DAYS)
        ).strftime("%Y-%m-%d")

        logger.info(f"\n[{kol_id}] {name} <{email}>")

        if not email or "@" not in email:
            logger.warning("  无有效邮箱，跳过")
            lark_notify(
                f"Content Brief 提醒\n\n"
                f"KOL {name} 已签约但无邮箱，请手动发 Brief\n"
                f"TG: {row['tg_handle'] or '未知'}"
            )
            continue

        # 生成 Brief
        logger.info("  正在生成 Content Brief...")
        brief_body = generate_brief(
            kol_name=name,
            platform=platform,
            subscribers=subscribers,
            content_type=content_type,
            deliverables=deliverables,
            total_value=total_value,
            utm_code=utm_code,
            publish_deadline=publish_deadline,
        )

        # 发送邮件
        ok = send_brief_email(email, name, brief_body)
        if ok:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with kol_db.get_db(db_path) as conn:
                # 写入 content 表
                conn.execute("""
                    INSERT INTO content (kol_id, contract_id, brief_sent_at, platform)
                    VALUES (?, ?, ?, ?)
                """, (kol_id, contract_id, today, platform))
                # 更新活动日志
                conn.execute("""
                    INSERT INTO activities (kol_id, type, content, operator, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (kol_id, "brief_sent", f"Content Brief 已发至 {email}", "auto", now_str))
                # 设置发布截止日期
                conn.execute("""
                    UPDATE contracts SET deliverables=?, notes=? WHERE id=?
                """, (
                    deliverables or "1 dedicated video",
                    f"publish_deadline:{publish_deadline}",
                    contract_id
                ))

            logger.info(f"  ✓ Content Brief 已发送 → {name}")
            sent_count += 1

            lark_notify(
                f"Content Brief 已发送\n\n"
                f"KOL：{name}\n"
                f"邮箱：{email}\n"
                f"合同金额：${total_value or 'TBD'}\n"
                f"发布截止：{publish_deadline}\n"
                f"追踪码：{utm_code or '—'}\n\n"
                f"请在发布前审核内容草稿。"
            )
        else:
            logger.error(f"  ✗ Brief 发送失败 → {name}")

    logger.info(f"\n完成！发送 {sent_count} 份 Content Brief")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
