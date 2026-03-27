#!/usr/bin/env python3
"""
BYDFi MoonX — KOL 邮件回复检测
每天 09:00 BJT 自动运行，扫描 kelly@bydfi.com 收件箱
匹配已发送 KOL 的邮箱地址，自动写入 SQLite（kol_crm.db）
同时保存邮件正文片段，供 classify_reply.py 做意图分类
"""

import imaplib
import email
import os
import re
import logging
from datetime import datetime, timedelta
from pathlib import Path
from email.header import decode_header
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env.outreach", override=True)

_ON_FLY  = bool(os.getenv("FLY_APP_NAME"))
DATA_DIR = Path("/data") if _ON_FLY else Path(__file__).parent
LOG_DIR  = DATA_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "reply_check.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

IMAP_HOST = "imap.zoho.com"
IMAP_PORT = 993
IMAP_USER = os.getenv("BYDFI_EMAIL", "kelly@bydfi.com")
IMAP_PASS = os.getenv("BYDFI_EMAIL_PASSWORD")

SCAN_FOLDERS = ["INBOX", "&V4NXPpCuTvY-", "Archive", "Notification"]


def _decode_header_str(h: str) -> str:
    """解码邮件 header（处理 base64/quoted-printable 编码）"""
    parts = decode_header(h or "")
    decoded = []
    for part, enc in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(enc or "utf-8", errors="ignore"))
        else:
            decoded.append(str(part))
    return " ".join(decoded)


def _extract_text_body(msg) -> str:
    """从 email.message 中提取纯文本正文"""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if ct == "text/plain" and "attachment" not in cd:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    body = payload.decode(charset, errors="ignore")
                    break
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            body = payload.decode(charset, errors="ignore")
    return body.strip()[:500]  # 只保留前 500 字供分类使用


def fetch_replies(days_back: int = 90) -> list[dict]:
    """
    连接 IMAP，扫描所有文件夹，返回最近 N 天内来信列表
    每封邮件返回: {from_email, from_name, subject, body_snippet}
    """
    results = []
    if not IMAP_PASS:
        logger.error("BYDFI_EMAIL_PASSWORD 未设置，跳过 IMAP 扫描")
        return results

    try:
        mail = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
        mail.login(IMAP_USER, IMAP_PASS)
        since_date = (datetime.now() - timedelta(days=days_back)).strftime("%d-%b-%Y")

        for folder in SCAN_FOLDERS:
            try:
                status, _ = mail.select(folder, readonly=True)
                if status != "OK":
                    continue
                _, data = mail.search(None, f'(SINCE "{since_date}")')
                msg_ids = data[0].split()
                logger.info(f"文件夹 {folder!r}: {len(msg_ids)} 封")

                for mid in msg_ids:
                    try:
                        # 先只拉 header，快速过滤
                        _, hdr_data = mail.fetch(mid, "(RFC822.HEADER)")
                        if not hdr_data or not hdr_data[0]:
                            continue
                        hdr_msg = email.message_from_bytes(hdr_data[0][1])
                        from_header = hdr_msg.get("From", "")
                        m = re.search(r"[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}", from_header)
                        if not m:
                            continue
                        from_email = m.group(0).lower()

                        # 跳过自己发出的邮件
                        if from_email == IMAP_USER.lower():
                            continue

                        subject = _decode_header_str(hdr_msg.get("Subject", ""))

                        # 拉完整邮件正文（只对需要的邮件）
                        body_snippet = ""
                        try:
                            _, full_data = mail.fetch(mid, "(RFC822)")
                            if full_data and full_data[0]:
                                full_msg = email.message_from_bytes(full_data[0][1])
                                body_snippet = _extract_text_body(full_msg)
                        except Exception:
                            pass

                        results.append({
                            "from_email":    from_email,
                            "subject":       subject,
                            "body_snippet":  body_snippet,
                        })
                    except Exception as e:
                        logger.debug(f"处理邮件 {mid} 出错: {e}")

            except Exception as e:
                logger.warning(f"扫描文件夹 {folder} 失败: {e}")

        mail.logout()
        logger.info(f"全部扫描完毕，共 {len(results)} 封邮件")

    except Exception as e:
        logger.error(f"IMAP 连接失败: {e}")

    return results


def main():
    from kol_db import get_kols_by_email, record_reply

    today = datetime.now().strftime("%Y-%m-%d")
    logger.info("=" * 60)
    logger.info(f"KOL 回复检测开始 — {today}")
    logger.info("=" * 60)

    # 1. 加载已发送 KOL（从 SQLite）
    sent_kols = get_kols_by_email()
    logger.info(f"已发送待回复 KOL: {len(sent_kols)} 个")

    if not sent_kols:
        logger.info("无待检测邮箱，退出")
        return

    # 2. 拉取 IMAP 收件箱
    inbox = fetch_replies(days_back=90)
    logger.info(f"收件箱邮件数: {len(inbox)} 封")

    # 3. 交叉匹配 + 记录到 SQLite
    matched = 0
    seen_emails = set()  # 同一发件人只记录一次

    for item in inbox:
        addr = item["from_email"]
        if addr not in sent_kols or addr in seen_emails:
            continue
        seen_emails.add(addr)

        kol = sent_kols[addr]
        reply_id = record_reply(
            kol_id       = kol["id"],
            email_from   = addr,
            subject      = item["subject"],
            body_snippet = item["body_snippet"],
        )
        logger.info(f"  ✓ 已回复 → {kol['name']} <{addr}>  [reply_id={reply_id}]")
        matched += 1

    logger.info(f"\n完成！本次新增回复记录 {matched} 条")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
