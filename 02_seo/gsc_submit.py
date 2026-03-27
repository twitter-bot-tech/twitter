#!/usr/bin/env python3
"""
Google 收录加速 — 两种方式：
1. IndexNow (Bing/Yandex，间接影响 Google)
2. Google Indexing API (OAuth2 服务账号，需 bydfi.com 在 GSC 已验证)
"""
import os
import json
import logging
import requests
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
PUB_FILE = LOG_DIR / "published_articles.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "gsc_submit.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# IndexNow key（任意字符串，需在 bydfi.com 根目录放同名 .txt 文件）
INDEXNOW_KEY  = os.getenv("INDEXNOW_KEY", "moonx-indexnow-2026")
INDEXNOW_HOST = "www.bydfi.com"  # bydfi.com 主域


def submit_indexnow(urls: list) -> bool:
    """批量提交到 IndexNow（Bing/Yandex 实时收录，间接推动 Google）"""
    payload = {
        "host":    INDEXNOW_HOST,
        "key":     INDEXNOW_KEY,
        "keyLocation": f"https://{INDEXNOW_HOST}/{INDEXNOW_KEY}.txt",
        "urlList": urls,
    }
    # 同时提交到多个搜索引擎
    endpoints = [
        "https://api.indexnow.org/indexnow",
        "https://www.bing.com/indexnow",
    ]
    success = True
    for endpoint in endpoints:
        try:
            resp = requests.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            logger.info(f"IndexNow {endpoint.split('/')[2]}: {resp.status_code}")
            if resp.status_code not in (200, 202):
                logger.warning(f"返回: {resp.text[:100]}")
                success = False
        except Exception as e:
            logger.error(f"IndexNow 提交失败: {e}")
            success = False
    return success


def ping_google_sitemap():
    """Ping Google 更新 dev.to 的 RSS/Atom feed（让 Google 尽快发现新内容）"""
    feeds = [
        "https://dev.to/feed/bydfi-moonx",
        "https://medium.com/feed/@ppmworker",
    ]
    for feed in feeds:
        try:
            resp = requests.get(
                f"https://www.google.com/ping",
                params={"sitemap": feed},
                timeout=10,
            )
            logger.info(f"Google ping {feed.split('/')[2]}: {resp.status_code}")
        except Exception as e:
            logger.error(f"Google ping 失败: {e}")


def submit_google_indexing_api(urls: list):
    """
    Google Indexing API — 需要 OAuth2 服务账号
    前提：bydfi.com 已在 Google Search Console 验证
    服务账号 JSON 路径：.env 中 GOOGLE_SERVICE_ACCOUNT_JSON
    """
    sa_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sa_path or not Path(sa_path).exists():
        logger.warning("Google Indexing API: 未配置服务账号（GOOGLE_SERVICE_ACCOUNT_JSON），跳过")
        return

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        creds = service_account.Credentials.from_service_account_file(
            sa_path,
            scopes=["https://www.googleapis.com/auth/indexing"],
        )
        service = build("indexing", "v3", credentials=creds)

        for url in urls:
            result = service.urlNotifications().publish(
                body={"url": url, "type": "URL_UPDATED"}
            ).execute()
            logger.info(f"✅ Google Indexing API: {url} → {result}")
    except Exception as e:
        logger.error(f"Google Indexing API 失败: {e}")


def run():
    logger.info("=== Google/IndexNow 收录提交 ===")

    # 读取最新发布的 URL
    if not PUB_FILE.exists():
        logger.error("published_articles.json 不存在")
        return

    records = json.loads(PUB_FILE.read_text())
    urls = [r["url"] for r in records if r.get("url")]

    if not urls:
        logger.info("没有 URL 需要提交")
        return

    logger.info(f"提交 {len(urls)} 个 URL")

    # 1. IndexNow（Bing + api.indexnow.org）
    submit_indexnow(urls)

    # 2. Ping Google sitemap/feed
    ping_google_sitemap()

    # 3. Google Indexing API（可选，需服务账号）
    submit_google_indexing_api(urls)

    logger.info("=== 完成 ===")


if __name__ == "__main__":
    run()
