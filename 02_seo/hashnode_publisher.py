#!/usr/bin/env python3
"""
Hashnode 自动发布 — 通过 GraphQL API 同步文章
Hashnode 文章在 Google 权重高，收录快
canonical_url 指向 Medium 原文，不影响原始 SEO
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
PUB_FILE   = LOG_DIR / "published_articles.json"
OUTBOX     = Path(__file__).parent.parent / "outbox"

HASHNODE_API = "https://gql.hashnode.com"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "hashnode_publisher.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def gql(query: str, variables: dict, token: str) -> dict:
    resp = requests.post(
        HASHNODE_API,
        json={"query": query, "variables": variables},
        headers={"Authorization": token, "Content-Type": "application/json"},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise ValueError(data["errors"])
    return data


def publish_post(token: str, pub_id: str, title: str, body: str, tags: list, canonical: str) -> str:
    mutation = """
    mutation PublishPost($input: PublishPostInput!) {
      publishPost(input: $input) {
        post {
          url
        }
      }
    }
    """
    # Hashnode tags: [{"name": "...", "slug": "..."}]
    hn_tags = [{"name": t, "slug": t.lower().replace(" ", "-")} for t in tags[:5]]

    variables = {
        "input": {
            "title":          title,
            "contentMarkdown": body,
            "publicationId":  pub_id,
            "tags":           hn_tags,
            "originalArticleURL": canonical,
        }
    }
    data = gql(mutation, variables, token)
    return data["data"]["publishPost"]["post"]["url"]


def extract_body(md_path: Path) -> str:
    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    body, in_body = [], False
    for line in lines:
        if not in_body:
            if line.startswith("# ") and not line.startswith("# SEO Article"):
                in_body = True
        if in_body:
            body.append(line)
    return "\n".join(body).strip()


def load_published() -> list:
    return json.loads(PUB_FILE.read_text()) if PUB_FILE.exists() else []


def already_published(records: list, title: str) -> bool:
    return any(
        r.get("title") == title and "hashnode" in r.get("platform", "").lower()
        for r in records
    )


def run():
    logger.info("=== Hashnode 发布 ===")
    token  = os.getenv("HASHNODE_TOKEN")
    pub_id = os.getenv("HASHNODE_PUBLICATION_ID")

    if not token or not pub_id:
        logger.error("缺少 HASHNODE_TOKEN 或 HASHNODE_PUBLICATION_ID，请在 .env 配置")
        return

    records = load_published()
    # 取所有有 dev.to URL 且有 canonical（=Medium URL）的文章
    devto_articles = [r for r in records if "dev.to" in r.get("platform", "") and r.get("canonical")]

    if not devto_articles:
        logger.info("没有可同步到 Hashnode 的文章")
        return

    new_records = []
    for art in devto_articles:
        title    = art["title"]
        canonical = art["canonical"]

        if already_published(records, title):
            logger.info(f"已发布到 Hashnode，跳过: {title[:50]}")
            continue

        # 找对应的 md 文件
        md_files = list(OUTBOX.glob(f"*SEO文章*.md"))
        md_path  = None
        for f in md_files:
            content = f.read_text(encoding="utf-8")
            if title[:30] in content or canonical.split("/")[-1][:20] in f.name:
                md_path = f
                break

        if not md_path:
            logger.warning(f"找不到 md 文件: {title[:50]}")
            continue

        body = extract_body(md_path)
        tags = art.get("target_keyword", ["crypto", "predictionmarkets"])
        if isinstance(tags, str):
            tags = [tags]

        logger.info(f"发布到 Hashnode: {title[:60]}")
        try:
            url = publish_post(token, pub_id, title, body, tags, canonical)
            logger.info(f"✅ {url}")
            new_records.append({
                "title":       title,
                "url":         url,
                "platform":    "hashnode",
                "canonical":   canonical,
                "published_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            })
        except Exception as e:
            logger.error(f"Hashnode 发布失败: {e}")

    if new_records:
        records.extend(new_records)
        PUB_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False))

    logger.info(f"=== 完成，新发布 {len(new_records)} 篇 ===")


if __name__ == "__main__":
    run()
