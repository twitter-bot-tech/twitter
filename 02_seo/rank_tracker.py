#!/usr/bin/env python3
"""
Rand — SEO关键词排名追踪
每周自动查询Google排名，记录变化，生成周报
"""
import json
import os
import requests
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
DATA_FILE   = Path(__file__).parent / "logs" / "rank_history.json"
REPORT_DIR  = Path(__file__).parent.parent / "outbox"

# 追踪的关键词列表
KEYWORDS = [
    {"keyword": "polymarket alternative",            "priority": "A"},
    {"keyword": "prediction market smart money",     "priority": "A"},
    {"keyword": "crypto prediction market tracker",  "priority": "A"},
    {"keyword": "prediction market signals crypto",  "priority": "B"},
    {"keyword": "meme coin smart money signals",     "priority": "B"},
    {"keyword": "moonx bydfi",                       "priority": "A"},
    {"keyword": "whale tracking crypto",             "priority": "B"},
]

TARGET_DOMAIN = "bydfi.com"


def check_rank(keyword: str) -> dict:
    """查询关键词在Google的排名，返回排名信息"""
    params = {
        "engine":   "google",
        "q":        keyword,
        "num":      100,
        "hl":       "en",
        "gl":       "us",
        "api_key":  SERPAPI_KEY,
    }
    try:
        resp = requests.get("https://serpapi.com/search", params=params, timeout=15)
        data = resp.json()
    except Exception as e:
        return {"keyword": keyword, "rank": None, "url": None, "error": str(e)}

    results = data.get("organic_results", [])
    for i, r in enumerate(results, 1):
        link = r.get("link", "")
        if TARGET_DOMAIN in link:
            return {
                "keyword": keyword,
                "rank":    i,
                "url":     link,
                "title":   r.get("title", ""),
                "error":   None,
            }

    return {"keyword": keyword, "rank": None, "url": None, "title": None, "error": None}


def load_history() -> list:
    if DATA_FILE.exists():
        return json.loads(DATA_FILE.read_text())
    return []


def save_history(history: list):
    DATA_FILE.parent.mkdir(exist_ok=True)
    DATA_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False))


def get_prev_rank(history: list, keyword: str):
    """从历史记录里找上次排名"""
    for entry in reversed(history):
        for r in entry.get("results", []):
            if r["keyword"] == keyword:
                return r["rank"]
    return None


def run():
    print("=== Rand — 关键词排名追踪 ===")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    history = load_history()

    results = []
    for item in KEYWORDS:
        kw = item["keyword"]
        print(f"  查询: {kw} ...", end=" ", flush=True)
        result = check_rank(kw)
        result["priority"] = item["priority"]
        prev = get_prev_rank(history, kw)
        result["prev_rank"] = prev
        results.append(result)

        rank_str = f"#{result['rank']}" if result["rank"] else "未进Top100"
        change = ""
        if prev and result["rank"]:
            diff = prev - result["rank"]
            change = f" (↑{diff})" if diff > 0 else f" (↓{abs(diff)})" if diff < 0 else " (−)"
        elif prev and not result["rank"]:
            change = f" (↓ 掉出Top100，上次#{prev})"
        print(f"{rank_str}{change}")

    # 保存历史
    history.append({"date": today, "results": results})
    save_history(history)

    # 生成周报 Markdown
    report_path = REPORT_DIR / f"{today}_SEO排名周报.md"
    lines = [
        f"# SEO关键词排名周报 — {today}\n",
        f"域名：`{TARGET_DOMAIN}` | 搜索引擎：Google US\n",
        "| 优先级 | 关键词 | 本次排名 | 上次排名 | 变化 | 命中URL |",
        "|--------|--------|---------|---------|------|---------|",
    ]
    for r in results:
        rank     = f"#{r['rank']}" if r["rank"] else "未进Top100"
        prev     = f"#{r['prev_rank']}" if r["prev_rank"] else "—"
        if r["rank"] and r["prev_rank"]:
            diff = r["prev_rank"] - r["rank"]
            chg  = f"↑{diff}" if diff > 0 else f"↓{abs(diff)}" if diff < 0 else "—"
        else:
            chg = "新" if not r["prev_rank"] else "—"
        url  = r["url"] or "—"
        lines.append(f"| {r['priority']} | {r['keyword']} | {rank} | {prev} | {chg} | {url} |")

    lines += [
        "",
        "## 已发布文章",
    ]
    pub_file = Path(__file__).parent / "logs" / "published_articles.json"
    if pub_file.exists():
        articles = json.loads(pub_file.read_text())
        for a in articles:
            lines.append(f"- [{a['title']}]({a['url']}) — {a['published_at']}")

    lines += ["", f"*自动生成 by Rand | {today}*"]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n周报已生成: {report_path.name}")


if __name__ == "__main__":
    import sys as _sys; _sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.pid_lock import acquire_lock; acquire_lock()
    run()
