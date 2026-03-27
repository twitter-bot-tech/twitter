#!/usr/bin/env python3
"""
MoonX OKR 周报推送
每周五 09:00 BJT 读取 05_strategy/okr_data.json，格式化后发到 LARK_LEAD 群
"""
import os, json, urllib.request
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

BASE = Path(__file__).parent
load_dotenv(BASE / ".env", override=True)

BJT = ZoneInfo("Asia/Shanghai")
OKR_FILE = BASE / "05_strategy" / "okr_data.json"
LEAD_WEBHOOK = os.getenv("LARK_LEAD")


def send_card(webhook_url: str, card: dict) -> bool:
    payload = json.dumps({"msg_type": "interactive", "card": card}, ensure_ascii=False).encode()
    req = urllib.request.Request(webhook_url, data=payload, headers={"Content-Type": "application/json"})
    try:
        res = json.loads(urllib.request.urlopen(req, timeout=10).read())
        ok = res.get("code") == 0
        if not ok:
            print(f"  send error: {res.get('msg','')}")
        return ok
    except Exception as e:
        print(f"  send error: {e}")
        return False


def _md(content: str) -> dict:
    return {"tag": "markdown", "content": content}

def _hr() -> dict:
    return {"tag": "hr"}

def _tbl(cols: list, rows: list) -> dict:
    return {
        "tag": "table",
        "page_size": 20,
        "row_height": "low",
        "header_style": {
            "text_align": "left",
            "background_color": "grey",
            "bold": True,
        },
        "columns": [
            {"tag": "column", "name": n, "display_name": d,
             "data_type": "text", "width": "auto", "horizontal_align": "left"}
            for n, d in cols
        ],
        "rows": [
            {k: str(v) for k, v in row.items()}
            for row in rows
        ],
    }


def build_okr_card() -> dict:
    data = json.loads(OKR_FILE.read_text(encoding="utf-8"))
    now = datetime.now(BJT)
    week_num = now.isocalendar()[1]
    updated = data.get("updated", now.strftime("%Y-%m-%d"))
    wiki_url = data.get("wiki_url", "")

    rows = data.get("rows", [])

    # 按市场分组，统计进展状态
    total = len(rows)
    has_status = sum(1 for r in rows if r.get("status", "").strip())

    # 构建表格行
    tbl_rows = []
    for r in rows:
        # 把换行符替换成空格，避免表格太高
        tbl_rows.append({
            "market":   r.get("market", ""),
            "channel":  r.get("channel", ""),
            "platform": r.get("platform", "").replace("\n", " / "),
            "goal":     r.get("goal", "").replace("\n", " | "),
            "priority": r.get("priority", ""),
            "status":   r.get("status", "") or "—",
        })

    footer_parts = [f"数据更新：{updated}"]
    if wiki_url:
        footer_parts.append(f"[查看完整OKR表格]({wiki_url})")

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"📋 执行OKR 周报  W{week_num} · {now.strftime('%Y-%m-%d')}"},
            "template": "orange",
        },
        "elements": [
            _md(f"**{data.get('title', '执行OKR')}** — 本周进展汇总  共 **{total}** 个执行方向，**{has_status}** 个有进展更新"),
            _hr(),
            _tbl(
                [
                    ("market",   "市场"),
                    ("channel",  "渠道类型"),
                    ("platform", "平台"),
                    ("goal",     "目标"),
                    ("priority", "优先级"),
                    ("status",   "当前进展"),
                ],
                tbl_rows,
            ),
            _hr(),
            _md("  ".join(footer_parts)),
        ],
    }


def run():
    if not LEAD_WEBHOOK:
        print("❌ LARK_LEAD webhook 未配置")
        return
    if not OKR_FILE.exists():
        print(f"❌ OKR 数据文件不存在: {OKR_FILE}")
        return

    print(f"📋 OKR 周报推送 — {datetime.now(BJT).strftime('%Y-%m-%d %H:%M')}")
    card = build_okr_card()
    ok = send_card(LEAD_WEBHOOK, card)
    print(f"  {'✅ 已发送到 LARK_LEAD' if ok else '❌ 发送失败'}")


if __name__ == "__main__":
    run()
