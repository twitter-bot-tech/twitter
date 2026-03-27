#!/usr/bin/env python3
"""
MoonX 团队监控面板 — 浏览器可视化
运行: python3 dashboard.py → 自动打开 http://localhost:8888
"""
import os
import json
import threading
import webbrowser
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from http.server import HTTPServer, BaseHTTPRequestHandler
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

BJT  = ZoneInfo("Asia/Shanghai")
BASE = Path(__file__).parent


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except:
        return []


def get_agent_data():
    now = datetime.now(BJT)
    today = now.strftime("%Y-%m-%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")

    agents = []

    # ── Lead ─────────────────────────────────────────────────────────────────
    rank = read_json(BASE / "02_seo/logs/rank_history.json")
    comp = read_json(BASE / "01_social_media/logs/competitor_data.json")
    rank_date = rank[-1]["date"] if rank else "—"
    comp_date = comp[-1]["date"] if comp else "—"
    ranked_kw = len([r for r in (rank[-1]["results"] if rank else []) if r.get("rank")])
    agents.append({
        "name":   "Lead",
        "role":   "策略 & 统筹",
        "emoji":  "👔",
        "status": "active",
        "tasks": [
            f"SEO排名追踪 — 最后更新 {rank_date}，上榜 {ranked_kw} 个关键词",
            f"竞品分析 — 最后追踪 {comp_date}，{len(comp[-1]['data']) if comp else 0} 个账号",
        ],
        "next": "周一 09:00 自动跑竞品+排名报告",
    })

    # ── Nate ─────────────────────────────────────────────────────────────────
    pub = read_json(BASE / "02_seo/logs/published_articles.json")
    devto  = len([r for r in pub if "dev.to" in r.get("platform","")])
    medium = len([r for r in pub if "Medium" in r.get("platform","")])
    queue_file = BASE / "02_seo/logs/article_queue_index.json"
    queue_idx = 0
    if queue_file.exists():
        queue_idx = json.loads(queue_file.read_text()).get("index", 0)
    QUEUE = ["kalshi vs polymarket","crypto prediction market tracker",
             "smart money crypto signals","meme coin smart money",
             "how to use prediction markets","polymarket whale tracker",
             "prediction market arbitrage","kalshi prediction market review"]
    next_kw = QUEUE[queue_idx % len(QUEUE)]
    agents.append({
        "name":   "Nate",
        "role":   "策略 & 数据",
        "emoji":  "📊",
        "status": "active",
        "tasks": [
            f"已发布文章: dev.to {devto}篇 | Medium {medium}篇",
            f"关键词排名: {ranked_kw} 个上榜 (moonx bydfi #1)",
            f"下一篇选题: {next_kw}",
        ],
        "next": "周一/周四 10:00 自动生成文章",
    })

    # ── SEO专员 ──────────────────────────────────────────────────────────────
    reddit_log = BASE / "02_seo/logs/reddit_posts.json"
    hashnode_pub = [r for r in pub if "hashnode" in r.get("platform","").lower()]
    reddit_posts = read_json(reddit_log)
    agents.append({
        "name":   "SEO专员",
        "role":   "内容分发",
        "emoji":  "📝",
        "status": "partial",
        "tasks": [
            f"dev.to / Medium 已同步",
            f"Hashnode: {len(hashnode_pub)} 篇（待配置 credentials）",
            f"Reddit: {len(reddit_posts)} 篇（待配置 credentials）",
            "IndexNow: 待技术配合部署 key 文件",
        ],
        "next": "填好 Reddit/Hashnode credentials 后自动激活",
    })

    # ── Gary ─────────────────────────────────────────────────────────────────
    tweet_h = read_json(BASE / "01_social_media/logs/tweet_history.json")
    comment_h = read_json(BASE / "01_social_media/logs/comment_history.json") if (BASE / "01_social_media/logs/comment_history.json").exists() else []
    kol_files = sorted((BASE / "03_kol_media").glob("MoonX_KOL名单_*.xlsx"))
    kol_count = 0
    if kol_files:
        try:
            import openpyxl
            wb = openpyxl.load_workbook(kol_files[-1])
            kol_count = wb.active.max_row - 1
        except:
            pass
    today_tweets = [t for t in tweet_h if t.get("date") == today]
    agents.append({
        "name":   "Gary",
        "role":   "Twitter 增长 & KOL",
        "emoji":  "🎯",
        "status": "active" if today_tweets else "partial",
        "tasks": [
            f"今日推文: {'✅ 已发' if today_tweets else '⏳ 待发 09:30'}",
            f"历史推文: {len(tweet_h)} 条",
            f"自动评论: 待账号预热后激活（403 限制中）",
            f"KOL名单: {kol_count} 人",
        ],
        "next": "每天 09:30 自动发推 | 账号预热后评论bot激活",
    })

    # ── Sean ─────────────────────────────────────────────────────────────────
    campaigns = read_json(BASE / "01_social_media/logs/campaigns.json")
    current_campaign = campaigns[-1] if campaigns else None
    camp_status = "无" if not current_campaign else current_campaign.get("status","?")
    camp_market = current_campaign["market"]["question"][:45] + "..." if current_campaign else "—"
    agents.append({
        "name":   "Sean",
        "role":   "活动 & 用户运营",
        "emoji":  "🎪",
        "status": "active",
        "tasks": [
            f"本周竞猜: {camp_status} — {camp_market}",
            f"奖励: $10 USDT | Twitter + TG 同步",
            f"TG 日常内容: 08/14/20 BJT 自动推送",
        ],
        "next": "周五 20:00 自动生成获奖者草稿",
    })

    return agents


def build_html():
    agents = get_agent_data()
    now = datetime.now(BJT).strftime("%Y-%m-%d %H:%M BJT")

    status_color = {"active": "#22c55e", "partial": "#f59e0b", "inactive": "#ef4444"}
    status_label = {"active": "运行中", "partial": "部分就绪", "inactive": "未启动"}

    cards = ""
    for a in agents:
        color  = status_color.get(a["status"], "#6b7280")
        label  = status_label.get(a["status"], "—")
        tasks  = "".join(f"<li>{t}</li>" for t in a["tasks"])
        cards += f"""
        <div class="card">
            <div class="card-header">
                <span class="emoji">{a['emoji']}</span>
                <div>
                    <div class="name">{a['name']}</div>
                    <div class="role">{a['role']}</div>
                </div>
                <div class="badge" style="background:{color}">{label}</div>
            </div>
            <ul class="tasks">{tasks}</ul>
            <div class="next">⏭ {a['next']}</div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="30">
<title>MoonX 团队监控面板</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; padding: 24px; }}
  h1 {{ font-size: 22px; font-weight: 700; margin-bottom: 4px; color: #f8fafc; }}
  .subtitle {{ color: #64748b; font-size: 13px; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }}
  .card-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }}
  .emoji {{ font-size: 28px; }}
  .name {{ font-size: 16px; font-weight: 600; }}
  .role {{ font-size: 12px; color: #64748b; margin-top: 2px; }}
  .badge {{ margin-left: auto; font-size: 11px; font-weight: 600; color: white; padding: 3px 10px; border-radius: 20px; white-space: nowrap; }}
  .tasks {{ list-style: none; padding: 0; }}
  .tasks li {{ font-size: 13px; color: #94a3b8; padding: 5px 0; border-bottom: 1px solid #1e293b; line-height: 1.5; }}
  .tasks li:last-child {{ border: none; }}
  .next {{ margin-top: 14px; font-size: 12px; color: #475569; background: #0f172a; padding: 8px 12px; border-radius: 8px; }}
  .refresh {{ color: #475569; font-size: 11px; margin-top: 20px; text-align: center; }}
</style>
</head>
<body>
<h1>🌙 MoonX 团队监控面板</h1>
<div class="subtitle">最后更新: {now} · 每30秒自动刷新</div>
<div class="grid">{cards}</div>
<div class="refresh">页面每30秒自动刷新 · <a href="/" style="color:#3b82f6">手动刷新</a></div>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        html = build_html().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(html))
        self.end_headers()
        self.wfile.write(html)

    def log_message(self, *args):
        pass  # 静默日志


def run():
    port = 8888
    server = HTTPServer(("localhost", port), Handler)
    url = f"http://localhost:{port}"
    print(f"✅ 监控面板已启动: {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n面板已关闭")


if __name__ == "__main__":
    run()
