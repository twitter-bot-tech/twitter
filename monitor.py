#!/usr/bin/env python3
"""
Kelly 每日监控报告 — 汇总所有 agent 动态
每天 08:00 BJT 发到 Telegram，也可手动运行查看
"""
import os
import json
import urllib.request
import urllib.parse
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent / ".env.outreach")

BJT      = ZoneInfo("Asia/Shanghai")
BASE     = Path(__file__).parent
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID  = "-5133903231"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def yesterday_str():
    return (datetime.now(BJT) - timedelta(days=1)).strftime("%Y-%m-%d")


def today_str():
    return datetime.now(BJT).strftime("%Y-%m-%d")


# ── 各模块状态读取 ────────────────────────────────────────────────────────────

def check_tweets() -> str:
    path = BASE / "01_social_media/logs/tweet_history.json"
    if not path.exists():
        return "Twitter ❌ 无记录"
    data = json.loads(path.read_text())
    today = [t for t in data if t.get("date") == today_str()]
    yesterday = [t for t in data if t.get("date") == yesterday_str()]
    total = len(data)
    if today:
        return f"Twitter ✅ 今日已发 | 总计 {total} 条"
    elif yesterday:
        return f"Twitter ✅ 昨日发出 | 总计 {total} 条"
    return f"Twitter ⚠️ 今日未发 | 总计 {total} 条"


def check_seo() -> str:
    pub = BASE / "02_seo/logs/published_articles.json"
    if not pub.exists():
        return "SEO ❌ 无发布记录"
    data = json.loads(pub.read_text())
    devto = [r for r in data if "dev.to" in r.get("platform","")]
    medium = [r for r in data if "Medium" in r.get("platform","")]
    return f"SEO ✅ 文章: dev.to {len(devto)}篇 | Medium {len(medium)}篇"


def check_rank() -> str:
    path = BASE / "02_seo/logs/rank_history.json"
    if not path.exists():
        return "排名 ❌ 无数据"
    data = json.loads(path.read_text())
    if not data:
        return "排名 ❌ 无数据"
    latest = data[-1]
    date = latest.get("date", "?")
    ranked = [r for r in latest.get("results", []) if r.get("rank")]
    return f"排名 ✅ 最后更新 {date} | 上榜关键词: {len(ranked)} 个"


def check_campaign() -> str:
    path = BASE / "01_social_media/logs/campaigns.json"
    if not path.exists():
        return "活动 ❌ 无记录"
    data = json.loads(path.read_text())
    if not data:
        return "活动 ❌ 无记录"
    latest = data[-1]
    week   = latest.get("week","?")
    status = latest.get("status","?")
    market = latest.get("market",{}).get("question","?")[:50]
    return f"活动 ✅ {week} | {status} | {market}..."


def check_competitor() -> str:
    path = BASE / "01_social_media/logs/competitor_data.json"
    if not path.exists():
        return "竞品 ❌ 无数据"
    data = json.loads(path.read_text())
    if not data:
        return "竞品 ❌ 无数据"
    latest = data[-1]
    date = latest.get("date","?")
    count = len(latest.get("data",[]))
    return f"竞品 ✅ 最后追踪 {date} | {count} 个账号"


def check_kol() -> str:
    excels = list((BASE / "03_kol_media").glob("MoonX_KOL名单_*.xlsx"))
    if not excels:
        return "KOL ❌ 无名单"
    latest = sorted(excels)[-1]
    try:
        import openpyxl
        wb = openpyxl.load_workbook(latest)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        total = len(rows) - 1
        return f"KOL ✅ {latest.name} | {total} 人"
    except:
        return f"KOL ✅ {latest.name}"


def check_logs_for_errors() -> list:
    """扫描所有 log 文件，找最近 24h 内的 ERROR"""
    errors = []
    log_dirs = [
        BASE / "01_social_media/logs",
        BASE / "02_seo/logs",
    ]
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    for log_dir in log_dirs:
        if not log_dir.exists():
            continue
        for log_file in log_dir.glob("*.log"):
            try:
                lines = log_file.read_text(errors="ignore").split("\n")
                for line in lines[-200:]:  # 只看最后200行
                    if "[ERROR]" in line:
                        errors.append(f"  ⚠️ {log_file.name}: {line[20:80]}")
            except:
                continue
    return errors[:5]  # 最多报5条


def scheduled_today() -> str:
    now = datetime.now(BJT)
    weekday = now.weekday()  # 0=周一
    items = ["• 每日推文 09:30", "• TG 早/午/晚 3条"]
    if weekday == 0:
        items += ["• SEO文章生成", "• 竞品追踪", "• KOL收集", "• 竞猜活动发起", "• 排名追踪"]
    if weekday == 3:
        items.append("• SEO文章生成（周四）")
    if weekday == 4:
        items.append("• 竞猜结果公布")
    return "\n".join(items)


# ── 报告组装 ──────────────────────────────────────────────────────────────────

def build_report() -> str:
    now = datetime.now(BJT)
    errors = check_logs_for_errors()
    error_section = "\n" + "\n".join(errors) if errors else " 无"

    report = f"""📊 <b>MoonX 每日监控 — {now.strftime('%m/%d %H:%M')} BJT</b>

<b>── Agent 状态 ──</b>
{check_tweets()}
{check_seo()}
{check_rank()}
{check_campaign()}
{check_competitor()}
{check_kol()}

<b>── 今日计划任务 ──</b>
{scheduled_today()}

<b>── 近24h 异常 ──</b>{error_section}

<b>── 待你处理 ──</b>
• Reddit credentials（填好后@我）
• 周五查看竞猜结果草稿 → 手动发布获奖者"""

    return report


def send_telegram(text: str):
    try:
        data = urllib.parse.urlencode({
            "chat_id":    CHAT_ID,
            "text":       text,
            "parse_mode": "HTML",
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", data=data
        )
        res = json.loads(urllib.request.urlopen(req, timeout=10).read())
        if res.get("ok"):
            logger.info("✅ 监控报告已发送到 TG")
        else:
            logger.error(f"TG 发送失败: {res}")
    except Exception as e:
        logger.error(f"TG 错误: {e}")


def send_lark(text: str):
    """发送到 Lark 个人群"""
    webhook = os.getenv("LARK_WEBHOOK")
    if not webhook:
        return
    # 去掉 HTML 标签，Lark 用纯文本
    import re
    clean = re.sub(r'<[^>]+>', '', text)
    try:
        payload = json.dumps({
            "msg_type": "text",
            "content":  {"text": clean}
        }).encode()
        req = urllib.request.Request(
            webhook,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        res = json.loads(urllib.request.urlopen(req, timeout=10).read())
        if res.get("code") == 0:
            logger.info("✅ 监控报告已发送到 Lark")
        else:
            logger.error(f"Lark 发送失败: {res}")
    except Exception as e:
        logger.error(f"Lark 错误: {e}")


def run():
    report = build_report()
    print(report)
    send_lark(report)
    send_telegram(report)


if __name__ == "__main__":
    run()
