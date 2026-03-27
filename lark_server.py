#!/usr/bin/env python3
"""
MoonX Lark 自主讨论服务器 v2 — 决策闭环版
支持指令：
  @moonx 开会 [主题]        → 多轮会议
  @moonx 开会N轮 [主题]     → 指定轮数
  @moonx 我选A / 我选B      → 执行会议结论中的决策
  @moonx 执行               → 全员开始执行当前任务
  @moonx 状态               → 各部门汇报进度
  @moonx 日报               → 发今日各部门日报
  @moonx @员工X [问题]      → 指定员工回答
  @moonx @lead [问题]       → 直接问 Lead
"""
import os, json, re, threading, urllib.request, time, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import claude_cli as anthropic

# ── 实时数据缓存（内存，TTL 10 分钟）────────────────────────────────────────
_cache: dict = {}
_CACHE_TTL = 600  # seconds

def _cache_get(key: str):
    entry = _cache.get(key)
    if entry and time.time() - entry["ts"] < _CACHE_TTL:
        return entry["val"]
    return None

def _cache_set(key: str, val):
    _cache[key] = {"val": val, "ts": time.time()}

# KOL 数据由本地脚本 POST 推送
_kol_stats: dict = {}  # {"date": "2026-03-19", "sent_today": 5, "sent_total": 130, "kol_total": 320}

load_dotenv(Path(__file__).parent / ".env", override=True)
load_dotenv(Path(__file__).parent / ".env.outreach", override=True)

app    = Flask(__name__)
BJT    = ZoneInfo("Asia/Shanghai")
CLAUDE = anthropic.Anthropic()
BASE   = Path(__file__).parent

WEBHOOKS = {
    "lead":     os.getenv("LARK_LEAD"),
    "social":   os.getenv("LARK_SOCIAL"),
    "seo":      os.getenv("LARK_SEO"),
    "kol":      os.getenv("LARK_KOL"),
    "growth":   os.getenv("LARK_GROWTH"),
    "strategy": os.getenv("LARK_STRATEGY"),
}

processed_events = set()

# ── KOL 报价邮件配置（来自 .env.outreach）────────────────────────────────────
_SMTP_HOST    = os.getenv("BYDFI_SMTP_HOST", "smtp.zoho.com")
_SMTP_PORT    = int(os.getenv("BYDFI_SMTP_PORT", "465"))
_SMTP_USER    = os.getenv("BYDFI_EMAIL", "kelly@bydfi.com")
_SMTP_PASS    = os.getenv("BYDFI_EMAIL_PASSWORD")
_SENDER_NAME  = os.getenv("SENDER_NAME", "Kelly")
_SENDER_TITLE = os.getenv("SENDER_TITLE", "Head of Marketing")
_SENDER_TG    = os.getenv("SENDER_TG", "@BDkelly")
_ON_FLY       = bool(os.getenv("FLY_APP_NAME"))
_CRM_BASE_URL = "https://moonx-lark-server.fly.dev" if _ON_FLY else "http://localhost:8090"

# ── 上下文记忆（存在内存，重启会清空；如需持久化可存文件）──────────────────────
class MeetingContext:
    def __init__(self):
        self.topic      = ""
        self.conclusion = ""
        self.option_a   = ""
        self.option_b   = ""
        self.decision   = ""       # "A" / "B" / ""
        self.tasks      = {}       # {dept_key: task_description}
        self.status     = {}       # {dept_key: "pending"/"doing"/"done"}
        self.timestamp  = ""

    def has_pending_decision(self):
        return bool(self.option_a or self.option_b) and not self.decision

    def save(self):
        path = BASE / "context.json"
        data = self.__dict__.copy()
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def load(self):
        path = BASE / "context.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for k, v in data.items():
                    setattr(self, k, v)
            except Exception:
                pass

CTX = MeetingContext()
CTX.load()

# ── 员工配置 ────────────────────────────────────────────────────────────────────
EMPLOYEES = [
    {
        "key": "social", "emoji": "📱", "name": "员工1号｜社媒运营",
        "aliases": ["员工1", "社媒", "social"],
        "system": "你是MoonX社媒运营，见过Solana/Coinbase从0到百万粉的操盘手。从社媒增长角度简洁回答，不超过100字。",
        "task_system": "你是MoonX社媒运营。Kelly已做出决策，你需要：1.明确承接你的任务 2.说明具体执行方式 3.给出时间节点。语气果断，不超过80字。",
        "default_task": "今日推文节奏优化 + 配合决策出1条相关内容",
    },
    {
        "key": "seo", "emoji": "🔍", "name": "员工2号｜SEO专家",
        "aliases": ["员工2", "seo", "SEO"],
        "system": "你是MoonX SEO专家，帮CoinGecko做到月均3000万流量的操盘手。从SEO/内容角度简洁回答，不超过100字。",
        "task_system": "你是MoonX SEO专家。Kelly已做出决策，你需要：1.明确承接你的任务 2.说明具体执行方式 3.给出时间节点。语气果断，不超过80字。",
        "default_task": "输出1篇配合决策方向的SEO文章选题",
    },
    {
        "key": "kol", "emoji": "🤝", "name": "员工3号｜KOL媒体",
        "aliases": ["员工3", "kol", "KOL", "媒体"],
        "system": "你是MoonX KOL媒体负责人，帮Coinbase上市前把故事塞进WSJ的操盘手。从KOL/媒体角度简洁回答，不超过100字。",
        "task_system": "你是MoonX KOL媒体负责人。Kelly已做出决策，你需要：1.明确承接你的任务 2.说明具体执行方式 3.给出时间节点。语气果断，不超过80字。",
        "default_task": "今日10条Twitter DM + 跟进已有回复",
    },
    {
        "key": "growth", "emoji": "📈", "name": "员工4号｜增长运营",
        "aliases": ["员工4", "增长", "growth"],
        "system": "你是MoonX增长运营，设计过Uniswap空投/Blur积分等百万人激励机制。从增长/裂变角度简洁回答，不超过100字。",
        "task_system": "你是MoonX增长运营。Kelly已做出决策，你需要：1.明确承接你的任务 2.说明具体执行方式 3.给出时间节点。语气果断，不超过80字。",
        "default_task": "设计配合决策的增长激励方案",
    },
    {
        "key": "strategy", "emoji": "📊", "name": "员工5号｜策略数据",
        "aliases": ["员工5", "策略", "数据", "strategy"],
        "system": "你是MoonX策略分析师，a16z crypto研究团队风格。从数据/竞品角度简洁回答，不超过100字。",
        "task_system": "你是MoonX策略数据分析师。Kelly已做出决策，你需要：1.明确承接你的任务 2.说明具体衡量指标 3.给出追踪节点。语气果断，不超过80字。",
        "default_task": "设定决策的衡量指标 + 下周五更新OKR",
    },
]

LEAD_SYSTEM  = "你是MoonX Team Lead，融合了He Yi和Coinbase IPO营销负责人的思维。给出简洁有力的回答或决策，不超过120字。"
LEAD_ASSIGN  = """你是MoonX Team Lead。Kelly已选择方案{option}：{decision_content}。
现在向全员分配任务，格式：
━━ 任务分配 ━━
• 员工1 社媒：[具体任务]
• 员工2 SEO：[具体任务]
• 员工3 KOL：[具体任务]
• 员工4 增长：[具体任务]
• 员工5 策略：[具体任务]
━━ 截止：本周五 ━━
语气果断，不废话。"""


# ── 工具函数 ────────────────────────────────────────────────────────────────────
def send_lark(bot_key: str, text: str):
    url = WEBHOOKS.get(bot_key)
    if not url:
        return
    payload = json.dumps({"msg_type": "text", "content": {"text": text}}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"Lark error ({bot_key}): {e}")

def notify_lark_kol(kol_name: str, action_type: str, summary: str = ""):
    """向 Lark KOL 频道发送带深链的待处理通知"""
    labels = {
        "quote_needed":    "💰 报价待确认",
        "contract_review": "📄 合同待审核",
        "content_review":  "🎬 内容草稿待审核",
    }
    label = labels.get(action_type, action_type)
    url   = f"{_CRM_BASE_URL}/crm/pending"
    text  = (
        f"KOL 有新的待处理事项\n\n"
        f"KOL：{kol_name}\n"
        f"类型：{label}\n"
        f"{('说明：' + summary + chr(10)) if summary else ''}"
        f"点此处理：{url}"
    )
    send_lark("kol", text)


def send_quote_email(to_email: str, kol_name: str, price_usd: float,
                     deliverables: str, payment_terms: str = "",
                     notes: str = "",
                     original_subject: str = "") -> tuple[bool, str]:
    """
    向 KOL 发送报价邮件。
    返回 (success: bool, error_msg: str)
    """
    if not _SMTP_PASS:
        return False, "SMTP 密码未配置 (BYDFI_EMAIL_PASSWORD)"
    if not to_email:
        return False, "KOL 邮箱为空"

    _payment_terms = payment_terms or "50% upfront, 50% after publishing"
    subject = f"Re: {original_subject}" if original_subject else f"MoonX x {kol_name} — Partnership Terms"
    body = (
        f"Hi {kol_name},\n\n"
        f"Thanks for your interest in partnering with MoonX!\n\n"
        f"Here are the partnership terms we'd like to propose:\n\n"
        f"Deliverables: {deliverables or '(to be confirmed)'}\n"
        f"Partnership fee: ${price_usd:,.0f} USD\n"
        f"Payment terms: {_payment_terms}\n"
        f"{('Notes: ' + notes + chr(10)) if notes else ''}"
        f"\nPlease review and let me know if you'd like to move forward.\n"
        f"Feel free to reach me on Telegram: {_SENDER_TG}\n\n"
        f"Best,\n"
        f"{_SENDER_NAME}\n"
        f"{_SENDER_TITLE}, BYDFi MoonX"
    )
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"{_SENDER_NAME} <{_SMTP_USER}>"
        msg["To"]      = to_email
        msg.attach(MIMEText(body, "plain", "utf-8"))
        if _SMTP_PORT == 465:
            with smtplib.SMTP_SSL(_SMTP_HOST, _SMTP_PORT, timeout=30) as srv:
                srv.login(_SMTP_USER, _SMTP_PASS)
                srv.sendmail(_SMTP_USER, to_email, msg.as_string())
        else:
            with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT, timeout=30) as srv:
                srv.starttls()
                srv.login(_SMTP_USER, _SMTP_PASS)
                srv.sendmail(_SMTP_USER, to_email, msg.as_string())
        return True, ""
    except Exception as e:
        return False, str(e)


def ask_claude(system: str, prompt: str, max_tokens: int = 300) -> str:
    try:
        resp = CLAUDE.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        return f"（AI响应失败: {e}）"

def now_str():
    return datetime.now(BJT).strftime("%m/%d %H:%M")


# ── 指令处理函数 ─────────────────────────────────────────────────────────────────

def handle_meeting(topic: str, rounds: int = 2):
    from lark_meeting import run_meeting
    run_meeting(topic, rounds=rounds)
    # 会议结束后，更新上下文（从 lark_meeting 里获取结论）
    CTX.topic     = topic
    CTX.timestamp = now_str()
    CTX.decision  = ""
    CTX.tasks     = {}
    CTX.status    = {emp["key"]: "pending" for emp in EMPLOYEES}
    CTX.save()


def handle_decision(choice: str, extra: str = ""):
    """Kelly 说 '我选A' 或 '我选B'"""
    choice = choice.upper()
    if choice not in ("A", "B"):
        send_lark("lead", "👔 Team Lead\n\n请明确选择：我选A 或 我选B")
        return

    CTX.decision = choice
    CTX.save()

    # Layer 4 闭环：通知 scheduler 标记今日决策已完成
    try:
        import scheduler as _sched_mod
        _sched_mod.mark_decision(choice)
        return  # scheduler 已接管后续任务分配
    except Exception:
        pass  # 降级到原有流程

    decision_content = CTX.option_a if choice == "A" else CTX.option_b
    if not decision_content:
        decision_content = extra or f"方案{choice}"

    now = now_str()
    send_lark("lead", f"👔 Team Lead\n\n✅ 收到，Kelly 选择方案 {choice}\n\n正在分配任务...")

    import time
    time.sleep(1)

    # Lead 分配任务
    assign_prompt = f"Kelly选择了方案{choice}：{decision_content}。议题：{CTX.topic}"
    assign_msg = ask_claude(LEAD_ASSIGN.format(option=choice, decision_content=decision_content), assign_prompt, 400)
    send_lark("lead", f"👔 Team Lead｜任务分配 {now}\n\n{assign_msg}")
    time.sleep(1.5)

    # 各员工承接任务
    for emp in EMPLOYEES:
        prompt = f"Kelly选择了方案{choice}（{decision_content}），议题：{CTX.topic}。你的默认任务方向：{emp['default_task']}。请承接并说明执行计划。"
        response = ask_claude(emp["task_system"], prompt)
        send_lark(emp["key"], f"{emp['emoji']} {emp['name']}\n\n{response}")
        CTX.tasks[emp["key"]]  = response
        CTX.status[emp["key"]] = "doing"
        time.sleep(1.5)

    CTX.save()
    send_lark("lead", f"👔 Team Lead\n\n全员已接收任务 ✅\n发送「状态」随时查看进度\n发送「@moonx 完成 [部门]」标记某部门完成")


def handle_status():
    """查看各部门当前任务状态"""
    if not CTX.topic:
        send_lark("lead", "👔 Team Lead\n\n当前没有进行中的任务。\n发送「开会 [主题]」启动新议题。")
        return

    status_icons = {"pending": "⏳", "doing": "🔄", "done": "✅"}
    lines = [f"👔 Team Lead｜任务状态 {now_str()}", "", f"议题：{CTX.topic}"]
    if CTX.decision:
        lines.append(f"决策：方案 {CTX.decision}")
    lines.append("")

    for emp in EMPLOYEES:
        key    = emp["key"]
        icon   = status_icons.get(CTX.status.get(key, "pending"), "⏳")
        task   = CTX.tasks.get(key, "待分配")[:40] + ("..." if len(CTX.tasks.get(key, "")) > 40 else "")
        lines.append(f"{icon} {emp['emoji']} {emp['name'].split('｜')[1]}")
        if task and task != "待分配":
            lines.append(f"   └ {task}")

    send_lark("lead", "\n".join(lines))


def handle_complete(dept_name: str):
    """标记某部门任务完成"""
    matched = None
    for emp in EMPLOYEES:
        if any(a in dept_name for a in emp["aliases"]) or dept_name in emp["name"]:
            matched = emp
            break
    if not matched:
        send_lark("lead", f"👔 Team Lead\n\n没找到部门「{dept_name}」，请用：员工1~5、社媒、SEO、KOL、增长、策略")
        return

    CTX.status[matched["key"]] = "done"
    CTX.save()

    done_count = sum(1 for s in CTX.status.values() if s == "done")
    total = len(EMPLOYEES)
    send_lark(matched["key"], f"{matched['emoji']} {matched['name']}\n\n✅ 任务完成！\n{done_count}/{total} 个部门已完成")

    if done_count == total:
        import time
        time.sleep(1)
        send_lark("lead", f"👔 Team Lead\n\n🎉 全员任务完成！\n\n议题「{CTX.topic}」执行收尾。\n下次会议发「开会 [新主题]」继续。")


def handle_employee_question(emp: dict, question: str):
    response = ask_claude(emp["system"], question)
    send_lark(emp["key"], f"{emp['emoji']} {emp['name']}\n\n{response}")


def handle_lead_question(question: str):
    response = ask_claude(LEAD_SYSTEM, question)
    send_lark("lead", f"👔 Team Lead\n\n{response}")


def handle_daily_report():
    from lark_reporter import run
    run()


# ── KOL CRM Bot 指令 ─────────────────────────────────────────────────────────────
# 指令格式（发到 Lark，任意群均可）：
#   kol 查 张三               → 显示 KOL 详情
#   kol 列表 [状态]           → 列出 KOL（可加筛选：谈判中、TG接触 等）
#   kol tg 张三 @handle       → 记录 TG handle，状态 → TG接触
#   kol 谈判 张三 $500 视频    → 记录报价，状态 → 谈判中
#   kol 签约 张三 $800        → 记录合同，状态 → 已签约
#   kol 审核 张三 https://... → 内容提交审核，状态 → 审核中
#   kol 发布 张三 https://... → 记录发布，状态 → 已发布
#   kol 完成 张三             → 状态 → 已完成
#   kol 拒绝 张三             → 状态 → 已拒绝
#   kol 冷却 张三             → 状态 → 冷却
#   kol 付款 张三 $800        → 记录付款待付

def _kol_db_path():
    if os.getenv("FLY_APP_NAME"):
        return Path("/data") / "kol_crm.db"
    return BASE / "03_kol_media" / "kol_crm.db"

def _load_kol_db():
    import sys as _sys
    _sys.path.insert(0, str(BASE / "03_kol_media"))
    import kol_db
    return kol_db

def _resolve_kol(name: str, kol_db) -> dict | None:
    """模糊搜索 KOL，如果只有一条结果则直接返回；多条则通知 Kelly 指定"""
    results = kol_db.find_kol_by_name(name, path=_kol_db_path())
    if not results:
        send_lark("kol", f"KOL Bot\n\n找不到 KOL：「{name}」\n请检查名字拼写")
        return None
    if len(results) > 1:
        lines = "\n".join(
            f"  [{r['id']}] {r['name']}（{r['status']}）" for r in results[:5]
        )
        send_lark("kol", f"KOL Bot\n\n找到多个匹配「{name}」，请用更精确的名字：\n{lines}")
        return None
    return results[0]


def handle_kol_cmd(cmd: str):
    """处理 `kol <cmd>` 指令"""
    try:
        _handle_kol_cmd_inner(cmd)
    except Exception as e:
        import traceback
        send_lark("kol", f"KOL Bot\n\n执行出错: {e}\n{traceback.format_exc()[-300:]}")


def _handle_kol_cmd_inner(cmd: str):
    try:
        kol_db = _load_kol_db()
    except Exception as e:
        send_lark("kol", f"KOL Bot\n\n加载数据库失败: {e}")
        return

    parts = cmd.strip().split()
    if not parts:
        _kol_help()
        return

    action = parts[0]

    # ── kol 查 [name] ──────────────────────────────────────────────
    if action in ("查", "查看", "detail"):
        name = " ".join(parts[1:])
        if not name:
            send_lark("kol", "KOL Bot\n\n用法：kol 查 [名字]")
            return
        kol = _resolve_kol(name, kol_db)
        if not kol:
            return
        detail = kol_db.get_kol_detail(kol["id"], path=_kol_db_path())
        k = detail["kol"]
        neg = detail.get("negotiation") or {}
        contract = detail.get("contract") or {}
        pays = detail.get("payments") or []
        lines = [
            f"KOL 详情：{k['name']}",
            f"状态：{k['status']}",
            f"平台：{k['platform']} | 粉丝：{k.get('subscribers', 0):,}",
            f"邮箱：{k.get('email') or '—'}",
            f"TG：{k.get('tg_handle') or '—'}",
        ]
        if neg.get("price_usd"):
            lines.append(f"报价：${neg['price_usd']} | {neg.get('content_type','')}")
        if contract.get("total_value_usd"):
            lines.append(f"合同：${contract['total_value_usd']} ({contract.get('status','')})")
        if pays:
            pending = [p for p in pays if p.get("status") == "pending"]
            lines.append(f"待付款：{len(pending)} 笔")
        if k.get("notes"):
            lines.append(f"备注：{k['notes']}")
        send_lark("kol", "KOL Bot\n\n" + "\n".join(lines))

    # ── kol 列表 [状态] ────────────────────────────────────────────
    elif action in ("列表", "list", "ls"):
        status_filter = " ".join(parts[1:]) or None
        with kol_db.get_db(_kol_db_path()) as conn:
            if status_filter:
                rows = conn.execute(
                    "SELECT id, name, status, tier, tg_handle FROM kols WHERE status=? ORDER BY updated_at DESC LIMIT 20",
                    (status_filter,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT status, COUNT(*) as cnt FROM kols GROUP BY status ORDER BY cnt DESC"
                ).fetchall()
                lines = ["KOL 状态汇总："] + [f"  {r['status'] or '未知'}：{r['cnt']} 人" for r in rows]
                send_lark("kol", "KOL Bot\n\n" + "\n".join(lines))
                return
        if not rows:
            send_lark("kol", f"KOL Bot\n\n「{status_filter}」暂无记录")
            return
        lines = [f"KOL 列表（{status_filter}，共 {len(rows)} 条）："]
        for r in rows:
            tg = f" | TG:{r['tg_handle']}" if r.get("tg_handle") else ""
            lines.append(f"  [{r['id']}] {r['name']} ({r['tier']or''}){tg}")
        send_lark("kol", "KOL Bot\n\n" + "\n".join(lines))

    # ── kol tg [name] @handle ──────────────────────────────────────
    elif action in ("tg", "TG"):
        if len(parts) < 3:
            send_lark("kol", "KOL Bot\n\n用法：kol tg [名字] @handle")
            return
        tg_handle = parts[-1]
        name = " ".join(parts[1:-1])
        kol = _resolve_kol(name, kol_db)
        if not kol:
            return
        kol_db.update_kol_tg(kol["id"], tg_handle, path=_kol_db_path())
        notify_lark_kol(kol["name"], "quote_needed", f"TG: {tg_handle}")
        send_lark("kol", f"KOL Bot\n\n已记录\n{kol['name']} → TG接触\nTG: {tg_handle}")

    # ── kol 谈判 [name] $[amount] [type] ──────────────────────────
    elif action in ("谈判", "报价"):
        if len(parts) < 3:
            send_lark("kol", "KOL Bot\n\n用法：kol 谈判 [名字] $[金额] [内容类型]\n例：kol 谈判 张三 $500 视频评测")
            return
        # 解析金额
        price = None
        content_type = ""
        name_parts = []
        for p in parts[1:]:
            if p.startswith("$") or p.startswith("＄"):
                try:
                    price = float(p.lstrip("$＄").replace(",", ""))
                except ValueError:
                    pass
            elif price is not None:
                content_type += (" " + p)
            else:
                name_parts.append(p)
        name = " ".join(name_parts)
        kol = _resolve_kol(name, kol_db)
        if not kol:
            return
        kol_db.upsert_negotiation(kol["id"], {
            "stage": "initial", "price_usd": price,
            "content_type": content_type.strip(), "notes": ""
        }, path=_kol_db_path())
        kol_db.change_kol_status(kol["id"], "谈判中", operator="kelly", path=_kol_db_path())
        price_str = f"${price}" if price else "待定"
        send_lark("kol", f"KOL Bot\n\n已记录谈判\n{kol['name']} → 谈判中\n报价：{price_str} {content_type.strip()}")

    # ── kol 签约 [name] $[amount] ─────────────────────────────────
    elif action in ("签约", "合同"):
        if len(parts) < 2:
            send_lark("kol", "KOL Bot\n\n用法：kol 签约 [名字] $[金额]\n例：kol 签约 张三 $800")
            return
        price = None
        name_parts = []
        for p in parts[1:]:
            if p.startswith("$") or p.startswith("＄"):
                try:
                    price = float(p.lstrip("$＄").replace(",", ""))
                except ValueError:
                    pass
            else:
                name_parts.append(p)
        name = " ".join(name_parts)
        kol = _resolve_kol(name, kol_db)
        if not kol:
            return
        kol_db.add_contract(kol["id"], total_value_usd=price, path=_kol_db_path())
        price_str = f"${price}" if price else "待定"
        send_lark("kol", f"KOL Bot\n\n已签约\n{kol['name']} → 已签约\n合同金额：{price_str}")

    # ── kol 审核 [name] [url] ─────────────────────────────────────
    elif action in ("审核", "review"):
        if len(parts) < 3:
            send_lark("kol", "KOL Bot\n\n用法：kol 审核 [名字] [草稿链接]")
            return
        url = parts[-1]
        name = " ".join(parts[1:-1])
        kol = _resolve_kol(name, kol_db)
        if not kol:
            return
        now = datetime.now(BJT).strftime("%Y-%m-%d %H:%M:%S")
        with kol_db.get_db(_kol_db_path()) as conn:
            conn.execute(
                "INSERT INTO content (kol_id, draft_url, draft_submitted_at) VALUES (?,?,?)",
                (kol["id"], url, now[:10])
            )
            conn.execute(
                "UPDATE kols SET status='审核中', updated_at=? WHERE id=?", (now, kol["id"])
            )
            conn.execute(
                "INSERT INTO activities (kol_id, type, content, operator, created_at) VALUES (?,?,?,?,?)",
                (kol["id"], "draft_submitted", f"草稿: {url}", "kelly", now)
            )
        send_lark("kol", f"KOL Bot\n\n已进入审核\n{kol['name']} → 审核中\n草稿：{url}")

    # ── kol 发布 [name] [url] ─────────────────────────────────────
    elif action in ("发布", "published"):
        if len(parts) < 3:
            send_lark("kol", "KOL Bot\n\n用法：kol 发布 [名字] [发布链接]")
            return
        url = parts[-1]
        name = " ".join(parts[1:-1])
        kol = _resolve_kol(name, kol_db)
        if not kol:
            return
        kol_db.record_content_published(kol["id"], url, path=_kol_db_path())
        send_lark("kol", f"KOL Bot\n\n已发布\n{kol['name']} → 已发布\nURL：{url}")

    # ── kol 完成 [name] ───────────────────────────────────────────
    elif action in ("完成", "done"):
        name = " ".join(parts[1:])
        kol = _resolve_kol(name, kol_db)
        if not kol:
            return
        kol_db.change_kol_status(kol["id"], "已完成", operator="kelly", path=_kol_db_path())
        send_lark("kol", f"KOL Bot\n\n{kol['name']} → 已完成")

    # ── kol 拒绝 [name] ───────────────────────────────────────────
    elif action in ("拒绝", "reject"):
        name = " ".join(parts[1:])
        kol = _resolve_kol(name, kol_db)
        if not kol:
            return
        kol_db.change_kol_status(kol["id"], "已拒绝", operator="kelly", path=_kol_db_path())
        send_lark("kol", f"KOL Bot\n\n{kol['name']} → 已拒绝，已记录")

    # ── kol 冷却 [name] ───────────────────────────────────────────
    elif action in ("冷却", "cold"):
        name = " ".join(parts[1:])
        kol = _resolve_kol(name, kol_db)
        if not kol:
            return
        kol_db.change_kol_status(kol["id"], "冷却", operator="kelly", path=_kol_db_path())
        kol_db.schedule_followup(kol["id"], days=30, path=_kol_db_path())
        send_lark("kol", f"KOL Bot\n\n{kol['name']} → 冷却（30天后自动提醒跟进）")

    # ── kol 付款 [name] $[amount] ─────────────────────────────────
    elif action in ("付款", "pay", "payment"):
        if len(parts) < 2:
            send_lark("kol", "KOL Bot\n\n用法：kol 付款 [名字] $[金额]\n例：kol 付款 张三 $800")
            return
        price = None
        name_parts = []
        for p in parts[1:]:
            if p.startswith("$") or p.startswith("＄"):
                try:
                    price = float(p.lstrip("$＄").replace(",", ""))
                except ValueError:
                    pass
            else:
                name_parts.append(p)
        name = " ".join(name_parts)
        kol = _resolve_kol(name, kol_db)
        if not kol:
            return
        due = (datetime.now(BJT) + __import__("datetime").timedelta(days=7)).strftime("%Y-%m-%d")
        kol_db.add_payment(kol["id"], amount_usd=price or 0, due_date=due, path=_kol_db_path())
        price_str = f"${price}" if price else "待定"
        send_lark("kol", f"KOL Bot\n\n已记录付款\n{kol['name']} | {price_str}\n到期：{due}")

    # ── kol roi [name] 点击:N 注册:N 交易:N ────────────────────────
    elif action in ("roi",):
        # 格式：kol roi 张三 点击:100 注册:20 交易:5
        if len(parts) < 2:
            send_lark("kol", "KOL Bot\n\n用法：kol roi [名字] 点击:100 注册:20 交易:5")
            return
        name_parts, kv = [], {}
        for p in parts[1:]:
            if ":" in p:
                k, v = p.split(":", 1)
                try:
                    kv[k.strip()] = int(v.strip())
                except ValueError:
                    pass
            else:
                name_parts.append(p)
        name = " ".join(name_parts)
        kol = _resolve_kol(name, kol_db)
        if not kol:
            return
        clicks    = kv.get("点击", kv.get("clicks", 0))
        signups   = kv.get("注册", kv.get("signups", 0))
        trades    = kv.get("交易", kv.get("trades", 0))
        revenue   = kv.get("收益", kv.get("revenue", 0))
        db_path = _kol_db_path()
        with kol_db.get_db(db_path) as conn:
            # 更新或插入 performance 记录
            existing = conn.execute(
                "SELECT id FROM performance WHERE kol_id=?", (kol["id"],)
            ).fetchone()
            now_str = datetime.now(BJT).strftime("%Y-%m-%d %H:%M:%S")
            if existing:
                conn.execute(
                    "UPDATE performance SET clicks=?, signups=?, trades=?, revenue_usd=?, updated_at=? WHERE kol_id=?",
                    (clicks, signups, trades, revenue or None, now_str, kol["id"])
                )
            else:
                conn.execute(
                    "INSERT INTO performance (kol_id, clicks, signups, trades, revenue_usd, updated_at) VALUES (?,?,?,?,?,?)",
                    (kol["id"], clicks, signups, trades, revenue or None, now_str)
                )
            conn.execute(
                "INSERT INTO activities (kol_id, type, content, operator, created_at) VALUES (?,?,?,?,?)",
                (kol["id"], "roi_updated", f"点击:{clicks} 注册:{signups} 交易:{trades}", "kelly", now_str)
            )
        send_lark("kol", f"KOL Bot\n\n已更新 ROI 数据\n{kol['name']}\n点击:{clicks} | 注册:{signups} | 交易:{trades}")

    # ── kol 付款确认 [name] ──────────────────────────────────────────
    elif action in ("付款确认", "付款完成", "已付款"):
        if len(parts) < 2:
            send_lark("kol", "KOL Bot\n\n用法：kol 付款确认 [名字]")
            return
        name = " ".join(parts[1:])
        kol = _resolve_kol(name, kol_db)
        if not kol:
            return
        db_path = _kol_db_path()
        with kol_db.get_db(db_path) as conn:
            # 将最近一笔 pending 付款标记为 paid
            pay_row = conn.execute(
                "SELECT id, amount_usd FROM payments WHERE kol_id=? AND status='pending' ORDER BY due_date ASC LIMIT 1",
                (kol["id"],)
            ).fetchone()
            if not pay_row:
                send_lark("kol", f"KOL Bot\n\n{kol['name']} 没有待付款记录")
                return
            now_str = datetime.now(BJT).strftime("%Y-%m-%d %H:%M:%S")
            conn.execute(
                "UPDATE payments SET status='paid', paid_at=? WHERE id=?",
                (now_str, pay_row["id"])
            )
            conn.execute(
                "INSERT INTO activities (kol_id, type, content, operator, created_at) VALUES (?,?,?,?,?)",
                (kol["id"], "payment_confirmed", f"付款已确认 ${pay_row['amount_usd']}", "kelly", now_str)
            )
        send_lark("kol", f"KOL Bot\n\n付款已确认\n{kol['name']} | ${pay_row['amount_usd']}")

    else:
        _kol_help()


def _kol_help():
    help_text = """KOL Bot 指令列表：

kol 查 [名字]            → 查看 KOL 详情
kol 列表 [状态]          → 列出 KOL（不加状态=看汇总）
kol tg [名字] @handle    → 记录 TG，状态→TG接触
kol 谈判 [名字] $金额 [类型] → 报价，状态→谈判中
kol 签约 [名字] $金额    → 签合同，状态→已签约
kol 审核 [名字] [链接]   → 草稿审核，状态→审核中
kol 发布 [名字] [链接]   → 记录发布，状态→已发布
kol 完成 [名字]          → 状态→已完成
kol 拒绝 [名字]          → 状态→已拒绝
kol 冷却 [名字]          → 状态→冷却（30天后提醒）
kol 付款 [名字] $金额    → 记录待付款
kol roi [名字] 点击:N 注册:N 交易:N → 填入 ROI 数据
kol 付款确认 [名字]      → 标记最近一笔付款已完成"""
    send_lark("kol", help_text)


# ── 消息路由 ────────────────────────────────────────────────────────────────────
# 短确认词，不响应
_IGNORE_WORDS = {
    "好", "好的", "好吧", "嗯", "嗯嗯", "哦", "哦哦", "ok", "OK", "Ok",
    "收到", "知道了", "明白", "明白了", "好了", "行", "行吧", "好好",
    "谢谢", "谢", "感谢", "👍", "👌", "✅", "😊", "🙏",
}

def _should_ignore(text: str) -> bool:
    """过滤无需响应的短确认消息"""
    if len(text) <= 2:
        return True
    if text.lower() in _IGNORE_WORDS:
        return True
    return False


def parse_and_dispatch(text: str):
    text = text.strip()
    print(f"[{now_str()}] 收到: {text}")

    # 忽略短确认词
    if _should_ignore(text):
        print(f"[{now_str()}] 忽略确认词: {text}")
        return

    # 开会
    m = re.match(r"开会(\d+)轮\s*(.+)", text)
    if m:
        rounds, topic = int(m.group(1)), m.group(2).strip()
        send_lark("lead", f"👔 Team Lead\n\n📋 收到！启动 {rounds} 轮讨论：{topic}")
        threading.Thread(target=handle_meeting, args=(topic, rounds), daemon=True).start()
        return
    if text.startswith("开会"):
        topic = text[2:].strip()
        if not topic:
            send_lark("lead", "👔 Team Lead\n\n请告诉我议题，例如：\n开会 本周KOL外联策略")
            return
        send_lark("lead", f"👔 Team Lead\n\n📋 收到！启动 2 轮讨论：{topic}")
        threading.Thread(target=handle_meeting, args=(topic,), daemon=True).start()
        return

    # 决策
    m = re.match(r"我选([ABab])\s*(.*)", text)
    if m:
        threading.Thread(target=handle_decision, args=(m.group(1), m.group(2).strip()), daemon=True).start()
        return

    # 执行（直接开始，不选A/B）
    if text in ("执行", "开始执行", "全员执行"):
        if not CTX.decision and CTX.topic:
            send_lark("lead", "👔 Team Lead\n\n⚠️ 还没有选择方案，请先说「我选A」或「我选B」")
        else:
            threading.Thread(target=handle_decision, args=("A", "直接执行"), daemon=True).start()
        return

    # 状态
    if text in ("状态", "进度", "status"):
        handle_status()
        return

    # 完成
    m = re.match(r"完成\s*(.+)", text)
    if m:
        threading.Thread(target=handle_complete, args=(m.group(1).strip(),), daemon=True).start()
        return

    # 日报
    if text in ("日报", "发日报"):
        send_lark("lead", "👔 Team Lead\n\n正在汇总各部门日报...")
        threading.Thread(target=handle_daily_report, daemon=True).start()
        return

    # 全流程（讨论→结论→执行→复盘→日报完整闭环）
    if text in ("全流程", "启动全流程", "今日全流程") or text.startswith("全流程 "):
        topic = text[4:].strip() if text.startswith("全流程 ") else None
        def _run_full():
            import scheduler as _s
            _s.run_full_workflow(topic)
        threading.Thread(target=_run_full, daemon=True).start()
        return

    # KOL CRM 指令（必须在员工 alias 匹配之前，否则 "kol" alias 会拦截）
    if text.lower().startswith("kol "):
        threading.Thread(target=handle_kol_cmd, args=(text[4:].strip(),), daemon=True).start()
        return

    # @员工X
    for emp in EMPLOYEES:
        for alias in emp["aliases"]:
            if text.startswith(f"@{alias}") or text.lower().startswith(alias.lower()):
                question = re.sub(rf"@?{re.escape(alias)}\s*", "", text, count=1).strip()
                if not question:
                    question = "请介绍你的工作和近期计划"
                threading.Thread(target=handle_employee_question, args=(emp, question), daemon=True).start()
                return

    # @lead
    if re.match(r"@?lead\s*", text, re.IGNORECASE) or text.startswith("@团队长"):
        question = re.sub(r"@?(lead|团队长)\s*", "", text, flags=re.IGNORECASE).strip()
        if not question:
            question = "现在最重要的事是什么？"
        threading.Thread(target=handle_lead_question, args=(question,), daemon=True).start()
        return

    # 未识别 → 直接交给 Lead 处理（回答问题 or 提议开会）
    def _lead_smart_reply(q: str):
        prompt = f"""Kelly 发来一条消息：「{q}」

判断她的意图并回应：
- 如果是问题/话题 → 先给出你的直接观点（2-3句），然后问「要开会深入讨论吗？发「开会 {q[:15]}」启动」
- 如果是指令但格式不对 → 直接告诉正确格式
- 如果是聊天 → 正常回复

不超过100字。"""
        response = ask_claude(LEAD_SYSTEM, prompt)
        send_lark("lead", f"👔 Team Lead\n\n{response}")

    threading.Thread(target=_lead_smart_reply, args=(text,), daemon=True).start()


# ── Flask 路由 ──────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return json.dumps({"status": "ok", "time": datetime.now(BJT).isoformat(),
                       "context": CTX.topic or "无进行中议题"})


def _fetch_twitter_today() -> int:
    """用 Bearer Token 查今日 @moonx_bydfi 发推数（缓存10分钟）"""
    cached = _cache_get("twitter_today")
    if cached is not None:
        return cached
    try:
        bearer = os.getenv("TWITTER_BEARER_TOKEN", "")
        t = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
        url = (f"https://api.twitter.com/2/tweets/search/recent"
               f"?query=from%3Amoonx_bydfi&max_results=10"
               f"&start_time={t}&tweet.fields=created_at")
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {bearer}"})
        res = json.loads(urllib.request.urlopen(req, timeout=8).read())
        count = res.get("meta", {}).get("result_count", 0)
        _cache_set("twitter_today", count)
        return count
    except Exception:
        return 0


def _fetch_devto_today() -> int:
    """用 DEVTO_API_KEY 查今日发布文章数（缓存10分钟）"""
    cached = _cache_get("devto_today")
    if cached is not None:
        return cached
    try:
        api_key = os.getenv("DEVTO_API_KEY", "")
        t = datetime.now(BJT).strftime("%Y-%m-%d")
        url = "https://dev.to/api/articles/me?per_page=10&state=published"
        req = urllib.request.Request(url, headers={"api-key": api_key})
        articles = json.loads(urllib.request.urlopen(req, timeout=8).read())
        count = sum(1 for a in articles if (a.get("published_at") or "")[:10] == t)
        _cache_set("devto_today", count)
        return count
    except Exception:
        return 0


@app.route("/push/stats", methods=["POST"])
def push_stats():
    """本地脚本推送 KOL 邮件等无 API 的数据"""
    global _kol_stats
    data = request.get_json(silent=True) or {}
    _kol_stats = {**data, "updated_at": datetime.now(BJT).isoformat()}
    return jsonify({"ok": True})


@app.route("/dash", methods=["GET"])
def dash():
    """实时数据面板 — 飞书卡片按钮跳转"""
    t   = datetime.now(BJT).strftime("%Y-%m-%d")
    now = datetime.now(BJT).strftime("%H:%M")

    tweets   = _fetch_twitter_today()
    articles = _fetch_devto_today()

    kol_today = _kol_stats.get("sent_today", "—") if _kol_stats.get("date") == t else "—"
    kol_total = _kol_stats.get("sent_total", "—")
    kol_names = _kol_stats.get("kol_total",  "—")
    kol_upd   = _kol_stats.get("updated_at", "未推送")[:16] if _kol_stats else "未推送"

    def stat(val, label, color):
        return f"""<div class="card">
      <div class="val {color}">{val}</div>
      <div class="lbl">{label}</div>
    </div>"""

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><title>MoonX 今日数据</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        background:#F1F5F9;padding:24px;}}
  .header{{margin-bottom:24px;}}
  .header h2{{font-size:22px;font-weight:800;color:#0F172A;}}
  .header p{{font-size:13px;color:#94A3B8;margin-top:4px;}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;}}
  .card{{background:#fff;border-radius:14px;padding:20px 18px;
         box-shadow:0 1px 3px rgba(0,0,0,.06);}}
  .val{{font-size:40px;font-weight:800;line-height:1;}}
  .lbl{{font-size:12px;color:#64748B;margin-top:8px;font-weight:500;}}
  .blue{{color:#3B82F6;}} .green{{color:#10B981;}}
  .orange{{color:#F59E0B;}} .purple{{color:#8B5CF6;}}
  .section{{margin-top:24px;}}
  .section h3{{font-size:13px;font-weight:600;color:#64748B;
               text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px;}}
  .row{{background:#fff;border-radius:10px;padding:12px 16px;
        display:flex;justify-content:space-between;align-items:center;
        margin-bottom:8px;box-shadow:0 1px 2px rgba(0,0,0,.04);}}
  .row-label{{font-size:14px;color:#0F172A;font-weight:500;}}
  .row-val{{font-size:14px;color:#64748B;}}
  .footer{{margin-top:20px;font-size:11px;color:#CBD5E1;text-align:right;}}
</style>
</head>
<body>
  <div class="header">
    <h2>MoonX 今日数据</h2>
    <p>{t} &nbsp;·&nbsp; 更新于 {now} BJT</p>
  </div>

  <div class="grid">
    {stat(tweets,   "Twitter 推文",  "blue")}
    {stat(articles, "SEO 文章发布",  "green")}
    {stat(kol_today,"KOL 邮件（今日）","orange")}
    {stat(kol_names,"KOL 名单总量",  "purple")}
  </div>

  <div class="section">
    <h3>KOL 邮件详情</h3>
    <div class="row">
      <span class="row-label">今日发送</span>
      <span class="row-val">{kol_today} 封</span>
    </div>
    <div class="row">
      <span class="row-label">累计发送</span>
      <span class="row-val">{kol_total} 封</span>
    </div>
    <div class="row">
      <span class="row-label">数据更新时间</span>
      <span class="row-val">{kol_upd}</span>
    </div>
  </div>

  <div class="footer">数据来源：Twitter API · dev.to API · 本地推送</div>
</body></html>"""
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


# ─────────────────────────────────────────────────────────────────────────────
# CRM 看板
# ─────────────────────────────────────────────────────────────────────────────

def _crm_db():
    """读取 SQLite 数据供 CRM 页面使用"""
    import sys, sqlite3
    sys.path.insert(0, str(BASE / "03_kol_media"))
    _ON_FLY = bool(os.getenv("FLY_APP_NAME"))
    db_path = "/data/kol_crm.db" if _ON_FLY else str(BASE / "03_kol_media" / "kol_crm.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/crm", methods=["GET"])
def crm_dashboard():
    """KOL CRM 看板 v2"""
    try:
        conn = _crm_db()
    except Exception as e:
        return f"<pre>DB Error: {e}</pre>", 500

    try:
        from datetime import date, timedelta
        _today     = date.today().isoformat()
        _yesterday = (date.today() - timedelta(days=1)).isoformat()
        _before_y  = (date.today() - timedelta(days=2)).isoformat()

        def _n(sql, p=()):
            return conn.execute(sql, p).fetchone()[0] or 0

        # ── 核心统计 ─────────────────────────────────────────────────────────
        stats = {
            "total":         _n("SELECT COUNT(*) FROM kols"),
            "email":         _n("SELECT COUNT(*) FROM kols WHERE email!='' AND email IS NOT NULL"),
            "sent":          _n("SELECT COUNT(*) FROM kols WHERE status IN ('已发送','已回复','谈判中','已签约','已发布','已完成')"),
            "replied":       _n("SELECT COUNT(*) FROM kols WHERE status IN ('已回复','谈判中','已签约','已发布','已完成')"),
            "negotiating":   _n("SELECT COUNT(*) FROM kols WHERE status='谈判中'"),
            "signed":        _n("SELECT COUNT(*) FROM kols WHERE status='已签约'"),
            "rejected":      _n("SELECT COUNT(*) FROM kols WHERE status='已拒绝'"),
            "pending":       _n("SELECT COUNT(*) FROM kols WHERE status='待发送' AND email!=''"),
            "media_total":   _n("SELECT COUNT(*) FROM media"),
            "media_email":   _n("SELECT COUNT(*) FROM media WHERE email!=''"),
            "media_inquired":_n("SELECT COUNT(*) FROM media WHERE status LIKE '已询价%'"),
            "media_replied": _n("SELECT COUNT(*) FROM media WHERE status LIKE '已回复%'"),
        }

        # ── 汇总表数据 ────────────────────────────────────────────────────────
        kol_rows = [
            {
                "label":  "KOL 收集总数",
                "total":  _n("SELECT COUNT(*) FROM kols"),
                "today":  _n("SELECT COUNT(*) FROM kols WHERE collect_date=?", (_today,)),
                "yest":   _n("SELECT COUNT(*) FROM kols WHERE collect_date=?", (_yesterday,)),
                "before": _n("SELECT COUNT(*) FROM kols WHERE collect_date=?", (_before_y,)),
            },
            {
                "label":  "KOL 有邮箱",
                "total":  _n("SELECT COUNT(*) FROM kols WHERE email!=''"),
                "today":  _n("SELECT COUNT(*) FROM kols WHERE collect_date=? AND email!=''", (_today,)),
                "yest":   _n("SELECT COUNT(*) FROM kols WHERE collect_date=? AND email!=''", (_yesterday,)),
                "before": _n("SELECT COUNT(*) FROM kols WHERE collect_date=? AND email!=''", (_before_y,)),
            },
            {
                "label":  "KOL 已发送",
                "total":  _n("SELECT COUNT(*) FROM contacts"),
                "today":  _n("SELECT COUNT(*) FROM contacts WHERE sent_at LIKE ?", (f"{_today}%",)),
                "yest":   _n("SELECT COUNT(*) FROM contacts WHERE sent_at LIKE ?", (f"{_yesterday}%",)),
                "before": _n("SELECT COUNT(*) FROM contacts WHERE sent_at LIKE ?", (f"{_before_y}%",)),
            },
            {
                "label":  "KOL 已回复",
                "total":  _n("SELECT COUNT(*) FROM replies WHERE intent!='migrated'"),
                "today":  _n("SELECT COUNT(*) FROM replies WHERE detected_at LIKE ? AND intent!='migrated'", (f"{_today}%",)),
                "yest":   "—",
                "before": "—",
            },
        ]
        media_rows = [
            {
                "label":  "媒体收录总数",
                "total":  _n("SELECT COUNT(*) FROM media"),
                "today":  _n("SELECT COUNT(*) FROM media WHERE collect_date=?", (_today,)),
                "yest":   _n("SELECT COUNT(*) FROM media WHERE collect_date=?", (_yesterday,)),
                "before": _n("SELECT COUNT(*) FROM media WHERE collect_date=?", (_before_y,)),
            },
            {
                "label":  "媒体有邮箱",
                "total":  _n("SELECT COUNT(*) FROM media WHERE email!=''"),
                "today":  _n("SELECT COUNT(*) FROM media WHERE collect_date=? AND email!=''", (_today,)),
                "yest":   _n("SELECT COUNT(*) FROM media WHERE collect_date=? AND email!=''", (_yesterday,)),
                "before": _n("SELECT COUNT(*) FROM media WHERE collect_date=? AND email!=''", (_before_y,)),
            },
            {
                "label":  "媒体已询价",
                "total":  _n("SELECT COUNT(*) FROM media WHERE status LIKE '已询价%'"),
                "today":  _n("SELECT COUNT(*) FROM media WHERE status LIKE '已询价%' AND updated_at LIKE ?", (f"{_today}%",)),
                "yest":   _n("SELECT COUNT(*) FROM media WHERE status LIKE '已询价%' AND updated_at LIKE ?", (f"{_yesterday}%",)),
                "before": _n("SELECT COUNT(*) FROM media WHERE status LIKE '已询价%' AND updated_at LIKE ?", (f"{_before_y}%",)),
            },
            {
                "label":  "媒体已回复",
                "total":  _n("SELECT COUNT(*) FROM media WHERE status LIKE '已回复%'"),
                "today":  _n("SELECT COUNT(*) FROM media WHERE status LIKE '已回复%' AND updated_at LIKE ?", (f"{_today}%",)),
                "yest":   _n("SELECT COUNT(*) FROM media WHERE status LIKE '已回复%' AND updated_at LIKE ?", (f"{_yesterday}%",)),
                "before": _n("SELECT COUNT(*) FROM media WHERE status LIKE '已回复%' AND updated_at LIKE ?", (f"{_before_y}%",)),
            },
        ]

        # ── 看板管道 ──────────────────────────────────────────────────────────
        pipeline_stages = [
            ("待发送",  "#94A3B8", ["待发送"]),
            ("已发送",  "#3B82F6", ["已发送"]),
            ("已回复",  "#F59E0B", ["已回复"]),
            ("谈判中",  "#8B5CF6", ["谈判中"]),
            ("已签约",  "#10B981", ["已签约", "已发布", "已完成"]),
        ]
        pipeline_data = {}
        for label, color, statuses in pipeline_stages:
            ph = ",".join("?" * len(statuses))
            rows = conn.execute(f"""
                SELECT k.id, k.name, k.tier, k.subscribers, k.email,
                       k.channel_url, k.status, k.utm_code, k.collect_date,
                       k.tg_status, k.source,
                       (SELECT COUNT(*) FROM replies WHERE kol_id=k.id) as reply_cnt,
                       (SELECT intent FROM replies WHERE kol_id=k.id ORDER BY detected_at DESC LIMIT 1) as last_intent
                FROM kols k WHERE k.status IN ({ph})
                ORDER BY k.subscribers DESC LIMIT 60
            """, statuses).fetchall()
            pipeline_data[label] = {
                "color": color,
                "kols":  [dict(r) for r in rows],
                "total": _n(f"SELECT COUNT(*) FROM kols WHERE status IN ({ph})", statuses),
            }

        # ── 最近回复 ──────────────────────────────────────────────────────────
        recent_replies = conn.execute("""
            SELECT r.id, r.detected_at, r.intent, r.subject, r.body_snippet,
                   r.auto_response_sent, k.name, k.email, k.tier
            FROM replies r JOIN kols k ON k.id=r.kol_id
            WHERE r.intent != 'migrated'
            ORDER BY r.detected_at DESC LIMIT 50
        """).fetchall()
        recent_replies = [dict(r) for r in recent_replies]

        # ── KOL 列表 ──────────────────────────────────────────────────────────
        all_kols = conn.execute("""
            SELECT k.name, k.tier, k.subscribers, k.status, k.email,
                   k.source, k.channel_url, k.collect_date
            FROM kols k ORDER BY k.subscribers DESC LIMIT 500
        """).fetchall()
        all_kols = [dict(r) for r in all_kols]

        # ── 媒体列表 ──────────────────────────────────────────────────────────
        all_media = conn.execute("""
            SELECT name, type, status, email, priority, notes
            FROM media ORDER BY priority, name LIMIT 300
        """).fetchall()
        all_media = [dict(r) for r in all_media]

        conn.close()
    except Exception as e:
        conn.close()
        return f"<pre>Query Error: {e}</pre>", 500

    # ── 渲染辅助函数 ──────────────────────────────────────────────────────────
    def _subs(n):
        try:
            n = int(n or 0)
            if n >= 100_0000: return f"{n/100_0000:.1f}M"
            if n >= 10000:    return f"{n/10000:.0f}万"
            return str(n)
        except:
            return "—"

    TIER_COLORS = {
        "A级": "#F59E0B", "B级": "#10B981", "C级": "#3B82F6",
        "D级": "#94A3B8", "KALSHI": "#8B5CF6", "PM": "#EC4899",
    }
    AVATAR_COLORS = ["#FF6B00","#3B82F6","#10B981","#8B5CF6","#F59E0B","#EC4899","#06B6D4"]

    def _avatar_color(name):
        return AVATAR_COLORS[sum(ord(c) for c in (name or "?")) % len(AVATAR_COLORS)]

    def _tier_tag(tier):
        c = TIER_COLORS.get(tier, "#94A3B8")
        return f'<span class="tier-tag" style="--t-c:{c};background:{c}">{tier or "—"}</span>'

    INTENT_MAP = {
        "interested": ("感兴趣", "badge-green"),
        "rejected":   ("拒绝",   "badge-red"),
        "ooo":        ("不在",   "badge-gray"),
        "inquiry":    ("询价",   "badge-amber"),
        "unknown":    ("待分类", "badge-gray"),
    }
    def _intent_badge(intent, small=False):
        lbl, cls = INTENT_MAP.get(intent or "unknown", ("—", "badge-gray"))
        sz = ' style="font-size:10px"' if small else ""
        return f'<span class="badge {cls}"{sz}>{lbl}</span>'

    def _status_pill(status):
        m = {
            "待发送": ("#94A3B8","#F1F5F9"), "已发送": ("#1D4ED8","#DBEAFE"),
            "已回复": ("#92400E","#FEF3C7"), "谈判中": ("#6D28D9","#EDE9FE"),
            "已签约": ("#166534","#DCFCE7"), "已发布": ("#166534","#DCFCE7"),
            "已完成": ("#166534","#DCFCE7"), "已拒绝": ("#991B1B","#FEE2E2"),
        }
        fg, bg = m.get(status, ("#374151","#F1F5F9"))
        return f'<span class="status-pill" style="color:{fg};background:{bg}">{status or "—"}</span>'

    def _pct(a, b):
        if not b: return "—"
        return f"{round(a/b*100)}%"

    def _num_td(v):
        cls = "td-num zero" if v == 0 or v == "—" else "td-num"
        return f'<td class="{cls}">{v}</td>'

    # ── HTML 构建 ─────────────────────────────────────────────────────────────

    # 汇总表行
    def _summary_row(r):
        return (f'<tr><td class="td-label">{r["label"]}</td>'
                f'{_num_td(r["total"])}{_num_td(r["today"])}'
                f'{_num_td(r["yest"])}{_num_td(r["before"])}</tr>')

    kol_rows_html   = "".join(_summary_row(r) for r in kol_rows)
    media_rows_html_tbl = "".join(_summary_row(r) for r in media_rows)

    # 看板卡片
    def _kol_card(k):
        ac = _avatar_color(k["name"])
        initials = (k["name"] or "?")[:2].upper()
        tc = TIER_COLORS.get(k["tier"], "#94A3B8")
        intent_html = _intent_badge(k.get("last_intent"), small=True) if k.get("last_intent") else ""
        tg_html = '<span class="tg-dot">● TG</span>' if k.get("tg_status") == "added" else ""
        yt_link = (f'<a href="{k["channel_url"]}" target="_blank" class="expand-link">YouTube ↗</a>'
                   if k.get("channel_url") else "")
        return f"""<div class="kol-card" data-name="{k['name']}" data-tier="{k.get('tier','')}" onclick="toggleCard(this)">
  <div class="kol-card-top">
    <div class="kol-avatar" style="background:{ac}">{initials}</div>
    <div class="kol-info">
      <div class="kol-name"><a href="/crm/kol/{k['id']}" onclick="event.stopPropagation()">{k['name']}</a></div>
      <div class="kol-sub-row">
        {_tier_tag(k.get('tier',''))}
        <span class="subs-txt">{_subs(k['subscribers'])}</span>
        {tg_html}{intent_html}
      </div>
    </div>
  </div>
  <div class="kol-expand">
    <div class="expand-row"><span class="expand-lbl">邮箱</span><span class="expand-val">{k['email'] or '—'}</span></div>
    <div class="expand-row"><span class="expand-lbl">UTM</span><span class="expand-val" style="color:#94A3B8">{k.get('utm_code') or '—'}</span></div>
    <div class="expand-row"><span class="expand-lbl">链接</span>{yt_link or '<span class="expand-val">—</span>'}</div>
  </div>
</div>"""

    # 看板列
    pipeline_cols_html = ""
    for label, info in pipeline_data.items():
        cards_html = "".join(_kol_card(k) for k in info["kols"])
        overflow = (f'<div class="overflow-note">还有 {info["total"] - len(info["kols"])} 条</div>'
                    if info["total"] > len(info["kols"]) else "")
        pipeline_cols_html += f"""<div class="col" style="--col-c:{info['color']}">
  <div class="col-header">
    <div class="col-header-left">
      <div class="col-status-dot"></div>
      <span class="col-name">{label}</span>
    </div>
    <span class="col-badge">{info['total']}</span>
  </div>
  <div class="col-body">{cards_html}{overflow}</div>
</div>"""

    # 回复记录行
    def _reply_row(r):
        auto_html = '<span class="badge badge-auto">TG已发</span>' if r.get("auto_response_sent") else ""
        snippet = (r.get("body_snippet") or "")[:100].replace("<", "&lt;")
        subj    = (r.get("subject") or "")[:50].replace("<", "&lt;")
        dt = (r.get("detected_at") or "")[:10]
        return f"""<tr>
  <td style="white-space:nowrap;color:#64748B;font-size:12px">{dt}</td>
  <td><div style="font-weight:600;font-size:13px">{r['name']}</div>
      {_tier_tag(r.get('tier',''))}</td>
  <td>{_intent_badge(r.get('intent'))}{auto_html}</td>
  <td><div style="font-size:12px;font-weight:500;color:#374151;margin-bottom:2px">{subj}</div>
      <div class="reply-snippet">{snippet}</div></td>
</tr>"""

    replies_rows_html = ("".join(_reply_row(r) for r in recent_replies)
                         or "<tr><td colspan=4 style='color:#94A3B8;text-align:center;padding:24px'>暂无回复记录</td></tr>")

    # KOL 列表行
    kol_list_html = ""
    for k in all_kols:
        kol_list_html += f"""<tr>
  <td style="font-weight:500">{k['name']}</td>
  <td>{_tier_tag(k.get('tier',''))}</td>
  <td style="text-align:right;font-variant-numeric:tabular-nums">{_subs(k['subscribers'])}</td>
  <td>{_status_pill(k.get('status',''))}</td>
  <td style="font-size:12px;color:#64748B;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{k.get('email') or '—'}</td>
  <td style="font-size:12px;color:#94A3B8">{k.get('source') or '—'}</td>
</tr>"""

    # 媒体列表行
    PRIORITY_COLORS = {"A": "#EF4444", "B": "#F59E0B", "C": "#94A3B8"}
    media_list_html = ""
    for m in all_media:
        pc = PRIORITY_COLORS.get((m.get('priority') or 'C').upper(), "#94A3B8")
        media_list_html += f"""<tr>
  <td style="font-weight:500">{m['name']}</td>
  <td style="font-size:12px;color:#64748B">{m.get('type') or '—'}</td>
  <td style="font-size:12px">{m.get('status') or '待联系'}</td>
  <td style="font-size:12px;color:#64748B">{m.get('email') or '—'}</td>
  <td><span style="font-weight:700;color:{pc}">{m.get('priority') or '—'}</span></td>
  <td style="font-size:12px;color:#94A3B8;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{m.get('notes') or '—'}</td>
</tr>"""

    now_str = datetime.now(BJT).strftime("%Y-%m-%d %H:%M BJT")

    # 漏斗转化率
    f_email_pct   = _pct(stats['email'],   stats['total'])
    f_sent_pct    = _pct(stats['sent'],    stats['email'])
    f_replied_pct = _pct(stats['replied'], stats['sent'])
    f_signed_pct  = _pct(stats['signed'],  stats['replied'])

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MoonX KOL CRM</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","SF Pro Text","Segoe UI",sans-serif;
     background:#F8FAFC;color:#0F172A;font-size:14px;min-height:100vh}}

/* ── Nav ── */
.nav{{background:#fff;border-bottom:1px solid #E2E8F0;padding:0 24px;
     display:flex;align-items:center;position:sticky;top:0;z-index:100;
     box-shadow:0 1px 4px rgba(0,0,0,.05)}}
.nav-brand{{display:flex;align-items:center;gap:10px;padding:14px 0;margin-right:20px;
            text-decoration:none;flex-shrink:0}}
.nav-logo{{width:32px;height:32px;background:#FF6B00;border-radius:8px;
           display:flex;align-items:center;justify-content:center;
           font-size:15px;font-weight:800;color:#fff}}
.nav-title{{font-size:15px;font-weight:700;color:#0F172A;letter-spacing:-.3px}}
.nav-tabs{{display:flex;height:56px;overflow-x:auto}}
.nav-tab{{padding:0 14px;display:flex;align-items:center;font-size:13px;
          font-weight:500;color:#64748B;cursor:pointer;border-bottom:2px solid transparent;
          transition:all .15s;white-space:nowrap;user-select:none;flex-shrink:0}}
.nav-tab:hover{{color:#0F172A}}
.nav-tab.active{{color:#FF6B00;border-bottom-color:#FF6B00;font-weight:600}}
.nav-time{{margin-left:auto;font-size:12px;color:#94A3B8;padding-left:16px;
           white-space:nowrap;flex-shrink:0}}

/* ── Pages ── */
.page{{display:none;padding:20px 24px 32px;animation:fadein .15s ease}}
.page.active{{display:block}}
@keyframes fadein{{from{{opacity:0;transform:translateY(3px)}}to{{opacity:1;transform:none}}}}

/* ── Funnel ── */
.funnel{{background:#fff;border-radius:14px;padding:20px 24px;margin-bottom:16px;
         box-shadow:0 1px 3px rgba(0,0,0,.05);border:1px solid #F1F5F9}}
.funnel-label{{font-size:11px;font-weight:600;color:#94A3B8;letter-spacing:.5px;
               text-transform:uppercase;margin-bottom:16px}}
.funnel-steps{{display:flex;align-items:stretch}}
.funnel-step{{flex:1;text-align:center;padding:12px 8px;position:relative}}
.funnel-step:not(:last-child)::after{{content:'›';position:absolute;right:-2px;top:50%;
  transform:translateY(-50%);font-size:22px;color:#CBD5E1;pointer-events:none}}
.funnel-val{{font-size:28px;font-weight:800;line-height:1;margin-bottom:3px}}
.funnel-name{{font-size:11px;color:#64748B;font-weight:500}}
.funnel-rate{{font-size:11px;color:#94A3B8;margin-top:3px}}
.fc-gray{{color:#94A3B8}}.fc-blue{{color:#3B82F6}}.fc-orange{{color:#FF6B00}}
.fc-amber{{color:#F59E0B}}.fc-green{{color:#10B981}}

/* ── KPI Grid ── */
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:12px;margin-bottom:16px}}
.kpi{{background:#fff;border-radius:12px;padding:16px 18px;
      box-shadow:0 1px 3px rgba(0,0,0,.05);border:1px solid #F1F5F9;
      position:relative;overflow:hidden;border-left:3px solid var(--kc,#E2E8F0)}}
.kpi-val{{font-size:30px;font-weight:800;line-height:1;color:var(--kc,#0F172A)}}
.kpi-lbl{{font-size:12px;color:#64748B;margin-top:5px;font-weight:500}}
.kpi-sub{{font-size:11px;color:#CBD5E1;margin-top:2px}}

/* ── Table wrap ── */
.tbl-wrap{{background:#fff;border-radius:14px;box-shadow:0 1px 3px rgba(0,0,0,.05);
           border:1px solid #F1F5F9;overflow:hidden;margin-bottom:16px}}
.tbl-head{{display:flex;align-items:center;justify-content:space-between;
           padding:14px 20px;border-bottom:1px solid #F1F5F9}}
.tbl-head-title{{font-size:14px;font-weight:700;color:#0F172A}}
.tbl-head-sub{{font-size:12px;color:#94A3B8}}
table{{width:100%;border-collapse:collapse}}
th{{background:#FAFAFA;font-size:11px;font-weight:600;color:#94A3B8;
    padding:10px 20px;text-align:left;text-transform:uppercase;
    letter-spacing:.4px;border-bottom:1px solid #F1F5F9}}
td{{padding:11px 20px;border-bottom:1px solid #F8FAFC;font-size:13px;vertical-align:middle}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:#FFFBF7}}
.td-label{{color:#374151;font-weight:500}}
.td-num{{text-align:right;font-weight:600;color:#0F172A;font-variant-numeric:tabular-nums}}
.td-num.zero{{color:#CBD5E1;font-weight:400}}
.td-group td{{font-size:10px;font-weight:700;color:#94A3B8;letter-spacing:.6px;
              text-transform:uppercase;padding:8px 20px;border-bottom:1px solid #F1F5F9;
              background:#FAFAFA}}

/* ── Toolbar ── */
.toolbar{{display:flex;align-items:center;gap:10px;margin-bottom:14px;flex-wrap:wrap}}
.search-box{{position:relative;flex:1;min-width:180px;max-width:280px}}
.search-box input{{width:100%;padding:8px 12px 8px 34px;border:1px solid #E2E8F0;
                   border-radius:8px;font-size:13px;background:#fff;outline:none;font-family:inherit}}
.search-box input:focus{{border-color:#FF6B00;box-shadow:0 0 0 3px rgba(255,107,0,.08)}}
.search-ico{{position:absolute;left:11px;top:50%;transform:translateY(-50%);color:#CBD5E1;font-size:15px}}
.tier-filter{{display:flex;gap:6px;flex-wrap:wrap}}
.tbtn{{padding:6px 12px;border-radius:20px;font-size:12px;font-weight:500;
       cursor:pointer;border:1px solid #E2E8F0;background:#fff;color:#64748B;
       transition:all .15s;font-family:inherit}}
.tbtn:hover,.tbtn.active{{background:#FF6B00;color:#fff;border-color:#FF6B00}}

/* ── Kanban ── */
.kanban{{display:flex;gap:14px;overflow-x:auto;align-items:flex-start;padding-bottom:6px}}
.kanban::-webkit-scrollbar{{height:4px}}
.kanban::-webkit-scrollbar-track{{background:#F1F5F9;border-radius:2px}}
.kanban::-webkit-scrollbar-thumb{{background:#CBD5E1;border-radius:2px}}
.col{{background:#F8FAFC;border-radius:14px;width:256px;flex-shrink:0;
      border:1px solid #E2E8F0;border-top:3px solid var(--col-c)}}
.col-header{{padding:13px 14px 11px;display:flex;justify-content:space-between;
             align-items:center;border-bottom:1px solid #E2E8F0}}
.col-header-left{{display:flex;align-items:center;gap:8px}}
.col-status-dot{{width:8px;height:8px;border-radius:50%;background:var(--col-c);flex-shrink:0}}
.col-name{{font-size:13px;font-weight:700;color:#0F172A}}
.col-badge{{font-size:11px;padding:2px 9px;border-radius:20px;
            background:var(--col-c);color:#fff;font-weight:700}}
.col-body{{padding:10px;max-height:68vh;overflow-y:auto}}
.col-body::-webkit-scrollbar{{width:3px}}
.col-body::-webkit-scrollbar-thumb{{background:#E2E8F0;border-radius:2px}}

/* ── KOL Card ── */
.kol-card{{background:#fff;border:1px solid #E2E8F0;border-radius:10px;
           padding:12px;margin-bottom:8px;cursor:pointer;transition:all .15s}}
.kol-card:hover{{border-color:#FF6B00;box-shadow:0 2px 10px rgba(255,107,0,.1);
                 transform:translateY(-1px)}}
.kol-card-top{{display:flex;align-items:flex-start;gap:10px}}
.kol-avatar{{width:36px;height:36px;border-radius:9px;flex-shrink:0;
             display:flex;align-items:center;justify-content:center;
             font-size:13px;font-weight:800;color:#fff}}
.kol-info{{flex:1;min-width:0}}
.kol-name{{font-size:13px;font-weight:600;white-space:nowrap;overflow:hidden;
           text-overflow:ellipsis;color:#0F172A;margin-bottom:4px}}
.kol-name a{{color:#0F172A;text-decoration:none}}
.kol-name a:hover{{color:#FF6B00}}
.kol-sub-row{{display:flex;align-items:center;gap:5px;flex-wrap:wrap}}
.tier-tag{{font-size:10px;padding:1px 6px;border-radius:4px;color:#fff;font-weight:700;flex-shrink:0}}
.subs-txt{{font-size:11px;color:#94A3B8}}
.tg-dot{{font-size:10px;color:#10B981;font-weight:700}}
.kol-expand{{border-top:1px solid #F1F5F9;margin-top:10px;padding-top:10px;display:none}}
.expand-row{{display:flex;gap:8px;margin-bottom:5px;align-items:baseline}}
.expand-lbl{{font-size:10px;color:#94A3B8;width:32px;flex-shrink:0;font-weight:600;text-transform:uppercase}}
.expand-val{{font-size:11px;color:#374151;word-break:break-all}}
.expand-link{{font-size:11px;color:#FF6B00;text-decoration:none;font-weight:500}}
.overflow-note{{text-align:center;padding:10px;font-size:11px;color:#94A3B8;
                border-top:1px solid #F1F5F9;margin-top:4px}}

/* ── Badges ── */
.badge{{display:inline-flex;align-items:center;padding:2px 8px;border-radius:20px;
        font-size:11px;font-weight:600;white-space:nowrap}}
.badge-green{{background:#DCFCE7;color:#166534}}
.badge-red{{background:#FEE2E2;color:#991B1B}}
.badge-amber{{background:#FEF3C7;color:#92400E}}
.badge-gray{{background:#F1F5F9;color:#475569}}
.badge-blue{{background:#DBEAFE;color:#1E40AF}}
.badge-auto{{background:#F0FDF4;color:#15803D;font-size:10px;margin-left:4px}}
.status-pill{{display:inline-block;padding:2px 8px;border-radius:20px;
              font-size:11px;font-weight:600}}
.reply-snippet{{font-size:11px;color:#94A3B8;overflow:hidden;text-overflow:ellipsis;
                white-space:nowrap;max-width:340px}}
</style>
</head>
<body>

<nav class="nav">
  <a class="nav-brand" href="/crm">
    <div class="nav-logo">M</div>
    <span class="nav-title">KOL CRM</span>
  </a>
  <div class="nav-tabs">
    <div class="nav-tab active" data-tab="overview">总览</div>
    <div class="nav-tab" data-tab="kanban">看板</div>
    <div class="nav-tab" data-tab="kollist">KOL 列表 <span style="font-size:10px;color:#CBD5E1;margin-left:4px">{stats['total']}</span></div>
    <div class="nav-tab" data-tab="media">媒体库 <span style="font-size:10px;color:#CBD5E1;margin-left:4px">{stats['media_total']}</span></div>
    <div class="nav-tab" data-tab="replies">回复记录</div>
  </div>
  <div class="nav-time">{now_str}</div>
</nav>

<!-- ═══ 总览 ═══ -->
<div id="tab-overview" class="page active">

  <div class="funnel">
    <div class="funnel-label">转化漏斗</div>
    <div class="funnel-steps">
      <div class="funnel-step">
        <div class="funnel-val fc-gray">{stats['total']}</div>
        <div class="funnel-name">KOL 收集</div>
        <div class="funnel-rate">100%</div>
      </div>
      <div class="funnel-step">
        <div class="funnel-val fc-blue">{stats['email']}</div>
        <div class="funnel-name">有邮箱</div>
        <div class="funnel-rate">{f_email_pct}</div>
      </div>
      <div class="funnel-step">
        <div class="funnel-val fc-orange">{stats['sent']}</div>
        <div class="funnel-name">已外联</div>
        <div class="funnel-rate">{f_sent_pct}</div>
      </div>
      <div class="funnel-step">
        <div class="funnel-val fc-amber">{stats['replied']}</div>
        <div class="funnel-name">已回复</div>
        <div class="funnel-rate">{f_replied_pct}</div>
      </div>
      <div class="funnel-step">
        <div class="funnel-val fc-green">{stats['signed']}</div>
        <div class="funnel-name">已签约</div>
        <div class="funnel-rate">{f_signed_pct}</div>
      </div>
    </div>
  </div>

  <div class="kpi-grid">
    <div class="kpi" style="--kc:#94A3B8"><div class="kpi-val">{stats['total']}</div><div class="kpi-lbl">KOL 总收录</div><div class="kpi-sub">全渠道</div></div>
    <div class="kpi" style="--kc:#3B82F6"><div class="kpi-val">{stats['email']}</div><div class="kpi-lbl">有邮箱</div><div class="kpi-sub">可触达</div></div>
    <div class="kpi" style="--kc:#FF6B00"><div class="kpi-val">{stats['sent']}</div><div class="kpi-lbl">已外联</div><div class="kpi-sub">已发邮件</div></div>
    <div class="kpi" style="--kc:#F59E0B"><div class="kpi-val">{stats['replied']}</div><div class="kpi-lbl">已回复</div><div class="kpi-sub">含谈判&amp;签约</div></div>
    <div class="kpi" style="--kc:#8B5CF6"><div class="kpi-val">{stats['negotiating']}</div><div class="kpi-lbl">谈判中</div><div class="kpi-sub">进行中</div></div>
    <div class="kpi" style="--kc:#10B981"><div class="kpi-val">{stats['signed']}</div><div class="kpi-lbl">已签约</div><div class="kpi-sub">合作达成</div></div>
    <div class="kpi" style="--kc:#EF4444"><div class="kpi-val">{stats['rejected']}</div><div class="kpi-lbl">已拒绝</div><div class="kpi-sub">无效线索</div></div>
    <div class="kpi" style="--kc:#3B82F6"><div class="kpi-val">{stats['pending']}</div><div class="kpi-lbl">待发送</div><div class="kpi-sub">有邮箱未联系</div></div>
  </div>

  <div class="tbl-wrap">
    <div class="tbl-head">
      <span class="tbl-head-title">收集 &amp; 外联汇总</span>
      <span class="tbl-head-sub">{now_str}</span>
    </div>
    <table>
      <thead><tr><th>指标</th><th style="text-align:right">累计</th><th style="text-align:right">今日</th><th style="text-align:right">昨日</th><th style="text-align:right">前日</th></tr></thead>
      <tbody>
        <tr class="td-group"><td colspan="5">KOL 外联</td></tr>
        {kol_rows_html}
        <tr class="td-group"><td colspan="5">媒体库</td></tr>
        {media_rows_html_tbl}
      </tbody>
    </table>
  </div>

</div>

<!-- ═══ 看板 ═══ -->
<div id="tab-kanban" class="page">
  <div class="toolbar">
    <div class="search-box">
      <span class="search-ico">⌕</span>
      <input type="text" placeholder="搜索 KOL 名称…" oninput="filterCards(this.value)">
    </div>
    <div class="tier-filter">
      <button class="tbtn active" onclick="filterTier('all',this)">全部</button>
      <button class="tbtn" onclick="filterTier('A级',this)">A级</button>
      <button class="tbtn" onclick="filterTier('B级',this)">B级</button>
      <button class="tbtn" onclick="filterTier('C级',this)">C级</button>
      <button class="tbtn" onclick="filterTier('KALSHI',this)">Kalshi</button>
    </div>
  </div>
  <div class="kanban">{pipeline_cols_html}</div>
</div>

<!-- ═══ KOL 列表 ═══ -->
<div id="tab-kollist" class="page">
  <div class="toolbar">
    <div class="search-box">
      <span class="search-ico">⌕</span>
      <input type="text" placeholder="搜索 KOL…" oninput="filterTbl(this.value,'kol-tbody')">
    </div>
  </div>
  <div class="tbl-wrap">
    <table>
      <thead><tr><th>名称</th><th>级别</th><th style="text-align:right">订阅数</th><th>状态</th><th>邮箱</th><th>来源</th></tr></thead>
      <tbody id="kol-tbody">{kol_list_html}</tbody>
    </table>
  </div>
</div>

<!-- ═══ 媒体库 ═══ -->
<div id="tab-media" class="page">
  <div class="toolbar">
    <div class="search-box">
      <span class="search-ico">⌕</span>
      <input type="text" placeholder="搜索媒体…" oninput="filterTbl(this.value,'media-tbody')">
    </div>
  </div>
  <div class="tbl-wrap">
    <table>
      <thead><tr><th>媒体名称</th><th>类型</th><th>状态</th><th>邮箱</th><th>优先级</th><th>备注</th></tr></thead>
      <tbody id="media-tbody">{media_list_html}</tbody>
    </table>
  </div>
</div>

<!-- ═══ 回复记录 ═══ -->
<div id="tab-replies" class="page">
  <div class="tbl-wrap">
    <div class="tbl-head">
      <span class="tbl-head-title">回复记录</span>
      <span class="tbl-head-sub">最近 {len(recent_replies)} 条</span>
    </div>
    <table>
      <thead><tr><th>日期</th><th>KOL</th><th>意图</th><th>内容摘要</th></tr></thead>
      <tbody>{replies_rows_html}</tbody>
    </table>
  </div>
</div>

<script>
// Tab 切换
document.querySelectorAll('.nav-tab').forEach(t => {{
  t.addEventListener('click', () => {{
    document.querySelectorAll('.nav-tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.page').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('tab-' + t.dataset.tab).classList.add('active');
  }});
}});

// 展开卡片
function toggleCard(card) {{
  const d = card.querySelector('.kol-expand');
  d.style.display = d.style.display === 'none' ? 'block' : 'none';
}}

// 看板搜索
function filterCards(q) {{
  q = q.toLowerCase();
  document.querySelectorAll('.kol-card').forEach(c => {{
    c.style.display = (c.dataset.name||'').toLowerCase().includes(q) ? '' : 'none';
  }});
}}

// 看板 Tier 过滤
function filterTier(tier, btn) {{
  document.querySelectorAll('.tbtn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.kol-card').forEach(c => {{
    c.style.display = (tier === 'all' || c.dataset.tier === tier) ? '' : 'none';
  }});
}}

// 表格搜索
function filterTbl(q, id) {{
  q = q.toLowerCase();
  document.getElementById(id).querySelectorAll('tr').forEach(r => {{
    r.style.display = r.textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}}
</script>
</body></html>"""

    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/crm/kol/<int:kol_id>", methods=["GET"])
def crm_kol_detail(kol_id: int):
    """KOL 详情页"""
    import sys
    sys.path.insert(0, str(BASE / "03_kol_media"))
    try:
        from kol_db import get_kol_detail
        data = get_kol_detail(kol_id)
    except Exception as e:
        return f"<pre>DB Error: {e}</pre>", 500

    if not data:
        return "<h2>KOL 不存在</h2>", 404

    kol  = data["kol"]
    neg  = data["negotiation"]
    acts = data["activities"]
    ctrs = data["contacts"]
    reps = data["replies"]
    flw  = data["followup"]
    now_str = datetime.now(BJT).strftime("%Y-%m-%d %H:%M BJT")

    TIER_COLORS = {
        "A级": "#F59E0B", "B级": "#10B981", "C级": "#3B82F6",
        "D级": "#94A3B8", "KALSHI": "#8B5CF6", "PM": "#EC4899",
    }
    STATUS_STYLE = {
        "待发送": ("#94A3B8","#F1F5F9"), "已发送": ("#1D4ED8","#DBEAFE"),
        "已回复": ("#92400E","#FEF3C7"), "谈判中": ("#6D28D9","#EDE9FE"),
        "已签约": ("#166534","#DCFCE7"), "已拒绝": ("#991B1B","#FEE2E2"),
    }
    AVATAR_COLORS = ["#FF6B00","#3B82F6","#10B981","#8B5CF6","#F59E0B","#EC4899","#06B6D4"]

    def _avatar_c(name):
        return AVATAR_COLORS[sum(ord(c) for c in (name or "?")) % len(AVATAR_COLORS)]

    def _subs(n):
        try:
            n = int(n or 0)
            if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
            if n >= 10_000:    return f"{n/10_000:.0f}万"
            return str(n)
        except: return "—"

    tc = TIER_COLORS.get(kol.get("tier",""), "#94A3B8")
    sc_fg, sc_bg = STATUS_STYLE.get(kol.get("status",""), ("#374151","#F1F5F9"))
    av_c = _avatar_c(kol.get("name",""))
    initials = (kol.get("name") or "?")[:2].upper()

    # 合并时间线：activities（已包含 email_sent / reply_received） + contacts/replies 兜底
    # activities 是主数据源，contacts/replies 仅补充 activities 为空的情况
    timeline = []
    act_types_with_data = {a["type"] for a in acts}

    # 如果 activities 里没有 email_sent（历史数据），从 contacts 补
    if "email_sent" not in act_types_with_data:
        for c in ctrs:
            timeline.append({
                "type": "email_sent",
                "content": f"模板: {c.get('template','')} | 主题: {c.get('subject','')}",
                "operator": "auto",
                "created_at": c.get("sent_at",""),
            })
    # 如果 activities 里没有 reply_received，从 replies 补
    if "reply_received" not in act_types_with_data:
        INTENT_ZH = {"interested":"感兴趣","rejected":"拒绝","ooo":"不在","inquiry":"询价","unknown":"待分类"}
        for r in reps:
            if r.get("intent") == "migrated": continue
            timeline.append({
                "type": "reply_received",
                "content": f"意图: {INTENT_ZH.get(r.get('intent',''),'?')} | {r.get('body_snippet','')[:80]}",
                "operator": "auto",
                "created_at": r.get("detected_at",""),
            })

    for a in acts:
        timeline.append(dict(a))

    timeline.sort(key=lambda x: x.get("created_at",""), reverse=True)

    TYPE_ICON = {
        "email_sent":    ("📧", "#3B82F6", "发送邮件"),
        "reply_received":("💬", "#F59E0B", "收到回复"),
        "note":          ("📝", "#8B5CF6", "添加备注"),
        "status_change": ("🔄", "#94A3B8", "状态变更"),
        "negotiation":   ("🤝", "#10B981", "谈判更新"),
        "call":          ("📞", "#EC4899", "通话记录"),
    }

    def _tl_item(item):
        icon, color, label = TYPE_ICON.get(item["type"], ("•", "#94A3B8", item["type"]))
        dt = (item.get("created_at") or "")[:16]
        op_badge = ""
        if item.get("operator") == "kelly":
            op_badge = '<span style="font-size:10px;background:#FFF3E0;color:#FF6B00;padding:1px 6px;border-radius:10px;margin-left:6px">Kelly</span>'
        return f"""<div class="tl-item">
  <div class="tl-dot" style="background:{color}">{icon}</div>
  <div class="tl-body">
    <div class="tl-header">
      <span class="tl-label" style="color:{color}">{label}</span>{op_badge}
      <span class="tl-time">{dt}</span>
    </div>
    <div class="tl-content">{(item.get('content') or '').replace('<','&lt;')}</div>
  </div>
</div>"""

    timeline_html = "".join(_tl_item(t) for t in timeline) or \
        '<div style="color:#94A3B8;text-align:center;padding:32px">暂无活动记录</div>'

    # 谈判阶段选项
    NEG_STAGES = [
        ("initial", "初始接触"),
        ("price_quoted", "已报价"),
        ("terms_agreed", "条款确认"),
        ("signed", "已签约"),
    ]
    stage_opts = "".join(
        f'<option value="{v}" {"selected" if neg.get("stage")==v else ""}>{l}</option>'
        for v, l in NEG_STAGES
    )
    content_type_opts = "".join(
        f'<option value="{v}" {"selected" if neg.get("content_type")==v else ""}>{v or "—"}</option>'
        for v in ["", "YouTube 视频", "YouTube Short", "Twitter 推文", "Twitter Space", "其他"]
    )

    # 跟进状态
    followup_scheduled = flw.get("scheduled_at","")
    followup_status    = flw.get("status","")
    followup_step      = kol.get("sequence_step", 0)
    followup_html = ""
    if followup_scheduled and followup_status == "pending":
        followup_html = f'<div class="followup-badge">⏰ 第{followup_step+1}封跟进 · {followup_scheduled}</div>'
    elif followup_status == "replied":
        followup_html = '<div class="followup-badge replied">✓ 已回复，跟进序列结束</div>'

    # 预计算含反斜杠的表达式，Python<3.12 不允许在 f-string {} 内使用反斜杠
    _no_contacts = '<div style="padding:16px 20px;color:#94A3B8;font-size:12px">暂无外联记录</div>'
    ctrs_html = "".join(
        '<div style="padding:10px 20px;border-bottom:1px solid #F8FAFC;font-size:12px">'
        '<span style="color:#94A3B8">' + (c.get("sent_at") or "")[:16] + "</span>"
        " &nbsp; 模板: " + c.get("template", "—") + "</div>"
        for c in ctrs
    ) or _no_contacts

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{kol.get('name','')} — MoonX KOL CRM</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","SF Pro Text",sans-serif;
     background:#F8FAFC;color:#0F172A;font-size:14px}}

/* Nav */
.nav{{background:#fff;border-bottom:1px solid #E2E8F0;padding:0 24px;height:52px;
     display:flex;align-items:center;gap:12px;position:sticky;top:0;z-index:100;
     box-shadow:0 1px 4px rgba(0,0,0,.05)}}
.back-btn{{display:flex;align-items:center;gap:6px;font-size:13px;color:#64748B;
           text-decoration:none;padding:6px 10px;border-radius:7px;transition:all .15s}}
.back-btn:hover{{background:#F1F5F9;color:#0F172A}}
.nav-sep{{color:#E2E8F0;font-size:18px}}
.nav-kol-name{{font-size:14px;font-weight:600;color:#0F172A}}
.nav-time{{margin-left:auto;font-size:12px;color:#94A3B8}}

/* Layout */
.detail-wrap{{display:grid;grid-template-columns:340px 1fr;gap:20px;
              padding:20px 24px 40px;max-width:1400px;margin:0 auto}}

/* Profile Card */
.card{{background:#fff;border-radius:14px;border:1px solid #F1F5F9;
       box-shadow:0 1px 3px rgba(0,0,0,.05);overflow:hidden;margin-bottom:16px}}
.card-head{{padding:16px 20px;border-bottom:1px solid #F8FAFC;
            font-size:12px;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:.5px}}
.card-body{{padding:16px 20px}}

.profile-top{{display:flex;align-items:center;gap:14px;margin-bottom:16px}}
.profile-avatar{{width:52px;height:52px;border-radius:12px;flex-shrink:0;
                 display:flex;align-items:center;justify-content:center;
                 font-size:20px;font-weight:800;color:#fff;background:{av_c}}}
.profile-name{{font-size:18px;font-weight:800;color:#0F172A;margin-bottom:5px}}
.profile-badges{{display:flex;gap:6px;align-items:center;flex-wrap:wrap}}
.tier-badge{{font-size:11px;padding:2px 8px;border-radius:5px;color:#fff;
             font-weight:700;background:{tc}}}
.status-badge{{font-size:11px;padding:2px 8px;border-radius:20px;font-weight:600;
               color:{sc_fg};background:{sc_bg}}}

.info-row{{display:flex;gap:8px;margin-bottom:8px;align-items:flex-start}}
.info-lbl{{font-size:11px;color:#94A3B8;width:52px;flex-shrink:0;padding-top:2px;
           font-weight:600;text-transform:uppercase}}
.info-val{{font-size:13px;color:#374151;word-break:break-all}}
.info-link{{font-size:13px;color:#FF6B00;text-decoration:none;font-weight:500}}
.info-link:hover{{text-decoration:underline}}
.info-mono{{font-size:11px;color:#94A3B8;font-family:monospace;background:#F8FAFC;
            padding:2px 6px;border-radius:4px}}

.followup-badge{{margin-top:12px;padding:8px 12px;background:#FFF3E0;color:#FF6B00;
                 border-radius:8px;font-size:12px;font-weight:600;border:1px solid #FFD4A3}}
.followup-badge.replied{{background:#F0FDF4;color:#166534;border-color:#BBF7D0}}

/* Action buttons */
.actions{{display:flex;gap:8px;flex-wrap:wrap;margin-top:4px}}
.btn{{padding:7px 14px;border-radius:8px;font-size:12px;font-weight:600;cursor:pointer;
      border:1px solid #E2E8F0;background:#fff;color:#374151;font-family:inherit;
      transition:all .15s}}
.btn:hover{{background:#F8FAFC;border-color:#CBD5E1}}
.btn-primary{{background:#FF6B00;color:#fff;border-color:#FF6B00}}
.btn-primary:hover{{background:#E55A00}}
.btn-danger{{background:#fff;color:#EF4444;border-color:#FCA5A5}}
.btn-danger:hover{{background:#FEF2F2}}
.btn-sm{{padding:5px 10px;font-size:11px}}

/* Form */
select,input,textarea{{width:100%;padding:8px 10px;border:1px solid #E2E8F0;
  border-radius:7px;font-size:13px;font-family:inherit;outline:none;background:#fff;
  color:#0F172A}}
select:focus,input:focus,textarea:focus{{border-color:#FF6B00;box-shadow:0 0 0 3px rgba(255,107,0,.08)}}
textarea{{resize:vertical;min-height:72px}}
.form-row{{margin-bottom:12px}}
.form-lbl{{font-size:11px;font-weight:600;color:#64748B;margin-bottom:4px;
           text-transform:uppercase;letter-spacing:.3px}}
.form-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}

/* Status change */
.status-select{{width:auto;padding:6px 28px 6px 10px;font-size:12px;
                appearance:auto;display:inline-block}}

/* Modal overlay */
.modal-overlay{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.3);
                z-index:200;align-items:center;justify-content:center}}
.modal-overlay.open{{display:flex}}
.modal{{background:#fff;border-radius:14px;padding:24px;width:420px;max-width:90vw;
        box-shadow:0 8px 32px rgba(0,0,0,.12)}}
.modal-title{{font-size:16px;font-weight:700;margin-bottom:16px}}
.modal-actions{{display:flex;gap:8px;justify-content:flex-end;margin-top:16px}}

/* Timeline */
.tl-wrap{{padding:4px 0}}
.tl-item{{display:flex;gap:12px;padding:12px 0;border-bottom:1px solid #F8FAFC}}
.tl-item:last-child{{border-bottom:none}}
.tl-dot{{width:30px;height:30px;border-radius:8px;flex-shrink:0;
         display:flex;align-items:center;justify-content:center;
         font-size:14px;background:#F8FAFC}}
.tl-body{{flex:1;min-width:0}}
.tl-header{{display:flex;align-items:center;gap:6px;margin-bottom:3px;flex-wrap:wrap}}
.tl-label{{font-size:12px;font-weight:700}}
.tl-time{{font-size:11px;color:#94A3B8;margin-left:auto}}
.tl-content{{font-size:12px;color:#64748B;word-break:break-word;line-height:1.5}}

.empty-tl{{padding:40px;text-align:center;color:#CBD5E1;font-size:13px}}
</style>
</head>
<body>

<nav class="nav">
  <a class="back-btn" href="/crm">← 返回看板</a>
  <span class="nav-sep">|</span>
  <span class="nav-kol-name">{kol.get('name','')}</span>
  <div class="nav-time">{now_str}</div>
</nav>

<div class="detail-wrap">

  <!-- ══ 左侧面板 ══ -->
  <div>

    <!-- Profile -->
    <div class="card">
      <div class="card-body">
        <div class="profile-top">
          <div class="profile-avatar">{initials}</div>
          <div>
            <div class="profile-name">{kol.get('name','')}</div>
            <div class="profile-badges">
              <span class="tier-badge">{kol.get('tier','—')}</span>
              <span class="status-badge">{kol.get('status','—')}</span>
              {'<span style="font-size:11px;background:#F0FDF4;color:#15803D;padding:2px 8px;border-radius:20px;font-weight:600">● TG</span>' if kol.get('tg_status')=='added' else ''}
            </div>
          </div>
        </div>

        <div class="info-row"><span class="info-lbl">粉丝</span><span class="info-val">{_subs(kol.get('subscribers',0))}</span></div>
        <div class="info-row"><span class="info-lbl">邮箱</span><span class="info-val">{kol.get('email') or '—'}</span></div>
        {'<div class="info-row"><span class="info-lbl">Twitter</span><span class="info-val">'+kol.get('twitter','')+'</span></div>' if kol.get('twitter') else ''}
        {'<div class="info-row"><span class="info-lbl">频道</span><a class="info-link" href="'+kol.get('channel_url','')+'" target="_blank">YouTube ↗</a></div>' if kol.get('channel_url') else ''}
        <div class="info-row"><span class="info-lbl">地区</span><span class="info-val">{kol.get('country') or '—'}</span></div>
        <div class="info-row"><span class="info-lbl">来源</span><span class="info-val">{kol.get('source','')}</span></div>
        <div class="info-row"><span class="info-lbl">UTM</span><span class="info-mono">{kol.get('utm_code','—')}</span></div>
        <div class="info-row"><span class="info-lbl">收集</span><span class="info-val">{kol.get('collect_date','—')}</span></div>

        {followup_html}

        <div class="actions" style="margin-top:16px">
          <button class="btn btn-primary btn-sm" onclick="openModal('note')">+ 添加备注</button>
          <button class="btn btn-sm" onclick="openModal('status')">改状态</button>
          <button class="btn btn-sm" onclick="openModal('followup')">安排跟进</button>
        </div>
      </div>
    </div>

    <!-- 谈判详情 -->
    <div class="card">
      <div class="card-head">谈判详情</div>
      <div class="card-body">
        <form id="neg-form" onsubmit="saveNeg(event)">
          <div class="form-row">
            <div class="form-lbl">谈判阶段</div>
            <select name="stage">{stage_opts}</select>
          </div>
          <div class="form-grid">
            <div class="form-row">
              <div class="form-lbl">报价 (USD)</div>
              <input type="number" name="price_usd" placeholder="500"
                     value="{neg.get('price_usd') or ''}">
            </div>
            <div class="form-row">
              <div class="form-lbl">合作形式</div>
              <select name="content_type">{content_type_opts}</select>
            </div>
          </div>
          <div class="form-row">
            <div class="form-lbl">交付物</div>
            <input type="text" name="deliverables" placeholder="如：1条视频 10分钟"
                   value="{(neg.get('deliverables') or '').replace(chr(34), '&quot;')}">
          </div>
          <div class="form-row">
            <div class="form-lbl">备注</div>
            <textarea name="notes" placeholder="谈判进展、对方诉求…">{neg.get('notes') or ''}</textarea>
          </div>
          <button type="submit" class="btn btn-primary" style="width:100%">保存谈判详情</button>
        </form>
        <div id="neg-msg" style="margin-top:8px;font-size:12px;color:#10B981;display:none">已保存 ✓</div>
      </div>
    </div>

    <!-- 外联历史 -->
    <div class="card">
      <div class="card-head">外联记录 ({len(ctrs)} 封)</div>
      <div class="card-body" style="padding:0">
        {ctrs_html}
      </div>
    </div>

  </div>

  <!-- ══ 右侧面板：时间线 ══ -->
  <div>
    <div class="card" style="height:calc(100vh - 100px);display:flex;flex-direction:column">
      <div class="card-head" style="flex-shrink:0">
        活动时间线
        <span style="font-weight:400;color:#CBD5E1;margin-left:6px">({len(timeline)} 条记录)</span>
      </div>
      <div class="card-body" style="overflow-y:auto;flex:1;padding-top:8px">
        <div class="tl-wrap">
          {timeline_html}
        </div>
      </div>
    </div>
  </div>

</div>

<!-- Modals -->
<div class="modal-overlay" id="modal-note">
  <div class="modal">
    <div class="modal-title">添加备注</div>
    <div class="form-row">
      <textarea id="note-content" placeholder="记录谈判进展、对方反馈、下一步计划…" style="min-height:100px"></textarea>
    </div>
    <div class="modal-actions">
      <button class="btn" onclick="closeModal('note')">取消</button>
      <button class="btn btn-primary" onclick="submitNote()">保存备注</button>
    </div>
  </div>
</div>

<div class="modal-overlay" id="modal-status">
  <div class="modal">
    <div class="modal-title">变更状态</div>
    <div class="form-row">
      <div class="form-lbl">新状态</div>
      <select id="new-status">
        <option value="待发送">待发送</option>
        <option value="已发送">已发送</option>
        <option value="已回复">已回复</option>
        <option value="TG接触">TG接触</option>
        <option value="谈判中">谈判中</option>
        <option value="已签约">已签约</option>
        <option value="审核中">审核中</option>
        <option value="已发布">已发布</option>
        <option value="已完成">已完成</option>
        <option value="已拒绝">已拒绝</option>
        <option value="冷却">冷却</option>
      </select>
    </div>
    <div class="modal-actions">
      <button class="btn" onclick="closeModal('status')">取消</button>
      <button class="btn btn-primary" onclick="submitStatus()">确认变更</button>
    </div>
  </div>
</div>

<div class="modal-overlay" id="modal-followup">
  <div class="modal">
    <div class="modal-title">安排跟进</div>
    <div class="form-row">
      <div class="form-lbl">几天后跟进</div>
      <input type="number" id="followup-days" value="7" min="1" max="30">
    </div>
    <div class="modal-actions">
      <button class="btn" onclick="closeModal('followup')">取消</button>
      <button class="btn btn-primary" onclick="submitFollowup()">确认</button>
    </div>
  </div>
</div>

<script>
const KOL_ID = {kol_id};

function openModal(name) {{
  document.getElementById('modal-' + name).classList.add('open');
}}
function closeModal(name) {{
  document.getElementById('modal-' + name).classList.remove('open');
}}
document.querySelectorAll('.modal-overlay').forEach(m => {{
  m.addEventListener('click', e => {{ if (e.target === m) m.classList.remove('open'); }});
}});

async function postAction(payload) {{
  const r = await fetch('/crm/kol/' + KOL_ID + '/action', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify(payload)
  }});
  return r.json();
}}

async function submitNote() {{
  const content = document.getElementById('note-content').value.trim();
  if (!content) return;
  await postAction({{action: 'add_note', content}});
  closeModal('note');
  location.reload();
}}

async function submitStatus() {{
  const status = document.getElementById('new-status').value;
  await postAction({{action: 'change_status', status}});
  closeModal('status');
  location.reload();
}}

async function submitFollowup() {{
  const days = parseInt(document.getElementById('followup-days').value) || 7;
  await postAction({{action: 'schedule_followup', days}});
  closeModal('followup');
  location.reload();
}}

async function saveNeg(e) {{
  e.preventDefault();
  const fd = new FormData(e.target);
  const payload = {{action: 'save_negotiation'}};
  for (const [k,v] of fd.entries()) payload[k] = v;
  if (payload.price_usd) payload.price_usd = parseFloat(payload.price_usd);
  await postAction(payload);
  const msg = document.getElementById('neg-msg');
  msg.style.display = 'block';
  setTimeout(() => msg.style.display = 'none', 2000);
}}
</script>
</body></html>"""

    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/crm/pending", methods=["GET"])
def crm_pending():
    """待处理事项页 — Kelly 每日必看"""
    import sys, json as _json
    sys.path.insert(0, str(BASE / "03_kol_media"))
    try:
        from kol_db import get_pending_actions, get_pending_counts
    except Exception as e:
        return f"<pre>Import Error: {e}</pre>", 500

    try:
        actions   = get_pending_actions(status="open")
        counts    = get_pending_counts()
        now_str   = datetime.now(BJT).strftime("%Y-%m-%d %H:%M BJT")
    except Exception as e:
        return f"<pre>DB Error: {e}</pre>", 500

    def _subs(n):
        try:
            n = int(n or 0)
            if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
            if n >= 10_000:    return f"{n/10_000:.0f}万"
            return str(n)
        except: return "—"

    TIER_COLORS = {"A级":"#F59E0B","B级":"#10B981","C级":"#3B82F6","D级":"#94A3B8"}
    AVATAR_COLORS = ["#FF6B00","#3B82F6","#10B981","#8B5CF6","#F59E0B","#EC4899"]

    def _avatar_c(name):
        return AVATAR_COLORS[sum(ord(c) for c in (name or "?")) % len(AVATAR_COLORS)]

    def _kol_card_header(a):
        name = a.get("name","?")
        av   = _avatar_c(name)
        tc   = TIER_COLORS.get(a.get("tier",""), "#94A3B8")
        subs = _subs(a.get("subscribers",0))
        plat = a.get("platform","")
        tier = a.get("tier","")
        url  = a.get("channel_url","#")
        return f"""
<div class="card-header">
  <div class="avatar" style="background:{av}">{(name[:2]).upper()}</div>
  <div class="card-meta">
    <a class="kol-name" href="/crm/kol/{a['kol_id']}" target="_blank">{name}</a>
    <div class="tags">
      <span class="tag" style="background:{tc}20;color:{tc}">{tier}</span>
      <span class="tag tag-gray">{plat}</span>
      <span class="tag tag-gray">{subs} 粉丝</span>
    </div>
  </div>
</div>"""

    # ── 分类 ──────────────────────────────────────────────────────────────────
    quote_actions    = [a for a in actions if a["type"] == "quote_needed"]
    contract_actions = [a for a in actions if a["type"] == "contract_review"]
    content_actions  = [a for a in actions if a["type"] == "content_review"]

    # ── 生成各区块 HTML ───────────────────────────────────────────────────────
    def _quote_cards():
        if not quote_actions:
            return '<p class="empty-tip">暂无待报价的 KOL</p>'
        out = []
        for a in quote_actions:
            hdr   = _kol_card_header(a)
            reply = (a.get("context", {}).get("reply_snippet") or a.get("last_reply") or "")[:120]
            aid   = a["id"]
            kid   = a["kol_id"]
            ctx   = a.get("context", {})
            out.append(f"""
<div class="action-card" id="card-{aid}">
  {hdr}
  <div class="reply-snippet">"{reply or '（无回复片段）'}"</div>
  <form class="action-form" onsubmit="submitQuote(event,{aid},{kid})">
    <div class="form-row">
      <label>报价金额</label>
      <div class="input-group">
        <input type="number" name="price_usd" placeholder="0" min="0" step="1" required>
        <span class="input-suffix">USD</span>
      </div>
    </div>
    <div class="form-row">
      <label>交付物</label>
      <input type="text" name="deliverables" placeholder="例：1条YouTube视频 + 3条Twitter提及">
    </div>
    <div class="form-row">
      <label>付款条件</label>
      <input type="text" name="payment_terms" placeholder="例：50% 预付，50% 发布后">
    </div>
    <div class="form-row">
      <label>备注</label>
      <input type="text" name="notes" placeholder="可选">
    </div>
    <div class="btn-row">
      <button type="submit" class="btn-primary">确认发送报价</button>
      <button type="button" class="btn-ghost" onclick="dismissAction({aid})">暂跳过</button>
    </div>
  </form>
</div>""")
        return "".join(out)

    def _contract_cards():
        if not contract_actions:
            return '<p class="empty-tip">暂无待审核合同</p>'
        out = []
        for a in contract_actions:
            hdr  = _kol_card_header(a)
            aid  = a["id"]
            kid  = a["kol_id"]
            ctx  = a.get("context", {})
            amt  = ctx.get("total_value_usd", "—")
            dels = ctx.get("deliverables", "—")
            url  = ctx.get("contract_url", "")
            link = f'<a href="{url}" target="_blank" class="doc-link">查看合同文件</a>' if url else ""
            out.append(f"""
<div class="action-card" id="card-{aid}">
  {hdr}
  <div class="contract-info">
    <div class="info-row"><span>合同金额</span><strong>${amt}</strong></div>
    <div class="info-row"><span>交付物</span><span>{dels}</span></div>
    {link}
  </div>
  <form class="action-form" onsubmit="submitContractAction(event,{aid},{kid})">
    <div class="form-row">
      <label>审核意见（拒绝时填写）</label>
      <input type="text" name="notes" placeholder="可选">
    </div>
    <input type="hidden" name="decision" id="contract-decision-{aid}" value="">
    <div class="btn-row">
      <button type="button" class="btn-primary" onclick="setDecision('contract-decision-{aid}','approve',this.closest('form'))">通过</button>
      <button type="button" class="btn-danger"  onclick="setDecision('contract-decision-{aid}','reject',this.closest('form'))">打回</button>
      <button type="button" class="btn-ghost"   onclick="dismissAction({aid})">暂跳过</button>
    </div>
  </form>
</div>""")
        return "".join(out)

    def _content_cards():
        if not content_actions:
            return '<p class="empty-tip">暂无待审核内容草稿</p>'
        out = []
        for a in content_actions:
            hdr  = _kol_card_header(a)
            aid  = a["id"]
            kid  = a["kol_id"]
            ctx  = a.get("context", {})
            draft_url = ctx.get("draft_url","")
            plat      = ctx.get("platform","")
            link = f'<a href="{draft_url}" target="_blank" class="doc-link">查看草稿</a>' if draft_url else '<span class="empty-tip">（无链接）</span>'
            out.append(f"""
<div class="action-card" id="card-{aid}">
  {hdr}
  <div class="contract-info">
    <div class="info-row"><span>平台</span><span>{plat or "—"}</span></div>
    <div class="info-row"><span>草稿链接</span>{link}</div>
  </div>
  <form class="action-form" onsubmit="submitContentAction(event,{aid},{kid})">
    <div class="form-row">
      <label>修改意见（打回时填写）</label>
      <input type="text" name="notes" placeholder="可选">
    </div>
    <input type="hidden" name="decision" id="content-decision-{aid}" value="">
    <div class="btn-row">
      <button type="button" class="btn-primary" onclick="setDecision('content-decision-{aid}','approve',this.closest('form'))">通过</button>
      <button type="button" class="btn-danger"  onclick="setDecision('content-decision-{aid}','reject',this.closest('form'))">打回修改</button>
      <button type="button" class="btn-ghost"   onclick="dismissAction({aid})">暂跳过</button>
    </div>
  </form>
</div>""")
        return "".join(out)

    total = counts["total"]
    badge = f'<span class="badge-count">{total}</span>' if total else ""

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>待处理 — MoonX KOL CRM</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0F172A;color:#E2E8F0;min-height:100vh}}
a{{color:inherit;text-decoration:none}}
.topbar{{background:#1E293B;border-bottom:1px solid #334155;padding:0 20px;height:52px;display:flex;align-items:center;gap:16px;position:sticky;top:0;z-index:100}}
.topbar-logo{{font-weight:700;font-size:15px;color:#FF6B00;letter-spacing:.5px}}
.topbar-nav a{{font-size:13px;color:#94A3B8;padding:6px 12px;border-radius:6px;transition:.15s}}
.topbar-nav a:hover{{background:#334155;color:#E2E8F0}}
.topbar-nav a.active{{background:#FF6B0020;color:#FF6B00}}
.topbar-time{{margin-left:auto;font-size:11px;color:#64748B}}
.page{{max-width:720px;margin:0 auto;padding:24px 16px 60px}}
.page-header{{margin-bottom:28px}}
.page-title{{font-size:22px;font-weight:700;display:flex;align-items:center;gap:10px}}
.badge-count{{background:#EF4444;color:#fff;font-size:11px;font-weight:700;padding:2px 7px;border-radius:999px;min-width:22px;text-align:center}}
.page-sub{{font-size:13px;color:#64748B;margin-top:6px}}
.section{{margin-bottom:36px}}
.section-title{{font-size:13px;font-weight:600;color:#94A3B8;letter-spacing:.8px;text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;gap:8px}}
.section-count{{background:#334155;color:#E2E8F0;font-size:11px;padding:2px 8px;border-radius:999px}}
.section-count.has-items{{background:#FF6B0030;color:#FF6B00}}
.action-card{{background:#1E293B;border:1px solid #334155;border-radius:12px;padding:20px;margin-bottom:14px;transition:.2s}}
.action-card.resolving{{opacity:.5;pointer-events:none}}
.card-header{{display:flex;align-items:center;gap:12px;margin-bottom:14px}}
.avatar{{width:40px;height:40px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px;color:#fff;flex-shrink:0}}
.kol-name{{font-size:15px;font-weight:600;color:#F1F5F9}}
.kol-name:hover{{color:#FF6B00}}
.tags{{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px}}
.tag{{font-size:11px;padding:2px 8px;border-radius:999px;font-weight:500}}
.tag-gray{{background:#334155;color:#94A3B8}}
.reply-snippet{{font-size:13px;color:#94A3B8;background:#0F172A;border-left:3px solid #FF6B00;padding:10px 14px;border-radius:0 6px 6px 0;margin-bottom:14px;line-height:1.5}}
.contract-info{{background:#0F172A;border-radius:8px;padding:12px 14px;margin-bottom:14px}}
.info-row{{display:flex;justify-content:space-between;align-items:center;font-size:13px;padding:5px 0;border-bottom:1px solid #1E293B}}
.info-row:last-child{{border-bottom:none}}
.info-row span:first-child{{color:#64748B}}
.doc-link{{color:#60A5FA;font-size:13px}}
.action-form{{display:flex;flex-direction:column;gap:10px}}
.form-row{{display:flex;flex-direction:column;gap:4px}}
.form-row label{{font-size:12px;color:#64748B}}
.form-row input{{background:#0F172A;border:1px solid #334155;border-radius:8px;padding:9px 12px;color:#E2E8F0;font-size:14px;outline:none;transition:.15s}}
.form-row input:focus{{border-color:#FF6B00}}
.input-group{{display:flex;gap:0}}
.input-group input{{border-radius:8px 0 0 8px;flex:1}}
.input-suffix{{background:#334155;border:1px solid #334155;border-left:none;border-radius:0 8px 8px 0;padding:9px 12px;font-size:13px;color:#94A3B8;white-space:nowrap}}
.btn-row{{display:flex;gap:8px;flex-wrap:wrap;margin-top:4px}}
.btn-primary{{background:#FF6B00;color:#fff;border:none;border-radius:8px;padding:9px 20px;font-size:13px;font-weight:600;cursor:pointer;transition:.15s}}
.btn-primary:hover{{background:#E55F00}}
.btn-danger{{background:#EF444420;color:#EF4444;border:1px solid #EF444440;border-radius:8px;padding:9px 20px;font-size:13px;font-weight:600;cursor:pointer;transition:.15s}}
.btn-danger:hover{{background:#EF444430}}
.btn-ghost{{background:transparent;color:#64748B;border:1px solid #334155;border-radius:8px;padding:9px 16px;font-size:13px;cursor:pointer;transition:.15s}}
.btn-ghost:hover{{background:#334155;color:#E2E8F0}}
.empty-tip{{font-size:13px;color:#475569;padding:16px 0}}
.toast{{position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:#1E293B;border:1px solid #334155;border-radius:10px;padding:12px 20px;font-size:14px;color:#E2E8F0;box-shadow:0 8px 32px #00000060;z-index:9999;display:none}}
.toast.show{{display:block;animation:fadeUp .2s ease}}
@keyframes fadeUp{{from{{opacity:0;transform:translateX(-50%) translateY(8px)}}to{{opacity:1;transform:translateX(-50%) translateY(0)}}}}
@media(max-width:480px){{.btn-row{{flex-direction:column}}.btn-primary,.btn-danger,.btn-ghost{{width:100%;text-align:center}}}}
</style>
</head>
<body>
<div class="topbar">
  <span class="topbar-logo">MoonX KOL</span>
  <nav class="topbar-nav" style="display:flex;gap:4px">
    <a href="/crm/pending" class="active">待处理 {badge}</a>
    <a href="/crm">流水线</a>
    <a href="/crm/kols">KOL 列表</a>
    <a href="/crm/finance">财务</a>
    <a href="/crm/logs">日志</a>
  </nav>
  <span class="topbar-time">{now_str}</span>
</div>

<div class="page">
  <div class="page-header">
    <div class="page-title">待处理 {badge}</div>
    <div class="page-sub">需要你做决定的事项，处理完自动流转到下一步</div>
  </div>

  <div class="section">
    <div class="section-title">
      💰 报价待确认
      <span class="section-count {'has-items' if counts['quote_needed'] else ''}">{counts['quote_needed']}</span>
    </div>
    {_quote_cards()}
  </div>

  <div class="section">
    <div class="section-title">
      📄 合同待审核
      <span class="section-count {'has-items' if counts['contract_review'] else ''}">{counts['contract_review']}</span>
    </div>
    {_contract_cards()}
  </div>

  <div class="section">
    <div class="section-title">
      🎬 内容草稿待审核
      <span class="section-count {'has-items' if counts['content_review'] else ''}">{counts['content_review']}</span>
    </div>
    {_content_cards()}
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
function showToast(msg, ok=true) {{
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.borderColor = ok ? '#22C55E' : '#EF4444';
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2800);
}}

function resolveCard(actionId) {{
  const card = document.getElementById('card-' + actionId);
  if (card) {{
    card.classList.add('resolving');
    setTimeout(() => card.remove(), 600);
  }}
  // 更新顶栏徽章
  const b = document.querySelector('.badge-count');
  if (b) {{
    const n = parseInt(b.textContent) - 1;
    if (n <= 0) b.remove();
    else b.textContent = n;
  }}
}}

async function apiPost(url, data) {{
  const r = await fetch(url, {{
    method: 'POST',
    headers: {{'Content-Type':'application/json'}},
    body: JSON.stringify(data)
  }});
  return r.json();
}}

async function submitQuote(e, actionId, kolId) {{
  e.preventDefault();
  const f = e.target;
  const price = parseFloat(f.price_usd.value);
  if (!price || price <= 0) {{ showToast('请填写报价金额', false); return; }}
  const res = await apiPost('/crm/kol/' + kolId + '/pending-action', {{
    action_id: actionId,
    action_type: 'quote_needed',
    data: {{
      price_usd: price,
      deliverables: f.deliverables.value,
      payment_terms: f.payment_terms.value,
      notes: f.notes.value
    }}
  }});
  if (res.ok) {{ resolveCard(actionId); showToast('报价已发送给 KOL'); }}
  else if (res.warn) {{ showToast('⚠️ ' + res.error, false); }}
  else showToast('操作失败: ' + res.error, false);
}}

function setDecision(inputId, decision, form) {{
  document.getElementById(inputId).value = decision;
  form.requestSubmit();
}}

async function submitContractAction(e, actionId, kolId) {{
  e.preventDefault();
  const f = e.target;
  const decision = f.decision.value;
  if (!decision) {{ showToast('请点击通过或打回', false); return; }}
  const res = await apiPost('/crm/kol/' + kolId + '/pending-action', {{
    action_id: actionId,
    action_type: 'contract_review',
    data: {{ decision, notes: f.notes.value }}
  }});
  if (res.ok) {{ resolveCard(actionId); showToast(decision === 'approve' ? '合同已通过' : '已打回，备注已记录'); }}
  else showToast('操作失败: ' + res.error, false);
}}

async function submitContentAction(e, actionId, kolId) {{
  e.preventDefault();
  const f = e.target;
  const decision = f.decision.value;
  if (!decision) {{ showToast('请点击通过或打回修改', false); return; }}
  const res = await apiPost('/crm/kol/' + kolId + '/pending-action', {{
    action_id: actionId,
    action_type: 'content_review',
    data: {{ decision, notes: f.notes.value }}
  }});
  if (res.ok) {{ resolveCard(actionId); showToast(decision === 'approve' ? '内容已通过' : '已打回，修改意见已记录'); }}
  else showToast('操作失败: ' + res.error, false);
}}

async function dismissAction(actionId) {{
  const res = await apiPost('/crm/kol/0/pending-action', {{
    action_id: actionId,
    action_type: 'dismiss',
    data: {{}}
  }});
  if (res.ok) resolveCard(actionId);
}}
</script>
</body></html>"""

    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/crm/kol/<int:kol_id>/pending-action", methods=["POST"])
def crm_kol_pending_action(kol_id: int):
    """处理待办事项 API — 报价确认 / 合同审核 / 内容审核 / 暂跳过"""
    import sys, json as _json
    sys.path.insert(0, str(BASE / "03_kol_media"))
    try:
        from kol_db import (resolve_pending_action, upsert_negotiation,
                             change_kol_status, add_kol_note, log_activity)
    except Exception as e:
        return _json.dumps({"ok": False, "error": str(e)}), 500, {"Content-Type": "application/json"}

    payload     = request.get_json(silent=True) or {}
    action_id   = payload.get("action_id")
    action_type = payload.get("action_type", "")
    data        = payload.get("data", {})

    try:
        if action_type == "dismiss":
            resolve_pending_action(action_id, resolved_by="dismissed")

        elif action_type == "quote_needed":
            price = data.get("price_usd")
            if not price:
                return _json.dumps({"ok": False, "error": "price_usd required"}), 400, {"Content-Type": "application/json"}
            price_f       = float(price)
            deliverables  = data.get("deliverables", "")
            payment_terms = data.get("payment_terms", "")
            notes         = data.get("notes", "")
            # 获取 KOL 邮箱（用于发送报价邮件）
            from kol_db import get_kol_detail
            detail      = get_kol_detail(kol_id) or {}
            kol_info    = detail.get("kol", {})
            to_email    = kol_info.get("email", "")
            kol_name    = kol_info.get("name", "KOL")
            # 写入谈判记录
            upsert_negotiation(kol_id, {
                "stage":         "price_sent",
                "price_usd":     price_f,
                "deliverables":  deliverables,
                "payment_terms": payment_terms,
                "notes":         notes,
            })
            # 推进状态到谈判中
            change_kol_status(kol_id, "谈判中", operator="kelly")
            # 发送报价邮件（先发后 resolve，失败则不 resolve）
            ok_mail, err_mail = send_quote_email(
                to_email, kol_name, price_f, deliverables, payment_terms, notes
            )
            log_activity(kol_id,
                         "quote_sent" if ok_mail else "quote_email_failed",
                         f"报价 ${price_f:,.0f} USD | 交付物: {deliverables}"
                         + ("" if ok_mail else f" | SMTP错误: {err_mail}"),
                         operator="kelly")
            if not ok_mail:
                # 邮件未发出，不 resolve，前端提示错误
                return _json.dumps({
                    "ok": False,
                    "error": f"谈判记录已保存，但报价邮件发送失败：{err_mail}。请手动发送。",
                    "warn": True,
                }), 200, {"Content-Type": "application/json"}
            resolve_pending_action(action_id)

        elif action_type == "contract_review":
            decision = data.get("decision", "")
            notes    = data.get("notes", "")
            if decision == "approve":
                change_kol_status(kol_id, "已签约", operator="kelly")
                log_activity(kol_id, "contract_approved", notes or "合同已通过", operator="kelly")
            else:
                change_kol_status(kol_id, "已终止", operator="kelly")
                log_activity(kol_id, "contract_rejected", notes or "合同已打回", operator="kelly")
                if notes:
                    add_kol_note(kol_id, f"合同打回意见: {notes}", operator="kelly")
            resolve_pending_action(action_id)

        elif action_type == "content_review":
            decision = data.get("decision", "")
            notes    = data.get("notes", "")
            if decision == "approve":
                change_kol_status(kol_id, "已发布", operator="kelly")
                log_activity(kol_id, "content_approved", notes or "内容草稿已通过", operator="kelly")
            else:
                change_kol_status(kol_id, "内容修改中", operator="kelly")
                log_activity(kol_id, "content_rejected", notes or "内容草稿已打回", operator="kelly")
                if notes:
                    add_kol_note(kol_id, f"内容修改意见: {notes}", operator="kelly")
            resolve_pending_action(action_id)

        else:
            return _json.dumps({"ok": False, "error": "unknown action_type"}), 400, {"Content-Type": "application/json"}

        return _json.dumps({"ok": True}), 200, {"Content-Type": "application/json"}

    except Exception as e:
        return _json.dumps({"ok": False, "error": str(e)}), 500, {"Content-Type": "application/json"}


@app.route("/crm/kol/<int:kol_id>/activities", methods=["GET"])
def crm_kol_activities(kol_id: int):
    """返回 KOL 最近活动（供抽屉时间轴使用）"""
    import sys, json as _json
    sys.path.insert(0, str(BASE / "03_kol_media"))
    try:
        from kol_db import get_db
        with get_db(_kol_db_path()) as conn:
            rows = conn.execute(
                "SELECT type, content, operator, created_at FROM activities "
                "WHERE kol_id=? ORDER BY created_at DESC LIMIT 10",
                (kol_id,)
            ).fetchall()
        acts = [dict(r) for r in rows]
        return _json.dumps({"activities": acts}, ensure_ascii=False), 200, {"Content-Type": "application/json"}
    except Exception as e:
        return _json.dumps({"activities": [], "error": str(e)}), 200, {"Content-Type": "application/json"}


@app.route("/crm/kol/<int:kol_id>/tg-accepted", methods=["POST"])
def crm_kol_tg_accepted(kol_id: int):
    """KOL 通过 TG 接受报价 → 创建合同审核待处理事项"""
    import sys, json as _json
    sys.path.insert(0, str(BASE / "03_kol_media"))
    try:
        from kol_db import (create_pending_action, log_activity,
                            find_kol_by_name, get_kol_detail)
    except Exception as e:
        return _json.dumps({"ok": False, "error": str(e)}), 500, {"Content-Type": "application/json"}
    try:
        detail   = get_kol_detail(kol_id)
        if not detail:
            return _json.dumps({"ok": False, "error": "KOL not found"}), 404, {"Content-Type": "application/json"}
        kol_name = detail["kol"]["name"]
        action_id = create_pending_action(kol_id, "contract_review", {
            "note": "KOL 已通过 TG 接受报价，等待合同审核"
        })
        log_activity(kol_id, "tg_accepted",
                     "KOL 接受报价，合同审核待处理事项已创建", operator="kelly")
        notify_lark_kol(kol_name, "contract_review", "KOL 已接受报价，请审核合同")
        return _json.dumps({"ok": True, "action_id": action_id}), 200, {"Content-Type": "application/json"}
    except Exception as e:
        return _json.dumps({"ok": False, "error": str(e)}), 500, {"Content-Type": "application/json"}


@app.route("/crm/kol/<int:kol_id>/action", methods=["POST"])
def crm_kol_action(kol_id: int):
    """KOL 操作 API（JSON）"""
    import sys, json as _json
    sys.path.insert(0, str(BASE / "03_kol_media"))
    try:
        from kol_db import add_kol_note, change_kol_status, schedule_followup, upsert_negotiation
    except Exception as e:
        return _json.dumps({"ok": False, "error": str(e)}), 500, {"Content-Type": "application/json"}

    payload = request.get_json(silent=True) or {}
    action  = payload.get("action", "")

    try:
        if action == "add_note":
            add_kol_note(kol_id, payload.get("content",""), operator="kelly")
        elif action == "change_status":
            change_kol_status(kol_id, payload.get("status",""), operator="kelly")
        elif action == "schedule_followup":
            schedule_followup(kol_id, days=int(payload.get("days", 7)))
        elif action == "save_negotiation":
            upsert_negotiation(kol_id, {
                "stage":        payload.get("stage"),
                "price_usd":    payload.get("price_usd"),
                "content_type": payload.get("content_type"),
                "deliverables": payload.get("deliverables"),
                "notes":        payload.get("notes"),
            })
        else:
            return _json.dumps({"ok": False, "error": "unknown action"}), 400, {"Content-Type": "application/json"}
        return _json.dumps({"ok": True}), 200, {"Content-Type": "application/json"}
    except Exception as e:
        return _json.dumps({"ok": False, "error": str(e)}), 500, {"Content-Type": "application/json"}


def _crm_topbar(active: str, counts: dict = None) -> str:
    """通用顶部导航栏（所有 CRM 页面共用）"""
    counts = counts or {}
    total  = counts.get("total", 0)
    badge  = f' <span style="background:#FF6B00;color:#fff;border-radius:10px;padding:1px 6px;font-size:11px">{total}</span>' if total else ""
    links  = [
        ("待处理", "/crm/pending", badge),
        ("流水线", "/crm", ""),
        ("KOL 列表", "/crm/kols", ""),
        ("财务", "/crm/finance", ""),
        ("日志", "/crm/logs", ""),
    ]
    parts = []
    for label, href, extra in links:
        cls = ' class="active"' if active == href else ""
        parts.append(f'<a href="{href}"{cls}>{label}{extra}</a>')
    nav = "\n    ".join(parts)
    now = datetime.now(BJT).strftime("%Y-%m-%d %H:%M BJT")
    return f"""<div class="topbar">
  <span class="topbar-logo">MoonX KOL</span>
  <nav class="topbar-nav" style="display:flex;gap:4px">
    {nav}
  </nav>
  <span class="topbar-time">{now}</span>
</div>"""


_CRM_NAV_CSS = """
.topbar{background:#1E293B;border-bottom:1px solid #334155;padding:0 20px;height:52px;display:flex;align-items:center;gap:16px;position:sticky;top:0;z-index:100}
.topbar-logo{font-weight:700;font-size:15px;color:#FF6B00;letter-spacing:.5px}
.topbar-nav a{font-size:13px;color:#94A3B8;padding:6px 12px;border-radius:6px;transition:.15s;text-decoration:none}
.topbar-nav a:hover{background:#334155;color:#E2E8F0}
.topbar-nav a.active{background:#FF6B0020;color:#FF6B00}
.topbar-time{margin-left:auto;font-size:11px;color:#64748B}
"""


@app.route("/crm/kols", methods=["GET"])
def crm_kols():
    """KOL 列表页 — 表格+筛选+右侧抽屉"""
    import sys, json as _json
    sys.path.insert(0, str(BASE / "03_kol_media"))
    try:
        from kol_db import get_pending_counts
        counts = get_pending_counts()
    except Exception:
        counts = {}

    try:
        import sqlite3
        db_path = "/data/kol_crm.db" if _ON_FLY else str(BASE / "03_kol_media" / "kol_crm.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        kols = conn.execute("""
            SELECT k.id, k.name, k.platform, k.status, k.tier, k.subscribers,
                   k.email, k.tg_handle, k.channel_url,
                   (SELECT MAX(a.created_at) FROM activities a WHERE a.kol_id=k.id) as last_activity
            FROM kols k ORDER BY k.updated_at DESC
        """).fetchall()
        kols = [dict(r) for r in kols]
        conn.close()
    except Exception as e:
        return f"<pre>DB Error: {e}</pre>", 500

    STATUS_COLORS = {
        "待发送": "#64748B", "已发送": "#3B82F6", "已回复": "#8B5CF6",
        "TG接触": "#F59E0B", "谈判中": "#FF6B00", "已签约": "#10B981",
        "已发布": "#22C55E", "已终止": "#EF4444",
    }
    TIER_COLORS = {"A级": "#F59E0B", "B级": "#10B981", "C级": "#3B82F6", "D级": "#94A3B8"}
    AVATAR_COLORS = ["#FF6B00","#3B82F6","#10B981","#8B5CF6","#F59E0B","#EC4899"]

    def _avatar_c(name):
        return AVATAR_COLORS[sum(ord(c) for c in (name or "?")) % len(AVATAR_COLORS)]

    def _subs(n):
        try:
            n = int(n or 0)
            if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
            if n >= 10_000:    return f"{n//10_000}万"
            return str(n) if n else "—"
        except: return "—"

    def _days_ago(ts):
        if not ts: return "—"
        try:
            from datetime import date
            d = datetime.strptime(ts[:10], "%Y-%m-%d").date()
            diff = (date.today() - d).days
            if diff == 0: return "今天"
            if diff == 1: return "昨天"
            return f"{diff}天前"
        except: return "—"

    rows_html = []
    for k in kols:
        name   = k.get("name") or "?"
        status = k.get("status") or "未知"
        tier   = k.get("tier") or ""
        plat   = k.get("platform") or "—"
        sc     = STATUS_COLORS.get(status, "#64748B")
        tc     = TIER_COLORS.get(tier, "#94A3B8")
        av     = _avatar_c(name)
        subs   = _subs(k.get("subscribers"))
        last   = _days_ago(k.get("last_activity"))
        kid    = k["id"]
        rows_html.append(f"""<tr data-id="{kid}" data-status="{status}" data-platform="{plat}" onclick="openDrawer({kid})" style="cursor:pointer">
  <td><div style="display:flex;align-items:center;gap:10px">
    <div style="width:32px;height:32px;border-radius:50%;background:{av};display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;color:#fff;flex-shrink:0">{name[:2].upper()}</div>
    <div>
      <div style="font-weight:600;color:#E2E8F0;font-size:13px">{name}</div>
      <div style="font-size:11px;color:#64748B">{plat}</div>
    </div>
  </div></td>
  <td><span style="background:{sc}20;color:{sc};padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600">{status}</span></td>
  <td><span style="background:{tc}20;color:{tc};padding:2px 8px;border-radius:12px;font-size:11px">{tier or "—"}</span></td>
  <td style="color:#94A3B8;font-size:12px">{subs}</td>
  <td style="color:#64748B;font-size:12px">{last}</td>
</tr>""")

    all_statuses = sorted(set(k.get("status") or "未知" for k in kols))
    all_platforms = sorted(set(k.get("platform") or "" for k in kols if k.get("platform")))
    status_opts  = "".join(f'<option value="{s}">{s}</option>' for s in all_statuses)
    platform_opts = "".join(f'<option value="{p}">{p}</option>' for p in all_platforms)

    kols_json = _json.dumps([{
        "id": k["id"], "name": k.get("name",""), "status": k.get("status",""),
        "platform": k.get("platform",""), "tier": k.get("tier",""),
        "subscribers": k.get("subscribers",0), "email": k.get("email",""),
        "tg_handle": k.get("tg_handle",""), "channel_url": k.get("channel_url",""),
    } for k in kols], ensure_ascii=False)

    topbar = _crm_topbar("/crm/kols", counts)

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KOL 列表 — MoonX</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0F172A;color:#E2E8F0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-height:100vh}}
{_CRM_NAV_CSS}
.page{{padding:24px;max-width:1200px;margin:0 auto}}
.filters{{display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap;align-items:center}}
.filter-input{{background:#1E293B;border:1px solid #334155;border-radius:8px;color:#E2E8F0;padding:8px 12px;font-size:13px;outline:none}}
.filter-input:focus{{border-color:#FF6B00}}
select.filter-input option{{background:#1E293B}}
.count-badge{{color:#64748B;font-size:13px;margin-left:auto}}
table{{width:100%;border-collapse:collapse}}
thead th{{color:#64748B;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;padding:8px 12px;text-align:left;border-bottom:1px solid #1E293B}}
tbody tr{{border-bottom:1px solid #1E293B20;transition:.1s}}
tbody tr:hover{{background:#1E293B}}
td{{padding:12px}}
.drawer-overlay{{position:fixed;top:0;left:0;right:0;bottom:0;background:#00000060;z-index:200;display:none}}
.drawer{{position:fixed;top:0;right:0;width:420px;height:100vh;background:#1E293B;border-left:1px solid #334155;z-index:201;overflow-y:auto;transform:translateX(100%);transition:.25s ease}}
.drawer.open{{transform:translateX(0)}}
.drawer-header{{padding:20px;border-bottom:1px solid #334155;display:flex;justify-content:space-between;align-items:center}}
.drawer-close{{background:none;border:none;color:#94A3B8;font-size:20px;cursor:pointer;padding:4px}}
.drawer-body{{padding:20px}}
.drawer-section{{margin-bottom:20px}}
.drawer-label{{font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}}
.drawer-value{{font-size:13px;color:#E2E8F0}}
.btn-tg-accepted{{background:#FF6B00;color:#fff;border:none;border-radius:8px;padding:8px 16px;font-size:13px;cursor:pointer;width:100%;margin-top:12px}}
.btn-tg-accepted:hover{{background:#e55c00}}
.btn-detail{{display:block;text-align:center;background:#1E293B;border:1px solid #334155;color:#94A3B8;border-radius:8px;padding:8px 16px;font-size:13px;cursor:pointer;text-decoration:none;margin-top:8px}}
.btn-detail:hover{{border-color:#FF6B00;color:#FF6B00}}
</style>
</head>
<body>
{topbar}
<div class="page">
  <div class="filters">
    <input class="filter-input" id="searchInput" placeholder="搜索 KOL 名称..." oninput="filterTable()" style="width:220px">
    <select class="filter-input" id="statusFilter" onchange="filterTable()">
      <option value="">全部状态</option>
      {status_opts}
    </select>
    <select class="filter-input" id="platformFilter" onchange="filterTable()">
      <option value="">全部平台</option>
      {platform_opts}
    </select>
    <span class="count-badge" id="countBadge">共 {len(kols)} 个 KOL</span>
  </div>
  <table id="kolTable">
    <thead>
      <tr>
        <th>KOL</th>
        <th>状态</th>
        <th>级别</th>
        <th>粉丝数</th>
        <th>最近动作</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows_html)}
    </tbody>
  </table>
</div>

<div class="drawer-overlay" id="overlay" onclick="closeDrawer()"></div>
<div class="drawer" id="drawer">
  <div class="drawer-header">
    <span id="drawerName" style="font-weight:700;font-size:15px;color:#E2E8F0"></span>
    <button class="drawer-close" onclick="closeDrawer()">✕</button>
  </div>
  <div class="drawer-body" id="drawerBody"></div>
</div>

<script>
const KOLS = {kols_json};
const kolMap = Object.fromEntries(KOLS.map(k => [k.id, k]));

function filterTable() {{
  const q      = document.getElementById('searchInput').value.toLowerCase();
  const status = document.getElementById('statusFilter').value;
  const plat   = document.getElementById('platformFilter').value;
  const rows   = document.querySelectorAll('#kolTable tbody tr');
  let visible  = 0;
  rows.forEach(tr => {{
    const name = (tr.querySelector('td div div')?.textContent || '').toLowerCase();
    const s    = tr.dataset.status;
    const p    = tr.dataset.platform;
    const show = (!q || name.includes(q)) && (!status || s === status) && (!plat || p === plat);
    tr.style.display = show ? '' : 'none';
    if (show) visible++;
  }});
  document.getElementById('countBadge').textContent = `显示 ${{visible}} / {len(kols)} 个 KOL`;
}}

async function openDrawer(kolId) {{
  const k = kolMap[kolId];
  if (!k) return;
  document.getElementById('drawerName').textContent = k.name;
  const tgBtn = k.status === '谈判中' ? `<button class="btn-tg-accepted" onclick="tgAccepted(${{kolId}})">✓ KOL 已通过 TG 接受报价</button>` : '';
  document.getElementById('drawerBody').innerHTML = `
    <div class="drawer-section">
      <div class="drawer-label">平台</div>
      <div class="drawer-value">${{k.platform || '—'}}</div>
    </div>
    <div class="drawer-section">
      <div class="drawer-label">状态</div>
      <div class="drawer-value">${{k.status || '—'}}</div>
    </div>
    <div class="drawer-section">
      <div class="drawer-label">粉丝数</div>
      <div class="drawer-value">${{k.subscribers ? Number(k.subscribers).toLocaleString() : '—'}}</div>
    </div>
    <div class="drawer-section">
      <div class="drawer-label">邮箱</div>
      <div class="drawer-value">${{k.email || '—'}}</div>
    </div>
    ${{tgBtn}}
    <div class="drawer-section" id="drawer-timeline-${{kolId}}">
      <div class="drawer-label">历史动作</div>
      <div style="color:#64748B;font-size:12px">加载中…</div>
    </div>
    <a class="btn-detail" href="/crm/kol/${{kolId}}" target="_blank">查看完整详情 →</a>
  `;
  document.getElementById('overlay').style.display = 'block';
  document.getElementById('drawer').classList.add('open');

  // 异步加载活动时间轴
  try {{
    const res = await fetch(`/crm/kol/${{kolId}}/activities`);
    const data = await res.json();
    const tlEl = document.getElementById(`drawer-timeline-${{kolId}}`);
    if (!tlEl) return;
    if (!data.activities || data.activities.length === 0) {{
      tlEl.querySelector('div:last-child').textContent = '暂无记录';
      return;
    }}
    const icons = {{
      email_sent: '📧', reply_received: '💬', tg_contacted: '📱',
      quote_sent: '💰', contract_approved: '✅', contract_rejected: '❌',
      content_approved: '🎬', content_rejected: '🔄', status_changed: '🔀',
    }};
    const html = data.activities.slice(0, 8).map(a => {{
      const icon = icons[a.type] || '•';
      const dt = a.created_at ? a.created_at.slice(0,16).replace('T',' ') : '';
      return `<div style="display:flex;gap:8px;padding:6px 0;border-bottom:1px solid #1E293B">
        <span style="font-size:14px;flex-shrink:0">${{icon}}</span>
        <div>
          <div style="font-size:12px;color:#CBD5E1">${{a.content || a.type}}</div>
          <div style="font-size:11px;color:#475569">${{dt}}</div>
        </div>
      </div>`;
    }}).join('');
    tlEl.innerHTML = `<div class="drawer-label">历史动作</div>${{html}}`;
  }} catch(e) {{
    const tlEl = document.getElementById(`drawer-timeline-${{kolId}}`);
    if (tlEl) tlEl.querySelector('div:last-child').textContent = '加载失败';
  }}
}}

function closeDrawer() {{
  document.getElementById('drawer').classList.remove('open');
  document.getElementById('overlay').style.display = 'none';
}}

async function tgAccepted(kolId) {{
  const res = await fetch('/crm/kol/' + kolId + '/tg-accepted', {{method:'POST'}});
  const data = await res.json();
  if (data.ok) {{
    alert('已创建合同审核待处理事项，Lark 已通知');
    closeDrawer();
  }} else {{
    alert('操作失败: ' + data.error);
  }}
}}
</script>
</body>
</html>"""


@app.route("/crm/finance", methods=["GET"])
def crm_finance():
    """财务页 — 本季度 $20k 预算用量"""
    import sys
    sys.path.insert(0, str(BASE / "03_kol_media"))
    try:
        from kol_db import get_pending_counts
        counts = get_pending_counts()
    except Exception:
        counts = {}

    QUARTERLY_BUDGET = 20_000.0
    try:
        import sqlite3
        from datetime import date
        db_path = "/data/kol_crm.db" if _ON_FLY else str(BASE / "03_kol_media" / "kol_crm.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        today = date.today()
        q_start_month = ((today.month - 1) // 3) * 3 + 1
        q_start = date(today.year, q_start_month, 1).strftime("%Y-%m-%d")
        payments = conn.execute("""
            SELECT p.id, p.amount_usd, p.status, p.due_date, p.paid_at, p.notes,
                   k.name as kol_name
            FROM payments p JOIN kols k ON k.id=p.kol_id
            WHERE p.due_date >= ? OR p.paid_at >= ?
            ORDER BY p.due_date DESC
        """, (q_start, q_start)).fetchall()
        payments = [dict(r) for r in payments]
        conn.close()
    except Exception as e:
        return f"<pre>DB Error: {e}</pre>", 500

    spent   = sum(p["amount_usd"] or 0 for p in payments if p["status"] == "paid")
    pending = sum(p["amount_usd"] or 0 for p in payments if p["status"] == "pending")
    pct     = min(spent / QUARTERLY_BUDGET * 100, 100)
    over    = spent > QUARTERLY_BUDGET
    bar_color = "#EF4444" if over else ("#F59E0B" if pct > 80 else "#FF6B00")

    rows_html = []
    for p in payments:
        s = p.get("status","")
        sc = {"paid": "#10B981", "pending": "#F59E0B"}.get(s, "#64748B")
        rows_html.append(f"""<tr>
  <td style="color:#E2E8F0;font-size:13px">{p.get('kol_name','—')}</td>
  <td style="color:#E2E8F0;font-size:13px">${p.get('amount_usd',0):,.0f}</td>
  <td><span style="background:{sc}20;color:{sc};padding:2px 8px;border-radius:10px;font-size:11px">{s}</span></td>
  <td style="color:#64748B;font-size:12px">{p.get('due_date','—')}</td>
  <td style="color:#64748B;font-size:12px">{p.get('notes','') or '—'}</td>
</tr>""")

    no_data = '<tr><td colspan="5" style="text-align:center;color:#64748B;padding:24px">本季度暂无付款记录</td></tr>' if not payments else ""
    topbar = _crm_topbar("/crm/finance", counts)

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>财务 — MoonX KOL</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0F172A;color:#E2E8F0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-height:100vh}}
{_CRM_NAV_CSS}
.page{{padding:24px;max-width:900px;margin:0 auto}}
.budget-card{{background:#1E293B;border-radius:12px;padding:24px;margin-bottom:24px;border:1px solid #334155}}
.budget-title{{font-size:13px;color:#64748B;margin-bottom:4px}}
.budget-amount{{font-size:32px;font-weight:700;color:#E2E8F0;margin-bottom:16px}}
.budget-sub{{font-size:13px;color:#94A3B8}}
.progress-bar{{height:12px;background:#334155;border-radius:6px;overflow:hidden;margin:12px 0}}
.progress-fill{{height:100%;border-radius:6px;transition:.5s}}
.budget-meta{{display:flex;gap:24px;margin-top:16px;flex-wrap:wrap}}
.meta-item{{text-align:center}}
.meta-label{{font-size:11px;color:#64748B}}
.meta-value{{font-size:18px;font-weight:700;color:#E2E8F0;margin-top:2px}}
.section-title{{font-size:13px;font-weight:600;color:#94A3B8;text-transform:uppercase;letter-spacing:.5px;margin-bottom:12px}}
table{{width:100%;border-collapse:collapse;background:#1E293B;border-radius:12px;overflow:hidden}}
thead th{{color:#64748B;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;padding:10px 16px;text-align:left;border-bottom:1px solid #334155}}
tbody tr{{border-bottom:1px solid #334155}}
tbody tr:last-child{{border-bottom:none}}
td{{padding:12px 16px}}
.over-warning{{color:#EF4444;font-weight:700;margin-top:8px;font-size:13px}}
</style>
</head>
<body>
{topbar}
<div class="page">
  <div class="budget-card">
    <div class="budget-title">本季度 KOL 合作预算</div>
    <div class="budget-amount">${spent:,.0f} <span style="font-size:18px;color:#64748B">/ $20,000</span></div>
    <div class="progress-bar"><div class="progress-fill" style="width:{pct:.1f}%;background:{bar_color}"></div></div>
    <div class="budget-sub">已用 {pct:.1f}%{'  ⚠️ 超支！' if over else ''}</div>
    {'<div class="over-warning">⚠️ 已超过季度预算 $' + f'{spent-QUARTERLY_BUDGET:,.0f}' + '</div>' if over else ''}
    <div class="budget-meta">
      <div class="meta-item">
        <div class="meta-label">已付款</div>
        <div class="meta-value" style="color:#10B981">${spent:,.0f}</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">待付款</div>
        <div class="meta-value" style="color:#F59E0B">${pending:,.0f}</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">剩余预算</div>
        <div class="meta-value" style="color:{'#EF4444' if over else '#E2E8F0'}">${max(QUARTERLY_BUDGET-spent,0):,.0f}</div>
      </div>
    </div>
  </div>

  <div class="section-title">本季度付款明细</div>
  <table>
    <thead>
      <tr>
        <th>KOL</th>
        <th>金额 (USD)</th>
        <th>状态</th>
        <th>到期日</th>
        <th>备注</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows_html) or no_data}
    </tbody>
  </table>
</div>
</body>
</html>"""


@app.route("/crm/logs", methods=["GET"])
def crm_logs():
    """自动化日志页 — activities 表可视化"""
    import sys
    sys.path.insert(0, str(BASE / "03_kol_media"))
    try:
        from kol_db import get_pending_counts
        counts = get_pending_counts()
    except Exception:
        counts = {}

    try:
        import sqlite3
        db_path = "/data/kol_crm.db" if _ON_FLY else str(BASE / "03_kol_media" / "kol_crm.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        logs = conn.execute("""
            SELECT a.id, a.type, a.content, a.operator, a.created_at,
                   k.name as kol_name
            FROM activities a
            LEFT JOIN kols k ON k.id = a.kol_id
            ORDER BY a.created_at DESC
            LIMIT 200
        """).fetchall()
        logs = [dict(r) for r in logs]
        conn.close()
    except Exception as e:
        return f"<pre>DB Error: {e}</pre>", 500

    TYPE_LABELS = {
        "email_sent":        ("📧", "#3B82F6",  "外联邮件"),
        "reply_detected":    ("📨", "#8B5CF6",  "邮件回复"),
        "intent_classified": ("🔍", "#64748B",  "意图分类"),
        "tg_invited":        ("💬", "#F59E0B",  "TG 邀请"),
        "tg_contact":        ("💬", "#F59E0B",  "TG 接触"),
        "tg_accepted":       ("✅", "#10B981",  "TG 接受报价"),
        "quote_sent":        ("💰", "#FF6B00",  "报价已发"),
        "quote_email_failed":("❌", "#EF4444",  "报价邮件失败"),
        "contract_approved": ("📄", "#10B981",  "合同批准"),
        "contract_rejected": ("📄", "#EF4444",  "合同打回"),
        "content_approved":  ("🎬", "#10B981",  "内容通过"),
        "content_rejected":  ("🎬", "#EF4444",  "内容打回"),
        "content_published": ("🚀", "#22C55E",  "内容发布"),
        "note":              ("📝", "#94A3B8",  "备注"),
    }

    rows_html = []
    for log in logs:
        t     = log.get("type","")
        icon, color, label = TYPE_LABELS.get(t, ("•", "#64748B", t))
        ts    = (log.get("created_at") or "")[:16].replace("T", " ")
        kname = log.get("kol_name") or "—"
        cont  = (log.get("content") or "")[:100]
        op    = log.get("operator") or "auto"
        rows_html.append(f"""<tr>
  <td style="color:#64748B;font-size:11px;white-space:nowrap">{ts}</td>
  <td><span style="background:{color}20;color:{color};padding:2px 8px;border-radius:10px;font-size:11px;white-space:nowrap">{icon} {label}</span></td>
  <td style="color:#E2E8F0;font-size:12px">{kname}</td>
  <td style="color:#94A3B8;font-size:12px">{cont}</td>
  <td style="color:#64748B;font-size:11px">{op}</td>
</tr>""")

    no_data = '<tr><td colspan="5" style="text-align:center;color:#64748B;padding:24px">暂无日志记录</td></tr>'
    topbar = _crm_topbar("/crm/logs", counts)

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>自动化日志 — MoonX KOL</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0F172A;color:#E2E8F0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-height:100vh}}
{_CRM_NAV_CSS}
.page{{padding:24px;max-width:1100px;margin:0 auto}}
.page-header{{margin-bottom:20px}}
.page-title{{font-size:20px;font-weight:700;color:#E2E8F0}}
.page-sub{{font-size:13px;color:#64748B;margin-top:4px}}
table{{width:100%;border-collapse:collapse;background:#1E293B;border-radius:12px;overflow:hidden}}
thead th{{color:#64748B;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;padding:10px 16px;text-align:left;border-bottom:1px solid #334155}}
tbody tr{{border-bottom:1px solid #334155}}
tbody tr:last-child{{border-bottom:none}}
td{{padding:10px 16px}}
</style>
</head>
<body>
{topbar}
<div class="page">
  <div class="page-header">
    <div class="page-title">自动化日志</div>
    <div class="page-sub">系统最近 200 条操作记录（最新在上）</div>
  </div>
  <table>
    <thead>
      <tr>
        <th>时间</th>
        <th>类型</th>
        <th>KOL</th>
        <th>内容</th>
        <th>操作人</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows_html) or no_data}
    </tbody>
  </table>
</div>
</body>
</html>"""


@app.route("/lark/event", methods=["POST"])
def lark_event():
    body = request.get_json(silent=True) or {}

    if "challenge" in body:
        return json.dumps({"challenge": body["challenge"]})

    event_id = body.get("header", {}).get("event_id", "")
    if event_id:
        if event_id in processed_events:
            return json.dumps({"code": 0})
        processed_events.add(event_id)
        if len(processed_events) > 1000:
            processed_events.clear()

    event = body.get("event", {})
    msg   = event.get("message", {})
    if not msg:
        return json.dumps({"code": 0})

    try:
        content = json.loads(msg.get("content", "{}"))
        text = content.get("text", "").strip()
        text = re.sub(r"@\S+\s*", "", text).strip()
    except Exception:
        return json.dumps({"code": 0})

    if text:
        threading.Thread(target=parse_and_dispatch, args=(text,), daemon=True).start()

    return json.dumps({"code": 0})


if __name__ == "__main__":
    # ── 启动自主调度器（Layer 2）────────────────────────────────────────────
    import scheduler as _sched
    _sched.init(send_lark)
    _active_scheduler = _sched.setup()

    port = int(os.getenv("PORT", 8080))
    print(f"🚀 Lark Server v2 启动，端口 {port}")
    app.run(host="0.0.0.0", port=port)
