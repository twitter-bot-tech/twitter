#!/usr/bin/env python3
"""
MoonX 自主调度器 — Layer 2 + 3 + 4

Layer 2: 自动执行脚本（推文 / TG / KOL收集 / 日报）
Layer 3: 员工自思考 + 自工作（晨报 / 工作产出）
Layer 4: 员工自主相互讨论 + 闭环执行（讨论→决策→执行→复盘→循环）
"""
import os
import sys
import json
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import date, datetime

import claude_cli as anthropic
import requests as _requests
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

BASE         = Path(__file__).parent
TZ           = "Asia/Shanghai"
CTX_FILE     = BASE / "loop_context.json"    # 跨天上下文（决策 + 复盘结果）
LEARNING_FILE = BASE / "team_learning.json"  # 自学习：每日自动更新的技能改进

_send_lark = None   # 由 lark_server.py 注入


def init(send_lark_fn):
    global _send_lark
    _send_lark = send_lark_fn


def _notify(key: str, msg: str):
    if _send_lark:
        try:
            _send_lark(key, msg)
        except Exception:
            pass


# ── 跨天上下文（讨论→决策→复盘→明日晨报引用）────────────────────────────────
def _load_loop_ctx() -> dict:
    if CTX_FILE.exists():
        try:
            return json.loads(CTX_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_loop_ctx(data: dict):
    try:
        CTX_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        pass


# ── 员工配置 ──────────────────────────────────────────────────────────────────
EMPLOYEES = [
    {
        "key": "social", "emoji": "📱", "name": "员工1号｜社媒运营",
        "system": """你是 MoonX 社媒运营，见过 Solana/Coinbase 从0到百万粉的操盘手。
核心信念：每条推文都是一场微型战役，粉丝是赢来的不是攒来的。
你主动工作，不等被安排。发言直接，引用具体数据和案例。""",
        "work_prompt": """产出今日社媒工作成果：
1. 今日推文内容日历（3条，写出具体钩子开头，不超过20字/条）
2. 今日要评论的1个竞品或大V（@handle + 评论角度）
3. 本周待做的1件提升互动率的事
格式清晰，可直接执行。不超过200字。""",
    },
    {
        "key": "seo", "emoji": "🔍", "name": "员工2号｜SEO专家",
        "system": """你是 MoonX SEO专家，帮 CoinGecko 做到月均3000万自然流量。
核心信念：不做孤立文章，做内容集群。每篇必须攻打一个明确关键词。
你主动工作，不等被安排。发言引用具体关键词和数据。""",
        "work_prompt": """产出今日 SEO 工作成果：
1. 本周攻打的核心关键词（1个）+ 当前排名预估 + 竞品差距
2. 文章选题（标题 + 副标题 + 3句话大纲）
3. 需要内链的2篇相关文章方向
格式清晰，可直接交给写手执行。不超过250字。""",
    },
    {
        "key": "kol", "emoji": "🤝", "name": "员工3号｜KOL媒体",
        "system": """你是 MoonX KOL & 媒体负责人，帮 Coinbase 上市前把故事塞进 WSJ 和 FT。
核心信念：媒体不是广告位，是信任转移机器。
你主动工作，不等被安排。发言引用具体 KOL 名字和媒体。""",
        "work_prompt": """产出今日 KOL & 媒体工作成果：
1. 今日重点跟进的1个 KOL（名字 + 粉丝量 + 合作切入点）
2. 一封外联 DM 草稿（50字以内，英文，直接可发）
3. 本周的1个媒体报道角度（故事线 + 目标媒体名）
格式清晰，可直接执行。不超过200字。""",
    },
    {
        "key": "growth", "emoji": "📈", "name": "员工4号｜增长运营",
        "system": """你是 MoonX 增长运营，设计过 Uniswap空投/Blur积分等百万人参与激励机制。
核心信念：不做活动，设计增长飞轮。好的机制会自己跑。
你主动工作，不等被安排。发言引用具体案例和数字。""",
        "work_prompt": """产出今日增长工作成果：
1. 本周增长实验（实验名 + 核心假设 + 衡量指标）
2. 一个可3天内上线的裂变机制（具体玩法 + 预期拉新数）
3. 当前用户留存最大漏斗（哪个环节流失 + 修复建议）
格式清晰，可直接评估落地。不超过200字。""",
    },
    {
        "key": "strategy", "emoji": "📊", "name": "员工5号｜策略数据",
        "system": """你是 MoonX 策略分析师，a16z crypto 研究团队风格。
核心信念：不写报告，生产决策弹药。所有结论必须有数字支撑。
你主动工作，不等被安排。发言引用具体竞品数据和市场数字。""",
        "work_prompt": """产出今日策略工作成果：
1. 本周最重要的竞品动态（Polymarket/Kalshi/Manifold，1条具体事件 + 影响判断）
2. MoonX 当前最大的增长瓶颈（数据支撑 + 根本原因）
3. 给 Kelly 的一个选择题（A vs B，附数字依据）
格式清晰，是决策弹药不是报告。不超过200字。""",
    },
]

LEAD_SYSTEM = """你是 MoonX Team Lead，融合了 He Yi（币安CMO）和 Coinbase IPO营销负责人的思维。
你管理结果，不管理过程。语气果断，不废话，给选择题不给问答题。"""


# ── 自学习：读写改进记录 ──────────────────────────────────────────────────────
def _load_learning() -> dict:
    if LEARNING_FILE.exists():
        try:
            return json.loads(LEARNING_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_learning(data: dict):
    try:
        LEARNING_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        pass


def _build_system(emp: dict) -> str:
    """在员工基础 system prompt 上叠加自学习改进层"""
    base = emp["system"]
    learning = _load_learning()
    learned = learning.get("employees", {}).get(emp["key"], "")
    if learned:
        base += f"\n\n【自我升级 — 上次改进点】\n{learned}"
    return base


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def _ask(system: str, prompt: str, max_tokens: int = 400) -> str:
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        return f"（AI生成失败: {e}）"


def _ask_medium(system: str, prompt: str, max_tokens: int = 400) -> str:
    """Lead 综合/结论/分配 — Sonnet，推理质量更高"""
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        return f"（分析失败: {e}）"


def _ask_deep(system: str, prompt: str) -> str:
    """员工5专用 — Sonnet 深度分析"""
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        return f"（深度分析失败: {e}）"


def _fetch_market_ctx() -> str:
    """拉取全量实时数据（Polymarket + Twitter + CoinGecko + GSC + 神策）"""
    try:
        from data_feeds import format_context
        return format_context()
    except Exception as e:
        # 降级：只拉 Polymarket
        try:
            resp = _requests.get(
                "https://gamma-api.polymarket.com/markets",
                params={"closed": "false", "order": "volume24hr", "ascending": "false", "limit": 3},
                timeout=10,
            )
            if resp.status_code == 200:
                lines = []
                for m in resp.json()[:3]:
                    vol = m.get("volume24hr", 0) or 0
                    try:
                        prices = json.loads(m.get("outcomePrices", "[0.5]"))
                        yes = round(float(prices[0]) * 100, 1)
                    except Exception:
                        yes = 50
                    lines.append(f"- {m.get('question','')} | YES {yes}% | ${vol:,.0f}")
                return "\n".join(lines)
        except Exception:
            pass
        return f"（数据获取失败: {e}）"


def _today() -> str:
    return date.today().strftime("%m/%d")


def _fmt_speeches(speeches: list) -> str:
    return "\n".join(f"【{s['name']}】{s['content']}" for s in speeches)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3 — 员工自思考 + 自工作
# ═══════════════════════════════════════════════════════════════════════════════

def morning_standup():
    """09:30 BJT — 5员工各自群发晨报，引用昨日复盘结果"""
    today  = _today()
    market = _fetch_market_ctx()
    ctx    = _load_loop_ctx()

    yesterday_summary = ctx.get("yesterday_results", "（首次运行，暂无昨日数据）")
    yesterday_decision = ctx.get("yesterday_decision", "")

    standups = []
    print(f"[{today} 09:30] 🌅 员工晨报开始（并行）...")

    history_ctx = ""
    if yesterday_decision:
        history_ctx = f"\n昨日团队决策：{yesterday_decision}\n昨日执行结果：{yesterday_summary}\n"

    def _standup_worker(emp):
        prompt = f"""今天是 {today}，北京时间 09:30。
{history_ctx}
当前预测市场热门数据：
{market}

作为 {emp['name']}，发布今日晨报：
1. 基于市场数据和昨日结果，今天最关注什么
2. 今天要完成的最重要一件事（具体可执行）
3. 给其他部门一个建议或问题

不超过100字。语气主动直接，引用具体数字或案例。"""
        return emp, _ask(_build_system(emp), prompt)

    with ThreadPoolExecutor(max_workers=5) as ex:
        future_map = {ex.submit(_standup_worker, emp): emp for emp in EMPLOYEES}
        results = {future_map[f]["key"]: f.result()[1] for f in as_completed(future_map)}

    for emp in EMPLOYEES:
        response = results[emp["key"]]
        _notify(emp["key"], f"{emp['emoji']} {emp['name']}｜晨报 {today}\n\n{response}")
        standups.append({"name": emp["name"], "content": response})

    # 存储晨报供讨论用
    ctx["today_standups"] = standups
    ctx["today_market"]   = market
    _save_loop_ctx(ctx)

    print(f"[{today} 09:30] ✅ 晨报完成，15分钟后触发自主讨论")


def employee_work_output():
    """10:30 BJT — 5员工并行产出今日工作成果"""
    today  = _today()
    market = _fetch_market_ctx()
    print(f"[{today} 10:30] 💼 员工工作产出开始（并行）...")

    def _work_worker(emp):
        prompt = f"""今天是 {today}。当前市场数据：\n{market}\n\n{emp['work_prompt']}"""
        if emp["key"] == "strategy":
            return emp, _ask_deep(_build_system(emp), prompt)
        return emp, _ask(_build_system(emp), prompt, max_tokens=500)

    with ThreadPoolExecutor(max_workers=5) as ex:
        future_map = {ex.submit(_work_worker, emp): emp for emp in EMPLOYEES}
        results = {future_map[f]["key"]: f.result()[1] for f in as_completed(future_map)}

    for emp in EMPLOYEES:
        _notify(emp["key"], f"{emp['emoji']} {emp['name']}｜今日产出 {today}\n\n{results[emp['key']]}")
        print(f"   ✅ {emp['name']}")

    print(f"[{today} 10:30] ✅ 5个部门工作产出完成")


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 4 — 员工自主相互讨论 + 闭环执行
# ═══════════════════════════════════════════════════════════════════════════════

def auto_discussion():
    """09:45 BJT — 基于晨报自动发起讨论（所有员工发到 Lead 群，形成可见讨论流）"""
    today = _today()
    ctx   = _load_loop_ctx()

    standups = ctx.get("today_standups", [])
    market   = ctx.get("today_market", _fetch_market_ctx())

    print(f"[{today} 09:45] 💬 自主讨论开始...")

    # Step 1: Lead 从晨报中提炼今日讨论议题
    if standups:
        standup_text = _fmt_speeches(standups)
        topic_prompt = f"""基于以下晨报，提炼今日最重要的1个团队讨论议题（一句话，聚焦最大分歧或机会）：\n{standup_text}"""
        topic = _ask_medium(LEAD_SYSTEM, topic_prompt, max_tokens=60)
    else:
        topic = f"MoonX 本周最高优先级增长动作"

    _notify("lead", f"👔 Team Lead｜发起今日讨论 {today}\n\n━━ 议题 ━━\n{topic}\n\n请各部门发言 👇")
    time.sleep(2)

    # Step 2: 员工依次在 Lead 群发言，每人读取前面所有发言再回应
    speeches = []
    for emp in EMPLOYEES:
        prev_ctx = _fmt_speeches(speeches) if speeches else "（你是第一个发言）"

        prompt = f"""议题：{topic}

市场背景：
{market}

其他人已发言：
{prev_ctx}

作为 {emp['name']}，从你的职责角度：
- 对这个议题的核心观点（可以反驳或补充其他人）
- 你认为最应该优先做的一个具体动作

不超过80字。直接说观点，不说"我认为"开头的废话。"""

        response = _ask(_build_system(emp), prompt, max_tokens=200)

        # 发到 Lead 群（所有人都发这里，形成讨论流）
        _notify("lead", f"{emp['emoji']} {emp['name']}\n\n{response}")
        speeches.append({"name": emp["name"], "content": response})
        time.sleep(3)

    # Step 3: Lead 综合，给出 A/B 决策 + 推荐方案
    all_speeches = _fmt_speeches(speeches)
    conclusion_prompt = f"""议题：{topic}

讨论记录：
{all_speeches}

给出：
1. 一句话核心结论（提炼分歧点）
2. 方案A：[具体方案]
3. 方案B：[具体方案]
4. 你的推荐（A或B）+ 理由（一句话）

格式固定，语气果断。"""

    conclusion = _ask_medium(LEAD_SYSTEM, conclusion_prompt, max_tokens=300)

    # 提取推荐方案（默认A）
    recommended = "A" if "推荐A" in conclusion or "建议A" in conclusion or "选A" in conclusion else "B"
    if "推荐B" in conclusion or "建议B" in conclusion or "选B" in conclusion:
        recommended = "B"

    final_msg = f"""👔 Team Lead｜讨论结论 {today}

{conclusion}

━━━━━━━━━━━━━━━
💡 Kelly，请回复「我选A」或「我选B」
⏰ 2小时内无选择，自动执行推荐方案{recommended}"""

    _notify("lead", final_msg)

    # 存储讨论结果（供自动闭环使用）
    # Parse A/B options from conclusion text
    import re as _re
    m_a = _re.search(r'方案\s*A[：:]\s*(.+?)(?=\n.*方案\s*B|\n.*推荐|\n.*建议|$)', conclusion, _re.DOTALL)
    m_b = _re.search(r'方案\s*B[：:]\s*(.+?)(?=\n.*推荐|\n.*建议|\n\d|$)', conclusion, _re.DOTALL)
    option_a = m_a.group(1).strip()[:150] if m_a else ""
    option_b = m_b.group(1).strip()[:150] if m_b else ""

    ctx["today_topic"]       = topic
    ctx["today_discussion"]  = speeches
    ctx["today_conclusion"]  = conclusion
    ctx["today_option_a"]    = option_a
    ctx["today_option_b"]    = option_b
    ctx["today_recommended"] = recommended
    ctx["today_decided"]     = False   # Kelly 还没选
    ctx["discussion_time"]   = datetime.now().isoformat()
    _save_loop_ctx(ctx)

    print(f"[{today} 09:45] ✅ 讨论完成，推荐方案: {recommended}，等待 Kelly 2小时")


def auto_execute_if_no_decision():
    """11:45 BJT — Kelly 若未选择，自动执行讨论推荐方案（闭环执行）"""
    today = _today()
    ctx   = _load_loop_ctx()

    # 如果 Kelly 已经手动选了，跳过
    if ctx.get("today_decided"):
        print(f"[{today} 11:45] Kelly 已选择，跳过自动执行")
        return

    recommended = ctx.get("today_recommended", "A")
    topic       = ctx.get("today_topic", "今日议题")
    conclusion  = ctx.get("today_conclusion", "")

    _notify("lead", f"""👔 Team Lead｜自动执行 {today}

⏰ Kelly 2小时内未选择
自动执行推荐方案 {recommended}

━━ 开始分配任务... ━━""")

    time.sleep(1)
    _assign_and_execute(recommended, topic, conclusion, today, ctx)


def _assign_and_execute(choice: str, topic: str, conclusion: str, today: str, ctx: dict):
    """分配任务给各部门并执行（供手动选择 + 自动执行共用）"""
    assign_prompt = f"""Kelly 选择了方案 {choice}。
议题：{topic}
结论：{conclusion}

向全员分配具体任务，格式：
━━ 任务分配 ━━
• 员工1 社媒：[具体任务，一句话]
• 员工2 SEO：[具体任务，一句话]
• 员工3 KOL：[具体任务，一句话]
• 员工4 增长：[具体任务，一句话]
• 员工5 策略：[具体任务，一句话]
━━ 截止：今日 18:00 ━━
语气果断，每条任务要有可衡量的产出。"""

    assignment = _ask_medium(LEAD_SYSTEM, assign_prompt, max_tokens=400)
    _notify("lead", f"👔 Team Lead｜任务分配 {today}\n\n{assignment}")
    time.sleep(1.5)

    # 各员工并行承接任务
    def _accept_worker(emp):
        accept_prompt = f"""Kelly 选择了方案{choice}，议题：{topic}。
任务分配：{assignment}
你是 {emp['name']}，承接你的部分并说明今天具体怎么执行，不超过60字。"""
        return emp, _ask(_build_system(emp), accept_prompt, max_tokens=150)

    with ThreadPoolExecutor(max_workers=5) as ex:
        future_map = {ex.submit(_accept_worker, emp): emp for emp in EMPLOYEES}
        accept_results = {future_map[f]["key"]: f.result()[1] for f in as_completed(future_map)}

    for emp in EMPLOYEES:
        _notify(emp["key"], f"{emp['emoji']} {emp['name']}｜接收任务 {today}\n\n✅ {accept_results[emp['key']]}")

    ctx["today_decided"] = True
    ctx["today_choice"]  = choice
    ctx["today_assignment"] = assignment
    _save_loop_ctx(ctx)
    _notify("lead", f"👔 Team Lead\n\n✅ 全员已接收任务，今日 18:00 前完成\n\n发「状态」随时查看进度")


def mark_decision(choice: str):
    """被 lark_server.py 调用：Kelly 手动选择后，标记已决策并执行"""
    today = _today()
    ctx   = _load_loop_ctx()

    if ctx.get("today_decided"):
        return  # 已经决策过了

    topic      = ctx.get("today_topic", "今日议题")
    conclusion = ctx.get("today_conclusion", "")
    _assign_and_execute(choice, topic, conclusion, today, ctx)


def evening_review():
    """17:30 BJT — 员工复盘今日结果，发到 Lead 群，形成闭环数据"""
    today = _today()
    ctx   = _load_loop_ctx()

    topic      = ctx.get("today_topic", "今日工作")
    choice     = ctx.get("today_choice", ctx.get("today_recommended", ""))
    assignment = ctx.get("today_assignment", "")

    print(f"[{today} 17:30] 🌆 员工复盘开始...")
    _notify("lead", f"👔 Team Lead｜收工复盘 {today}\n\n各部门汇报今日成果 👇")
    time.sleep(1)

    reviews = []

    def _review_worker(emp):
        prompt = f"""今天是 {today}，北京时间 17:30，即将收工。
今日议题：{topic}
执行方案：{choice}
你的任务：{assignment}

作为 {emp['name']}，汇报今日复盘：
1. 今天实际完成了什么（具体数据/产出）
2. 遇到的最大障碍（如果有）
3. 明天最重要的一件事

不超过80字。如实汇报，有数字就用数字。"""
        return emp, _ask(_build_system(emp), prompt, max_tokens=200)

    with ThreadPoolExecutor(max_workers=5) as ex:
        future_map = {ex.submit(_review_worker, emp): emp for emp in EMPLOYEES}
        review_results = {future_map[f]["key"]: f.result()[1] for f in as_completed(future_map)}

    for emp in EMPLOYEES:
        response = review_results[emp["key"]]
        _notify("lead", f"{emp['emoji']} {emp['name']}｜复盘\n\n{response}")
        reviews.append({"name": emp["name"], "content": response})

    # Lead 综合复盘（Sonnet）
    all_reviews = _fmt_speeches(reviews)
    summary_prompt = f"""今日工作复盘：
{all_reviews}

给出今日总结：
1. 整体完成情况（一句话）
2. 最大亮点（具体）
3. 明日最高优先级（一件事）
4. 给 Kelly 的一个信号（需要关注的风险或机会）

不超过100字，果断直接。"""

    summary = _ask_medium(LEAD_SYSTEM, summary_prompt, max_tokens=300)
    _notify("lead", f"👔 Team Lead｜今日总结 {today}\n\n{summary}")

    # 存储今日复盘（18:30日报读取）+ 明日晨报引用
    ctx["today_review_summary"] = summary
    ctx["today_review_detail"]  = all_reviews
    ctx["yesterday_results"]    = summary        # 摘要，不存全量 all_reviews
    ctx["yesterday_decision"]   = f"方案{choice}（{topic}）"
    ctx["yesterday_summary"]    = summary
    # 清空今日临时数据
    ctx.pop("today_standups", None)
    ctx.pop("today_discussion", None)
    ctx.pop("today_decided", None)
    _save_loop_ctx(ctx)

    print(f"[{today} 17:30] ✅ 复盘完成，结果已存储供明日晨报引用")

    # 触发自我升级
    self_improve(today, ctx, all_reviews, summary)


def self_improve(today: str, ctx: dict, reviews: str, summary: str):
    """复盘结束后，Lead 自动为每个员工生成明日改进点，写入 team_learning.json"""
    print(f"[{today} 17:30] 🧠 自我升级分析中...")

    discussion  = _fmt_speeches(ctx.get("today_discussion", []))
    choice      = ctx.get("today_choice", ctx.get("today_recommended", ""))
    topic       = ctx.get("today_topic", "")
    kelly_prefs = _load_learning().get("kelly_preferences", "")

    learning = _load_learning()
    prev_improvements = learning.get("employees", {})

    # Lead 并行为每个员工生成改进点（Haiku 够用，5条简短建议）
    def _improve_worker(emp):
        improve_prompt = f"""今日议题：{topic}
Kelly 选择了方案：{choice}
讨论记录摘要：{discussion[:800]}
复盘总结：{reviews[:600]}
{emp['name']} 之前的改进点：{prev_improvements.get(emp['key'], '无')}

作为 Team Lead，为 {emp['name']} 生成明日工作的1~2条具体改进方向：
- 基于今日讨论，哪种切入角度更受 Kelly 青睐？
- 这个员工应该在哪里说得更有力？
不超过60字，直接写改进点，不要废话。"""
        return emp["key"], _ask(LEAD_SYSTEM, improve_prompt, max_tokens=150)

    # 同时并行提炼 Kelly 偏好（Sonnet，质量更重要）
    pref_prompt = f"""今日 Kelly 的决策行为：
议题：{topic}，选择：方案{choice}
结论：{summary[:300]}
之前已知偏好：{kelly_prefs}

补充或更新 Kelly 的决策偏好（她倾向于什么类型的方案？什么样的论据能打动她？）
不超过80字。"""

    with ThreadPoolExecutor(max_workers=6) as ex:
        improve_futures = {ex.submit(_improve_worker, emp): emp for emp in EMPLOYEES}
        pref_future = ex.submit(_ask_medium, LEAD_SYSTEM, pref_prompt, 150)
        employee_improvements = {}
        for f in as_completed(improve_futures):
            key, improvement = f.result()
            employee_improvements[key] = improvement
            print(f"   🔧 改进点已生成: {key}")
        kelly_pref_new = pref_future.result()

    # 保存
    learning["employees"]         = employee_improvements
    learning["kelly_preferences"] = kelly_pref_new
    learning["last_updated"]      = today
    _save_learning(learning)

    _notify("lead", f"👔 Team Lead｜自我升级完成 {today}\n\n🧠 已为5个员工生成明日改进点\n📌 Kelly 决策偏好已更新\n\n明日员工将以升级后的能力工作")
    print(f"[{today} 17:30] ✅ 自我升级完成，team_learning.json 已更新")


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2 — 脚本执行任务
# ═══════════════════════════════════════════════════════════════════════════════

def _run_once(script: str, args: list, timeout: int = 300) -> tuple:
    """执行一次脚本，返回 (success: bool, output: str)"""
    try:
        result = subprocess.run(
            [sys.executable, script] + args,
            capture_output=True, text=True, timeout=timeout,
            cwd=str(Path(script).parent),
            env={**os.environ},
        )
        out = (result.stdout or result.stderr or "").strip()
        return result.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, f"超时（>{timeout//60}分钟）"
    except Exception as e:
        return False, str(e)


def run_job(script: str, args: list, job_name: str, lark_key: str, _retry: bool = False, timeout: int = 300):
    exec_time = datetime.now().strftime("%H:%M")

    # s07 依赖检查：前置任务未成功则跳过
    met, missing = _deps_met(job_name)
    if not met:
        msg = f"⏭️ {job_name} 跳过\n\n前置任务未完成：{', '.join(missing)}"
        _notify(lark_key, msg)
        _record_execution(job_name, "⏭️ 跳过（依赖未满足）", str(missing), exec_time)
        print(f"[{exec_time}] {job_name} 跳过，缺依赖: {missing}")
        return

    success, out = _run_once(script, args, timeout=timeout)

    if success:
        lines = [l for l in out.split("\n") if l.strip()]
        last  = lines[-1][:100] if lines else "完成"
        _notify(lark_key, f"✅ {job_name}\n\n{last}")
        _record_execution(job_name, "✅ 成功", last, exec_time)
        # s07 触发下游任务
        _trigger_downstream(job_name)
        # s11 发 follow-up 任务到队列
        _emit_followups(job_name, out)
    else:
        err = out[-200:].strip()
        if not _retry:
            # 首次失败：15 分钟后自动重试一次
            _notify(lark_key, f"⚠️ {job_name} 异常，15分钟后自动重试\n\n{err[-150:]}")
            _record_execution(job_name, "⚠️ 异常（等待重试）", err[-80:], exec_time)
            import threading
            threading.Timer(900, run_job, args=[script, args, job_name, lark_key], kwargs={"_retry": True, "timeout": timeout}).start()
        else:
            # 重试仍失败：最终通知
            _notify(lark_key, f"❌ {job_name} 重试失败\n\n{err[-150:]}")
            _record_execution(job_name, "❌ 重试失败", err[-80:], exec_time)


def _record_execution(job_name: str, status: str, detail: str, exec_time: str):
    """将脚本执行结果写入 loop_context，供复盘和日报引用"""
    ctx = _load_loop_ctx()
    if "today_executions" not in ctx:
        ctx["today_executions"] = []
    ctx["today_executions"].append({
        "job": job_name, "status": status,
        "detail": detail, "time": exec_time,
    })
    _save_loop_ctx(ctx)


# ═══════════════════════════════════════════════════════════════════════════════
# s07 — 任务依赖图
# ═══════════════════════════════════════════════════════════════════════════════

# 下游任务 → 必须先成功的上游任务
TASK_DEPS: dict = {
    "📊 KOL 数据同步 Google Sheets": ["🐦 KOL 每日收集 Twitter/X"],
    "🔗 Kalshi KOL Twitter补全":     ["🎯 Kalshi KOL 专项收集"],
}

# 上游任务完成后 → 自动触发的下游任务 (job_name → (script, args, lark_key))
TASK_TRIGGERS: dict = {}   # 在脚本变量定义后填充（见下方 _init_triggers）


def _deps_met(job_name: str) -> tuple:
    """检查依赖是否已全部成功，返回 (met: bool, missing: list)"""
    deps = TASK_DEPS.get(job_name, [])
    if not deps:
        return True, []
    ctx = _load_loop_ctx()
    done = {e["job"] for e in ctx.get("today_executions", []) if e["status"].startswith("✅")}
    missing = [d for d in deps if d not in done]
    return len(missing) == 0, missing


def _trigger_downstream(job_name: str):
    """上游任务成功后，自动把下游任务加入队列"""
    for downstream, (script, args, lark_key, timeout) in TASK_TRIGGERS.items():
        if job_name in TASK_DEPS.get(downstream, []):
            # 所有依赖都已满足，立即触发
            met, missing = _deps_met(downstream)
            if met:
                import threading as _t
                _t.Thread(
                    target=run_job,
                    args=[script, args, downstream, lark_key],
                    kwargs={"timeout": timeout},
                    daemon=True,
                ).start()
                print(f"   → 自动触发下游: {downstream}")


# ═══════════════════════════════════════════════════════════════════════════════
# s11 — 自主任务队列
# ═══════════════════════════════════════════════════════════════════════════════

TASK_QUEUE_FILE = BASE / "task_queue.json"


def _load_queue() -> list:
    if TASK_QUEUE_FILE.exists():
        try:
            return json.loads(TASK_QUEUE_FILE.read_text())
        except Exception:
            pass
    return []


def _save_queue(q: list):
    TASK_QUEUE_FILE.write_text(json.dumps(q, ensure_ascii=False, indent=2))


def add_task(title: str, owner: str, priority: int = 5, context: str = "", source: str = ""):
    """向任务队列添加待认领任务。priority: 1=高 3=中 5=低"""
    q = _load_queue()
    # 去重：同 title + 同 owner 的 pending 任务不重复添加
    for t in q:
        if t["title"] == title and t["owner"] == owner and t["status"] == "pending":
            return
    q.append({
        "id":         datetime.now().strftime("%Y%m%d%H%M%S%f")[:17],
        "title":      title,
        "owner":      owner,
        "priority":   priority,
        "context":    context,
        "source":     source,
        "status":     "pending",
        "created_at": datetime.now().isoformat(),
    })
    q.sort(key=lambda x: x["priority"])
    _save_queue(q)
    print(f"   [queue] +任务: [{owner}] {title}")


def scan_and_claim_tasks():
    """每 15 分钟扫描一次：自主认领 pending 任务，发 Lark 通知给对应 Agent"""
    q = _load_queue()
    if not q:
        return
    pending = [t for t in q if t["status"] == "pending"]
    if not pending:
        return

    claimed_count = 0
    for task in q:
        if task["status"] != "pending":
            continue
        task["status"] = "claimed"
        task["claimed_at"] = datetime.now().isoformat()
        claimed_count += 1

        owner = task["owner"]
        pri_label = "🔴 高" if task["priority"] <= 2 else "🟡 中" if task["priority"] <= 4 else "🟢 低"
        ctx_line  = f"\n背景：{task['context']}" if task["context"] else ""
        src_line  = f"\n来源：{task['source']}" if task["source"] else ""
        _notify(owner, f"🤖 自主认领任务\n\n📌 {task['title']}\n优先级：{pri_label}{ctx_line}{src_line}")

    if claimed_count:
        _save_queue(q)
        print(f"[scan_queue] 认领 {claimed_count} 个任务")


# ── 脚本执行后触发的 follow-up 任务（在 _init_triggers 中定义）────────────────
TASK_FOLLOW_UPS: dict = {}   # job_name → [(title, owner, priority, context)]


def _emit_followups(job_name: str, output: str):
    """脚本成功后，自动向队列发 follow-up 任务"""
    for title, owner, priority, context in TASK_FOLLOW_UPS.get(job_name, []):
        add_task(title, owner, priority, context, source=job_name)


TWEET_SCRIPT    = str(BASE / "01_social_media" / "tweet_bot.py")
TG_SCRIPT       = str(BASE / "01_social_media" / "tg_daily_poster.py")
REPORTER_SCRIPT = str(BASE / "lark_reporter.py")
OKR_SCRIPT      = str(BASE / "lark_okr_push.py")
KOL_SCRIPT      = str(BASE / "03_kol_media" / "collect_kols_youtube.py")
KALSHI_SCRIPT   = str(BASE / "03_kol_media" / "collect_kalshi_kols.py")
KALSHI_ENRICH   = str(BASE / "03_kol_media" / "enrich_kalshi_twitter.py")
SEND_SCRIPT     = str(BASE / "03_kol_media" / "scheduled_send.py")
WEB3_KOL_SCRIPT = str(BASE / "03_kol_media" / "web3_kol_scraper.py")
SYNC_SHEETS_SCRIPT = str(BASE / "03_kol_media" / "sync_kol_to_sheets.py")
MEDIA_COLLECT_SCRIPT  = str(BASE / "03_kol_media" / "collect_media.py")
MEDIA_SEND_SCRIPT     = str(BASE / "03_kol_media" / "send_media_inquiry.py")
REPLY_CHECK_SCRIPT    = str(BASE / "03_kol_media" / "check_replies.py")
CLASSIFY_REPLY_SCRIPT = str(BASE / "03_kol_media" / "classify_reply.py")
FOLLOWUP_TRACKER_SCRIPT  = str(BASE / "03_kol_media" / "followup_tracker.py")
CONTENT_BRIEF_SCRIPT     = str(BASE / "03_kol_media" / "content_brief.py")
DEADLINE_REMINDER_SCRIPT = str(BASE / "03_kol_media" / "deadline_reminder.py")
ROI_TRACKER_SCRIPT       = str(BASE / "03_kol_media" / "roi_tracker.py")

def _init_triggers():
    """在脚本路径变量定义完成后，填充 TASK_TRIGGERS 和 TASK_FOLLOW_UPS"""
    # s07 依赖图：上游成功 → 自动触发下游（下游从 SCRIPT_JOBS 移除，改为事件驱动）
    TASK_TRIGGERS["📊 KOL 数据同步 Google Sheets"] = (SYNC_SHEETS_SCRIPT, [], "kol", 300)
    TASK_TRIGGERS["🔗 Kalshi KOL Twitter补全"]     = (KALSHI_ENRICH,      [], "kol", 300)

    # s11 follow-up 任务：脚本完成后自动向队列添加待认领任务
    TASK_FOLLOW_UPS["🐦 KOL 每日收集 Twitter/X"] = [
        ("审查今日 Web3 KOL 收集结果，标记高潜力名单", "kol", 2,
         "web3_kol_scraper.py 已完成，进入 Google Sheets 审查今日新增 KOL，打优先级标签"),
    ]
    TASK_FOLLOW_UPS["🤝 KOL 每日收集 YouTube"] = [
        ("筛选今日 YouTube KOL，更新外联优先级", "kol", 3,
         "collect_kols_youtube.py 已完成，对比昨日新增，识别值得今日外联的频道"),
    ]
    TASK_FOLLOW_UPS["📊 09:00 团队日报"] = [
        ("查看今日日报，确认各部门优先级是否对齐", "lead", 1,
         "lark_reporter.py 日报已发出，请确认数据异常项并做优先级调整"),
    ]

_init_triggers()


# 注意：SYNC_SHEETS_SCRIPT 和 KALSHI_ENRICH 已从 SCRIPT_JOBS 移除
# 改为由各自上游任务成功后通过 TASK_TRIGGERS 自动触发（s07 依赖图）
SCRIPT_JOBS = [
    (8,  0,  TG_SCRIPT,       ["morning"],       "📱 TG 08:00 早报",              "social"),
    (10, 0,  KOL_SCRIPT,            [],            "🤝 KOL 每日收集 YouTube",        "kol"),
    (10, 15, WEB3_KOL_SCRIPT,      [],            "🐦 KOL 每日收集 Twitter/X",      "kol"),
    (10, 30, KALSHI_SCRIPT,        [],            "🎯 Kalshi KOL 专项收集",         "kol"),
    (9,  0,  REPLY_CHECK_SCRIPT,    [],            "📬 KOL 回复检测",                "kol"),
    (9,  15, CLASSIFY_REPLY_SCRIPT,[],            "🤖 KOL 回复意图分类",            "kol"),
    (9,  30, FOLLOWUP_TRACKER_SCRIPT,  [],        "🔁 KOL TG跟进追踪",             "kol"),
    (9,  45, DEADLINE_REMINDER_SCRIPT, [],      "⏰ KOL 截止提醒+催稿",          "kol"),
    (10, 5,  CONTENT_BRIEF_SCRIPT,   [],        "📄 KOL Content Brief 自动发送", "kol"),
    (10, 15, ROI_TRACKER_SCRIPT,     [],        "📊 KOL ROI+付款+复购",          "kol"),
    (10, 45, MEDIA_COLLECT_SCRIPT, [],            "📰 媒体库 每日收集",              "kol"),
    (14, 5,  TG_SCRIPT,       ["afternoon"],     "📱 TG 14:05 下午报",            "social"),
    (9,  0,  REPORTER_SCRIPT, [],                "📊 09:00 团队日报",              "lead"),
    (20, 5,  TG_SCRIPT,       ["evening"],       "📱 TG 20:05 晚报",              "social"),
    (21, 7,  TWEET_SCRIPT,    ["--trending"],    "📱 推文 21:07 trending",         "social"),  # 08:07 EST
    (2,  23, TWEET_SCRIPT,    ["--smart-money"], "📱 推文 02:23 smart-money",      "social"),  # 13:23 EST
    (9,  41, TWEET_SCRIPT,    ["--closing-soon"],"📱 推文 09:41 closing-soon",     "social"),  # 20:41 EST
]


# ═══════════════════════════════════════════════════════════════════════════════
# KOL 收集进度日报（09:30 BJT → Lark KOL 群）
# ═══════════════════════════════════════════════════════════════════════════════

def kol_daily_report():
    """09:30 BJT — 从 SQLite 读取数据，发两张 Lark 卡片：汇总数据 + KOL 明细"""
    import urllib.request, json as _json

    today   = _today()
    webhook = os.getenv("LARK_KOL")
    if not webhook:
        print("[kol_daily_report] LARK_KOL not set"); return

    # ── 从 SQLite 读统计 ──────────────────────────────────────────────────────
    try:
        sys.path.insert(0, str(BASE / "03_kol_media"))
        from kol_db import get_kol_stats, get_media_stats, get_detail_kols, init_db
        init_db()
        kst  = get_kol_stats()["kol"]
        mst  = get_media_stats()["media"]
        dkols = get_detail_kols()
    except Exception as e:
        print(f"[kol_daily_report] SQLite 读取失败: {e}")
        return

    def _fmt_subs(n):
        try:
            n = int(n or 0)
            return f"{n/10000:.1f}万" if n >= 10000 else str(n)
        except:
            return "—"

    STATUS_ICON = {
        "已签约": "✅ 已签约", "谈判中": "🤝 谈判中",
        "已回复": "💬 已回复", "已发送": "📧 已发送",
        "已拒绝": "❌ 已拒绝",
    }

    def _send_card(card):
        payload = _json.dumps({"msg_type": "interactive", "card": card}, ensure_ascii=False).encode()
        req = urllib.request.Request(webhook, data=payload, headers={"Content-Type": "application/json"})
        res = _json.loads(urllib.request.urlopen(req, timeout=10).read())
        return res.get("msg", "ok")

    def _tbl(cols, rows):
        return {
            "tag": "table", "page_size": 50, "row_height": "low",
            "header_style": {"text_align": "left", "background_color": "grey", "bold": True},
            "columns": [{"tag": "column", "name": n, "display_name": d,
                         "data_type": "text", "width": "auto"} for n, d in cols],
            "rows": rows,
        }

    # ── 卡片1：汇总数据 ──────────────────────────────────────────────────────
    pending = max(kst["total_email"] - kst["total_sent"], 0)

    # 意图分布文字（如有）
    intents = kst.get("intents", {})
    intent_txt = "  ".join(
        f"{k}:{v}" for k, v in intents.items()
        if k not in ("migrated", "unknown") and v > 0
    )

    stats_rows = [
        {"metric": "KOL 收集总数", "total": str(kst["total"]),        "today": str(kst["today_total"]),   "yest": str(kst["yest_total"]),   "by": str(kst["by_total"])},
        {"metric": "KOL 有邮箱",   "total": str(kst["total_email"]),  "today": str(kst["today_email"]),   "yest": str(kst["yest_email"]),   "by": str(kst["by_email"])},
        {"metric": "KOL 已发送",   "total": str(kst["total_sent"]),   "today": str(kst["sent_today"]),    "yest": str(kst["sent_yest"]),    "by": str(kst["sent_by"])},
        {"metric": "KOL 已回复",   "total": str(kst["total_replied"]), "today": str(kst["replied_today"]), "yest": str(kst["replied_yest"]), "by": str(kst["replied_by"])},
        {"metric": "KOL 已签约",   "total": str(kst["total_signed"]), "today": "—",                       "yest": "—",                      "by": "—"},
        {"metric": "媒体收录总数", "total": str(mst["total"]),        "today": str(mst["today_total"]),   "yest": str(mst["yest_total"]),   "by": str(mst["by_total"])},
        {"metric": "媒体有邮箱",   "total": str(mst["total_email"]),  "today": str(mst["today_email"]),   "yest": str(mst["yest_email"]),   "by": str(mst["by_email"])},
        {"metric": "媒体已询价",   "total": str(mst["total_sent"]),   "today": str(mst["sent_today"]),    "yest": str(mst["sent_yest"]),    "by": str(mst["sent_by"])},
        {"metric": "媒体已回复",   "total": str(mst["total_replied"]), "today": str(mst["replied_today"]), "yest": str(mst["replied_yest"]), "by": str(mst["replied_by"])},
    ]

    footer_parts = [f"待发送：**{pending}** 封（C级优先）　22:00 UTC 自动发送（上限 10 封）"]
    if intent_txt:
        footer_parts.append(f"回复意图：{intent_txt}")

    card1 = {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": f"KOL 收集汇总  {today}  09:30 BJT"}, "template": "green"},
        "elements": [
            _tbl([("metric","指标"),("total","累计"),("today","今日"),("yest","昨日"),("by","前日")], stats_rows),
            {"tag": "markdown", "content": "\n".join(footer_parts)},
        ]
    }

    # ── 卡片2：KOL 明细 ──────────────────────────────────────────────────────
    detail_rows = []
    for k in dkols:
        sicon = STATUS_ICON.get(k["status"], "⏳ 待发送")
        detail_rows.append({
            "name":   k["name"],
            "link":   k["channel_url"] or "—",
            "tier":   f"{k['tier']} · {_fmt_subs(k['subscribers'])}",
            "email":  k["email"],
            "status": sicon,
            "utm":    k.get("utm_code", "—"),
        })

    card2 = {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": f"KOL 明细（有邮箱，共 {kst['total_email']} 个）"}, "template": "blue"},
        "elements": [
            _tbl([("name","博主名称"),("link","频道链接"),("tier","分级·订阅"),
                  ("email","邮箱"),("status","联系状态"),("utm","UTM码")],
                 detail_rows),
        ]
    }

    try:
        r1 = _send_card(card1)
        r2 = _send_card(card2)
        print(f"[{today} 09:30] KOL 汇总: {r1} | 明细: {r2}")
    except Exception as e:
        print(f"[kol_daily_report] 发送失败: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 启动调度器
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_workflow(topic: str = None):
    """手动触发完整作业流程：晨报 → 讨论 → 结论 → 任务分配
    执行+复盘+日报在当天定时自动完成，无需手动触发"""
    today = _today()
    _notify("lead", f"""👔 Team Lead｜今日完整作业流程启动 {today}

━━ 执行顺序 ━━
① 员工晨报（各自群）
② 自主讨论（此群）→ 形成结论
③ A/B 决策 → 任务分配（全员此群确认）
④ 各脚本按计划执行 → 实时汇报结果
⑤ 17:30 复盘（此群）
⑥ 18:30 综合日报（引用全天数据）

开始执行 ⬇️""")
    time.sleep(2)

    # 如果指定了议题，存入上下文
    if topic:
        ctx = _load_loop_ctx()
        ctx["force_topic"] = topic
        _save_loop_ctx(ctx)

    # ① 晨报
    morning_standup()
    time.sleep(5)
    # ② 讨论+结论（包含任务分配）
    auto_discussion()


def setup() -> BackgroundScheduler:
    sched = BackgroundScheduler(timezone=TZ)

    # Layer 2: 脚本任务
    for hour, minute, script, args, name, key in SCRIPT_JOBS:
        job_id = name.replace(" ", "_").replace(":", "").replace("/", "")
        sched.add_job(run_job, CronTrigger(hour=hour, minute=minute, timezone=TZ),
                      args=[script, args, name, key],
                      id=job_id, misfire_grace_time=300, coalesce=True)

    # YouTube KOL 外联 22:00 BJT — 单独注册，超时 30 分钟
    sched.add_job(
        run_job, CronTrigger(hour=22, minute=0, timezone=TZ),
        args=[SEND_SCRIPT, ["--youtube", "10"], "🤝 KOL 外联 22:00 YouTube", "kol"],
        kwargs={"timeout": 1800},
        id="kol_youtube_send", misfire_grace_time=300, coalesce=True,
    )

    # 媒体询价 22:30 BJT — 每日自动发送，超时 30 分钟
    sched.add_job(
        run_job, CronTrigger(hour=22, minute=30, timezone=TZ),
        args=[MEDIA_SEND_SCRIPT, [], "📰 媒体询价 22:30", "kol"],
        kwargs={"timeout": 1800},
        id="media_inquiry_send", misfire_grace_time=300, coalesce=True,
    )

    # Layer 3 & 4（讨论/晨报/复盘）已暂停
    # 每日只发 Layer 2 脚本 + 18:30 日报 + 异常提示
    # 如需手动触发讨论：_notify("lead", ...) 或直接调用 run_full_workflow()

    # 09:30 BJT — KOL 收集进度日报
    sched.add_job(
        kol_daily_report,
        CronTrigger(hour=9, minute=30, timezone=TZ),
        id="kol_daily_report", misfire_grace_time=300, coalesce=True,
    )

    # s11 — 每 15 分钟扫描任务队列，自主认领 pending 任务
    from apscheduler.triggers.interval import IntervalTrigger
    sched.add_job(
        scan_and_claim_tasks,
        IntervalTrigger(minutes=15, timezone=TZ),
        id="task_queue_scanner", misfire_grace_time=60, coalesce=True,
    )

    # 周三 21:07 BJT (08:07 EST) — MoonX Market Brief Thread（替代当天 trending）
    sched.add_job(
        run_job,
        CronTrigger(day_of_week="wed", hour=21, minute=7, timezone=TZ),
        args=[TWEET_SCRIPT, ["--thread"], "📱 推文 周三 Market Brief Thread", "social"],
        id="weekly_thread", misfire_grace_time=300, coalesce=True,
    )

    # 周五 09:00 BJT — OKR 周报推送到 LARK_LEAD
    sched.add_job(
        run_job,
        CronTrigger(day_of_week="fri", hour=9, minute=0, timezone=TZ),
        args=[OKR_SCRIPT, [], "📋 OKR 周报推送", "lead"],
        id="weekly_okr_push", misfire_grace_time=600, coalesce=True,
    )

    sched.start()

    total = len(SCRIPT_JOBS) + 1
    print(f"✅ 调度器已启动，共 {total} 个定时任务")
    print("  [Layer 2 — 脚本执行]")
    for h, m, _, _, name, _ in SCRIPT_JOBS:
        print(f"   {h:02d}:{m:02d} BJT — {name}")
    print("   周三 09:00 BJT — 📱 推文 周三 Market Brief Thread")
    print("  [Layer 3/4 — 讨论/晨报/复盘：已暂停，按需手动触发]")
    return sched
