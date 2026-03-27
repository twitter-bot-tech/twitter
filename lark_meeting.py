#!/usr/bin/env python3
"""
MoonX 团队 Lark 会议系统 — 多轮讨论版
用法：python3 lark_meeting.py "会议主题"
      python3 lark_meeting.py "会议主题" --rounds 3
"""
import os, sys, json, time, urllib.request
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import claude_cli as anthropic

load_dotenv(Path(__file__).parent / ".env", override=True)
load_dotenv(Path(__file__).parent / ".env.outreach", override=True)

BJT    = ZoneInfo("Asia/Shanghai")
BASE   = Path(__file__).parent
CLAUDE = anthropic.Anthropic()

WEBHOOKS = {
    "lead":     os.getenv("LARK_LEAD"),
    "social":   os.getenv("LARK_SOCIAL"),
    "seo":      os.getenv("LARK_SEO"),
    "kol":      os.getenv("LARK_KOL"),
    "growth":   os.getenv("LARK_GROWTH"),
    "strategy": os.getenv("LARK_STRATEGY"),
}

EMPLOYEES = [
    {
        "key": "social", "emoji": "📱", "name": "员工1号｜社媒运营",
        "system": """你是 MoonX 社媒运营，见过 Solana/Coinbase 从0到百万粉的操盘手。
核心信念：每条推文都是一场微型战役，粉丝是赢来的不是攒来的。
你的视角：这对 Twitter/Telegram 增长有什么影响？
发言要求：直接说观点，可以赞同或反驳其他人，引用具体数据或案例。不超过80字。""",
    },
    {
        "key": "seo", "emoji": "🔍", "name": "员工2号｜SEO专家",
        "system": """你是 MoonX SEO专家，帮 CoinGecko 做到月均3000万自然流量的操盘手。
核心信念：不做孤立文章，做内容集群；每篇必须攻打一个明确关键词。
你的视角：这对搜索流量/关键词排名有什么影响？
发言要求：直接说观点，可以赞同或反驳其他人，引用具体关键词或竞品数据。不超过80字。""",
    },
    {
        "key": "kol", "emoji": "🤝", "name": "员工3号｜KOL媒体",
        "system": """你是 MoonX KOL & 媒体负责人，帮 Coinbase 上市前把故事塞进 WSJ 和 FT 的操盘手。
核心信念：媒体不是广告位，是信任转移机器。一篇 WSJ 报道转移的信任比100篇自发内容多1000倍。
你的视角：这对KOL签约/媒体曝光有什么影响？
发言要求：直接说观点，可以赞同或反驳其他人，引用具体KOL名字或媒体。不超过80字。""",
    },
    {
        "key": "growth", "emoji": "📈", "name": "员工4号｜增长运营",
        "system": """你是 MoonX 增长运营，设计过 Uniswap空投/Blur积分等百万人参与激励机制的操盘手。
核心信念：不做活动，设计增长飞轮；好的机制会自己跑。
你的视角：这对用户增长/裂变/留存有什么影响？
发言要求：直接说观点，可以赞同或反驳其他人，引用具体增长案例或数据。不超过80字。""",
    },
    {
        "key": "strategy", "emoji": "📊", "name": "员工5号｜策略数据",
        "system": """你是 MoonX 策略分析师，a16z crypto 研究团队风格。
核心信念：不写报告，生产决策弹药；所有结论必须有数字支撑。
你的视角：数据说明什么？竞品在干什么？ROI 是多少？
发言要求：直接说观点，可以赞同或反驳其他人，引用具体竞品数据或市场数字。不超过80字。""",
    },
]

LEAD_ROUND_SYSTEM = """你是 MoonX Team Lead，融合了 He Yi（币安CMO）和 Coinbase IPO营销负责人的思维。
你在主持一场多轮讨论会议。现在需要你做轮次总结：
- 提炼本轮核心分歧点
- 向有分歧的部门提出一个尖锐问题，推动下轮深入
- 语气果断，不废话。不超过80字。"""

LEAD_FINAL_SYSTEM = """你是 MoonX Team Lead，现在所有部门已充分讨论完毕。
给出最终会议结论：
1. 一句话核心决策
2. 给 Kelly 的选择题（A 还是 B，不给问答题）
3. 各部门明确的下一步行动（一句话/部门）
语气果断，格式清晰。"""


def send_lark(bot_key: str, text: str):
    url = WEBHOOKS.get(bot_key)
    if not url:
        return
    payload = json.dumps({"msg_type": "text", "content": {"text": text}}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"  ⚠ Lark 错误 ({bot_key}): {e}")


def ask_claude(system: str, user_msg: str) -> str:
    try:
        resp = CLAUDE.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=250,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        return f"（生成失败: {e}）"


def format_history(history: list[dict]) -> str:
    """把历史发言格式化成上下文"""
    lines = []
    for msg in history:
        lines.append(f"【{msg['name']}】{msg['content']}")
    return "\n".join(lines)


def run_meeting(topic: str, rounds: int = 2):
    now = datetime.now(BJT).strftime("%m/%d %H:%M")
    history = []  # 全程发言记录

    print(f"\n📋 会议开始：{topic}（{rounds}轮讨论）\n")

    # ── 开场 ──
    opening = f"""👔 Team Lead｜主持会议 {now} BJT

━━ 📋 今日议题 ━━
{topic}

共 {rounds} 轮讨论，请各部门依次发言 👇
"""
    send_lark("lead", opening)
    print("  ✅ Lead 开场")
    time.sleep(1.5)

    # ── 多轮讨论 ──
    for round_num in range(1, rounds + 1):
        is_last_round = (round_num == rounds)

        round_header = f"━━ 第 {round_num} 轮{'（最终轮）' if is_last_round else ''} ━━"
        send_lark("lead", f"👔 Team Lead\n\n{round_header}")
        print(f"\n  [{round_num}轮]")
        time.sleep(1)

        round_speeches = []

        for emp in EMPLOYEES:
            ctx = format_history(history) if history else "（首轮发言，尚无其他人发言）"

            if round_num == 1:
                prompt = f"""会议议题：{topic}

请从你的职责角度发表初始观点。"""
            else:
                prompt = f"""会议议题：{topic}

之前各轮发言记录：
{ctx}

现在是第{round_num}轮。请回应其他人的观点：可以反驳、补充或提出新问题。聚焦分歧点。"""

            response = ask_claude(emp["system"], prompt)
            msg = f"{emp['emoji']} {emp['name']}（第{round_num}轮）\n\n{response}"
            send_lark(emp["key"], msg)

            history.append({"name": emp["name"], "content": response})
            round_speeches.append({"name": emp["name"], "content": response})
            print(f"    ✅ {emp['name']}")
            time.sleep(2)

        # 轮次结束 — Lead 点评（非最后轮）
        if not is_last_round:
            ctx = format_history(round_speeches)
            prompt = f"""议题：{topic}

本轮发言：
{ctx}

请做轮次总结，提炼分歧，向下一轮提问。"""
            summary = ask_claude(LEAD_ROUND_SYSTEM, prompt)
            msg = f"👔 Team Lead｜第{round_num}轮小结\n\n{summary}\n\n➡️ 进入第{round_num+1}轮讨论..."
            send_lark("lead", msg)
            history.append({"name": "Team Lead", "content": summary})
            print(f"    ✅ Lead 轮次小结")
            time.sleep(2)

    # ── 最终决策 ──
    full_ctx = format_history(history)
    prompt = f"""议题：{topic}

完整讨论记录：
{full_ctx}

请给出最终会议结论。"""
    conclusion = ask_claude(LEAD_FINAL_SYSTEM, prompt)

    final_msg = f"""👔 Team Lead｜最终决策

━━ 会议结论 ━━
{conclusion}

━━ 会议信息 ━━
议题：{topic}
时间：{now} BJT
轮数：{rounds} 轮
参与：5个部门"""
    send_lark("lead", final_msg)
    print(f"\n  ✅ Lead 最终决策")
    print(f"\n✅ 会议完成，共 {rounds} 轮，已发送到 Lark 群")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("用法：python3 lark_meeting.py \"会议主题\"")
        print("     python3 lark_meeting.py \"会议主题\" --rounds 3")
        sys.exit(1)

    rounds = 2
    if "--rounds" in args:
        idx = args.index("--rounds")
        rounds = int(args[idx + 1])
        args = [a for i, a in enumerate(args) if i != idx and i != idx + 1]

    topic = " ".join(args)
    run_meeting(topic, rounds=rounds)
