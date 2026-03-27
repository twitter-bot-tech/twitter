#!/usr/bin/env python3
"""
MoonX KOL 申请者回复脚本
来源：Google Form「MoonX KOL Recruitment」行 144–171（2026-03 批次）
功能：
  1. 向有 Twitter handle 的申请者发送 DM（tweepy，复用现有认证）
  2. 打印 Telegram handle 清单 + 消息模版，供手动发送

用法：
  python3 send_form_applicants.py          # 测试模式（不真实发送）
  python3 send_form_applicants.py --send   # 真实发送 Twitter DM
"""

import os, time, random, logging, sys
from pathlib import Path
from dotenv import load_dotenv
import tweepy

load_dotenv(Path(__file__).parent.parent / ".env", override=True)
load_dotenv(Path(__file__).parent.parent / ".env.outreach", override=True)

# ── Twitter 认证 ──
client = tweepy.Client(
    bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
    consumer_key=os.getenv("TWITTER_API_KEY"),
    consumer_secret=os.getenv("TWITTER_API_SECRET"),
    access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
    access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
)

MOONX_URL  = "https://www.bydfi.com/en/moonx/markets/trending"
REBATE_URL = "https://www.bydfi.com/zh/moonx/account/my-rebate?type=my-rebate"
SENDER_TG  = os.getenv("SENDER_TG", "@BDkelly")
DAILY_LIMIT = 10

# ── 日志 ──
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "form_applicant_dm.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── 申请者名单（来源：Google Form 截图，行 144–171）──
# telegram: column C 中有效的 @handle；twitter: column E 中提取的 handle
APPLICANTS = [
    {"name": "Kophonelay",           "telegram": "@duulay1197",    "twitter": "kophone13482",  "identity": "KOL"},
    {"name": "Khurram",              "telegram": None,              "twitter": "KhurramSha28342","identity": "KOL"},
    {"name": "Kacper Czuba",         "telegram": "@kacperairdrops", "twitter": None,            "identity": "KOL"},
    {"name": "Muhammad Firdaus Bin", "telegram": "@Firdauszack1",   "twitter": "Firdaus_Fm1",   "identity": "KOL, Youtuber, Website admin"},
    {"name": "Artem",                "telegram": "@kassichh",       "twitter": None,            "identity": "KOL"},
    {"name": "Natthawut Manthong",   "telegram": "@nxth132002",     "twitter": None,            "identity": "KOL"},
    {"name": "Pavel",                "telegram": "@escanorish",     "twitter": None,            "identity": "KOL"},
    {"name": "Kevin Arenth",         "telegram": "@Kay392",         "twitter": None,            "identity": "Website admin"},
    {"name": "Giorgi",               "telegram": "@GRAYGTS",        "twitter": None,            "identity": "KOL, Youtuber, Website admin"},
    {"name": "Lahiru Dilshan",       "telegram": "@mrdraco56",      "twitter": "MrDilZ2214",    "identity": "KOL, Crypto Community Promoter"},
    {"name": "Ramez",                "telegram": "@ramezelgendy",   "twitter": "ramezelgendy",  "identity": "KOL"},
    {"name": "Jack novak",           "telegram": "@Drankoutlet",    "twitter": None,            "identity": "KOL"},
    # 以下只有电话号或无效信息，跳过自动化
    # William Shawn Holtz, Marián Maco, เฉลิมชัย เรืองพุ่ม (x2), Chalermchai Rungphum,
    # Labbize houssem Eddin, Arnas, Monica, Christian (Christianuwu),
    # Christian Vollmost, Ivans, Kashif Rasheed, Chwnwichi,
    # Sofiane, Manuel Márquez Arias, Manuel
]

# ── 英文版模版 ──
DM_TEMPLATE_EN = (
    "Hey {name} 👋\n\n"
    "Thanks for applying to the MoonX KOL program — we received your application!\n\n"
    "I'm Kelly, head of partnerships at BYDFi. MoonX ({url}) tracks smart wallet activity "
    "across Solana and BNB Chain in real time — think GMGN-style smart money tracking, "
    "built for traders who want to catch moves early.\n\n"
    "Here's what our KOL partners get:\n"
    "✅ Free full access to MoonX\n"
    "✅ Personal referral link with rev-share ({rebate})\n"
    "✅ Alpha data feeds for your content\n"
    "✅ Early access to new features\n\n"
    "Love to get you set up — quick chat?\n"
    "TG: {tg}"
)

# ── 中文版模版 ──
DM_TEMPLATE_ZH = (
    "你好 {name}！\n\n"
    "感谢你申请 MoonX KOL 合作项目，我们已收到你的申请！\n\n"
    "我是 BYDFi 合作负责人 Kelly。MoonX（{url}）实时追踪 Solana 和 BNB Chain "
    "上的聪明钱钱包动向，类似 GMGN 的智能资金追踪，帮你第一时间抓住早期机会。\n\n"
    "KOL 合作权益：\n"
    "✅ 免费完整使用 MoonX\n"
    "✅ 专属推荐链接 + 佣金分成（{rebate}）\n"
    "✅ 原始数据支持内容创作\n"
    "✅ 新功能抢先体验\n\n"
    "期待进一步沟通，有空聊聊吗？\n"
    "TG: {tg}"
)


# ══════════════════════════════════════════
# Twitter DM
# ══════════════════════════════════════════

def send_twitter_dm(applicant: dict, dry_run: bool = True) -> bool:
    username = applicant["twitter"]
    first_name = applicant["name"].split()[0]
    if not username:
        return False
    try:
        user = client.get_user(username=username)
        if not user.data:
            logger.warning(f"  ✗ 用户不存在: @{username}")
            return False
        user_id = user.data.id
        dm_text = DM_TEMPLATE_EN.format(name=first_name, url=MOONX_URL, tg=SENDER_TG, rebate=REBATE_URL)

        if dry_run:
            logger.info(f"  [DRY RUN] @{username} ({applicant['name']})")
            logger.info(f"  预览: {dm_text[:120]}...")
            return True

        client.create_direct_message(participant_id=user_id, text=dm_text)
        logger.info(f"  ✓ Twitter DM 已发送 → @{username}")
        return True

    except tweepy.errors.Forbidden:
        logger.error(f"  ✗ @{username} 关闭了 DM")
        return False
    except tweepy.errors.TooManyRequests:
        logger.warning("  ⚠ Twitter 限速，等待 15 分钟...")
        time.sleep(900)
        return False
    except Exception as e:
        logger.error(f"  ✗ 失败 @{username}: {e}")
        return False


# ══════════════════════════════════════════
# Telegram 清单输出
# ══════════════════════════════════════════

def print_telegram_list():
    tg_list = [a for a in APPLICANTS if a["telegram"]]
    print("\n" + "="*60)
    print(f"Telegram 跟进名单（共 {len(tg_list)} 人）")
    print("="*60)
    for a in tg_list:
        twitter_note = f"  | Twitter: @{a['twitter']}" if a["twitter"] else ""
        print(f"  {a['telegram']:<22}  {a['name']:<28}  [{a['identity']}]{twitter_note}")
    print("="*60)
    print("\n【英文版消息模版】\n")
    print(DM_TEMPLATE_EN.format(name="[Name]", url=MOONX_URL, tg=SENDER_TG, rebate=REBATE_URL))
    print("\n" + "="*60)
    print("\n【中文版消息模版】\n")
    print(DM_TEMPLATE_ZH.format(name="[名字]", url=MOONX_URL, tg=SENDER_TG, rebate=REBATE_URL))
    print("="*60)


# ══════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════

def run(dry_run: bool = True):
    logger.info("=" * 60)
    logger.info("MoonX Form 申请者外联")
    logger.info("来源：Google Form 行 144-171")
    logger.info("模式: %s", "DRY RUN（测试）" if dry_run else "真实发送")
    logger.info("=" * 60)

    # 1. 输出 Telegram 清单
    print_telegram_list()

    # 2. Twitter DM
    twitter_targets = [a for a in APPLICANTS if a["twitter"]]
    logger.info(f"\nTwitter DM 目标：{len(twitter_targets)} 人")

    sent = 0
    for applicant in twitter_targets:
        if sent >= DAILY_LIMIT:
            logger.warning(f"已达每日上限 {DAILY_LIMIT} 条，停止")
            break
        logger.info(f"▶ {applicant['name']} (@{applicant['twitter']})")
        success = send_twitter_dm(applicant, dry_run=dry_run)
        if success:
            sent += 1
            if not dry_run and sent < len(twitter_targets):
                wait = random.randint(90, 200)
                logger.info(f"  等待 {wait} 秒...")
                time.sleep(wait)

    logger.info(f"\n完成！Twitter DM 发送 {sent} 条，剩余 Telegram {len([a for a in APPLICANTS if a['telegram']])} 条需手动发送")


if __name__ == "__main__":
    dry_run = "--send" not in sys.argv
    if dry_run:
        logger.info("测试模式 — 不会真实发送 DM")
        logger.info("真实发送：python3 send_form_applicants.py --send")
    run(dry_run=dry_run)
