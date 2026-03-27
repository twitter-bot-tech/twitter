#!/usr/bin/env python3
"""生成 MoonX KOL 外联完整框架 PDF"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path
import os

# ── 字体注册（支持中文）──────────────────────────────────────────────
FONT_PATHS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/Library/Fonts/Arial Unicode MS.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]
CN_FONT = "Helvetica"  # fallback

for fp in FONT_PATHS:
    if os.path.exists(fp):
        try:
            pdfmetrics.registerFont(TTFont("CNFont", fp))
            CN_FONT = "CNFont"
            break
        except Exception:
            continue

# ── 颜色 ─────────────────────────────────────────────────────────────
NAVY    = colors.HexColor("#1F4E79")
BLUE    = colors.HexColor("#2E75B6")
LBLUE   = colors.HexColor("#BDD7EE")
RED     = colors.HexColor("#C00000")
ORANGE  = colors.HexColor("#E26B0A")
YELLOW  = colors.HexColor("#FFD700")
GREEN   = colors.HexColor("#C6EFCE")
LGREEN  = colors.HexColor("#70AD47")
LGRAY   = colors.HexColor("#F2F2F2")
WHITE   = colors.white
BLACK   = colors.black
DKGRAY  = colors.HexColor("#404040")

# ── 样式 ─────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def s(name, **kw):
    return ParagraphStyle(name, fontName=CN_FONT, **kw)

TITLE_S   = s("Title",   fontSize=20, textColor=WHITE,   spaceAfter=4,  leading=26)
H1_S      = s("H1",      fontSize=14, textColor=WHITE,   spaceAfter=2,  leading=18)
H2_S      = s("H2",      fontSize=11, textColor=NAVY,    spaceBefore=8, spaceAfter=4, leading=15)
BODY_S    = s("Body",    fontSize=9,  textColor=DKGRAY,  leading=13)
CAPTION_S = s("Caption", fontSize=8,  textColor=colors.HexColor("#666666"), leading=11)
NOTE_S    = s("Note",    fontSize=8,  textColor=NAVY,    leading=11, leftIndent=8)

def cell(text, bold=False, size=9, color=BLACK, align="LEFT"):
    style = ParagraphStyle(
        "cell", fontName=CN_FONT, fontSize=size,
        textColor=color, leading=size+4,
        alignment={"LEFT":0,"CENTER":1,"RIGHT":2}[align]
    )
    if bold:
        text = f"<b>{text}</b>"
    return Paragraph(text, style)

def hcell(text):
    return cell(text, bold=True, size=9, color=WHITE, align="CENTER")

def build_table(headers, rows, col_widths, header_color=NAVY):
    data = [[hcell(h) for h in headers]]
    for row in rows:
        data.append([cell(str(c)) for c in row])
    t = Table(data, colWidths=[w*cm for w in col_widths], repeatRows=1)
    style = [
        ("BACKGROUND",  (0,0), (-1,0),  header_color),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, colors.HexColor("#EBF3FB")]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#AAAAAA")),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
    ]
    t.setStyle(TableStyle(style))
    return t

def section_header(title, subtitle=""):
    bg = Table(
        [[Paragraph(f"<b>{title}</b>", H1_S),
          Paragraph(subtitle, ParagraphStyle("sub", fontName=CN_FONT, fontSize=9, textColor=colors.HexColor("#BDD7EE"), leading=13))]],
        colWidths=[13*cm, 5*cm]
    )
    bg.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), NAVY),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING",(0,0),(-1,-1), 10),
        ("RIGHTPADDING",(0,0),(-1,-1),6),
    ]))
    return bg

def priority_badge(text, color):
    return Table([[Paragraph(f"<b>{text}</b>",
        ParagraphStyle("badge", fontName=CN_FONT, fontSize=8, textColor=WHITE, alignment=1))]],
        colWidths=[1.8*cm])

# ════════════════════════════════════════════════════════════════════════
# BUILD
# ════════════════════════════════════════════════════════════════════════
output_path = Path(__file__).parent / "2026-03-11_MoonX_KOL外联完整框架_v1.0.pdf"

doc = SimpleDocTemplate(
    str(output_path), pagesize=A4,
    leftMargin=1.8*cm, rightMargin=1.8*cm,
    topMargin=1.5*cm, bottomMargin=1.5*cm,
    title="MoonX KOL 外联完整框架 v1.0",
    author="Kelly · BYDFi MoonX"
)

story = []

# ── 封面标题块 ──────────────────────────────────────────────────────
cover = Table([[
    Paragraph("<b>MoonX KOL 外联完整框架</b>", s("ct", fontSize=22, textColor=WHITE, leading=28)),
    Paragraph("v1.0 · 2026-03-11", s("cv", fontSize=10, textColor=colors.HexColor("#BDD7EE"), leading=14, alignment=2))
]], colWidths=[12*cm, 6*cm])
cover.setStyle(TableStyle([
    ("BACKGROUND",   (0,0),(-1,-1), NAVY),
    ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
    ("TOPPADDING",   (0,0),(-1,-1), 14),
    ("BOTTOMPADDING",(0,0),(-1,-1), 14),
    ("LEFTPADDING",  (0,0),(-1,-1), 12),
]))
story.append(cover)
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("待执行 | 发件人：kelly@bydfi.com | 目标：Q1 签约 ≥ 5 个 KOL", CAPTION_S))
story.append(Spacer(1, 0.5*cm))

# ═══════════════════════════════════
# 一、收集策略
# ═══════════════════════════════════
story.append(section_header("一、收集策略", "Collection Strategy"))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("1.1 渠道规划", H2_S))
story.append(build_table(
    ["渠道", "状态", "节奏", "优先级"],
    [
        ["YouTube",    "✅ 运行中（212频道）",      "每日，扩词到100+",        "P0"],
        ["Twitter/X",  "⚠️ 限量重启",              "每周1次，每次≤10关键词",  "P1"],
        ["Substack",   "❌ 新增接入",              "每周，抓预测市场Newsletter", "P1"],
    ],
    [3.5, 5, 5.5, 2.5]
))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("1.2 关键词策略（15个 → 100个+）", H2_S))
story.append(build_table(
    ["类别", "关键词方向（示例）"],
    [
        ["预测市场",      "polymarket tutorial / kalshi trading / prediction markets crypto / manifold markets"],
        ["合约 / 永续",   "crypto futures trading / perpetual trading / crypto leverage / binance futures tutorial"],
        ["现货",          "spot trading crypto / crypto spot strategy / buy bitcoin / altcoin spot"],
        ["Meme 币",       "solana meme coin alpha / pump.fun tutorial / meme coin early / degen trading / gmgn"],
        ["链上 / 智能钱",  "on-chain analysis / smart money tracker / whale wallet tracking / crypto signals"],
        ["DeFi / DEX",    "defi trading / raydium / pancakeswap / uniswap trading / yield farming"],
        ["工具 / 数据",   "dexscreener / birdeye / crypto screener / trading signals crypto"],
        ["宏观 / BTC",    "bitcoin macro / crypto market outlook / btc cycle / altcoin season"],
    ],
    [3.5, 13]
))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("1.3 邮箱提取策略（当前 16.5% → 目标 30%+）", H2_S))
story.append(build_table(
    ["提取方法", "覆盖范围", "状态"],
    [
        ["Description 正则（含混淆还原 (at)(dot)）", "全部频道",        "✅ 已有"],
        ["yt-dlp About页抓取",                       "C级+（1万+订阅）", "✅ 已扩展"],
        ["官网 contact 页爬取",                      "B级+（10万+）",   "⬜ 待开发"],
        ["Linktree 页面解析",                        "有 Linktree 链接的", "⬜ 待开发"],
        ["视频描述提取（近3条）",                    "全部频道",         "⬜ 待开发"],
        ["Twitter Bio 交叉查询",                     "有 Twitter Handle 的", "⬜ 待开发"],
        ["Google 搜索兜底",                          "无邮箱 B/C 级",   "⬜ 待开发"],
    ],
    [6.5, 4.5, 2.5]
))
story.append(Spacer(1, 0.6*cm))

# ═══════════════════════════════════
# 二、KOL 分级
# ═══════════════════════════════════
story.append(section_header("二、KOL 分级与外联优先级", "KOL Tiering"))
story.append(Spacer(1, 0.3*cm))

tier_data = [
    ["Kalshi 前KOL\n（独立专项表）", "任意", "原有受众精准，竞品失联", "独立跟进，专属条件", "🔴 最高"],
    ["PM级",    "任意",         "Polymarket/Kalshi 垂类",    "独家数据合作，定制报告",      "🔴 最高"],
    ["C级",     "1万~10万",     "主力发送对象",              "工具体验 + 早期高返佣合作",   "🟠 高"],
    ["D级",     "1千~1万",      "精准垂类保留",              "免费Pro + 返佣",              "🟡 中"],
    ["B级",     "10万~100万",   "内容方向切入",              "独家数据，不谈广告费",        "🟡 中"],
    ["A级",     "100万+",       "暂缓",                      "等产品数据更强再谈",          "⚪ 低"],
]
story.append(build_table(
    ["级别", "粉丝量", "定位", "外联策略", "优先级"],
    tier_data,
    [3, 2.5, 3.5, 5, 1.8]
))
story.append(Spacer(1, 0.6*cm))

# ═══════════════════════════════════
# 三、发送策略
# ═══════════════════════════════════
story.append(section_header("三、发送策略", "Outreach Strategy"))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("3.1 基本参数", H2_S))
story.append(build_table(
    ["参数", "确认值"],
    [
        ["发件人",        "kelly@bydfi.com"],
        ["发送时间",      "北京时间 22:00（= 美东 09:00）"],
        ["每日发送量",    "10 封"],
        ["月覆盖量",      "~300 封"],
        ["跟进机制",      "3 轮跟进序列（Day 0 / 7 / 14）"],
    ],
    [4, 12.5]
))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("3.2 邮件模板（4套差异化）", H2_S))
story.append(build_table(
    ["对象", "钩子角度", "核心 CTA"],
    [
        ["PM级 / Kalshi前KOL", '"你上期讲的 Polymarket，我有它背后你没见过的数据层"',      "独家数据合作 + 定制报告"],
        ["B级 KOL",            '"你的受众正在交易这些市场，我有他们在看的实时数据"',       "内容合作 + 独家访问权"],
        ["C/D级 KOL",          '"我们早期合作伙伴返佣比行业高3倍，名额即将关闭"',         "免费Pro + 高返佣早期合作"],
        ["Meme KOL",           '"你的受众已在赌这些市场，只差一个工具和高返佣渠道"',      "rev-share + 早期代理优惠"],
    ],
    [3.5, 7, 5.5]
))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "💡 稀缺性话术：早期合作伙伴返佣比例更高 + 代理优惠政策，窗口关闭后恢复标准条款（不用'名额有限'）",
    NOTE_S
))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("3.3 跟进序列", H2_S))
story.append(build_table(
    ["轮次", "时间", "策略"],
    [
        ["第 1 封", "Day 0",  "正式外联，钩子切入"],
        ["第 2 封", "Day 7",  "换角度跟进：提供一条市场数据作为诚意"],
        ["第 3 封", "Day 14", "最终跟进：早期返佣窗口即将关闭"],
    ],
    [2.5, 2.5, 11.5]
))
story.append(Spacer(1, 0.6*cm))

# ═══════════════════════════════════
# 四、效果追踪
# ═══════════════════════════════════
story.append(section_header("四、效果追踪与成功标准", "Tracking & Success Metrics"))
story.append(Spacer(1, 0.3*cm))

story.append(build_table(
    ["指标", "追踪方式", "阶段目标", "成功标准"],
    [
        ["发送量",       "Excel 状态列自动更新",  "10封/天，300封/月",      "连续2周不断档"],
        ["邮箱命中率",   "有邮箱数/总收集数",     "≥ 30%（当前 16.5%）",   "稳定在 30%+"],
        ["回复率",       "手动标记已回复",        "≥ 10%（30封/月）",      "第2周起有稳定回复"],
        ["进入谈判",     "标记谈判中",            "月 ≥ 3 个",             "第4周前至少1个"],
        ["签约转化",     "标记已签约",            "Q1 ≥ 5 个",             "至少含1个Kalshi前KOL"],
        ["带来注册",     "MoonX后台 ref 统计",    "每KOL ≥ 50 注册",       "单KOL ROI 为正"],
        ["返佣收益",     "后台返佣记录",          "月 ≥ $500 总返佣",      "自我造血，不依赖预算"],
    ],
    [3, 4, 4, 5.5]
))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("复盘节奏", H2_S))
story.append(build_table(
    ["频率", "内容"],
    [
        ["每周五",  "发送量/回复率统计，更新 Excel"],
        ["每月1日", "月报：签约进展 + 带来用户数 + 返佣金额"],
        ["Q1末",    "复盘哪类 KOL 转化最高，调整 Q2 侧重"],
    ],
    [3, 13.5]
))
story.append(Spacer(1, 0.8*cm))

# ── 底部签名 ─────────────────────────────────────────────────────
story.append(HRFlowable(width="100%", thickness=0.5, color=LBLUE))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    "Kelly · Head of Marketing · BYDFi MoonX  |  TG: @BDkelly  |  kelly@bydfi.com",
    CAPTION_S
))

# ── Build ─────────────────────────────────────────────────────────
doc.build(story)
print(f"✅ PDF 已生成：{output_path}")
