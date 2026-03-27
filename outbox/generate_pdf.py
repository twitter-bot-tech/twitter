#!/usr/bin/env python3
"""将 MoonX SEO 完整计划书 Markdown 转换为格式化 PDF"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
import os, re

# ── 中文字体注册 ────────────────────────────────────────────────────────────────
# (path, subfontIndex)  .ttc 是字体集合，需要 subfontIndex；.ttf 填 None
FONT_SPECS = [
    ("/Library/Fonts/Arial Unicode.ttf", None),
    ("/System/Library/Fonts/STHeiti Medium.ttc", 0),
    ("/System/Library/Fonts/STHeiti Light.ttc", 0),
    ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
]
FONT_NAME = "Chinese"
font_registered = False
for fp, idx in FONT_SPECS:
    if os.path.exists(fp):
        try:
            if idx is not None:
                pdfmetrics.registerFont(TTFont(FONT_NAME, fp, subfontIndex=idx))
            else:
                pdfmetrics.registerFont(TTFont(FONT_NAME, fp))
            font_registered = True
            print(f"✅ 字体加载成功：{fp}")
            break
        except Exception as e:
            print(f"⚠️ 字体加载失败 {fp}: {e}")
            continue

if not font_registered:
    FONT_NAME = "Helvetica"
    print("❌ 未找到中文字体，将使用 Helvetica（中文会乱码）")

# ── 颜色方案 ────────────────────────────────────────────────────────────────────
C_DARK    = colors.HexColor("#0D1117")
C_BLUE    = colors.HexColor("#1E6FD9")
C_LIGHT   = colors.HexColor("#F6F8FA")
C_BORDER  = colors.HexColor("#D0D7DE")
C_GREEN   = colors.HexColor("#1A7F37")
C_ORANGE  = colors.HexColor("#E36209")
C_GRAY    = colors.HexColor("#57606A")
C_WHITE   = colors.white

W, H = A4
MARGIN = 20 * mm

# ── 样式定义 ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def S(name, **kw):
    kw.setdefault("fontName", FONT_NAME)
    return ParagraphStyle(name, **kw)

sTitle    = S("sTitle",    fontSize=24, textColor=C_DARK,  spaceAfter=4,  leading=32, alignment=TA_CENTER)
sSubtitle = S("sSubtitle", fontSize=11, textColor=C_GRAY,  spaceAfter=2,  leading=16, alignment=TA_CENTER)
sMeta     = S("sMeta",     fontSize=9,  textColor=C_GRAY,  spaceAfter=12, leading=14, alignment=TA_CENTER)
sH1       = S("sH1",       fontSize=16, textColor=C_WHITE, spaceAfter=8,  leading=22, spaceBefore=14)
sH2       = S("sH2",       fontSize=13, textColor=C_BLUE,  spaceAfter=6,  leading=18, spaceBefore=10)
sH3       = S("sH3",       fontSize=11, textColor=C_DARK,  spaceAfter=4,  leading=16, spaceBefore=8)
sBody     = S("sBody",     fontSize=9.5,textColor=C_DARK,  spaceAfter=4,  leading=15)
sBullet   = S("sBullet",   fontSize=9.5,textColor=C_DARK,  spaceAfter=3,  leading=15, leftIndent=12, firstLineIndent=0)
sCode     = ParagraphStyle("sCode", fontName="Courier", fontSize=8.5,
              textColor=colors.HexColor("#24292F"), spaceAfter=3, leading=13,
              backColor=C_LIGHT, leftIndent=8, rightIndent=8)
sQuote    = S("sQuote",    fontSize=9.5,textColor=C_GRAY,  spaceAfter=4,  leading=15, leftIndent=16,
              borderPad=4)
sTOC      = S("sTOC",      fontSize=10, textColor=C_DARK,  spaceAfter=3,  leading=16, leftIndent=0)
sTOCsub   = S("sTOCsub",  fontSize=9,  textColor=C_GRAY,  spaceAfter=2,  leading=14, leftIndent=16)

def safe(text):
    """转义 ReportLab 特殊字符"""
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("**", "")
                .replace("`", ""))

def h1_block(text):
    """带深色背景的一级标题"""
    t = Table([[Paragraph(safe(text), sH1)]], colWidths=[W - 2*MARGIN])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_BLUE),
        ("LEFTPADDING",  (0,0), (-1,-1), 12),
        ("RIGHTPADDING", (0,0), (-1,-1), 12),
        ("TOPPADDING",   (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0), (-1,-1), 8),
        ("ROUNDEDCORNERS", (0,0), (-1,-1), [4,4,4,4]),
    ]))
    return t

def make_table(headers, rows):
    """通用表格生成"""
    col_w = (W - 2*MARGIN) / len(headers)
    col_widths = [col_w] * len(headers)

    head_row = [Paragraph(f"<b>{safe(h)}</b>", S("th", fontSize=9, textColor=C_WHITE, fontName=FONT_NAME, leading=13))
                for h in headers]
    data = [head_row]
    for row in rows:
        data.append([Paragraph(safe(str(c)), S("td", fontSize=9, textColor=C_DARK, fontName=FONT_NAME, leading=13))
                     for c in row])

    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  C_BLUE),
        ("BACKGROUND",    (0,1), (-1,-1), C_WHITE),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [C_WHITE, C_LIGHT]),
        ("GRID",          (0,0), (-1,-1), 0.5, C_BORDER),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
    ]))
    return t

def cover_page():
    elems = []
    elems.append(Spacer(1, 40*mm))
    elems.append(Paragraph("MoonX SEO 完整战略计划书", sTitle))
    elems.append(Spacer(1, 4*mm))
    elems.append(HRFlowable(width="60%", thickness=2, color=C_BLUE, hAlign="CENTER"))
    elems.append(Spacer(1, 6*mm))
    elems.append(Paragraph("KuCoin 竞品分析 × MoonX SEO 战略 × 12 周执行计划", sSubtitle))
    elems.append(Spacer(1, 8*mm))
    elems.append(Paragraph("版本：v1.0　|　日期：2026-03-06　|　撰写：Team Lead + SEO 专家", sMeta))
    elems.append(Spacer(1, 6*mm))

    # 摘要框
    summary_text = (
        "现状：MoonX SEO 从零起步，内容发在第三方平台，权重不归我们。\n\n"
        "机会：预测市场赛道 Google 几乎无权威内容，窗口期 6~12 个月。\n\n"
        "战略：① 迁移内容到官方域名  ② 程序化SEO扩规模  ③ 数据体系形成飞轮。\n\n"
        "目标：Q2结束前自然流量 >5,000/月，polymarket alternative 进入 Google 前 20。"
    )
    box = Table([[Paragraph(summary_text, S("sum", fontSize=10, textColor=C_DARK,
                                             fontName=FONT_NAME, leading=17, spaceAfter=0))]],
                colWidths=[W - 2*MARGIN - 24*mm])
    box.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_LIGHT),
        ("LEFTPADDING",   (0,0), (-1,-1), 16),
        ("RIGHTPADDING",  (0,0), (-1,-1), 16),
        ("TOPPADDING",    (0,0), (-1,-1), 14),
        ("BOTTOMPADDING", (0,0), (-1,-1), 14),
        ("BOX",           (0,0), (-1,-1), 1.5, C_BLUE),
    ]))
    elems.append(box)
    elems.append(PageBreak())
    return elems

def toc():
    elems = []
    elems += [h1_block("目录"), Spacer(1, 6*mm)]
    chapters = [
        ("一", "KuCoin SEO 全景分析"),
        ("二", "MoonX SEO 现状诊断"),
        ("三", "MoonX SEO 完整战略"),
        ("四", "程序化 SEO 计划"),
        ("五", "SEO 数据体系搭建（Claude Code 指挥中心）"),
        ("六", "外链建设计划"),
        ("七", "12 周执行计划"),
        ("八", "OKR 与衡量标准"),
        ("九", "不做清单"),
        ("十", "需要 Kelly 确认的事项"),
    ]
    for num, title in chapters:
        elems.append(Paragraph(f"第{num}部分　{title}", sTOC))
        elems.append(HRFlowable(width="100%", thickness=0.3, color=C_BORDER))
    elems.append(PageBreak())
    return elems

def build_content():
    story = []

    # ── 第一部分 ────────────────────────────────────────────────────────────────
    story += [h1_block("第一部分：KuCoin SEO 全景分析"), Spacer(1,5*mm)]

    story.append(Paragraph("1.1 规模数据", sH2))
    story.append(make_table(
        ["维度", "KuCoin 现状", "战略意义"],
        [
            ["博客页面", "273 页（持续更新）", "话题权威建立的内容密度"],
            ["多语言覆盖", "18 种语言", "全球流量覆盖，每语言独立 DR 积累"],
            ["Sitemap 条目", "1,000+", "规模化 SEO 运营的标志"],
            ["程序化页面", "60+ 法币 × N 币种 = 30,000+ 页", "长尾流量收割机"],
            ["技术框架", "Next.js SSR", "Google 爬取友好"],
            ["Domain Rating", "~75（Ahrefs 估算）", "高 DR 让新页面天然有排名优势"],
        ]
    ))
    story.append(Spacer(1,4*mm))

    story.append(Paragraph("1.2 KuCoin 的核心武器拆解", sH2))
    story.append(Paragraph("武器一：程序化 SEO（最强）", sH3))
    story.append(Paragraph(
        "KuCoin 不靠人工写作赢，靠批量生成页面赢。价格页（/price/BTC）、换算页（/converter/BTC-to-USD）、"
        "交易对页（/trade/BTC-USDT）等，估算 30,000+ 程序化页面，每页覆盖 1~3 个长尾关键词。"
        "这是人工写作永远追不上的规模。", sBody))

    story.append(Paragraph("武器二：内容集群 + 内链转化链路", sH3))
    for line in ["市场洞察文章（流量入口）→ 价格预测页（承接流量）→ 实时价格页（数据展示）→ 交易入口（转化终点）"]:
        story.append(Paragraph(line, sCode))
    story.append(Paragraph("每层都有内链，流量不浪费，权重向转化页汇聚。", sBody))

    story.append(Paragraph("1.3 KuCoin 的弱点（MoonX 的机会）", sH2))
    story.append(make_table(
        ["KuCoin 弱点", "MoonX 的机会"],
        [
            ["预测市场赛道完全空白", "MoonX 可成为该赛道 Google 唯一权威"],
            ["内容同质化（大量价格预测文章）", "Google HCU 持续降权此类内容"],
            ["品牌权重依赖", "新站在细分赛道反而更灵活"],
            ["多语言稀释了内容深度", "专注单语言精品内容更高效"],
        ]
    ))
    story.append(Spacer(1,3*mm))
    story.append(Paragraph(
        "核心判断：KuCoin 打的是交易所战争，MoonX 打的是预测市场聚合战争。"
        "MoonX 不需要赢过 KuCoin，只需要在预测市场赛道成为 Google 眼中的第一权威。",
        S("key", fontSize=10, textColor=C_BLUE, fontName=FONT_NAME, leading=16,
          backColor=colors.HexColor("#EBF5FF"), leftIndent=8, rightIndent=8,
          borderPad=8, spaceAfter=6)))
    story.append(PageBreak())

    # ── 第二部分 ────────────────────────────────────────────────────────────────
    story += [h1_block("第二部分：MoonX SEO 现状诊断"), Spacer(1,5*mm)]

    story.append(Paragraph("2.1 当前资产盘点", sH2))
    story.append(make_table(
        ["资产", "现状", "问题"],
        [
            ["已发文章", "4 篇（Medium 2 + Dev.to 2）", "发在第三方平台，权重不归我们"],
            ["目标关键词", "polymarket alternative、prediction market smart money", "方向正确，KD 可打"],
            ["发布平台", "Medium、Dev.to", "Dev.to canonical 指向 Medium，双重稀释"],
            ["官方博客", "无（开发中，3/20 上线）", "最大的基础设施缺口"],
            ["GSC 接入", "未验证", "没有数据就没有优化依据"],
            ["GA4 接入", "未确认", "无法追踪转化路径"],
            ["内链结构", "无", "文章之间没有形成集群"],
            ["外链", "0", "文章发完没有推广获取外链"],
        ]
    ))
    story.append(Spacer(1,4*mm))

    story.append(Paragraph("2.2 关键词竞争力评估", sH2))
    story.append(make_table(
        ["关键词", "月搜索量（估算）", "KD", "当前能打吗？"],
        [
            ["polymarket alternative", "1,000~3,000", "~25", "✅ 可以打"],
            ["prediction market aggregator", "200~500", "~15", "✅ 强烈推荐"],
            ["kalshi vs polymarket", "500~1,500", "~20", "✅ 可以打"],
            ["crypto prediction market", "1,000~3,000", "~30", "✅ 可以打"],
            ["polymarket smart money", "300~800", "~20", "✅ 可以打"],
            ["best prediction market", "2,000~5,000", "~35", "⚠️ 边缘可打"],
            ["prediction market", "10,000~30,000", "~55", "❌ DR 不够"],
        ]
    ))
    story.append(PageBreak())

    # ── 第三部分 ────────────────────────────────────────────────────────────────
    story += [h1_block("第三部分：MoonX SEO 完整战略"), Spacer(1,5*mm)]

    story.append(Paragraph("3.1 战略定位", sH2))
    story.append(Paragraph(
        "成为 Google 眼中【预测市场聚合】赛道的唯一权威。不追 KuCoin 的规模，"
        "专注预测市场细分赛道，用 6 个月时间建立话题权威（Topical Authority）。",
        S("pos", fontSize=10.5, textColor=C_DARK, fontName=FONT_NAME, leading=17,
          backColor=colors.HexColor("#FFF8E7"), leftIndent=8, borderPad=8, spaceAfter=6)))

    story.append(Paragraph("3.2 三层 SEO 架构", sH2))
    for layer in [
        "第一层：内容权威（Topical Authority）— Pillar Page + Cluster Pages，覆盖预测市场所有搜索意图",
        "第二层：程序化规模（Programmatic SEO）— 平台聚合页 + 话题页批量生成",
        "第三层：数据驱动优化（Data Intelligence）— GSC + GA4 + Twitter 数据管道，Claude Code 分析",
    ]:
        story.append(Paragraph(f"• {layer}", sBullet))
    story.append(Spacer(1,4*mm))

    story.append(Paragraph("3.3 内容集群规划", sH2))
    story.append(Paragraph("核心 Pillar Pages（支柱页）", sH3))
    story.append(make_table(
        ["Pillar Page", "目标 URL", "目标关键词", "KD", "字数"],
        [
            ["Polymarket Alternative", "bydfi.com/en/moonx/blog/polymarket-alternative", "polymarket alternative", "25", "2,000~2,500"],
            ["Prediction Market Aggregator", "bydfi.com/en/moonx/blog/prediction-market-aggregator", "prediction market aggregator", "15", "1,800~2,000"],
        ]
    ))
    story.append(Spacer(1,3*mm))
    story.append(Paragraph("Cluster Pages（集群页）", sH3))
    story.append(make_table(
        ["文章标题", "目标关键词", "KD", "状态"],
        [
            ["How Prediction Market Smart Money Moves", "prediction market smart money", "20", "✅ 已有（需迁移）"],
            ["Kalshi vs Polymarket: Which Is Better?", "kalshi vs polymarket", "18", "待写"],
            ["Best Crypto Prediction Markets in 2026", "crypto prediction market", "30", "待写"],
            ["How to Read Polymarket Odds", "how to read polymarket odds", "15", "待写"],
            ["Manifold Markets Review 2026", "manifold markets review", "20", "待写"],
            ["What Is a Prediction Market?", "what is a prediction market", "25", "待写"],
            ["Polymarket Fees Explained", "polymarket fees", "15", "待写"],
        ]
    ))
    story.append(PageBreak())

    # ── 第四部分 ────────────────────────────────────────────────────────────────
    story += [h1_block("第四部分：程序化 SEO 计划"), Spacer(1,5*mm)]

    story.append(Paragraph("4.1 平台聚合页（第一批，3 个）", sH2))
    story.append(make_table(
        ["URL", "目标关键词", "月搜索估算", "内容核心"],
        [
            ["/en/moonx/markets/polymarket", "polymarket markets today", "2,000+", "Polymarket 热门市场实时展示 + MoonX 聚合优势"],
            ["/en/moonx/markets/kalshi", "kalshi markets", "1,500+", "Kalshi 合规市场展示 + 对比分析"],
            ["/en/moonx/markets/manifold", "manifold markets", "1,000+", "Manifold 玩法 + MoonX 跨平台价值"],
        ]
    ))
    story.append(Spacer(1,4*mm))

    story.append(Paragraph("4.2 话题聚合页（第二批，10 个）", sH2))
    story.append(make_table(
        ["URL", "目标关键词", "月搜索估算"],
        [
            ["/en/moonx/topic/us-politics", "political prediction market", "3,000+"],
            ["/en/moonx/topic/crypto", "crypto price prediction market", "2,500+"],
            ["/en/moonx/topic/elections", "election prediction market odds", "3,500+"],
            ["/en/moonx/topic/sports", "sports prediction market", "2,000+"],
            ["/en/moonx/topic/ai", "AI prediction market", "1,500+"],
            ["/en/moonx/topic/economics", "economic prediction market", "1,000+"],
            ["/en/moonx/topic/tech", "tech prediction market", "800+"],
            ["/en/moonx/topic/entertainment", "entertainment prediction market", "600+"],
            ["/en/moonx/topic/weather", "weather prediction market", "500+"],
            ["/en/moonx/topic/science", "science prediction market", "400+"],
        ]
    ))
    story.append(PageBreak())

    # ── 第五部分 ────────────────────────────────────────────────────────────────
    story += [h1_block("第五部分：SEO 数据体系（Claude Code 指挥中心）"), Spacer(1,5*mm)]

    story.append(Paragraph("5.1 项目结构", sH2))
    for line in [
        "moonx-seo/",
        "├── config.json                 # MoonX 配置文件",
        "├── fetchers/",
        "│   ├── fetch_gsc.py            # GSC：关键词排名、展示量、点击率",
        "│   ├── fetch_ga4.py            # GA4：流量来源、页面表现、转化",
        "│   └── fetch_twitter.py        # Twitter：推文热度、话题趋势",
        "├── data/",
        "│   ├── gsc/                    # 关键词数据 JSON",
        "│   ├── ga4/                    # 流量数据 JSON",
        "│   └── twitter/                # 推文数据 JSON",
        "├── reports/                    # Claude Code 生成的分析报告",
        "└── run_fetch.py                # 一键拉取所有数据",
    ]:
        story.append(Paragraph(line, sCode))
    story.append(Spacer(1,4*mm))

    story.append(Paragraph("5.2 搭建步骤（按优先级）", sH2))
    story.append(make_table(
        ["步骤", "任务", "时间"],
        [
            ["Step 1", "GSC 接入：Cloud Console 创建项目，启用 Search Console API，服务账户加入 GSC 属性", "3/20 博客上线当天"],
            ["Step 2", "GA4 接入：确认 tracking code，启用 Analytics Data API，服务账户加入 GA4 属性", "3/20 当天"],
            ["Step 3", "Twitter 数据接入：复用 tweet_bot.py 认证，写 fetch_twitter.py", "3/27"],
            ["Step 4", "第一次跨源分析：GSC 关键词 vs Twitter 话题热点对比报告", "4/03"],
        ]
    ))
    story.append(Spacer(1,4*mm))

    story.append(Paragraph("5.3 高价值跨源分析问题库", sH2))
    questions = [
        "【GSC × GA4】哪些页面展示量高但点击率低（CTR<2%）？对应 GA4 跳出率是多少？",
        "【GSC × Twitter】过去 7 天 Twitter 预测市场热门话题，对应哪些 GSC 关键词？有没有内容缺口？",
        "【内容缺口】对比已有文章关键词，找出月搜索量 >500 但完全没有内容覆盖的词。",
        "【竞品差距】Polymarket 和 Kalshi 在哪些关键词排名前 10 但我们完全缺席？",
    ]
    for q in questions:
        story.append(Paragraph(f"• {q}", sBullet))
    story.append(PageBreak())

    # ── 第六部分 ────────────────────────────────────────────────────────────────
    story += [h1_block("第六部分：外链建设计划"), Spacer(1,5*mm)]
    story.append(make_table(
        ["渠道", "方法", "预期效果", "ROI"],
        [
            ["KOL 引用", "发布后推给 3~5 个预测市场 KOL，请求引用", "每篇获得 1~3 条高 DR 外链", "⭐⭐⭐⭐⭐"],
            ["Reddit 内容营销", "r/PredictionMarkets、r/Crypto 发有价值内容", "低成本，高相关性外链", "⭐⭐⭐⭐"],
            ["行业目录", "提交 MoonX 到 CoinGecko、DeFiLlama 等", "稳定高 DR 外链", "⭐⭐⭐⭐"],
            ["竞品外链复制", "找 Polymarket 外链来源，逐一争取", "中期稳定外链", "⭐⭐⭐"],
            ["数据报告", "发布独家市场数据报告，媒体主动引用", "高质量外链爆发", "⭐⭐⭐⭐⭐"],
        ]
    ))
    story.append(PageBreak())

    # ── 第七部分 ────────────────────────────────────────────────────────────────
    story += [h1_block("第七部分：12 周执行计划"), Spacer(1,5*mm)]

    story.append(Paragraph("Phase 1：地基建设（Week 1~3，当前阶段）", sH2))
    story.append(make_table(
        ["周次", "任务", "负责", "衡量标准"],
        [
            ["Week 1 (3/06~3/10)", "Pillar Page 初稿完成，交 Kelly 审核", "SEO", "2,000 字初稿"],
            ["Week 1", "fetch_gsc.py + fetch_ga4.py 代码完成", "Tech", "代码就绪"],
            ["Week 2 (3/13~3/17)", "Kelly 审核完，Cluster 1+2 初稿完成", "SEO+Kelly", "3 篇内容备好"],
            ["Week 3 (3/20)", "博客上线，3 篇文章当天发布", "Tech+SEO", "Google 收录"],
            ["Week 3 (3/20)", "GSC + GA4 验证，数据开始跑", "Tech", "数据可读"],
        ]
    ))
    story.append(Spacer(1,3*mm))

    story.append(Paragraph("Phase 2：规模扩张（Week 4~8）", sH2))
    story.append(make_table(
        ["周次", "任务", "衡量标准"],
        [
            ["Week 4", "程序化平台聚合页开发（3个）", "3 个页面上线"],
            ["Week 5", "Pillar 2 发布（prediction market aggregator）", "发布 + Google 收录"],
            ["Week 6", "第一批程序化话题页（5个）", "5 个话题页上线"],
            ["Week 7", "外链 Outreach 第一轮（KOL 引用）", "获得 5+ 条外链"],
            ["Week 8", "第二批程序化话题页（5个）", "累计 13 个程序化页"],
        ]
    ))
    story.append(Spacer(1,3*mm))

    story.append(Paragraph("Phase 3：数据驱动优化（Week 9~12）", sH2))
    story.append(make_table(
        ["周次", "任务", "衡量标准"],
        [
            ["Week 9", "第一次完整跨源分析报告", "识别优化机会清单"],
            ["Week 10", "根据 GSC 数据优化标题/描述", "CTR 提升 >10%"],
            ["Week 11", "补写内容缺口关键词文章（3篇）", "3 篇发布"],
            ["Week 12", "Q1 SEO 复盘 + Q2 计划调整", "复盘报告产出"],
        ]
    ))
    story.append(PageBreak())

    # ── 第八部分 ────────────────────────────────────────────────────────────────
    story += [h1_block("第八部分：OKR 与衡量标准"), Spacer(1,5*mm)]

    story.append(Paragraph("Q1 OKR（剩余 4 周）", sH2))
    story.append(make_table(
        ["KR", "目标值", "衡量方式"],
        [
            ["KR1", "Pillar Page 发布并 Google 收录", "GSC 确认"],
            ["KR2", "polymarket alternative 进入前 50", "SerpAPI 追踪"],
            ["KR3", "官方博客累计 6+ 篇文章", "实际发布数"],
            ["KR4", "GSC + GA4 数据管道跑通", "能产出分析报告"],
            ["KR5", "获得 5+ 条外链", "Ahrefs 外链报告"],
        ]
    ))
    story.append(Spacer(1,4*mm))

    story.append(Paragraph("Q2 OKR（预设）", sH2))
    story.append(make_table(
        ["KR", "目标值"],
        [
            ["KR1", "自然流量 >5,000/月"],
            ["KR2", "polymarket alternative 进入前 20"],
            ["KR3", "程序化页面上线 13+ 个"],
            ["KR4", "Domain Rating 提升 5+ 点"],
            ["KR5", "SEO 带来的产品注册 >200/月"],
        ]
    ))
    story.append(PageBreak())

    # ── 第九部分 ────────────────────────────────────────────────────────────────
    story += [h1_block("第九部分：不做清单"), Spacer(1,5*mm)]
    story.append(make_table(
        ["不做的事", "原因"],
        [
            ["多语言内容", "DR 低阶段多语言稀释权重，Q2 后评估"],
            ["继续只发 Medium/Dev.to", "权重不归我们，停止作为主渠道"],
            ["打高 KD 关键词（KD>50）", "DR 不够，打了也白打"],
            ["价格预测类文章", "Google HCU 降权，低质量陷阱"],
            ["付费广告（现阶段）", "SEO 建立权威期，广告 ROI 低"],
            ["独立域名 moonx.com", "DR 从零开始，晚 12 个月出效果"],
            ["AI 可见度追踪付费工具", "用 Bing Webmaster Tools 免费方案先跑"],
        ]
    ))
    story.append(PageBreak())

    # ── 第十部分 ────────────────────────────────────────────────────────────────
    story += [h1_block("第十部分：已确认事项与下一步"), Spacer(1,5*mm)]
    story.append(make_table(
        ["事项", "Kelly 决策", "影响"],
        [
            ["博客 URL", "bydfi.com/en/moonx/blog/（已确认）", "继承 BYDFi 主域 DR，MoonX 品牌清晰"],
            ["博客上线时间", "2 周，3/20 上线", "倒排所有执行计划"],
            ["内容发布方式", "审核后发布", "流程：SEO 初稿 → Kelly 审核 → 发布"],
            ["GSC 验证", "3/20 与博客同步完成", "需技术团队配合加 meta tag"],
        ]
    ))
    story.append(Spacer(1,5*mm))

    story.append(Paragraph("下一步行动（本周）", sH2))
    actions = [
        "SEO 专家：3/10 前完成 Pillar Page 初稿（polymarket-alternative，2,000字），发给 Kelly 审核",
        "Kelly：通知开发团队，博客上线需同步配置 GSC 验证 meta tag",
        "SEO 专家：搭好 fetch_gsc.py + fetch_ga4.py，3/20 当天数据立刻开始跑",
        "Kelly：3/13 前完成 Pillar Page 审核，3/15 前完成 Cluster 1+2 审核",
    ]
    for i, a in enumerate(actions, 1):
        story.append(Paragraph(f"{i}. {a}", sBullet))

    story.append(Spacer(1,8*mm))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BORDER))
    story.append(Spacer(1,3*mm))
    story.append(Paragraph(
        "MoonX SEO 完整战略计划书 v1.0　|　Team Lead + SEO 专家　|　2026-03-06",
        S("footer", fontSize=8, textColor=C_GRAY, fontName=FONT_NAME, alignment=TA_CENTER, leading=12)
    ))

    return story

# ── 生成 PDF ────────────────────────────────────────────────────────────────────
OUTPUT = "/Users/coco/agent-twitter/outbox/MoonX_SEO完整战略计划书_v1.0.pdf"

def on_page(canvas, doc):
    """页眉页脚"""
    canvas.saveState()
    # 页眉
    canvas.setFillColor(C_BLUE)
    canvas.rect(0, H - 10*mm, W, 10*mm, fill=1, stroke=0)
    canvas.setFillColor(C_WHITE)
    canvas.setFont(FONT_NAME, 8)
    canvas.drawString(MARGIN, H - 6.5*mm, "MoonX SEO 完整战略计划书 v1.0")
    canvas.drawRightString(W - MARGIN, H - 6.5*mm, "2026-03-06 | Confidential")
    # 页脚
    canvas.setFillColor(C_LIGHT)
    canvas.rect(0, 0, W, 8*mm, fill=1, stroke=0)
    canvas.setFillColor(C_GRAY)
    canvas.setFont(FONT_NAME, 8)
    canvas.drawCentredString(W/2, 3*mm, f"第 {doc.page} 页")
    canvas.restoreState()

doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=18*mm, bottomMargin=14*mm,
    title="MoonX SEO 完整战略计划书",
    author="Team Lead + SEO 专家",
)

story = cover_page() + toc() + build_content()
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"✅ PDF 已生成：{OUTPUT}")
