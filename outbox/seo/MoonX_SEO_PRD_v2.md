# MoonX SEO 页面体系 产品需求文档 PRD v2.0
**版本：2.0 | 日期：2026-03-11**
**产品负责人：Kelly**
**文档状态：待评审**
**依据：SEO全站规范 + 基础能力优化P1 + GSC数据 + 竞品分析**

---

## 一、页面体系总览

MoonX SEO 由 **2 类人工内容** + **5 类程序化内容** 构成：

```
/en/moonx/
│
├── 【人工内容 A】预测学院
│   └── /en/moonx/learn/prediction/
│       ├── /en/moonx/learn/prediction/                  分类首页
│       └── /en/moonx/learn/prediction/{slug}            学院文章（15-20篇）
│
├── 【人工内容 B】Meme 学院
│   └── /en/moonx/learn/meme/
│       ├── /en/moonx/learn/meme/                        分类首页
│       └── /en/moonx/learn/meme/{slug}                  学院文章（10-15篇）
│
├── 【程序化 1】如何参与预测市场（事件级）
│   └── /en/moonx/guide/prediction/{event-slug}          每个热门预测市场一页
│
├── 【程序化 2】如何参与美股预测市场（股票级）
│   └── /en/moonx/markets/stocks/{ticker}                每支股票一页
│
├── 【程序化 3】如何购买 Meme
│   └── /en/moonx/guide/buy/{token-slug}                 每个 Token 一页
│
├── 【程序化 4】Token 行情页（已有架构）
│   └── /en/moonx/solana/token/{contract}
│
└── 【程序化 5】热门榜单页（分类展开）
    └── /en/moonx/markets/trending/{category}
```

---

## 二、人工内容 A — 预测学院

### 2.1 定位

攻打所有「预测市场」相关中高频词，建立 MoonX 在 Google 眼中的预测市场权威地位。

---

### 2.2 分类首页

**URL：** `/en/moonx/learn/prediction/`

**页面内容结构：**

```
─────────────────────────────────────────────────────────────
[H1] Prediction Market Academy — Learn to Trade  (SSR)

[说明段落]                                        (SSR)
Everything you need to know about prediction
markets: how they work, which platforms to use,
and how to find the best odds.

[文章分类 Tab]
  All | Beginner | Platform Guides | Strategies | US Stocks

[文章卡片列表]                                    (SSR)
┌────────────────────────────────────────────────┐
│ [分类标签] Beginner                             │
│ What Is a Prediction Market? Complete Guide    │
│ 5 min read · Updated 2026-03                   │
├────────────────────────────────────────────────┤
│ [分类标签] Platform Guide                      │
│ Polymarket vs Kalshi: Full Comparison          │
│ 8 min read · Updated 2026-03                   │
└────────────────────────────────────────────────┘
─────────────────────────────────────────────────────────────
```

---

### 2.3 文章页内容结构

```
Breadcrumb: Home > MoonX > Prediction Academy > {文章标题}
─────────────────────────────────────────────────────────────
Main（720px）                        Side（260px）
─────────────────────────────────────────────────────────────
[H1] 文章标题（含目标关键词）  (SSR)  ┌───────────────────────┐
                                      │  目录 TOC（sticky）   │
[Intro: 100-150词]          (SSR)     │  1. Section 1         │
说清楚文章回答什么问题                │  2. Section 2         │
                                      │  3. ...               │
[H2] Section 1              (SSR)     └───────────────────────┘
[正文]
                                      ┌───────────────────────┐
[对比表格]（如有）           (SSR)     │  MoonX CTA 卡片       │
                                      │  Compare odds on      │
[H2] Section 2              (SSR)     │  Polymarket & Kalshi  │
[正文]                                │                       │
                                      │  [Browse Markets →]   │
[H2] Section N              (SSR)     └───────────────────────┘
[正文]
                                      ┌───────────────────────┐
[FAQ] 5-8 问答              (SSR)     │  Related Articles     │
JSON-LD FAQPage Schema                │  • Article 1          │
                                      │  • Article 2          │
[CTA Banner]                (SSR)     │  • Article 3          │
→ Compare Markets on MoonX            └───────────────────────┘

[Related Articles]          (SSR)
← → Cluster 文章互链
─────────────────────────────────────────────────────────────
```

---

### 2.4 文章清单与关键词规格

**支柱页（Pillar）：**

| Slug | H1 | 目标关键词 | 字数 | 优先级 |
|------|----|----------|------|--------|
| `prediction-markets-guide` | Prediction Markets: The Complete Guide 2026 | what are prediction markets | 4,000 | **P0** |

**Cluster 文章：**

| # | Slug | H1 | 目标关键词 | 月搜索量 | 字数 | 优先级 |
|---|------|----|----------|---------|------|--------|
| 1 | `prediction-market-aggregator` | What Is a Prediction Market Aggregator? | prediction market aggregator | 1K-5K | 1,800 | **P0** |
| 2 | `polymarket-alternatives` | Best Polymarket Alternatives 2026 | polymarket alternative | 10K-50K | 2,200 | **P0** |
| 3 | `polymarket-vs-kalshi` | Polymarket vs Kalshi: Full Comparison | polymarket vs kalshi | 10K-50K | 2,000 | P1 |
| 4 | `how-to-trade-prediction-markets` | How to Trade Prediction Markets for Beginners | how to trade prediction markets | 5K-20K | 2,000 | P1 |
| 5 | `prediction-market-odds` | Prediction Market Odds Explained | prediction market odds | 1K-5K | 1,500 | P2 |
| 6 | `kalshi-review` | Kalshi Review 2026: Is It Worth Using? | kalshi review | 5K-10K | 2,000 | P1 |
| 7 | `polymarket-review` | Polymarket Review 2026 | polymarket review | 5K-20K | 2,000 | P2 |
| 8 | `us-election-prediction-market` | How to Bet on US Elections Using Prediction Markets | election prediction market | 10K-50K | 2,000 | P1 |
| 9 | `stock-prediction-markets` | How to Trade Stock Price Outcomes on Prediction Markets | stock prediction market | 5K-20K | 2,000 | P1 |

**内链规则：**
```
所有 Cluster 文章 → 支柱页 prediction-markets-guide（必须）
所有文章结尾 CTA → /en/moonx/markets/trending
美股相关文章 → /en/moonx/markets/stocks/{ticker}（程序化页面）
```

---

### 2.5 目标关键词矩阵（预测学院）

| 类型 | 关键词 | 月搜索量 | 竞争 | 目标页面 |
|------|--------|---------|------|---------|
| 品类独占 | prediction market aggregator | 1K-5K | 极低 | /learn/prediction/prediction-market-aggregator |
| 替代词 | polymarket alternative | 10K-50K | 中 | /learn/prediction/polymarket-alternatives |
| 对比词 | polymarket vs kalshi | 10K-50K | 中 | /learn/prediction/polymarket-vs-kalshi |
| 通用权威 | what are prediction markets | 50K+ | 高 | /learn/prediction/prediction-markets-guide |
| 入门词 | how to trade prediction markets | 5K-20K | 中 | /learn/prediction/how-to-trade-prediction-markets |
| 热门场景 | election prediction market | 10K-50K | 中 | /learn/prediction/us-election-prediction-market |
| 美股场景 | stock prediction market | 5K-20K | 中 | /learn/prediction/stock-prediction-markets |

---

## 三、人工内容 B — Meme 学院

### 3.1 定位

攻打 Solana meme 币、pump.fun、链上工具相关词。与 Token 行情页形成内链闭环。

---

### 3.2 分类首页

**URL：** `/en/moonx/learn/meme/`

**页面内容结构：**

```
─────────────────────────────────────────────────────────────
[H1] Meme Coin Academy — Solana Trading Guide     (SSR)

[说明段落]                                         (SSR)
Learn how to find, analyze, and trade Solana meme
coins. From pump.fun launches to smart money
tracking—everything in one place.

[文章分类 Tab]
  All | Getting Started | Tools | Strategies | Safety

[文章卡片列表]                                     (SSR)
─────────────────────────────────────────────────────────────
```

---

### 3.3 文章清单与关键词规格

**支柱页：**

| Slug | H1 | 目标关键词 | 字数 | 优先级 |
|------|----|----------|------|--------|
| `solana-meme-coin-guide` | Solana Meme Coin Trading: The Complete Guide 2026 | solana meme coin | 4,000 | **P0** |

**Cluster 文章：**

| # | Slug | H1 | 目标关键词 | 月搜索量 | 字数 |
|---|------|----|----------|---------|------|
| 1 | `what-is-smart-money-tracking` | What Is Smart Money Tracking in Crypto? | smart money tracker crypto | 5K-20K | 1,800 |
| 2 | `gmgn-alternative` | GMGN vs MoonX: Which Solana Tracker Is Better? | GMGN alternative | 1K-5K | 2,000 |
| 3 | `best-solana-meme-tracker` | Best Solana Meme Coin Trackers 2026 | best solana meme coin tracker | 2K-10K | 2,000 |
| 4 | `how-to-use-pump-fun` | How to Use Pump.fun: Complete Beginner Guide | how to use pump.fun | 5K-20K | 2,000 |
| 5 | `how-to-find-100x-meme-coin` | How to Find the Next 100x Meme Coin on Solana | how to find 100x meme coin | 2K-10K | 2,500 |
| 6 | `solana-meme-coin-risks` | Solana Meme Coin Risks: What You Need to Know | meme coin risks solana | 1K-5K | 1,500 |
| 7 | `what-are-meme-coins` | What Are Meme Coins? Complete Beginner Guide | what are meme coins | 10K-50K | 2,000 |

**内链规则：**
```
所有 Cluster → 支柱页 solana-meme-coin-guide
所有文章 CTA → /en/moonx/markets/trending 或 /en/moonx/pump
Smart money 类文章 → /en/moonx/monitor/dynamic/hot
Token 工具类文章 → /en/moonx/solana/token/{热门token}
```

---

## 四、程序化 1 — 如何参与预测市场（事件级）

### 4.1 定位

为每个热门预测市场事件生成独立指南页，捕获 `how to bet on [event]`、`[event] prediction market` 类搜索词。

**URL 结构：** `/en/moonx/guide/prediction/{event-slug}`

**示例：**
- `/en/moonx/guide/prediction/trump-2028-election`
- `/en/moonx/guide/prediction/fed-rate-march-2026`
- `/en/moonx/guide/prediction/super-bowl-2026`

---

### 4.2 触发与生成规则

| 维度 | 规则 |
|------|------|
| 触发条件 | Polymarket 或 Kalshi 上交易量 > $10,000 的市场 |
| 生成时间 | 市场上线后 24 小时内自动生成 |
| 下线规则 | 市场结束后 30 天内转为 noindex |
| 更新频率 | 市场数据（赔率/成交量）每日更新 |

---

### 4.3 页面内容结构

```
Breadcrumb: Home > MoonX > Prediction Guides > {Event Title}
─────────────────────────────────────────────────────────────
Main（720px）                        Side（280px）
─────────────────────────────────────────────────────────────
[H1]                         (SSR)   ┌───────────────────────┐
How to Trade {EVENT_TITLE}           │  Live Odds            │
Prediction Market                    │                       │
                                     │  YES  [██████] 68%    │
[Current Odds Block]         (SSR)   │  NO   [████  ] 32%    │
Market: {EVENT_TITLE}                │                       │
Platform: Polymarket / Kalshi        │  Vol: $X.XXM          │
YES: {PCT}% | NO: {PCT}%             │  [Trade on MoonX →]   │
Volume: ${VOLUME}                    └───────────────────────┘
Closes: {DATE}
                                     ┌───────────────────────┐
[说明段落]                   (SSR)   │  Compare Odds         │
{EVENT_TITLE} is one of the          │                       │
most-traded prediction markets       │  Polymarket  68%      │
right now...（~150词模板）           │  Kalshi      65%      │
                                     │  Manifold    70%      │
[H2] What Is This Market?   (SSR)   │                       │
{EVENT_DESC}（模板文字）             │  Best entry: Manifold  │
                                     └───────────────────────┘
[H2] How to Trade           (SSR)
Step 1: Connect to MoonX
Step 2: Find {EVENT_SHORT}
Step 3: Choose YES or NO
Step 4: Enter amount & confirm

[H2] Platform Comparison    (SSR)
| Platform | Odds | Fees | US? |
|----------|------|------|-----|
| Polymarket | {PCT}% | ~2% | ❌ |
| Kalshi | {PCT}% | ~6% | ✅ |

[H2] Price History          (SSR+CSR)
[历史赔率图表]（ApexCharts，CSR）
Key levels: {HIGH_ODDS}% high,
{LOW_ODDS}% low

[FAQ] 5 问答                (SSR)
JSON-LD FAQPage Schema

[Related Markets]           (CSR)
→ 同类热门预测市场
─────────────────────────────────────────────────────────────
```

---

### 4.4 关键词覆盖

| 关键词模式 | 示例 | 月搜索量级 |
|-----------|------|-----------|
| `{event} prediction market` | trump 2028 prediction market | 1K-100K |
| `how to bet on {event}` | how to bet on fed rate cut | 500-10K |
| `{event} odds` | super bowl 2026 odds prediction market | 1K-50K |
| `{event} polymarket` | election polymarket odds | 1K-20K |
| `{event} kalshi` | fed rate kalshi | 500-5K |

---

### 4.5 内链网络

```
/en/moonx/guide/prediction/{event-slug}
  ↔  /en/moonx/markets/trending                    （热门市场列表）
  →  /en/moonx/learn/prediction/how-to-trade-prediction-markets
  →  /en/moonx/learn/prediction/polymarket-vs-kalshi
  →  /en/moonx/guide/prediction/{related-event}    （同类其他市场）
```

---

## 五、程序化 2 — 美股预测市场页

### 5.1 定位

为每支有 Kalshi 预测市场的股票生成独立页面，结合股票实时价格 + 相关预测市场，捕获 `{TICKER} prediction market` / `{TICKER} price prediction` 类搜索词。

**URL 结构：** `/en/moonx/markets/stocks/{ticker}`

**示例：**
- `/en/moonx/markets/stocks/tsla`
- `/en/moonx/markets/stocks/googl`
- `/en/moonx/markets/stocks/nvda`
- `/en/moonx/markets/stocks/spy`（S&P 500 ETF）

---

### 5.2 触发规则

| 维度 | 规则 |
|------|------|
| 触发条件 | Kalshi 上有该股票相关的活跃预测市场 |
| 自动生成 | 有新股票市场上线时自动创建页面 |
| 更新频率 | 股票价格实时，预测市场数据每日更新 |

---

### 5.3 页面内容结构

```
Breadcrumb: Home > MoonX > Stock Markets > {TICKER}
─────────────────────────────────────────────────────────────
Main（720px）                        Side（280px）
─────────────────────────────────────────────────────────────
[H1]                         (SSR)   ┌───────────────────────┐
{TICKER} ({COMPANY}) Prediction      │  {TICKER} Live Price  │
Markets — Live Odds & Analysis       │  ${PRICE}             │
                                     │  24h: ▲ +X.X%         │
[Stock Price Card]           (SSR)   │                       │
Current: ${PRICE}                    │  [Trade Prediction    │
24h: +X.X% | 52w High: ${H}         │   Markets on MoonX →] │
52w Low: ${L}                        └───────────────────────┘
Source: Real-time market data
                                     ┌───────────────────────┐
[H2] Active Prediction       (SSR)   │  Quick Trade          │
Markets for {TICKER}                 │                       │
[预测市场列表，每条含：]              │  Will TSLA close      │
• "Will TSLA close above             │  above $300?          │
  $350 by March 31?"                 │  YES [████] 45%       │
  YES: 45% | Vol: $2.3M              │                       │
• "Will Tesla report Q1              │  [Trade YES / NO]     │
  revenue above $25B?"               └───────────────────────┘
  YES: 62% | Vol: $1.1M

[H2] {TICKER} Price          (SSR+CSR)
Prediction Analysis
[历史股价图表，1M/3M/1Y]    (CSR)
[程序化分析文字 ~200词]      (SSR)
基于当前预测市场隐含概率分析
股价走势...

[H2] Historical Accuracy     (SSR)
Past {TICKER} prediction markets
and their resolution outcomes
[历史表格]

[H2] How to Trade            (SSR)
{TICKER} Prediction Markets
Step 1-3 指南

[FAQ] 5-8 问答               (SSR)
• What is the {TICKER} prediction market?
• How do I bet on {TICKER} stock price?
• Is {TICKER} prediction market legal in the US?
JSON-LD FAQPage Schema

[Related Stocks]             (CSR)
→ 同类热门股票预测市场
─────────────────────────────────────────────────────────────
```

---

### 5.4 关键词覆盖

| 关键词模式 | 示例 | 月搜索量级 |
|-----------|------|-----------|
| `{TICKER} prediction market` | TSLA prediction market | 500-5K |
| `{TICKER} price prediction` | Tesla price prediction 2026 | 5K-50K |
| `{COMPANY} stock prediction` | Tesla stock prediction | 5K-50K |
| `bet on {TICKER}` | bet on Tesla stock | 500-5K |
| `{TICKER} kalshi` | TSLA kalshi | 500-2K |
| `{COMPANY} earnings prediction` | Tesla earnings prediction | 1K-10K |

**首批目标股票（有 Kalshi 活跃市场）：**
TSLA · GOOGL · NVDA · AAPL · META · AMZN · SPY · QQQ · BTC-related ETFs

---

### 5.5 内链网络

```
/en/moonx/markets/stocks/{ticker}
  ↔  /en/moonx/markets/trending                    （互链）
  →  /en/moonx/learn/prediction/stock-prediction-markets
  →  /en/moonx/guide/prediction/{related-event}    （该股票的具体预测市场）
  →  /en/moonx/learn/prediction/how-to-trade-prediction-markets
```

---

## 六、程序化 3 — 如何购买 Meme

### 6.1 定位

为每个满足条件的 Solana Token 生成独立的购买指南页，捕获 `how to buy {TOKEN}` 搜索词（高转化意图）。

**URL 结构：** `/en/moonx/guide/buy/{token-slug}`

**示例：**
- `/en/moonx/guide/buy/pepe-solana`
- `/en/moonx/guide/buy/wif-solana`
- `/en/moonx/guide/buy/bonk`

---

### 6.2 触发规则

| 维度 | 规则 |
|------|------|
| 触发条件 | 与 Token 行情页相同准入条件（LP≥$5K / Holders≥100 / 上线≥48h 等）|
| 特殊条件 | 优先为搜索量 > 500/月 的 Token 生成 |
| 与行情页关系 | 购买指南页 ↔ 行情页 互链，互补不重复 |

---

### 6.3 页面内容结构

```
Breadcrumb: Home > MoonX > Buy Crypto > How to Buy {TOKEN}
─────────────────────────────────────────────────────────────
Main（720px）                        Side（280px）
─────────────────────────────────────────────────────────────
[H1] How to Buy {TOKEN_SYMBOL}       ┌───────────────────────┐
on Solana — Step-by-Step    (SSR)    │  Buy {TOKEN} Now      │
                                     │                       │
[Quick Stats]               (SSR)    │  Price: ${PRICE}      │
Current Price: ${PRICE}              │  24h: ▲ +X%           │
24h Volume: ${VOL}                   │                       │
Market Cap: ${CAP}                   │  [Buy on MoonX →]     │
                                     └───────────────────────┘
[H2] What Is {TOKEN}?       (SSR)
~150词介绍（程序化模板）              ┌───────────────────────┐
                                     │  Price Chart          │
[H2] How to Buy {TOKEN}     (SSR)    │  [mini chart CSR]     │
Step 1: Set up a Solana wallet       └───────────────────────┘
Step 2: Fund with SOL
Step 3: Open MoonX
Step 4: Search {TOKEN}
Step 5: Enter amount & trade

[H2] Where to Buy {TOKEN}   (SSR)
| Platform | Type | Fees |
|----------|------|------|
| MoonX | DEX Aggregator | 0.3% |
| Raydium | DEX | 0.25% |

[H2] Is {TOKEN} Safe?       (SSR)
~100词风险说明（模板）

[FAQ] 5问答                 (SSR)
• How to buy {TOKEN} without KYC?
• What wallet do I need for {TOKEN}?
• What is {TOKEN} contract address?
JSON-LD FAQPage Schema

[Price Chart - Full]        (CSR)

[Related Tokens]            (CSR)
─────────────────────────────────────────────────────────────
```

---

### 6.4 关键词覆盖

| 关键词模式 | 示例 | 月搜索量级 |
|-----------|------|-----------|
| `how to buy {TOKEN}` | how to buy PEPE | 100-50K |
| `buy {TOKEN} solana` | buy WIF solana | 100-20K |
| `where to buy {TOKEN}` | where to buy BONK | 100-10K |
| `{TOKEN} how to purchase` | PEPE how to purchase | 100-5K |

---

### 6.5 内链网络

```
/en/moonx/guide/buy/{token-slug}
  ↔  /en/moonx/solana/token/{contract}         （行情页，互链）
  →  /en/moonx/pump                            （发现更多新币）
  →  /en/moonx/learn/meme/what-are-meme-coins  （教育内链）
  →  /en/moonx/monitor/dynamic/hot             （聪明钱验证）
```

---

## 七、程序化 4 — Token 行情页（扩展版）

**URL：** `/en/moonx/solana/token/{contract}`（已有架构）

在原有基础上新增四大价值板块（详见 PRD v1.2）：

| 板块 | 新增关键词覆盖 |
|------|-------------|
| Historical Price | `{TOKEN} price history` / `{TOKEN} all time high` |
| Price Prediction | `{TOKEN} price prediction 2026` |
| Calculator | `{TOKEN} to USD` / `1000 {TOKEN} in dollars` |
| Market Analysis | `{TOKEN} market analysis` / `{TOKEN} market cap` |

**与新程序化页面的关系：**
```
Token 行情页 ↔ 购买指南页（互链）
Token 行情页 → 预测学院文章（教育内链）
```

---

## 八、程序化 5 — 热门榜单页（分类展开）

### 8.1 从单一 Trending 页扩展为分类 Trending 体系

**URL 结构：**

| 页面 | URL | 目标关键词 |
|------|-----|-----------|
| 总榜 | `/en/moonx/markets/trending` | trending prediction markets |
| 政治类 | `/en/moonx/markets/trending/politics` | political prediction markets |
| 加密类 | `/en/moonx/markets/trending/crypto` | crypto prediction markets |
| 体育类 | `/en/moonx/markets/trending/sports` | sports prediction markets |
| 美股类 | `/en/moonx/markets/trending/stocks` | stock prediction markets today |
| 经济类 | `/en/moonx/markets/trending/economics` | economic prediction markets |
| Meme 热榜 | `/en/moonx/markets/trending/meme` | trending meme coins solana |

---

### 8.2 页面内容结构（以分类 Trending 为例）

```
Breadcrumb: Home > MoonX > Markets > Trending > Politics
─────────────────────────────────────────────────────────────
[H1] Trending Political Prediction Markets  (SSR)

[静态说明段落]                               (SSR)
Track the most active political prediction
markets right now. Compare odds from Polymarket,
Kalshi, and Manifold side by side.

[市场列表]                                   (CSR)
┌─────────────────────────────────────────────┐
│ Will Trump win 2028? YES 58% Vol $50M  ↑   │
│ Senate control 2026? DEM 52% Vol $20M  →   │
└─────────────────────────────────────────────┘

[FAQ] 3 问答                                 (SSR)
JSON-LD FAQPage Schema
─────────────────────────────────────────────────────────────
```

---

### 8.3 内链网络

```
/en/moonx/markets/trending/{category}
  ↔  /en/moonx/markets/trending               （总榜互链）
  ↔  /en/moonx/markets/stocks/{ticker}        （美股分类 ↔ 个股页面）
  →  /en/moonx/guide/prediction/{event}       （具体市场参与指南）
  →  /en/moonx/learn/prediction/{article}    （教育内容）
```

---

## 九、全站内链网络

```
                      ┌──────────────────────────┐
                      │      MoonX 首页           │
                      │   /en/moonx              │
                      └────────────┬─────────────┘
                                   │
          ┌──────────────┬─────────┴──────────┬──────────────┐
          │              │                    │              │
   ┌──────▼──────┐ ┌─────▼──────┐  ┌─────────▼──────┐ ┌────▼──────┐
   │  预测学院    │ │  Meme 学院  │  │  热门榜单体系  │ │ 功能落地页 │
   │/learn/      │ │/learn/     │  │/markets/       │ │/pump      │
   │prediction/  │ │meme/       │  │trending/       │ │/monitor.. │
   └──────┬──────┘ └─────┬──────┘  └────────┬───────┘ └───────────┘
          │              │                   │
     文章内链         文章内链          分类页内链
          │              │                   │
   ┌──────▼──────┐ ┌─────▼──────┐  ┌────────▼───────┐
   │  事件指南    │ │  购买指南   │  │  美股预测页    │
   │/guide/      │ │/guide/buy/ │  │/markets/       │
   │prediction/  │ │{token}     │  │stocks/{ticker} │
   └──────┬──────┘ └─────┬──────┘  └────────┬───────┘
          │              │                   │
          └──────────────┼───────────────────┘
                         │ 双向互链
                  ┌──────▼──────┐
                  │ Token 行情页 │
                  │/solana/token│
                  │/{contract}  │
                  └─────────────┘
```

**内链核心规则：**

| 规则 | 说明 |
|------|------|
| 学院文章 → 支柱页 | 每篇 Cluster 必须链接本学院支柱页 |
| 程序化页 → 学院 | 每个程序化页至少 1 个指向相关学院文章的教育内链 |
| 程序化页 → 产品页 | 每个程序化页有 1 个 CTA 指向 /markets/trending |
| 行情页 ↔ 购买指南 | 互链，覆盖不同搜索意图 |
| 美股页 ↔ 事件指南 | 互链，同一股票的不同角度 |

---

## 十、关键词矩阵（全量）

### A. 人工内容词（高价值，稳定流量）

| 关键词 | 月搜索量 | 竞争 | 优先级 | 目标页面 |
|--------|---------|------|--------|---------|
| what are prediction markets | 50K+ | 高 | P0 | /learn/prediction/prediction-markets-guide |
| polymarket alternative | 10K-50K | 中 | P0 | /learn/prediction/polymarket-alternatives |
| polymarket vs kalshi | 10K-50K | 中 | P1 | /learn/prediction/polymarket-vs-kalshi |
| prediction market aggregator | 1K-5K | 极低 | P0 | /learn/prediction/prediction-market-aggregator |
| what are meme coins | 10K-50K | 高 | P1 | /learn/meme/what-are-meme-coins |
| how to use pump.fun | 5K-20K | 中 | P1 | /learn/meme/how-to-use-pump-fun |
| smart money tracker crypto | 5K-20K | 中 | P0 | /monitor/dynamic/hot + /learn/meme/... |
| GMGN alternative | 1K-5K | 低 | P1 | /learn/meme/gmgn-alternative |

### B. 程序化词（规模化，长尾）

| 关键词模式 | 规模 | 竞争 | 目标页面 |
|-----------|------|------|---------|
| `{event} prediction market` | 每事件 1K-100K | 低 | /guide/prediction/{event} |
| `how to bet on {event}` | 每事件 500-10K | 低 | /guide/prediction/{event} |
| `{TICKER} prediction market` | 每股票 500-5K | 低 | /markets/stocks/{ticker} |
| `{TICKER} price prediction` | 每股票 5K-50K | 中 | /markets/stocks/{ticker} |
| `how to buy {TOKEN}` | 每 Token 100-50K | 极低 | /guide/buy/{token} |
| `{TOKEN} price solana` | 每 Token 100-50K | 极低 | /solana/token/{contract} |
| `{TOKEN} price prediction` | 每 Token 100-100K | 低 | /solana/token/{contract} |
| `{TOKEN} to USD` | 每 Token 50-10K | 极低 | /solana/token/{contract} |

---

## 十一、技术需求

### 11.1 SSR 改造优先级

| 优先级 | 页面类型 | 必须 SSR 的内容 |
|--------|---------|---------------|
| **P0** | /learn/prediction/ 及所有文章 | 全部正文 |
| **P0** | /learn/meme/ 及所有文章 | 全部正文 |
| **P0** | /solana/token/{contract} | H1、4大板块文字、FAQ |
| P1 | /guide/prediction/{event} | H1、当前赔率文字、How-to、FAQ |
| P1 | /markets/stocks/{ticker} | H1、价格文字、市场列表标题、FAQ |
| P1 | /guide/buy/{token} | H1、What Is 段落、How-to、FAQ |
| P1 | /markets/trending/{category} | H1、说明段落、FAQ |

### 11.2 程序化页面模板要求

每类程序化页面需要技术团队实现的：

| 页面类型 | 数据接口需求 | 模板变量 |
|---------|------------|---------|
| 事件指南 | Polymarket + Kalshi API：赔率、交易量、截止时间 | EVENT_TITLE, PCT, VOLUME, DATE |
| 美股页 | 股票实时价格 API + Kalshi 股票市场 API | TICKER, PRICE, CHANGE, 市场列表 |
| 购买指南 | Token 链上数据（同行情页数据源） | TOKEN_SYMBOL, PRICE, VOL, CONTRACT |
| 热门榜单分类 | 同 Trending 页数据，按 category 过滤 | CATEGORY, 市场列表 |

### 11.3 Sitemap 规格

| Sitemap 文件 | 内容 | 更新频率 |
|------------|------|---------|
| learn-prediction | /learn/prediction/ 所有文章 | 新增时更新 |
| learn-meme | /learn/meme/ 所有文章 | 新增时更新 |
| guide-prediction | /guide/prediction/ 有效事件页 | 每日更新 |
| markets-stocks | /markets/stocks/ 有效股票页 | 每周更新 |
| guide-buy | /guide/buy/ 有效购买指南页 | 每日更新 |
| token-pages | /solana/token/ 满足条件的 token | 每日更新 |

### 11.4 noindex 规则

| 页面类型 | noindex 条件 |
|---------|------------|
| 事件指南页 | 市场交易量 < $10,000 OR 已结束超过 30 天 |
| 美股页 | Kalshi 上该股票无活跃市场 |
| 购买指南页 | 同 Token 行情页准入条件 |
| Token 行情页 | LP<$5K / Holders<100 / 上线<48h 等 |

---

## 十二、分期计划

### Phase 1（第 1-4 周）：内容地基
- [ ] /learn/prediction/ 路径建好（SSR）
- [ ] /learn/meme/ 路径建好（SSR）
- [ ] 发布预测学院文章 #1-4
- [ ] 发布 Meme 学院文章 #1-3
- [ ] Token 行情页 SSR meta 注入

### Phase 2（第 5-8 周）：程序化启动
- [ ] 事件指南页模板上线（/guide/prediction/）
- [ ] 美股预测页模板上线（/markets/stocks/）
- [ ] Trending 分类页扩展（/markets/trending/{category}）
- [ ] 发布学院剩余文章

### Phase 3（第 9-16 周）：规模化
- [ ] 购买指南页模板上线（/guide/buy/）
- [ ] Token 行情页四大板块上线
- [ ] 所有 Sitemap 提交 GSC
- [ ] 内链网络完整建立

---

## 十三、验收标准

### 内容验收
- [ ] 预测学院：支柱页 + 8 篇 Cluster，全部 SSR 渲染，内链闭环
- [ ] Meme 学院：支柱页 + 6 篇 Cluster，全部 SSR 渲染，内链闭环
- [ ] 每篇文章：title 50-60字符，description 140-160字符，FAQPage Schema 无报错

### 程序化页面验收
- [ ] 事件指南页：赔率数据正确，模板变量全部替换，SSR 内容可见
- [ ] 美股页：股票价格实时，Kalshi 市场列表准确，FAQ 正常
- [ ] 购买指南页：与行情页互链正确，How-to 步骤完整
- [ ] 热门榜单分类：各分类数据正确，canonical 无重复内容

### 业务验收
- [ ] Phase 1 结束：GSC 中 /learn/ 页面出现展示量
- [ ] Phase 2 结束：程序化页面开始被索引，月自然流量 > 500
- [ ] Phase 3 结束：月自然流量 > 3,000，"polymarket alternative" 进 Top 50

---

*PRD v2.0 · 2026-03-11 · MoonX SEO 完整页面体系*
