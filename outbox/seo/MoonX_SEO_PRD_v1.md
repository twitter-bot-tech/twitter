# MoonX SEO 页面体系 产品需求文档（PRD）
**文档版本：v1.2**
**创建日期：2026-03-11**
**产品负责人：Kelly**
**文档状态：待评审**

---

## 目录
1. 需求背景
2. 产品目标
3. 用户场景
4. 信息架构
5. 页面详细设计
   - 5.1 Token 行情页
   - 5.2 Learn 学院文章页
   - 5.3 功能落地页（Trending / Smart Money / Pump）
6. 全站内链网络
7. 关键词矩阵
8. 内容规格
9. 技术需求
10. 数据埋点
11. 分期计划
12. 验收标准
13. 风险与依赖

---

## 一、需求背景

### 1.1 业务背景

MoonX 是 BYDFi 旗下预测市场聚合器，聚合 Polymarket、Kalshi、Manifold 数据，同时提供 Solana meme 代币交易功能。

**当前核心问题：MoonX 在 Google 的搜索可见度为零。**

| 指标 | 数值 |
|------|------|
| MoonX GSC 展示量（过去 28 天） | **0** |
| MoonX GSC 点击量 | **0** |
| bydfi.com 整站月展示量 | 7,880,000 |
| bydfi.com 整站平均 CTR | 0.6% |
| bydfi.com 整站平均排名 | 9.5 |

**根本原因：** MoonX 全页面 CSR 渲染，Google 爬虫抓到空壳 HTML，无法索引任何内容。

### 1.2 需求来源

- **业务需求：** 通过 SEO 建立可持续自然流量渠道
- **产品机会：** "prediction market aggregator" 无有效竞争对手，MoonX 可独占
- **技术机会：** Token 详情页架构已存在，加 SSR meta 注入即可规模化

---

## 二、产品目标

| 时间节点 | 月自然流量 | 有效索引页 | 核心词排名目标 |
|---------|-----------|----------|--------------|
| SSR 上线后 2 周 | > 0 | 10+ | — |
| M2 | 200-500 | 50+ | 开始出现 |
| M4 | 3,000+ | 500+ | polymarket alternative Top 50 |
| M6 | 15,000+ | 2,000+ | polymarket alternative Top 10 |
| M12 | 60,000+ | 10,000+ | polymarket alternative Top 3 |

**流量来源拆解（M12）：**

| 来源 | 占比 | 月流量 |
|------|------|--------|
| Token 行情页（规模化长尾） | 80% | 48,000+ |
| Learn 学院页（内容权威词） | 15% | 9,000+ |
| 功能落地页（产品功能词） | 5% | 3,000+ |

---

## 三、用户场景

### 场景 1：搜索预测市场平台的用户
> 搜索词：polymarket alternative / polymarket vs kalshi / best prediction market
> 用户在评估平台选择，希望找到对比文章做决策。
> **目标页面：** Learn 学院文章页（对比 / 替代品文章）

### 场景 2：搜索代币价格的用户
> 搜索词：{TOKEN} price solana / buy {TOKEN} pump.fun
> 用户想查实时价格和聪明钱动向，并在同一页面交易。
> **目标页面：** Token 行情页

### 场景 3：了解预测市场的新用户
> 搜索词：what are prediction markets / prediction market aggregator
> 用户刚接触预测市场，想了解基础知识和入门平台。
> **目标页面：** Learn 学院文章页（教育类文章）

---

## 四、信息架构

```
bydfi.com/en/moonx/
│
├── /en/moonx                              首页（功能介绍 + 品牌词）
│
├── 功能页
│   ├── /en/moonx/markets/trending         热门预测市场
│   ├── /en/moonx/monitor/dynamic/hot      聪明钱追踪
│   └── /en/moonx/pump                     Pump.fun 新币
│
├── Token 行情页（程序化，规模 500-10,000+）
│   └── /en/moonx/solana/token/{contract}
│
└── 学院页
    ├── /en/moonx/learn/                   学院首页（文章列表）
    └── /en/moonx/learn/{slug}             每篇文章（9篇）
```

---

## 五、页面详细设计

---

### 5.1 Token 行情页

**路径：** `/en/moonx/solana/token/{contract_address}`
**数量：** 500-10,000+（程序化生成）
**目标关键词：** `{TOKEN} price solana` / `buy {TOKEN}` / `{TOKEN} pump.fun`

---

#### 5.1.1 页面内容结构

```
Breadcrumb: Home > MoonX > Solana Tokens > {TOKEN_SYMBOL}
─────────────────────────────────────────────────────────────────
Main（700px）                          Side（340px）
─────────────────────────────────────────────────────────────────
[H1]                    (SSR)          ┌───────────────────────┐
{TOKEN_SYMBOL} ({TOKEN_NAME})          │  Quick Trade          │
— Live Price & Trading                 │  [{TOKEN}] → [USDC]   │
                                       │  Amount: [_________]  │
[Price Card]            (SSR)          │  ≈ $XX.XX             │
Current: $X.XXXXX                      │  [Trade on MoonX]     │
24h: ▲ +X.X%  |  Vol: $X.XXM          └───────────────────────┘
ATH: $X.XX  |  ATL: $X.XX
                                       ┌───────────────────────┐
[K 线图 / Price Chart]  (CSR)          │  Smart Money          │
(ApexCharts，1D/7D/1M 切换)            │  Top wallets buying:  │
                                       │  • 0x1a2b...  +$XXK   │
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─              │  • 0x3c4d...  +$XXK   │
【价值板块 1】                          │  • 0x5e6f...  -$XXK   │
[Token Info Block]      (SSR)          └───────────────────────┘
Contract: 0x...（可复制）
Chain: Solana                          ┌───────────────────────┐
Launch: YYYY-MM-DD                     │  {TOKEN} Calculator   │
Holders: X,XXX                         │                       │
LP Pool: $XXX,XXX                      │  I have [____] {TOKEN}│
                                       │  = $[  XX.XX  ]       │
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─              │  (实时汇率，CSR更新)  │
【价值板块 2】                          └───────────────────────┘
[Historical Price]      (SSR + CSR)
{TOKEN} Historical Price               ┌───────────────────────┐
[时间轴图表，ApexCharts]               │  运营 Banner          │
• All Time High: $X.XX                 │  （CMS 后台配置）     │
  on YYYY-MM-DD                        └───────────────────────┘
• All Time Low:  $X.XX
  on YYYY-MM-DD
• 7d change: ▲ +X.X%
• 30d change: ▼ -X.X%

─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
【价值板块 3】
[Price Prediction]      (SSR，程序化生成)
{TOKEN} Price Prediction 2026
基于以下指标生成 ~200词分析文字：
• Smart money 净流入/流出
• 持仓集中度变化
• 链上交易量趋势
• 市场情绪（Bullish / Neutral / Bearish 标签）

─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
【价值板块 4】
[Market Analysis]       (SSR，程序化生成)
{TOKEN} Market Analysis
• Holder Distribution（饼图，CSR）
• Volume vs Price 关系分析（~100词，SSR）
• 风险提示（固定文字，SSR）

─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
[About {TOKEN}]         (SSR)
~150词固定模板 + 代币名/合约

[How to Buy {TOKEN}]    (SSR)
Step 1-3 流程图

[FAQ] 5-8 问答          (SSR)
JSON-LD FAQPage Schema
（含价格预测、历史价格类问题）

[Related Tokens]        (CSR)
→ 同类热门代币推荐
─────────────────────────────────────────────────────────────────
```

**四大价值板块说明：**

| 板块 | 渲染方式 | 数据来源 | 目标关键词 |
|------|---------|---------|-----------|
| 历史价格 Historical Price | SSR（数字）+ CSR（图表） | 链上历史数据 | `{TOKEN} price history` / `{TOKEN} all time high` |
| 价格预测 Price Prediction | **SSR**（程序化文字生成） | 聪明钱信号 + 链上指标 | `{TOKEN} price prediction` / `{TOKEN} price forecast` |
| 计算器 Calculator | CSR（实时计算） | 实时汇率 | `{TOKEN} to USD` / `1000 {TOKEN} in dollars` |
| 市场分析 Market Analysis | SSR（文字）+ CSR（图表） | 持仓/交易量数据 | `{TOKEN} market analysis` / `{TOKEN} market cap` |

---

#### 5.1.2 内链网络

```
/en/moonx/solana/token/{contract}（本页）
  ↔  /en/moonx/markets/trending                   （互链，热门代币聚合页）
  →  /en/moonx/pump                               （同类新币发现）
  →  /en/moonx/monitor/dynamic/hot                （聪明钱详情，来自 Smart Money 板块）
  →  /en/moonx/learn/smart-money-tracking-crypto  （Price Prediction 板块教育内链）
  →  /en/moonx/learn/best-solana-meme-coin-tracker（Market Analysis 板块教育内链）
```

**内链逻辑：每个价值板块都承担一个内链出口，将用户引导至相关内容页，同时构建话题权威。**

---

#### 5.1.3 目标关键词（扩展版）

| 类型 | 关键词示例 | 月搜索量级 | 对应页面板块 |
|------|----------|-----------|------------|
| 价格词 | `{TOKEN} price` / `{TOKEN} price solana` | 100-50K | Price Card |
| 购买词 | `buy {TOKEN}` / `how to buy {TOKEN}` | 50-10K | How to Buy |
| 历史价格 | `{TOKEN} price history` / `{TOKEN} all time high` | 50-5K | Historical Price 板块 |
| 价格预测 | `{TOKEN} price prediction` / `{TOKEN} price forecast 2026` | 100-100K | Price Prediction 板块 |
| 计算器 | `{TOKEN} to USD` / `1000 {TOKEN} in dollars` | 50-10K | Calculator 板块 |
| 市场分析 | `{TOKEN} market analysis` / `{TOKEN} market cap` | 50-5K | Market Analysis 板块 |
| 发现词 | `{TOKEN} pump.fun` / `{TOKEN} contract address` | 50-5K | Token Info |

**关键洞察：** 加入四个价值板块后，每个 Token 页从攻打 3 类关键词扩展到 7 类，单页面关键词覆盖面翻倍以上。

---

#### 5.1.4 noindex 规则

满足以下**任意一条**则 noindex，不加入 sitemap：

| 维度 | 准入阈值 |
|------|---------|
| LP 流动性池 | < $5,000 USD |
| 持有人数 | < 100 |
| 上线时长 | < 48 小时 |
| 24h 交易量 | < $1,000 USD |
| 24h 交易笔数 | < 50 笔 |

---

#### 5.1.5 验收标准

- [ ] 页面源码中 H1、About段落、FAQ文字 可见（非 JS 渲染）
- [ ] title 格式正确：`{TOKEN} Price, Chart & Trading | MoonX`（50-60字符）
- [ ] Rich Results Test：FinancialProduct Schema 无报错
- [ ] 满足准入条件的 token 在 sitemap 中；不满足的显示 noindex
- [ ] 变量全部替换，无 `{TOKEN_SYMBOL}` 等未替换字符出现

---

### 5.2 Learn 学院文章页

**路径：** `/en/moonx/learn/{slug}`
**数量：** 9 篇（本期）
**目标关键词：** 预测市场中高频词 + Meme 工具词

---

#### 5.2.1 页面内容结构

```
Breadcrumb: Home > MoonX > Learn > {文章标题}
─────────────────────────────────────────────────────────────
Main（720px）                        Side（260px）
─────────────────────────────────────────────────────────────
[H1]                     (SSR)       ┌─────────────────────┐
文章标题（含目标关键词）              │  Table of Contents  │
                                     │  1. Section One     │
[Intro 段落]             (SSR)       │  2. Section Two     │
100-150词，含目标关键词，            │  3. Section Three   │
说清楚文章回答什么问题               │  ...                │
                                     │  （sticky 跟随）    │
[H2] Section One         (SSR)       └─────────────────────┘
[正文内容]
                                     ┌─────────────────────┐
[对比表格]（如有）        (SSR)       │  MoonX CTA 卡片     │
| 平台 | 特点 | 适合谁 |             │                     │
|------|------|--------|             │  Compare odds on    │
                                     │  Polymarket & Kalshi│
[H2] Section Two         (SSR)       │  in one place.      │
[正文内容]                           │                     │
                                     │  [Browse Markets →] │
[H2] Section Three       (SSR)       └─────────────────────┘
[正文内容]
                                     ┌─────────────────────┐
[Summary / 结论]         (SSR)       │  Related Articles   │
                                     │  • Article Title 1  │
[FAQ] 3-5 问答           (SSR)       │  • Article Title 2  │
JSON-LD FAQPage Schema               │  • Article Title 3  │
                                     └─────────────────────┘
[CTA Banner]             (SSR)
→ Browse All Markets on MoonX

[Related Articles]       (SSR)
← → 同 Cluster 其他文章
─────────────────────────────────────────────────────────────
```

---

#### 5.2.2 内链网络（以 polymarket-alternatives 为例）

```
/en/moonx/learn/polymarket-alternatives（本页）
  ↔  /en/moonx/learn/prediction-markets-guide    （支柱页，互链）
  ↔  /en/moonx/learn/polymarket-vs-kalshi        （同 Cluster，互链）
  →  /en/moonx/learn/prediction-market-aggregator（解释聚合器）
  →  /en/moonx/markets/trending                  （CTA 转化）
```

---

#### 5.2.3 文章清单与字段规格

| # | Slug | H1 | 目标关键词 | 字数 | 优先级 | 状态 |
|---|------|----|----------|------|--------|------|
| 1 | prediction-market-aggregator | What Is a Prediction Market Aggregator? | prediction market aggregator | 1,800 | **P0** | ✅ 完成 |
| 2 | polymarket-alternatives | Best Polymarket Alternatives 2026 | polymarket alternative | 2,200 | **P0** | ✅ 完成 |
| 3 | polymarket-vs-kalshi | Polymarket vs Kalshi: Full Comparison 2026 | polymarket vs kalshi | 2,000 | P1 | ✅ 完成 |
| 4 | prediction-markets-guide | Prediction Markets: The Complete Guide 2026 | what are prediction markets | 4,000 | P1 | ✅ 完成 |
| 5 | how-to-trade-prediction-markets | How to Trade Prediction Markets for Beginners | how to trade prediction markets | 2,000 | P2 | 待写 |
| 6 | prediction-market-odds | Prediction Market Odds Explained | prediction market odds | 1,500 | P2 | 待写 |
| 7 | smart-money-tracking-crypto | What Is Smart Money Tracking in Crypto? | smart money tracker crypto | 1,800 | P1 | 待写 |
| 8 | gmgn-vs-moonx | GMGN vs MoonX: Which Solana Tracker Is Better? | GMGN alternative | 2,000 | P1 | 待写 |
| 9 | best-solana-meme-coin-tracker | Best Solana Meme Coin Trackers 2026 | best solana meme coin tracker | 2,000 | P2 | 待写 |

---

#### 5.2.4 验收标准

- [ ] 页面源码中 H1、正文、FAQ 文字可见（非 JS 渲染）
- [ ] title 50-60 字符，description 140-160 字符
- [ ] Rich Results Test：Article + FAQPage Schema 无报错
- [ ] 每篇文章有 ≥ 3 个内链（含指向支柱页的链接）
- [ ] 每篇结尾有 CTA 链接指向 `/en/moonx/markets/trending`
- [ ] 进入 sitemap，GSC 可正常抓取

---

### 5.3 功能落地页

**路径：** `/en/moonx` / `/en/moonx/markets/trending` / `/en/moonx/monitor/dynamic/hot` / `/en/moonx/pump`
**目标：** 让功能页被 Google 理解和索引，为产品功能词建立排名

---

#### 5.3.1 Trending Markets 页面结构

```
─────────────────────────────────────────────────────────────
Main（全宽，内容区 ~800px）
─────────────────────────────────────────────────────────────
[H1]                                                  (SSR)
Trending Prediction Markets

[静态说明段落]                                         (SSR)
MoonX tracks the most active prediction markets
across Polymarket, Kalshi, and Manifold in real time.
Browse trending markets by category—politics, crypto,
sports—and compare odds side by side before you trade.

[分类 Tab]  Politics | Crypto | Sports | All          (CSR)

[市场列表]                                             (CSR)
┌──────────────────────────────────────────────────┐
│ Market Title          YES 72%  NO 28%  Vol $500K  │
│ Market Title          YES 45%  NO 55%  Vol $200K  │
│ ...                                               │
└──────────────────────────────────────────────────┘

[FAQ] 3 问答                                          (SSR)
JSON-LD FAQPage Schema
─────────────────────────────────────────────────────────────
```

**内链网络：**
```
/en/moonx/markets/trending（本页）
  ↔  /en/moonx                                      （首页互链）
  ↔  /en/moonx/solana/token/{contract}              （token 页互链）
  →  /en/moonx/learn/prediction-market-aggregator   （功能说明）
  →  /en/moonx/learn/polymarket-alternatives        （帮用户了解平台）
```

---

#### 5.3.2 Smart Money 页面结构

```
─────────────────────────────────────────────────────────────
[H1] Smart Money Tracker                              (SSR)

[静态说明段落]                                         (SSR)
Smart money tracking shows you what the most
profitable crypto wallets are trading right now.
MoonX monitors on-chain activity from top Solana
traders and surfaces their moves in real time.

[钱包列表]                                             (CSR)
┌──────────────────────────────────────────────────┐
│ Wallet 0x1a2b...   +342%   Bought: PEPE $50K     │
│ Wallet 0x3c4d...   +218%   Bought: WIF  $30K     │
│ ...                                               │
└──────────────────────────────────────────────────┘

[FAQ] 3 问答                                          (SSR)
─────────────────────────────────────────────────────────────
```

**内链网络：**
```
/en/moonx/monitor/dynamic/hot（本页）
  ↔  /en/moonx/solana/token/{contract}              （点击代币进详情）
  →  /en/moonx/learn/smart-money-tracking-crypto    （教育内链）
  →  /en/moonx/markets/trending                     （CTA）
```

---

#### 5.3.3 功能落地页通用 Meta 规格

| 页面 | Title（50-60字符） | Description（140-160字符） |
|------|------------------|--------------------------|
| 首页 | `MoonX — Prediction Market Aggregator \| Polymarket, Kalshi & More` | `MoonX aggregates prediction markets from Polymarket, Kalshi, and Manifold in one place. Compare odds, track smart money, and trade meme tokens on Solana.` |
| Trending | `Trending Prediction Markets Today — Live Odds \| MoonX` | `Track the hottest prediction markets right now. MoonX shows trending markets from Polymarket, Kalshi, and Manifold with live odds and 24h volume. Updated in real time.` |
| Smart Money | `Smart Money Crypto Tracker — Follow Top Wallets \| MoonX` | `Track what smart money wallets are buying and selling in real time. MoonX monitors top Solana traders so you can spot trends before they go viral. Free to use.` |
| Pump | `Pump.fun Token Trading — New Launches on Solana \| MoonX` | `Trade the latest pump.fun token launches on Solana. MoonX shows new token launches with real-time price, volume, and smart money activity. Find the next 100x early.` |

---

## 六、全站内链网络

```
                    ┌─────────────────────┐
                    │  支柱页              │
                    │  /learn/            │
                    │  prediction-markets │
                    │  -guide             │
                    └──────────┬──────────┘
                               │ 双向内链
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
  /learn/prediction-   /learn/polymarket-   /learn/polymarket-
  market-aggregator     alternatives         vs-kalshi
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │ 所有文章 CTA 指向
                               ▼
                 /en/moonx/markets/trending（产品页）
                               │
                    ┌──────────┼──────────┐
                    │          │          │
                    ▼          ▼          ▼
              /en/moonx    /en/moonx   /en/moonx
              /pump        /monitor    /solana/token
                           /dynamic    /{contract}
                           /hot
```

**内链规则说明：**

| 关系类型 | 说明 | 标记 |
|---------|------|------|
| 互链 | 两个页面相互引用，权重互传 | ↔ |
| 单向内链 | 从当前页指向目标页 | → |
| CTA 转化链接 | 文章结尾指向产品页，目的是转化 | → CTA |
| 支柱页链接 | Cluster 文章必须链接支柱页 | → Pillar |

---

## 七、关键词矩阵

### 赛道 A：预测市场

| 类型 | 关键词 | 月搜索量 | 竞争 | 优先级 | 目标页面 |
|------|--------|---------|------|--------|---------|
| 品类词 | prediction market aggregator | 1K-5K | 极低 | **P0** | /learn/prediction-market-aggregator |
| 替代词 | polymarket alternative | 10K-50K | 中 | **P0** | /learn/polymarket-alternatives |
| 对比词 | polymarket vs kalshi | 10K-50K | 中 | P1 | /learn/polymarket-vs-kalshi |
| 通用词 | what are prediction markets | 50K+ | 高 | P1 | /learn/prediction-markets-guide |
| 入门词 | how to trade prediction markets | 5K-20K | 中 | P2 | /learn/how-to-trade-prediction-markets |
| 替代词 | kalshi alternative | 5K-10K | 低 | P1 | /learn/polymarket-alternatives |

### 赛道 B：Meme Token 工具

| 类型 | 关键词 | 月搜索量 | 竞争 | 优先级 | 目标页面 |
|------|--------|---------|------|--------|---------|
| 功能词 | smart money tracker crypto | 5K-20K | 中 | **P0** | /monitor/dynamic/hot |
| 竞品词 | GMGN alternative | 1K-5K | 低 | P1 | /learn/gmgn-vs-moonx |
| 工具词 | best solana meme coin tracker | 2K-10K | 低 | P1 | /learn/best-solana-meme-coin-tracker |
| 发现词 | pump.fun trading | 5K-20K | 中 | P2 | /pump |

### 赛道 C：Token 长尾（规模化）

| 类型 | 关键词模式 | 单词量级 | 竞争 | 对应页面板块 |
|------|----------|---------|------|------------|
| 价格词 | `{TOKEN} price solana` | 100-50K | 极低 | Price Card |
| 购买词 | `buy {TOKEN} solana` | 50-10K | 极低 | How to Buy |
| 历史价格 | `{TOKEN} price history` | 50-5K | 极低 | Historical Price 板块 |
| 历史高低 | `{TOKEN} all time high / low` | 50-10K | 极低 | Historical Price 板块 |
| 价格预测 | `{TOKEN} price prediction 2026` | 100-100K | 低 | Price Prediction 板块 |
| 计算器 | `{TOKEN} to USD` / `1000 {TOKEN} in dollars` | 50-10K | 极低 | Calculator 板块 |
| 市场分析 | `{TOKEN} market analysis` | 50-5K | 极低 | Market Analysis 板块 |
| 发现词 | `{TOKEN} pump.fun` / `{TOKEN} contract` | 50-5K | 极低 | Token Info |

**规模化价值：** 每个 Token 页 × 7 类关键词 × 10,000 个 Token = 70,000 个长尾排名入口

---

## 八、内容规格

### 8.1 文章写作标准

| 要素 | 标准 |
|------|------|
| 语言 | 英文（目标用户以英语为主，美国占 62%） |
| 语气 | 直接、客观、有明确立场，避免营销腔 |
| 关键词位置 | 目标词出现在：H1、前 100 词、至少 2 个 H2 标题 |
| 内链数量 | 每篇 ≥ 3 个内链 |
| 必须内链 | 每篇必须有 1 个指向支柱页 `/learn/prediction-markets-guide` |
| CTA | 每篇结尾 1 个，指向 `/en/moonx/markets/trending` |
| FAQ | 每篇 ≥ 3 个问答，使用 FAQPage Schema 标记 |
| 年份更新 | 含"2026"的内容跨年时同步更新 |

---

## 九、技术需求

### 9.1 SSR 改造优先级

| 优先级 | 页面 | 必须 SSR 的内容 |
|--------|------|---------------|
| **P0** | `/en/moonx/learn/` 及所有文章页 | 全部正文 |
| **P0** | `/en/moonx/solana/token/{contract}` | H1、About段落、How to Buy、FAQ、Schema |
| P1 | `/en/moonx/markets/trending` | H1、说明段落、FAQ Schema |
| P1 | `/en/moonx`（首页） | H1、功能介绍段落 |
| P2 | `/en/moonx/monitor/dynamic/hot` | H1、说明段落、FAQ Schema |
| P2 | `/en/moonx/pump` | H1、说明段落 |

### 9.2 Token 页四大价值板块技术规格

| 板块 | SSR 必须输出内容 | CSR 动态内容 | 数据接口 |
|------|----------------|------------|---------|
| Historical Price | ATH/ATL 数字、日期、7d/30d 涨跌幅文字 | 历史K线图表 | 链上历史价格 API |
| Price Prediction | ~200词分析文字（基于信号模板生成）、Bullish/Neutral/Bearish 标签 | 无（全SSR） | 聪明钱净流入、持仓集中度、交易量趋势 |
| Calculator | 计算器说明文字（SSR），输入框 + 实时计算结果（CSR） | 实时汇率计算 | 实时价格 API |
| Market Analysis | ~100词量价关系文字、风险提示（固定模板） | 持仓分布饼图、量价图 | 持仓数据、交易量数据 |

**Price Prediction 文字生成逻辑（程序化模板）：**
```
根据信号条件，服务端选择对应模板段落：
IF 聪明钱净流入 > +20% AND 持仓集中度下降（分散）
  → 输出看涨模板（Bullish）
IF 聪明钱净流出 > -20% OR 持仓集中度上升（集中）
  → 输出看跌模板（Bearish）
ELSE
  → 输出中性模板（Neutral）

模板示例（Bullish）：
"{TOKEN} is showing signs of strength as of {DATE}.
Smart money wallets have net accumulated ${AMOUNT} in the
past 7 days, while holder distribution has broadened to
{HOLDERS} addresses. Based on current on-chain activity,
{TOKEN} may face resistance near ${RESISTANCE_LEVEL}.
Always conduct your own research before trading."
```

**注意：** Price Prediction 文字属于 SSR 内容，Google 可索引，是捕获 `{TOKEN} price prediction` 关键词的核心模块。

### 9.3 Canonical 修复

| 问题 URL | 修复方式 |
|---------|---------|
| `/en/moonx/solana/token?address={x}` | canonical → `/en/moonx/solana/token/{contract}` |
| `/en/moonx/pump?network=SOLANA` | canonical → `/en/moonx/pump` |
| `/en/moonx/trade/dynamic?network=SOLANA` | canonical → `/en/moonx/trade/dynamic` |
| `/en/moonx/account/my-position` | noindex + robots.txt Disallow |

### 9.3 Sitemap

| Sitemap | 内容 | 更新频率 |
|---------|------|---------|
| Learn 文章 | 所有 /learn/ 文章 | 新增文章时 |
| Token 页 | 满足准入条件的 token | **每日自动** |

Token sitemap 路径：`/events/cms/sitemap/pseo/moonx-token-summary.xml`

---

## 十、数据埋点

| 事件名 | 触发条件 | 用途 |
|-------|---------|------|
| seo_article_view | 访问 /learn/ 任意文章 | 衡量内容流量 |
| seo_cta_click | 点击文章内 CTA 链接 | 衡量内容→产品转化率 |
| token_page_view | 访问 /token/ 任意页面 | 衡量 Token 页流量 |
| token_trade_click | Token 页点击交易按钮 | 衡量 Token 页→交易转化率 |

---

## 十一、分期计划

### Phase 1（第 1-4 周）：打地基
**目标：** GSC 中 MoonX 出现第一条展示量

| 任务 | 负责方 | 优先级 |
|------|--------|--------|
| /en/moonx/learn/ 路径建好（SSR） | Tech | **P0** |
| Token 页 SSR meta 注入上线 | Tech | **P0** |
| Canonical 修复（token?address= 问题） | Tech | **P0** |
| 发布文章 #1-4 | Content | **P0** |
| 提交 sitemap 至 GSC | Tech + Marketing | **P0** |

### Phase 2（第 5-8 周）：首次排名
**目标：** 核心文章出现展示量，Token 页索引数 ≥ 100

| 任务 | 负责方 | 优先级 |
|------|--------|--------|
| 功能落地页 SSR 上线（trending / 首页） | Tech | P1 |
| Token sitemap 自动更新机制上线 | Tech | P1 |
| 发布文章 #5-7 | Content | P1 |
| 根据 GSC 展示数据优化 CTR（改 description） | Marketing | P1 |

### Phase 3（第 9-16 周）：规模化
**目标：** Token 页 500+，核心词进 Top 20

| 任务 | 负责方 | 优先级 |
|------|--------|--------|
| 功能落地页 SSR 全部完成 | Tech | P2 |
| 发布文章 #8-9 | Content | P2 |
| Token 页收录规模达 500+ | Tech | P1 |

---

## 十二、验收标准汇总

### 技术验收
- [ ] 所有 SSR 页面：Google Search Console "检查网址"，SSR 内容在渲染 HTML 可见
- [ ] Token 页：满足条件的有 index，不满足的有 noindex
- [ ] Canonical：GSC Coverage 无重复内容警告
- [ ] Schema：所有页面通过 Google Rich Results Test，无 Error 报错
- [ ] Sitemap：Learn 文章 + Token 页已提交 GSC

### 内容验收
- [ ] 9 篇文章发布，title/description 符合字符规格
- [ ] 每篇有 ≥ 3 个内链，集群内链结构完整
- [ ] 所有文章有 CTA，链接正确

### 业务验收
- [ ] Phase 1 结束：GSC 展示量 > 0
- [ ] Phase 2 结束：月点击量 > 200
- [ ] Phase 3 结束："polymarket alternative" 进入 Top 50

---

## 十三、风险与依赖

| 风险 | 影响 | 概率 | 应对 |
|------|------|------|------|
| SSR 改造排期延后 | Phase 1 目标延后 | 中 | 提前锁定排期；内容提前写好待发 |
| /learn/ 路径建设滞后 | 文章无法落地 | 中 | /learn/ 路径列为 P0 依赖 |
| Google 重新爬取周期长（2-4 周） | 数据滞后 | 必然 | 提交 sitemap 后用覆盖率报告追踪 |
| Token 页 noindex 比例高 | 规模化受阻 | 低 | 提升平台活跃度，增加符合条件的 token |

### 关键依赖项

| 依赖 | 状态 |
|------|------|
| Tech SSR 改造排期确认 | ⏳ 待确认具体时间 |
| /en/moonx/learn/ 路径上线时间 | ⏳ 待确认 |
| Kelly 审核文章 #1-4 | ⏳ 待审核 |
| GSC 账户访问权限 | ⏳ 待提供 |

---

*PRD v1.1 · 2026-03-11 · MoonX SEO 页面体系*
