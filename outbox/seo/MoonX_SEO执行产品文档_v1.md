# MoonX SEO 执行产品文档 v1.0
**版本：1.0 | 日期：2026-03-11**
**文档性质：跨团队执行参考文档（Tech + Content + Marketing）**
**数据来源：GSC 截图 + SEO全站规范 + 基础能力优化P1 + SerpAPI 竞品分析 + MoonX 平台数据**

---

## 一、现状诊断

### 1.1 bydfi.com 整体数据（过去 28 天）

| 指标 | 数值 | 解读 |
|------|------|------|
| 总点击 | 46,100 | 月均约 5 万 |
| 总展示 | 7,880,000 | 月均约 800 万 |
| 平均 CTR | 0.6% | 行业均值 2-3%，**显著偏低** |
| 平均排名 | 9.5 | 第一页底部，有提升空间 |

**核心问题：展示量大但点击率低。说明 bydfi 主站有一定域名权重，但页面 title/description 吸引力不足，或排名词不精准。**

### 1.2 MoonX 当前 SEO 状态

| 指标 | 数值 |
|------|------|
| GSC 展示量 | **0** |
| GSC 点击量 | **0** |
| Google 索引页面数 | **0**（估计） |
| 自然流量占比 | **0%** |

**根本原因：MoonX 全页面 CSR 渲染，Google 爬取到的是空壳 HTML，无任何可索引内容。**

### 1.3 机会判断

| 机会 | 依据 |
|------|------|
| bydfi.com 有域名权重 | 7.88M 月展示，新 MoonX 页面起步更快 |
| "prediction market aggregator" 无竞争 | SerpAPI 分析：该词无强对手，MoonX = 品类定义者 |
| Token 页规模化 | DEXScreener 模式，MoonX 已有架构，缺 SSR meta |
| CSR→SSR 改造后从 0 起步 | 所有增量都是纯新增，增长曲线清晰 |

---

## 二、目标定义

### 2.1 分阶段流量目标

| 节点 | 月自然流量 | 索引页面数 | 核心词排名 |
|------|-----------|-----------|-----------|
| SSR 上线后 2 周 | 首次 > 0 | 10+ | 无 |
| M2（第 2 个月） | 200-500 | 50+ | 开始出现 |
| M4（第 4 个月） | 3,000+ | 500+ | polymarket alternative Top 50 |
| M6（第 6 个月） | 15,000+ | 2,000+ | polymarket alternative Top 10 |
| M12（第 12 个月） | 60,000+ | 10,000+ | polymarket alternative Top 3 |

### 2.2 流量来源拆解（M12 目标）

| 来源 | 占比 | 月流量 | 驱动方式 |
|------|------|--------|---------|
| Token 行情页（规模化） | 80% | 48,000+ | 程序化 SEO，Token 页 × 10,000 |
| 内容文章（Learn 页） | 15% | 9,000+ | 编辑 SEO，10 篇核心文章 |
| 功能落地页 | 5% | 3,000+ | 产品词排名 |

### 2.3 核心 KPI（月度追踪）

- GSC 展示量（先行指标，最快 2-4 周出现）
- Google 有效索引页面数
- "polymarket alternative" 排名位次
- 月自然点击量

---

## 三、页面体系设计

MoonX 需要建设 **4 类核心页面**，每类服务不同搜索意图。

---

### 类型 A：Token 行情页（规模化流量引擎）

**角色：** 引擎 B，主力流量来源（占 M12 目标 80%）

**URL 结构：**
```
/en/moonx/solana/token/{contract_address}
示例：/en/moonx/solana/token/EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v
```

**页面内容规格：**

| 内容块 | 渲染方式 | 内容 |
|--------|---------|------|
| Title（meta） | **SSR 必须** | `{TOKEN} Price, Chart & Trading \| MoonX` |
| Description（meta） | **SSR 必须** | `Trade {TOKEN} on MoonX. Live price ${PRICE}, 24h {DIRECTION} {PCT}%, volume ${VOL}. Track smart money and trade Solana tokens in real time.` |
| H1 | **SSR 必须** | `{TOKEN} ({TOKEN_NAME}) — Live Price & Trading` |
| 功能说明段落 | **SSR 必须** | 固定文字（见附录 A） |
| FAQ Schema | **SSR 必须** | 3 个问题（见附录 A） |
| 实时价格/K 线 | CSR 可以 | 动态数据 |
| Smart Money 动向 | CSR 可以 | 动态数据 |
| 持有人/交易量 | CSR 可以 | 动态数据 |

**Sitemap 准入规则（满足全部条件才收录）：**

| 维度 | 准入阈值 |
|------|---------|
| LP 流动性池 | ≥ $5,000 USD |
| 持有人数 | ≥ 100 |
| 上线时长 | ≥ 48 小时 |
| 24h 交易量 | ≥ $1,000 USD |
| 24h 交易笔数 | ≥ 50 笔 |

**不满足条件：** `<meta name="robots" content="noindex, nofollow" />`，不放入 sitemap。

**攻打关键词：**
- `{TOKEN} price solana`
- `buy {TOKEN} solana`
- `{TOKEN} pump.fun`
- `{TOKEN} contract address`

---

### 类型 B：学院页 / Learn（品牌权威）

**角色：** 引擎 A，建立 Google 对 MoonX 的认知，同时驱动预测市场相关词排名

**URL 结构：**
```
/en/moonx/learn/                          ← 学院首页（文章列表）
/en/moonx/learn/{slug}                    ← 每篇文章
示例：/en/moonx/learn/polymarket-vs-kalshi
```

**所有文章统一规格：**

| 要素 | 规格 |
|------|------|
| Title | 目标关键词 + 年份 + `\| MoonX`，50-60 字符 |
| Description | 解答搜索意图 + 钩子 + MoonX 价值，140-160 字符 |
| H1 | 与 Title 一致或相近 |
| 字数 | Cluster 文章 1,500-2,500 词；Pillar 页 3,000-4,000 词 |
| 内链 | 每篇必须链接到支柱页；支柱页链接到所有 Cluster |
| Schema | Article Schema + FAQPage Schema（≥3 个问题） |
| CTA | 每篇结尾放 MoonX 产品页链接 |

**技术要求：** `/en/moonx/learn/` 路径必须 SSR 渲染（参考 bydfi 现有 /learn/ 体系）。

---

### 类型 C：功能落地页（产品词排名）

**角色：** 地基，告诉 Google MoonX 是什么产品

**页面清单：**

| 页面 | URL | 核心关键词 |
|------|-----|-----------|
| MoonX 首页 | `/en/moonx` | prediction market aggregator, moonx |
| 热门预测市场 | `/en/moonx/markets/trending` | trending prediction markets |
| Smart Money | `/en/moonx/monitor/dynamic/hot` | smart money tracker crypto |
| Pump.fun | `/en/moonx/pump` | pump.fun trading, pump.fun token tracker |

**每个功能页必须有的静态内容（SSR 注入）：**
1. 功能说明段落（3-5 句，描述页面是什么）
2. FAQ Schema（3 个问题）
3. 正确的 title / description / canonical

**动态内容（CSR 渲染）：** 实时行情、排行、价格

---

### 类型 D：对比页（决策词截流）

**角色：** 引擎 A 的一部分，放在 /learn/ 下，单独列出因为搜索量大

**URL 结构：**
```
/en/moonx/learn/polymarket-vs-kalshi
/en/moonx/learn/polymarket-alternatives
/en/moonx/learn/moonx-vs-gmgn
```

**内容规格：** 同学院页规格，重点是比较表格 + 明确推荐结论 + MoonX 聚合方案作为最优解。

---

## 四、技术执行规格

### 4.1 SSR 改造优先级

| 优先级 | 页面 | 理由 |
|--------|------|------|
| **P0** | `/en/moonx/learn/` 路径 | 文章发布的前提 |
| **P0** | Token 详情页 `/token/{contract}` | 最大流量来源，不改造等于放弃 |
| **P1** | `/en/moonx/markets/trending` | 核心功能页，产品词排名 |
| **P1** | `/en/moonx` 首页 | 品牌词和聚合器词 |
| **P2** | `/en/moonx/monitor/dynamic/hot` | Smart money 词 |
| **P2** | `/en/moonx/pump` | Pump.fun 词 |

**改造原则：**
- 静态内容（title、h1、功能说明、FAQ 文字、Schema）= 服务端渲染
- 动态内容（实时价格、排行、K 线）= 客户端渲染
- Schema JSON-LD 必须服务端注入 `<head>`，禁止客户端异步写入

### 4.2 Meta 标签规格

**Token 页 Title 模板（50-60 字符）：**
```
{TOKEN_SYMBOL} Price, Chart & Trading | MoonX
示例：PEPE Price, Chart & Trading | MoonX
```

**Token 页 Description 模板（140-160 字符）：**
```
Trade {TOKEN_SYMBOL} on MoonX. Live price ${PRICE}, 24h {up/down} {PCT}%, volume ${VOL_24H}. Track smart money and trade Solana tokens in real time.
```

**功能页 Meta 规格：**

| 页面 | Title | Description |
|------|-------|-------------|
| 首页 | `MoonX — Prediction Market Aggregator \| Trade Polymarket, Kalshi & More` | `MoonX aggregates prediction markets from Polymarket, Kalshi, and Manifold in one place. Compare odds, track smart money, and trade meme tokens on Solana.` |
| Trending | `Trending Prediction Markets Today — Live Odds \| MoonX` | `Track the hottest prediction markets right now. MoonX shows trending markets from Polymarket, Kalshi, and Manifold with live odds and 24h volume.` |
| Smart Money | `Smart Money Crypto Tracker — Follow Top Wallets \| MoonX` | `Track what smart money wallets are buying and selling in real time. MoonX monitors top Solana traders so you can spot trends before they go viral.` |
| Pump | `Pump.fun Token Trading — New Launches on Solana \| MoonX` | `Trade the latest pump.fun token launches on Solana. MoonX shows new token launches with real-time price, volume, and smart money activity.` |

### 4.3 Canonical 修复清单

| 问题 URL | 修复方式 |
|---------|---------|
| `/en/moonx/trade/dynamic?network=SOLANA` | canonical → `/en/moonx/trade/dynamic` |
| `/en/moonx/pump?network=SOLANA` | canonical → `/en/moonx/pump` |
| `/en/moonx/solana/token?address={x}` | canonical → `/en/moonx/solana/token/{contract}` |
| `/en/moonx/account/my-position` | noindex + robots.txt Disallow |

### 4.4 Sitemap 规格

**Token 页 Sitemap 文件：**
```
www.bydfi.com/events/cms/sitemap/pseo/moonx-token-summary.xml
```
- `changefreq: daily`
- `priority: 0.6`
- `lastmod`：精确到秒，带 UTC 时区
- 仅收录满足准入条件的 token 页

**提交到 GSC 的 sitemap：**
- 主站 sitemap（已有）
- MoonX /learn/ 文章 sitemap（新建）
- MoonX token 页 sitemap（新建，动态更新）

### 4.5 Schema 规格

| 页面类型 | Schema 类型 |
|---------|------------|
| 首页 | WebPage + Organization |
| 功能落地页 | WebPage + FAQPage |
| Token 行情页 | FinancialProduct + BreadcrumbList + FAQPage |
| 学院文章 | Article + FAQPage |
| 对比页 | Article + FAQPage |

---

## 五、关键词执行矩阵

### 5.1 赛道 A：预测市场（品牌权威）

| 关键词 | 月搜索量 | 竞争 | 优先级 | 目标页面 URL |
|--------|---------|------|--------|------------|
| prediction market aggregator | 1K-5K | 极低 | **P0** | /learn/prediction-market-aggregator |
| polymarket alternative | 10K-50K | 中 | **P0** | /learn/polymarket-alternatives |
| polymarket alternatives 2026 | 10K-50K | 中 | **P0** | /learn/polymarket-alternatives |
| polymarket vs kalshi | 10K-50K | 中 | **P1** | /learn/polymarket-vs-kalshi |
| best prediction market 2026 | 5K-20K | 中 | **P1** | /learn/prediction-markets-guide |
| what are prediction markets | 50K+ | 高 | **P1** | /learn/prediction-markets-guide |
| how prediction markets work | 5K-20K | 中 | P1 | /learn/prediction-markets-guide |
| kalshi alternative | 5K-10K | 低 | P1 | /learn/polymarket-alternatives |
| manifold markets alternative | 1K-5K | 低 | P2 | /learn/polymarket-alternatives |
| how to trade prediction markets | 5K-20K | 中 | P2 | /learn/how-to-trade-prediction-markets |
| prediction market odds | 1K-5K | 低 | P2 | /learn/prediction-market-odds |
| is polymarket legal | 5K-10K | 低 | P2 | /learn/polymarket-vs-kalshi |
| prediction market aggregator | 1K-5K | 极低 | **P0** | /en/moonx（首页） |

### 5.2 赛道 B：Meme Token 工具（产品流量）

| 关键词 | 月搜索量 | 竞争 | 优先级 | 目标页面 URL |
|--------|---------|------|--------|------------|
| smart money tracker crypto | 5K-20K | 中 | **P0** | /monitor/dynamic/hot（功能页优化）|
| GMGN alternative | 1K-5K | 低 | P1 | /learn/gmgn-vs-moonx |
| best solana meme coin tracker | 2K-10K | 低 | P1 | /learn/best-solana-meme-coin-tracker |
| solana token tracker | 1K-5K | 低 | P1 | /learn/best-solana-meme-coin-tracker |
| pump.fun trading | 5K-20K | 中 | P2 | /pump（功能页优化）|
| trending meme coins solana | 2K-10K | 低 | P2 | /markets/trending（功能页优化）|
| how to find 100x meme coins | 1K-5K | 低 | P2 | /learn/how-to-find-100x-meme-coin |

### 5.3 赛道 C：规模化长尾（Token 页）

| 关键词模式 | 单词月搜索量 | 竞争 | 规模 |
|-----------|-----------|------|------|
| {TOKEN} price solana | 100-10K | 极低 | × N 个 token |
| buy {TOKEN} solana | 100-5K | 极低 | × N 个 token |
| {TOKEN} pump.fun | 50-5K | 极低 | × N 个 token |
| {TOKEN} contract address | 100-2K | 极低 | × N 个 token |

**规模预估：** 满足准入条件的活跃 token 约 500-2,000 个，每周新增 50-200 个。

---

## 六、内容执行计划

### 6.1 内容架构（集群模型）

```
支柱页 (Pillar)
└── Prediction Markets: The Complete Guide 2026
    URL: /en/moonx/learn/prediction-markets-guide
    目标词: what are prediction markets / best prediction market 2026
    字数: ~4,000 词
    状态: ✅ 初稿完成

Cluster A：聚合器定位（护城河）
├── What Is a Prediction Market Aggregator?
│   URL: /learn/prediction-market-aggregator
│   目标词: prediction market aggregator
│   状态: ✅ 初稿完成
└── （后续：Prediction Market Aggregator vs Single Platform）

Cluster B：竞品替代（最高流量）
├── Best Polymarket Alternatives 2026
│   URL: /learn/polymarket-alternatives
│   目标词: polymarket alternative
│   状态: ✅ 初稿完成
└── Polymarket vs Kalshi: Full Comparison
    URL: /learn/polymarket-vs-kalshi
    目标词: polymarket vs kalshi
    状态: ✅ 初稿完成

Cluster C：操作指南（转化）
├── How to Trade Prediction Markets for Beginners
│   URL: /learn/how-to-trade-prediction-markets
│   目标词: how to trade prediction markets
│   状态: 待写（#5）
└── Prediction Market Odds Explained
    URL: /learn/prediction-market-odds
    目标词: prediction market odds
    状态: 待写（#6）

Cluster D：Meme Token 工具（产品词）
├── What Is Smart Money Tracking in Crypto?
│   URL: /learn/smart-money-tracking-crypto
│   目标词: smart money tracker crypto
│   状态: 待写（#7）
├── GMGN vs MoonX: Which Is Better?
│   URL: /learn/gmgn-vs-moonx
│   目标词: GMGN alternative
│   状态: 待写（#8）
└── Best Solana Meme Coin Tracker 2026
    URL: /learn/best-solana-meme-coin-tracker
    目标词: best solana meme coin tracker
    状态: 待写（#9）
```

### 6.2 文章发布计划

| # | 文章标题 | 目标关键词 | 字数 | 优先级 | 状态 |
|---|---------|-----------|------|--------|------|
| 1 | What Is a Prediction Market Aggregator? | prediction market aggregator | ~1,800 | **P0** | ✅ 完成 |
| 2 | Best Polymarket Alternatives 2026 | polymarket alternative | ~2,200 | **P0** | ✅ 完成 |
| 3 | Polymarket vs Kalshi: Full Comparison | polymarket vs kalshi | ~2,000 | P1 | ✅ 完成 |
| 4 | Prediction Markets: The Complete Guide | prediction markets / what are prediction markets | ~4,000 | P1 | ✅ 完成 |
| 5 | How to Trade Prediction Markets | how to trade prediction markets | ~2,000 | P2 | 待写 |
| 6 | Prediction Market Odds Explained | prediction market odds | ~1,500 | P2 | 待写 |
| 7 | What Is Smart Money Tracking? | smart money tracker crypto | ~1,800 | P1 | 待写 |
| 8 | GMGN vs MoonX: Which Is Better? | GMGN alternative | ~2,000 | P1 | 待写 |
| 9 | Best Solana Meme Coin Tracker 2026 | best solana meme coin tracker | ~2,000 | P2 | 待写 |

### 6.3 内链规则

- 每篇 Cluster 文章 **必须** 链接到支柱页（Prediction Markets Guide）
- 支柱页链接到所有 Cluster 文章
- 所有文章结尾 CTA 链接到 MoonX 产品页：`/en/moonx/markets/trending`
- Token 页内链：Token 页 → /markets/trending → MoonX 首页

---

## 七、执行时间线

### 第 1-2 周：技术评估与内容并行

**Tech（技术团队）：**
- [ ] 确认 SSR 改造排期（哪些页面，第几周上线）
- [ ] 确认 `/en/moonx/learn/` 路径何时建好
- [ ] 开始 Canonical 修复（不依赖 SSR，现在就能做）
- [ ] 参考技术规格文档：`tech_ssr_spec_for_moonx.md`

**Content（内容团队）：**
- [ ] Kelly 审核文章 #1-4（已完成初稿）
- [ ] 根据反馈调整语气/风格
- [ ] 开始写文章 #5（How to Trade Prediction Markets）

**验收标准：** 确认 SSR 上线时间，/learn/ 路径建好时间

---

### 第 3-4 周：首次上线

**Tech：**
- [ ] `/en/moonx/learn/` 路径上线（SSR）
- [ ] 功能页 meta 标签写入（trending / smart money / pump）
- [ ] Canonical 修复完成

**Content：**
- [ ] 发布文章 #1-3（learn 路径一上线立即发布）
- [ ] 发布文章 #4（支柱页）
- [ ] GSC 提交 sitemap，申请抓取

**验收标准：** GSC 中 MoonX 开始出现展示量（哪怕个位数）

---

### 第 5-8 周：Token 页上线 + 内容继续

**Tech：**
- [ ] Token 详情页 SSR meta 注入上线
- [ ] Token sitemap 自动更新机制建好
- [ ] 提交 token sitemap 到 GSC
- [ ] MoonX 首页 SSR 上线

**Content：**
- [ ] 发布文章 #5-7
- [ ] 监控 GSC：哪些词开始出现展示 → 优化对应页面 description（提升 CTR）

**验收标准：** Token 页开始被 Google 收录，learn 页面出现关键词展示量

---

### 第 9-16 周：规模化

**Tech：**
- [ ] Token sitemap 持续更新（满足准入条件的 token 自动进入）
- [ ] 修复 noindex token 页（不满足条件的自动标记）

**Content：**
- [ ] 发布文章 #8-9
- [ ] 根据 GSC 数据：优化 CTR 低但展示量高的页面
- [ ] 开始外链建设（进入现有 polymarket alternative 榜单）

**验收标准：**
- Token 页有效收录 ≥ 500
- "polymarket alternative" 出现在 Top 50
- 月自然流量 ≥ 1,000

---

## 八、成功指标追踪

### 每月复盘看板

| 指标 | M1 目标 | M2 目标 | M4 目标 | M6 目标 | M12 目标 |
|------|--------|--------|--------|--------|---------|
| GSC 月展示量 | >0 | 5,000+ | 50,000+ | 200,000+ | 1,000,000+ |
| GSC 月点击量 | >0 | 200+ | 3,000+ | 15,000+ | 60,000+ |
| 有效索引页面数 | 5+ | 50+ | 500+ | 2,000+ | 10,000+ |
| 其中 token 页 | 0 | 0 | 500+ | 2,000+ | 10,000+ |
| 其中 learn 页 | 4 | 8+ | 10 | 10 | 10+ |
| "polymarket alternative" 排名 | — | Top 100 | Top 50 | Top 10 | Top 3 |
| 平均 CTR | — | 0.5%+ | 1%+ | 1.5%+ | 2%+ |

### 最重要的先行指标（Leading Indicators）

1. **GSC 展示量出现** — SSR 上线后 1-2 周应出现，这是一切的起点
2. **索引页面数增长** — 每周看 Google Search Console Coverage 报告
3. **长尾词开始排名** — 先是第 50-100 位，然后爬升，说明内容被认可

---

## 附录 A：Token 页固定静态内容规格

### 固定说明段落（所有 token 页一致，SSR 注入）
```
Trade {TOKEN_SYMBOL} on MoonX, the Solana meme token trading platform.
View live price charts, track smart money wallet activity, and execute trades
directly. MoonX aggregates liquidity across Solana DEXs for best execution.
```

### FAQ Schema（每个 token 页统一 3 个问题）
```json
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {
      "@type": "Question",
      "name": "What is {TOKEN_SYMBOL}?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "{TOKEN_SYMBOL} ({TOKEN_NAME}) is a token on the Solana blockchain. View live price, trading volume, and smart money activity on MoonX."
      }
    },
    {
      "@type": "Question",
      "name": "How to buy {TOKEN_SYMBOL} on MoonX?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "To buy {TOKEN_SYMBOL} on MoonX: 1) Connect your Solana wallet, 2) Search for {TOKEN_SYMBOL}, 3) Enter the amount and confirm the trade. MoonX routes through the best available DEX liquidity."
      }
    },
    {
      "@type": "Question",
      "name": "Is {TOKEN_SYMBOL} safe to trade?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Always research before trading any meme token. Check liquidity pool size, holder distribution, and smart money activity on MoonX before entering a position."
      }
    }
  ]
}
```

### FinancialProduct Schema 模板
```json
{
  "@context": "https://schema.org",
  "@type": "FinancialProduct",
  "name": "{TOKEN_SYMBOL} Meme Token — Live Chart & Trading",
  "url": "https://www.bydfi.com/en/moonx/solana/token/{CONTRACT}",
  "provider": {
    "@type": "Organization",
    "name": "MoonX by BYDFi",
    "url": "https://www.bydfi.com/en/moonx"
  }
}
```

---

## 附录 B：已产出内容文件清单

| 文件 | 说明 |
|------|------|
| `MoonX_SEO计划书_v2.md` | 三轨战略总计划 |
| `tech_ssr_spec_for_moonx.md` | 技术团队 SSR meta 规格 |
| `article_01_prediction_market_aggregator.md` | 文章 #1 初稿 |
| `article_02_polymarket_alternatives.md` | 文章 #2 初稿 |
| `article_03_polymarket_vs_kalshi.md` | 文章 #3 初稿 |
| `article_04_prediction_markets_guide.md` | 文章 #4 支柱页初稿 |
| `05_outreach_emails.md` | 5 封进榜单外联邮件 |

---

## 附录 C：需要 Kelly 确认的事项

| # | 问题 | 选项 A | 选项 B | 截止 |
|---|------|--------|--------|------|
| 1 | 文章 #1-4 语气/风格是否满意？ | 满意，继续写 #5-9 | 需要调整，先改 | 本周 |
| 2 | Tech SSR 改造排期确认了吗？ | 已确认时间节点 | 还在协调中 | 本周 |
| 3 | /en/moonx/learn/ 路径何时建好？ | 已有时间节点 | 还在协调中 | 本周 |

---

*文档版本：v1.0 · 2026-03-11 · 基于 GSC数据 + SEO全站规范 + 基础能力优化P1 + SerpAPI分析*
