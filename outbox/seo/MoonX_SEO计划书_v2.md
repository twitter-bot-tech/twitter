# MoonX SEO 总计划书 v2.0
**版本：2.0 | 日期：2026-03-11**
**基于：SerpAPI 竞品分析 + SEO全站规范 + 基础能力优化P1**
**核心修订：加入技术基础修复 + Token 页规模化，三轨并行**

---

## 一、现状诊断（先于一切）

读完技术文档后，MoonX 的 SEO 问题比我们预想的更复杂，也有更大的机会：

| 维度 | 现状 | 影响 |
|------|------|------|
| 渲染方式 | CSR 客户端渲染 | Google 抓到空壳，关键页面无法被索引 |
| Token 页面 | 已有 `/en/moonx/solana/token/{contract}`，但 CSR + 无 meta | 数千个长尾流量入口被浪费 |
| 重复 URL | `?address=` vs `/token/{contract}` 并存 | 权重分散，已部分处理 |
| 参数页被索引 | `?network=SOLANA` 被 Google 收录 | 与主页形成重复内容竞争 |
| 内容站 | `/learn/` + pSEO 已存在 | MoonX 内容可接入现有基础，不需要从零搭 |
| Sitemap | MoonX 页面暂未入 sitemap | 前提：先完成 SSR 改造 |

---

## 二、战略定位（不变）

> **MoonX 是世界第一个预测市场聚合器。**
> 但在内容策略生效之前，必须先解决技术基础问题。
> 技术不修，内容是无源之水。

---

## 三、三轨并行执行框架

```
Track 1：技术修复（P0，阻塞一切）
    └── SSR 改造 → Token 页 meta 注入 → Canonical 修复 → Sitemap 收录

Track 2：Token 页规模化（最大流量来源，依赖 Track 1）
    └── 满足准入条件的 token 页 → sitemap → 长尾排名

Track 3：内容/编辑 SEO（品牌定位，可与 Track 1 并行启动）
    └── Prediction market aggregator 内容集群
```

---

## 四、Track 1：技术修复（第 1-4 周，P0）

这是整个 SEO 战略的地基。**必须优先于内容发布。**

### 1.1 SSR 改造（最高优先级）

**问题：** MoonX 全页面 CSR，Google 抓不到内容
**目标页面（按优先级）：**

| 页面 | URL | 静态注入内容 | 动态内容 |
|------|-----|------------|---------|
| MoonX 首页 | /en/moonx | 功能介绍、FAQ、meta | 实时市场列表 |
| Trending 市场 | /en/moonx/markets/trending | 标题、说明段落、FAQSchema | 实时代币排行 |
| New Coin | /en/moonx/markets/new-coin | 说明段落、HowTo Schema | 实时新币列表 |
| Smart Money | /en/moonx/monitor/dynamic/hot | 功能说明、FAQ | 实时追踪数据 |
| Pump | /en/moonx/pump | 固定说明段落、meta | 代币列表 |
| **Token 详情** | /en/moonx/solana/token/{contract} | title/desc/Schema（SSR注入） | 实时价格/K线 |

**实施原则：**
- 静态内容（标题、功能描述、FAQ）= SSR 服务端注入
- 动态内容（实时价格、排行）= 客户端渲染
- Schema 必须 SSR 注入（禁止客户端异步写入，已在规范中明确）

### 1.2 Canonical 修复

| 问题 URL | 修复方式 |
|---------|---------|
| `/en/moonx/trade/dynamic?network=SOLANA` | canonical → `/en/moonx/trade/dynamic` |
| `/en/moonx/pump?network=SOLANA` | canonical → `/en/moonx/pump` |
| `/en/moonx/solana/token?address={x}` | canonical → `/en/moonx/solana/token/{contract}` |
| `/en/moonx/account/my-position` | noindex + robots.txt Disallow |

### 1.3 Token 页 Meta 标准（SSR 动态生成）

**Title 模板（50-60 字符）：**
`{TOKEN} Price, Chart & Trading | MoonX`

**Description 模板（140-160 字符）：**
`Trade {TOKEN} on MoonX. Live price ${PRICE}, 24h change {CHANGE}%, volume ${VOLUME}. Track smart money and trade Solana meme tokens in one platform.`

**Schema：** FinancialProduct + BreadcrumbList（参考 SEO全站规范 10.4）

### 1.4 MoonX 主要功能页 Meta

**markets/trending：**
- Title: `Trending Meme Tokens Today — Live Rankings | MoonX`
- Description: `Track the hottest meme tokens on Solana and BSC in real time. MoonX shows trending coins, smart money moves, and pump.fun launches. Updated every minute.`
- FAQSchema + HowToSchema

**monitor/dynamic/hot（Smart Money）：**
- Title: `Smart Money Tracker — Crypto Wallet Tracking | MoonX`
- Description: `Follow what smart money wallets are buying and selling. MoonX tracks top Solana traders in real time so you can spot trends before they go viral.`

---

## 五、Track 2：Token 页规模化（第 3-8 周）

这是 MoonX SEO 的最大流量来源，也是技术上最独特的机会。

### 5.1 为什么 Token 页是核心资产

- DEXScreener 月流量 2,000万+，70% 来自 token 详情页长尾搜索
- CoinGecko 月流量 5,000万+，大量来自 `{token} price` 长尾词
- MoonX 已有 token 详情页架构，**缺的只是 SSR meta 注入 + sitemap**
- 每个符合条件的 token = 1 个潜在 `{TOKEN} price solana` 排名机会

### 5.2 Token 入 sitemap 准入规则（沿用 P1 文档）

| 维度 | 准入阈值 |
|------|---------|
| 流动性 | LP池 ≥ $5,000 USD |
| 持有人数 | Holders ≥ 100 |
| 存活时长 | 上线 ≥ 48小时 |
| 24h 交易量 | Vol ≥ $1,000 USD |
| 24h 交易笔数 | Tx ≥ 50笔 |

### 5.3 Token 页 SEO 增强（在 SSR meta 基础上）

每个 token 页面需要增加：
1. **固定文字块**（SSR 注入，Google 爬虫能读到）：
   - Token 简介段落（合约地址、链、上线时间）
   - 交易说明（如何在 MoonX 交易此代币）
2. **FAQ Schema**（每个 token 页统一 3 个问题）：
   - "What is {TOKEN}?"
   - "How to buy {TOKEN} on MoonX?"
   - "Is {TOKEN} safe to trade?"
3. **内链**：token 页 → markets/trending → MoonX 首页
4. **死币处理**：不满足准入条件 → noindex（已在 P1 定义）

### 5.4 预期规模

- 当前满足准入条件的活跃 token：估计 500-2,000 个
- 每周新增符合条件的 token：50-200 个
- 12 个月后累计有效 token 页：5,000-20,000 个
- 每个页面平均月流量预估（保守）：5-20 访问
- **12 个月 token 页总贡献：25,000-400,000 月访问**

---

## 六、Track 3：内容/编辑 SEO（第 2-12 周）

### 6.1 内容落地路径

基于 SEO全站规范，内容应接入已有 SSR 内容站体系：

**推荐路径：** `bydfi.com/en/learn/prediction-markets/{slug}`

原因：
- `/learn/` 已有 SSR 渲染
- 已有 sitemap 收录机制（`/events/cms/sitemap/...`）
- 可继承现有域名权重和爬取配额
- 不需要技术额外搭建新目录

**备选路径：** `bydfi.com/en/moonx/learn/{slug}`（若技术更易实现）

→ **需要与技术确认：MoonX 内容页放在 /learn/ 下，还是 /moonx/learn/ 下？**

### 6.2 关键词战略地图（沿用 v1.0，补充 meme token 视角）

**Track 3A：Prediction Market 聚合器（品牌定位）**

| 关键词 | 月搜索量 | 优先级 | 目标文章 |
|--------|---------|-------|---------|
| prediction market aggregator | 1K-5K | P0 | 定义品类文章 |
| polymarket alternative | 10K-50K | P0 | 替代品对比页 |
| best prediction market 2026 | 5K-20K | P1 | 榜单文章 |
| polymarket vs kalshi | 10K-50K | P1 | 对比文章 |
| how to trade prediction markets | 5K-20K | P2 | 入门教程 |

**Track 3B：Meme Token 工具（产品流量）**

| 关键词 | 月搜索量 | 优先级 | 目标文章 |
|--------|---------|-------|---------|
| smart money tracker crypto | 5K-20K | P0 | Smart money 功能页 meta 优化 |
| GMGN alternative | 1K-5K | P1 | 对比文章 |
| best solana meme coin tracker | 2K-10K | P1 | 工具对比文章 |
| pump.fun trading | 5K-20K | P2 | 教程文章 |
| trending meme coins solana | 2K-10K | P2 | trending 页 meta 优化 |

### 6.3 内容架构（修订版）

```
Pillar A：预测市场（品牌定位）
└── Prediction Markets: The Complete Guide 2026（支柱页）
    ├── Best Polymarket Alternatives 2026
    ├── Polymarket vs Kalshi: Full Comparison
    ├── What Is a Prediction Market Aggregator?（MoonX 护城河）
    ├── How to Trade Prediction Markets for Beginners
    └── Prediction Market Odds Explained

Pillar B：Meme Token 工具（产品流量）
└── Best Solana Meme Coin Trading Tools 2026（支柱页）
    ├── What Is Smart Money Tracking in Crypto?
    ├── GMGN vs MoonX: Which Is Better?（吸引竞品流量）
    ├── How to Find the Next 100x Meme Coin (Solana)
    ├── How to Use Pump.fun: Complete Guide
    └── Best Solana Token Trackers Compared
```

---

## 七、执行时间线（修订版）

```
Week 1-2：P0 技术评估
  ├── 技术同学确认 SSR 改造排期
  ├── 确认内容落地路径（/learn/ or /moonx/learn/）
  ├── 开始 Canonical 修复（不需要等 SSR）
  └── 启动 Track 3 内容写作（与技术并行，不相互等待）

Week 3-4：技术修复 + 内容发布第一批
  ├── Canonical 修复完成
  ├── MoonX 主要功能页 meta 写入（即使暂时 CSR 也先写好）
  ├── 发布 3 篇核心内容：Pillar + 聚合器 + Polymarket Alternative
  └── 外联邮件发出（进入现有榜单）

Week 5-8：SSR 改造 + Token 页上线
  ├── MoonX 首页/trending/smart-money SSR 上线
  ├── Token 页 SSR meta 注入
  ├── Token sitemap 开始收录（满足准入条件的 token）
  └── 内容发布第二批（5 篇 Cluster 文章）

Week 9-16：Token 规模化 + 内容建权威
  ├── Token 页从 500 → 2,000+ 有效收录
  ├── 完成全部 Cluster 文章（10 篇）
  ├── 开始媒体外链（CoinDesk/Decrypt 新闻稿）
  └── "prediction market aggregator" 排名目标 Top 10
```

---

## 八、目标修订（更保守但更真实）

| 指标 | M2 | M4 | M6 | M12 |
|------|----|----|----|----|
| 月自然流量 | 200+ | 3,000+ | 15,000+ | 60,000+ |
| 其中 Token 页贡献 | 0 | 2,000+ | 12,000+ | 50,000+ |
| 其中内容页贡献 | 200+ | 1,000+ | 3,000+ | 10,000+ |
| Google 索引有效页 | 50 | 500+ | 2,000+ | 10,000+ |
| "polymarket alternative" 排名 | Top 50 | Top 20 | Top 5 | Top 3 |
| Token 页有效收录数 | 0 | 500+ | 2,000+ | 10,000+ |

---

## 九、需要 Kelly 现在做 2 个决策

**决策 1：SSR 改造排期**
这是整个 SEO 战略的硬性前提。Token 页和 MoonX 功能页的 SEO 完全依赖于此。
- **需要：** 和技术负责人确认 SSR 改造的时间节点
- **最低可行：** 先对 trending 页 + token 详情页实施 SSR，其他页面后续跟上

**决策 2：内容落地路径**
- **A 方案：** `bydfi.com/en/learn/prediction-markets/` — 接入现有内容站，SEO 权重继承更好
- **B 方案：** `bydfi.com/en/moonx/learn/` — MoonX 品牌独立，但需要新建路径

---

## 十、一句话总结

> 这个 SEO 战略的 80% 价值来自 **Token 页规模化**（技术驱动）；
> 20% 来自**预测市场聚合器内容**（品牌定位驱动）。
> 两者需要并行，但先把技术地基打好，内容才能发挥作用。

---

*计划书 v2.0 · 2026-03-11 · 基于 SEO全站规范 + 基础能力优化P1 + SerpAPI 竞品分析*
