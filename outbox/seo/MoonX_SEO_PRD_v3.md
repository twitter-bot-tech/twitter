# MoonX SEO 内容产品需求文档 V1.0

---

## 一、需求介绍

| | |
|---|---|
| **所属业务线** | MoonX 预测市场聚合器 |
| **需求负责人** | Kelly（市场负责人） |
| **需求介绍** | 本文档描述 MoonX SEO 内容体系的完整产品需求，覆盖两类人工内容（预测学院、Meme 学院）和五类程序化 SEO 页面（如何参与预测市场、如何参与美股、如何购买 Meme、价格页、热门榜单页）的前端页面结构、数据调用逻辑、后台管理工具和验收标准。 |

---

## 二、需求背景

### 2.1 需求价值

MoonX 当前自然流量为 0，bydfi.com 整站自然流量 46.1K/月，MoonX 无任何 SEO 内容承接。

核心问题：
- MoonX 全部页面为 CSR 渲染，Google 无法抓取内容，无自然流量
- 没有内容落地页，用户无法通过搜索发现 MoonX
- 竞品 laikalabs.ai（DA 仅 19）已开始布局，存在先发危机

本需求的价值：
1. 建立 MoonX 内容 SEO 体系，目标 6 个月月均自然流量 15,000+
2. 以内容集群方式建立"预测市场聚合器"品牌权威
3. 程序化页面形成规模化流量（Token 价格页 + 预测事件页）

### 2.2 目标用户

站外自然搜索流量：
- 搜索 "polymarket alternative"、"prediction market aggregator" 等词的用户
- 搜索 "how to buy [meme coin]"、"[TOKEN] price prediction" 的 Meme 币用户
- 搜索 "TSLA prediction market odds"、"Tesla prediction market" 的股票用户

### 2.3 典型使用场景

1. 用户搜索"polymarket vs kalshi"进入对比文章 → 了解 MoonX 聚合两者 → 点击进入 MoonX 平台
2. 用户搜索"how to buy PEPE on Solana"进入 Meme 购买指南 → 看到预测市场赔率 → 跳转 MoonX 下注
3. 用户搜索"TSLA prediction market"进入股票预测页 → 看到 Polymarket 上 TSLA 相关市场 → 跳转交易
4. 用户搜索"top prediction markets today"进入热门榜单 → 发现感兴趣市场 → 点击参与

### 2.4 SEO 关键词矩阵（已验证，Ahrefs 真实数据）

| 页面类型 | 核心关键词 | 月搜索量(US) | 竞争度 | 优先级 |
|---------|-----------|------------|--------|--------|
| 预测学院文章 | polymarket vs kalshi | 1,700 | 中 | P0 |
| 预测学院文章 | polymarket alternative | 50 | 低 | P0 |
| 预测学院文章 | prediction market aggregator | 50 | 极低 | P0 |
| 预测学院文章 | what are prediction markets | 500 | 中（AI Overview） | P1 |
| 程序化·价格页 | TSLA prediction market odds | 待测 | 低 | P1 |
| 程序化·Meme 购买 | {TOKEN} price prediction | 中长尾 | 低 | P1 |
| 程序化·热门榜单 | top prediction markets today | 待测 | 中 | P2 |

---

## 三、需求目标

### 3.1 需求目标

1. 建立 MoonX 内容 SEO 基础设施（CMS + 程序化模板 + SSR）
2. 发布第一批 SEO 文章，实现关键词排名突破
3. 程序化页面上线，形成规模化长尾流量

### 3.2 衡量指标

| 指标 | 计算方式 | 目标值 |
|------|---------|--------|
| 月自然流量 | GSC Click 数 | M2: 200+ / M4: 3,000+ / M6: 15,000+ |
| 关键词排名 | GSC Avg Position | "polymarket vs kalshi" M2 进 Top 20，M4 进 Top 5 |
| 内容页收录率 | GSC Coverage 已索引数/发布数 | ≥ 95% |
| 程序化页面数 | sitemap 有效 URL 数 | M3: 500+ / M6: 5,000+ |
| 内容→交易转化率 | 文章页点击 CTA / 进入 MoonX 平台数 | ≥ 3% |

---

## 四、需求范围

### 需求列表

| # | 需求 | 模块 | 涉及终端 | 优先级 |
|---|------|------|---------|--------|
| 1 | 【预测学院】文章列表页前端改造 | 新功能开发 | 客户端 | P0 |
| 2 | 【预测学院】文章详情页前端改造 | 新功能开发 | 客户端 | P0 |
| 3 | 【Meme学院】文章列表页前端改造 | 新功能开发 | 客户端 | P0 |
| 4 | 【Meme学院】文章详情页前端改造 | 新功能开发 | 客户端 | P0 |
| 5 | 【内容 CMS】文章管理后台 | 新功能开发 | 后台管理系统 | P0 |
| 6 | 【程序化·预测事件指南】页面模板 | 新功能开发 | 客户端 | P1 |
| 7 | 【程序化·美股预测页】页面模板 | 新功能开发 | 客户端 | P1 |
| 8 | 【程序化·Meme购买指南】页面模板 | 新功能开发 | 客户端 | P1 |
| 9 | 【程序化·Token价格页】SSR meta 注入 | 已有功能改造 | 客户端 | P1 |
| 10 | 【程序化·热门榜单页】页面模板 | 新功能开发 | 客户端 | P2 |
| 11 | 全站 SEO 技术基础（SSR + Sitemap + Schema） | 新功能开发 | 服务端 | P0 |
| 12 | 【导航入口】bydfi.com 顶部 Nav 新增 Learn 菜单 | 已有功能改造 | 客户端 | P0 |

---

## 五、需求详情

> **通用技术要求（所有 SEO 页面）：**
> - 所有 SEO 页面必须 **SSR 渲染**（服务端渲染），Google 爬虫能直接读取 HTML 内容
> - 每个页面必须有独立的 `<title>`、`<meta description>`、`<canonical>`
> - 所有内链使用 `<a href>` 标签，不使用 JS 跳转
> - 页面加载速度 LCP ≤ 2.5s（Core Web Vitals）

---

### 5.1 人工内容页面

#### 5.1.1 【预测学院】文章列表页

**页面路径：** `/en/moonx/learn/prediction/`
**SEO 定位：** 预测市场内容集群入口，传递权重给子页面

**页面线框图：**
```
┌─────────────────────────────────────────────┐
│  面包屑：MoonX > Learn > Prediction         │
├─────────────────────────────────────────────┤
│  H1: Prediction Market Academy              │
│  副标题：Learn how prediction markets work  │
├────────────────────┬────────────────────────┤
│  分类 Tab：         │                        │
│  All | Beginner |  │  精选文章卡片（大）     │
│  Advanced | Guide  │                        │
├────────────────────┴────────────────────────┤
│  文章卡片列表（每行3个）                      │
│  [封面图] [标题] [摘要] [日期] [阅读时间]    │
│  [封面图] [标题] [摘要] [日期] [阅读时间]    │
│  [封面图] [标题] [摘要] [日期] [阅读时间]    │
├─────────────────────────────────────────────┤
│  分页：< 1 2 3 ... >                         │
└─────────────────────────────────────────────┘
```

**页面元素规格：**

| 页面元素 | 前端功能逻辑 | 数据调用逻辑 | 其他说明 |
|---------|-----------|------------|---------|
| 页面路径导航 | 1. MoonX 首页 > Learn > Prediction<br>2. 点击「MoonX 首页」打开 `/en/moonx/markets/trending`<br>3. 「Prediction」页签不可点击 | 无后台数据调用，前端写死 | |
| SEO Meta | 1. Title: `Prediction Market Academy \| MoonX`<br>2. Meta: Learn how prediction markets work, compare Polymarket vs Kalshi, and find the best platforms. | 前端写死 | SSR 必须 |
| H1 标题 | `Prediction Market Academy` | 前端写死 | |
| 分类 Tab | All / Beginner / Advanced / Comparison / Guide，点击筛选对应分类文章 | 调用 CMS API：`GET /api/articles?category=prediction&tag={tab}` | 默认选中 All |
| 文章卡片 | 显示：封面图、标题、摘要（前120字）、发布日期、预计阅读时间（字数/200）<br>点击跳转文章详情页 | 调用 CMS API：`GET /api/articles?category=prediction&page={n}&limit=12`<br>SSR 渲染前12篇，剩余 CSR 加载 | 图片需 alt 属性 = 文章标题 |
| 内部推荐 | 右侧栏显示「Hot Topics」：3个热门内链文章 | CMS 后台手动配置 | |
| 分页 | 每页12篇，支持翻页，URL 参数：`?page=2` | 前端分页，SEO 无影响 | |

---

#### 5.1.2 【预测学院】文章详情页

**页面路径：** `/en/moonx/learn/prediction/{slug}`
**SEO 定位：** 单篇文章攻打具体关键词
**示例 URL：** `/en/moonx/learn/prediction/polymarket-vs-kalshi`

**页面线框图：**
```
┌─────────────────────────────────────────────────────┐
│  面包屑：MoonX > Learn > Prediction > {文章标题}    │
├───────────────────────────┬─────────────────────────┤
│                           │  右侧栏（Sticky）        │
│  H1: {文章标题}            │  ┌───────────────────┐  │
│  日期 | 作者 | 阅读时间    │  │  目录 TOC          │  │
│                           │  │  - H2 锚点 1       │  │
│  {文章封面图}              │  │  - H2 锚点 2       │  │
│                           │  │  - H2 锚点 3       │  │
│  {文章正文 Markdown 渲染}  │  └───────────────────┘  │
│  - H2 段落                │                         │
│  - H3 子段落              │  ┌───────────────────┐  │
│  - 表格 / 列表            │  │  相关文章          │  │
│                           │  │  推荐3篇           │  │
│  FAQ 折叠模块（Schema）    │  └───────────────────┘  │
│                           │                         │
│  CTA 模块                 │  ┌───────────────────┐  │
│  「Compare on MoonX →」   │  │  MoonX 平台推广位  │  │
│                           │  │  运营配置          │  │
├───────────────────────────┴─────────────────────────┤
│  相关文章推荐（底部，3篇横排）                        │
│  [卡片] [卡片] [卡片]                                │
└─────────────────────────────────────────────────────┘
```

**页面元素规格：**

| 页面元素 | 前端功能逻辑 | 数据调用逻辑 | 其他说明 |
|---------|-----------|------------|---------|
| SEO Meta（SSR） | 1. Title：`{文章SEO标题} \| MoonX`（≤60字符）<br>2. Meta Description：文章 meta（≤155字符）<br>3. Canonical：`https://www.bydfi.com/en/moonx/learn/prediction/{slug}`<br>4. og:title / og:description / og:image | 调用 CMS API：`GET /api/articles/{slug}`<br>**必须 SSR**，不可 CSR | canonical 必须正确，防止重复 |
| 面包屑 | MoonX > Learn > Prediction > {文章标题}<br>每级均为可点击内链 | 前端写死前3级，最后一级从 CMS 取文章标题 | BreadcrumbList Schema 必须注入 |
| H1 | 文章标题，与 Title Tag 可不同（H1 更长更自然） | CMS 字段：`h1_title` | H1 唯一，不可重复 |
| 发布信息 | 显示：发布日期、最近更新日期、作者名（可配置）、预计阅读时间 | CMS 字段 | 日期格式：Month DD, YYYY（利于信任度） |
| 文章正文 | Markdown/富文本渲染，支持：H2/H3、表格、列表、图片、代码块、加粗、内链 | CMS 富文本字段，SSR 渲染 | 图片必须有 alt，内链必须是相对路径 |
| 目录 TOC | 自动提取正文所有 H2/H3 生成，点击锚点跳转 | 前端自动解析 H2/H3，不调用后台 | 右侧 sticky，随滚动高亮当前章节 |
| FAQ 模块 | 折叠式展示 Q&A，每条 Q 点击展开 A | CMS 字段：FAQ 列表（question + answer）<br>**SSR 渲染** | **FAQPage Schema 必须注入**，提升 PAA 覆盖机会 |
| CTA 模块 | 1. 固定文案：「See {keyword} markets on MoonX →」<br>2. 点击跳转 MoonX 对应市场页<br>3. 文章内最多出现 2 次 CTA | CMS 字段：cta_text + cta_url，前端写死跳转逻辑 | CTA 按钮需 nofollow 防止 PageRank 流失 |
| 相关文章推荐 | 1. 右侧栏：同分类文章3篇<br>2. 文章底部：3篇横排卡片 | CMS API：`GET /api/articles?category=prediction&exclude={current_slug}&limit=3`<br>按相关 tag 匹配度排序 | |
| 运营推广位 | 右侧栏可配置的运营 Banner | 后台 Banner 管理工具配置，调取对应语言和适用范围的 banner | 见 5.3.3 Banner 管理 |
| Article Schema | 注入 Article Schema：headline / datePublished / dateModified / author / publisher | SSR 从 CMS 数据生成 | |

---

#### 5.1.3 【Meme学院】文章列表页

**页面路径：** `/en/moonx/learn/meme/`

规格与【预测学院】列表页相同，差异点：

| 差异项 | 预测学院 | Meme 学院 |
|-------|---------|---------|
| H1 | Prediction Market Academy | Meme Coin Academy |
| Meta | Learn prediction markets... | Learn how to research, buy and trade meme coins... |
| 分类 Tab | All / Beginner / Comparison / Guide | All / Research / Buy Guide / Price Prediction / Community |
| CMS 分类参数 | `category=prediction` | `category=meme` |
| 关联推广位 | Polymarket / Kalshi CTA | MoonX Meme 市场 CTA |

---

#### 5.1.4 【Meme学院】文章详情页

**页面路径：** `/en/moonx/learn/meme/{slug}`

规格与【预测学院】详情页相同，差异点：

| 差异项 | 预测学院详情页 | Meme 学院详情页 |
|-------|-------------|--------------|
| CTA 文案 | Compare on MoonX → | See {TOKEN} on MoonX → |
| CTA 跳转 | MoonX 预测市场页 | MoonX Token 价格页 `/en/moonx/solana/token/{contract}` |
| 右侧推广位 | 预测市场 Banner | Meme 代币交易 Banner |
| 相关推荐来源 | `category=meme` 文章 | |

---

### 5.2 程序化 SEO 页面

> **程序化页面通用规则：**
> - 所有模板变量用 `{变量名}` 标注
> - 每个页面的 Title / H1 / Meta 均为程序化生成（模板 + 变量）
> - 所有价格/赔率数据 SSR 生成静态快照，CSR 负责实时更新
> - noindex 规则：事件已结束 >30天 / Token 已 rug / 数据不完整 → 自动加 noindex

---

#### 5.2.1 【程序化·如何参与预测市场】

**页面路径：** `/en/moonx/guide/prediction/{event-slug}`
**示例 URL：** `/en/moonx/guide/prediction/will-trump-win-2026-midterms`
**目标关键词模式：** `how to bet on {event}` / `{event} prediction market odds`
**触发条件：** Polymarket / Kalshi 上 Volume ≥ $100K 的热门市场自动生成页面

**页面线框图：**
```
┌─────────────────────────────────────────────────────┐
│  面包屑：MoonX > Guide > {事件名称}                   │
├─────────────────────────────────────────────────────┤
│  H1: {事件名称}: Prediction Market Odds & How to Bet │
│  最后更新：{更新时间}                                  │
├─────────────────────┬───────────────────────────────┤
│                     │  当前赔率卡片（SSR快照）         │
│  事件介绍            │  ┌─────────────────────────┐   │
│  {事件背景说明}      │  │ YES: 72% | NO: 28%      │   │
│                     │  │ Polymarket Volume: $2.1M │   │
│  如何参与（步骤）    │  │ Kalshi Odds: 70%         │   │
│  Step 1: ...        │  └─────────────────────────┘   │
│  Step 2: ...        │                                 │
│  Step 3: ...        │  [Bet on MoonX →]               │
│                     │                                 │
│  平台对比           │  相关市场                        │
│  Polymarket vs      │  [市场1] [市场2] [市场3]         │
│  Kalshi odds        │                                 │
│                     │                                 │
│  FAQ                │                                 │
└─────────────────────┴───────────────────────────────┘
```

**页面元素规格：**

| 页面元素 | 前端功能逻辑 | 数据调用逻辑 | 其他说明 |
|---------|-----------|------------|---------|
| SEO Meta（SSR） | Title: `{事件名称}: Prediction Market Odds \| MoonX`<br>Meta: `What are the current odds for {事件}? Compare Polymarket ({YES%}) and Kalshi ({YES%}) odds. Trade on MoonX.` | **SSR** 从 Polymarket/Kalshi API 获取赔率生成 meta | 赔率每日更新，meta 同步更新 |
| H1 | `{事件名称}: Prediction Market Odds & How to Bet` | 模板+事件名，SSR 生成 | |
| 事件介绍 | 2-3段文字介绍事件背景，带内链到相关文章 | **SSR 模板文字**（固定结构）+ 事件标题/截止日期（API 动态） | 模板由市场团队配置，变量自动填充 |
| 当前赔率卡片 | 1. 显示 YES%/NO%<br>2. 分别展示 Polymarket 和 Kalshi 赔率<br>3. 显示 Volume 和 Open Interest | SSR 快照：渲染时拉取 Polymarket API + Kalshi API 最新数据<br>CSR：每60秒更新一次赔率显示 | SSR 快照确保 Google 能索引初始赔率 |
| 如何参与步骤 | 固定模板，3步：1.访问MoonX 2.连接钱包/注册 3.搜索市场下注 | 前端写死模板，带 MoonX 跳转链接 | |
| 平台赔率对比 | 表格展示：平台 / 当前赔率 / 手续费 / 流动性 | SSR + CSR 实时更新 Polymarket + Kalshi 数据 | |
| 相关市场 | 3个同主题相关预测市场，横排卡片 | MoonX API：`GET /api/markets/related?event_id={id}&limit=3` | |
| FAQ | 3-5条 FAQ，程序化生成常见问题（谁赢/截止日期/如何交易） | 模板 FAQ + 事件变量，SSR 渲染，FAQPage Schema | |
| CTA 按钮 | `Bet on This Market →` 跳转 MoonX 对应市场 | 前端写死跳转 `/en/moonx/markets/{market_id}` | |

**入库准入规则：**
- Polymarket Volume ≥ $100K 或 Kalshi Volume ≥ $50K
- 市场截止日期 > 当前日期
- 已结束且 >30 天的市场：加 `noindex`，保留页面历史记录用

---

#### 5.2.2 【程序化·如何参与美股预测市场】

**页面路径：** `/en/moonx/markets/stocks/{ticker}`
**示例 URL：** `/en/moonx/markets/stocks/TSLA`、`/en/moonx/markets/stocks/GOOGL`
**目标关键词模式：** `{TICKER} prediction market odds` / `{TICKER} Polymarket` / `bet on {company} stock`

**页面线框图：**
```
┌──────────────────────────────────────────────────────┐
│  面包屑：MoonX > Markets > Stocks > {TICKER}          │
├──────────────────────────────────────────────────────┤
│  股票图标  {公司名} ({TICKER}) Prediction Market Odds │
│  H1: Bet on {公司名} Stock: Polymarket Odds & Markets │
├─────────────────────┬────────────────────────────────┤
│                     │  实时数据卡片（SSR快照）          │
│  公司简介           │  ┌──────────────────────────┐   │
│  {公司简介2句话}    │  │ 股价: $399.24（CSR实时）  │   │
│                     │  │ 今日涨幅: +0.14%          │   │
│  预测市场概况       │  └──────────────────────────┘   │
│  Polymarket 相关    │                                  │
│  市场列表           │  ┌──────────────────────────┐   │
│                     │  │ Polymarket 市场：          │   │
│  分析师预测 vs      │  │ · Will TSLA > $400?        │   │
│  预测市场对比       │  │   YES 45% / NO 55%         │   │
│                     │  │ · TSLA Q1 earnings beat?  │   │
│  如何下注步骤       │  │   YES 38% / NO 62%         │   │
│                     │  └──────────────────────────┘   │
│  FAQ                │  [See All TSLA Markets →]        │
└─────────────────────┴────────────────────────────────┘
```

**页面元素规格：**

| 页面元素 | 前端功能逻辑 | 数据调用逻辑 | 其他说明 |
|---------|-----------|------------|---------|
| SEO Meta（SSR） | Title: `{公司名} ({TICKER}) Prediction Market: Polymarket Odds \| MoonX`<br>Meta: `Trade prediction markets on {公司名} stock. Current Polymarket odds: YES {%}. Compare markets on MoonX.` | **SSR** 生成，赔率从 Polymarket API 取最热市场第一条 | 每日重新生成 |
| H1 | `{公司名} ({TICKER}) Prediction Market Odds` | 模板+股票名称，SSR | |
| 实时股价 | 显示当前股价和涨跌幅 | CSR：调用 Yahoo Finance / Polygon.io API<br>SSR 快照：T-1日收盘价 | CSR 实时刷新，SSR 保底数据 |
| Polymarket 市场列表 | 显示 Polymarket 上与该股票相关的所有活跃市场<br>每条：市场名称 / YES% / NO% / Volume / 截止日期 | **SSR** 生成：Polymarket API 搜索 `{TICKER}` 相关市场<br>CSR 每30分钟更新赔率 | |
| 分析师预测 vs 预测市场 | 对比模块：分析师平均目标价 vs 预测市场对应事件赔率 | 分析师数据：Yahoo Finance API（或手动配置）<br>预测市场：Polymarket API | 高价值内容，提升停留时间 |
| 如何下注步骤 | 固定模板 3 步引导用户到 MoonX 参与 | 前端写死，带 CTA 跳转 | |
| 公司简介 | 2-3句话介绍公司 | **后台手动配置**（初期）或 Wikipedia API 自动拉取 | 不超过3句，聚焦 SEO 目的 |
| FAQ | 3-5条：`Is it legal to bet on {TICKER}?` / `What are current {TICKER} prediction market odds?` | 模板+变量，SSR，FAQPage Schema | |

**当前支持股票列表（Phase 1）：** TSLA、GOOGL、AAPL、MSFT、NVDA、META、AMZN、SPY

---

#### 5.2.3 【程序化·如何购买 Meme 币】

**页面路径：** `/en/moonx/guide/buy/{token-slug}`
**示例 URL：** `/en/moonx/guide/buy/pepe-solana`、`/en/moonx/guide/buy/dogecoin`
**目标关键词模式：** `{TOKEN} price prediction` / `{TOKEN} Polymarket odds` / `bet on {TOKEN}`

> ⚠️ **策略说明：** 经 SERP 分析，"how to buy {TOKEN}" 类词被 Binance/MetaMask/Gemini 垄断，MoonX 不做正面竞争。本页面定位改为：**{TOKEN} 的预测市场赔率 + 如何在 MoonX 上下注该币的走势**，差异化切入。

**页面线框图：**
```
┌──────────────────────────────────────────────────────┐
│  面包屑：MoonX > Guide > Buy > {TOKEN}                │
├──────────────────────────────────────────────────────┤
│  {代币图标} {TOKEN NAME} Price Prediction & Odds      │
│  H1: {TOKEN}: Price Prediction, Polymarket Odds      │
│       & How to Trade                                  │
├─────────────────────┬────────────────────────────────┤
│                     │  价格快照（SSR）                  │
│  代币简介           │  ┌──────────────────────────┐   │
│  什么是 {TOKEN}     │  │ 当前价格: $0.0000123      │   │
│                     │  │ 24h 涨幅: +12.3%          │   │
│  价格预测           │  │ 市值: $1.2B               │   │
│  看涨/看空分析      │  └──────────────────────────┘   │
│  （程序化生成）     │                                  │
│                     │  预测市场赔率                    │
│  预测市场赔率       │  ┌──────────────────────────┐   │
│  Polymarket 相关    │  │ Will PEPE reach ATH?      │   │
│  市场               │  │ YES 23% / NO 77%          │   │
│                     │  └──────────────────────────┘   │
│  如何在 MoonX 上    │                                  │
│  下注该币走势       │  [Bet on PEPE on MoonX →]        │
│                     │                                  │
│  购买指南简版       │                                  │
│  （3个主流平台）    │                                  │
│                     │                                  │
│  FAQ                │                                  │
└─────────────────────┴────────────────────────────────┘
```

**页面元素规格：**

| 页面元素 | 前端功能逻辑 | 数据调用逻辑 | 其他说明 |
|---------|-----------|------------|---------|
| SEO Meta（SSR） | Title: `{TOKEN} Price Prediction & Polymarket Odds \| MoonX`<br>Meta: `{TOKEN} is currently ${price}, up/down {%} in 24h. See Polymarket prediction market odds and how to trade {TOKEN} on MoonX.` | **SSR** 从 CoinGecko API 取价格，Polymarket API 取赔率 | 每日更新 |
| H1 | `{TOKEN NAME}: Price Prediction & How to Trade on MoonX` | 模板+代币名，SSR | |
| 价格快照 | SSR 快照：当前价格、24h 涨幅、市值<br>CSR：实时刷新（60秒） | SSR：CoinGecko API `/simple/price?ids={id}&vs_currencies=usd&include_24hr_change=true`<br>CSR 同接口实时更新 | SSR 保证 Google 能爬到初始价格 |
| 价格预测模块 | 程序化生成价格预测文字，基于技术信号：<br>- 看涨（价格在20日均线之上 + 成交量增加）<br>- 中性（震荡区间）<br>- 看跌（价格在20日均线之下） | **SSR**：CoinGecko 历史价格 API + 服务端模板选择 | 声明"非投资建议"免责 |
| 预测市场赔率 | 显示 Polymarket 上该代币相关市场（如"Will X reach Y price"） | Polymarket API 搜索代币相关市场<br>**SSR** + CSR 更新 | 无相关市场时显示"No active markets"并内链到热门市场 |
| 如何在 MoonX 下注 | 固定模板 3 步引导 + CTA 按钮 | 前端写死 | |
| 购买指南简版 | 3个主流平台（Binance/Coinbase/OKX）的简短购买说明 + 各自跳转链接 | 前端写死，带 rel="nofollow" | 不和 How to buy 页面竞争，仅作参考 |
| FAQ | `What is {TOKEN}?` / `Will {TOKEN} go up?` / `Where can I bet on {TOKEN}?` | 模板+变量，FAQPage Schema | |

**Token 页面准入规则：**
- 市值 ≥ $10M
- 近7日有 Polymarket/MoonX 相关预测市场，或
- CoinGecko 日均交易量 ≥ $1M
- 已 rug / 停止交易 >30天：加 noindex

---

#### 5.2.4 【程序化·Token 价格详情页 SSR 改造】

**页面路径：** `/en/moonx/solana/token/{contract}`（现有页面，需 SSR 改造）
**SEO 定位：** `{TOKEN} price`、`{TOKEN} chart`、`{TOKEN} market cap`

> **说明：** 本页面已存在（CSR），本需求为改造为 SSR，并注入 SEO meta 和 Schema。

**改造要点：**

| 改造项 | 现状 | 目标 |
|--------|------|------|
| 渲染方式 | CSR（Google 看不到内容） | **SSR** + CSR 混合 |
| Title | 无 / 通用标题 | `{TOKEN} Price Today: ${price} \| MoonX` |
| Meta Description | 无 | `{TOKEN} is currently ${price}, market cap ${mcap}. View real-time chart and prediction market odds on MoonX.` |
| H1 | 无 | `{TOKEN NAME} ({TICKER}) Price` |
| Canonical | 无 | `https://www.bydfi.com/en/moonx/solana/token/{contract}` |
| Schema | 无 | FinancialProduct Schema + BreadcrumbList Schema |
| Sitemap | 未收录 | 满足准入条件的 Token 自动加入 sitemap |

**Token Sitemap 准入阈值：**
- LP ≥ $5K
- Holders ≥ 100
- 上线 ≥ 48小时
- 日均成交量 ≥ $1K
- 日均交易笔数 ≥ 50

---

#### 5.2.5 【程序化·热门榜单页】

**页面路径：** `/en/moonx/markets/trending/{category}`
**示例 URL：** `/en/moonx/markets/trending/crypto`、`/en/moonx/markets/trending/politics`
**目标关键词模式：** `top prediction markets {category}` / `trending prediction markets today`

**页面线框图：**
```
┌──────────────────────────────────────────────────────┐
│  面包屑：MoonX > Markets > Trending > {分类}          │
├──────────────────────────────────────────────────────┤
│  H1: Top {分类} Prediction Markets Today             │
│  更新时间：{更新时间}                                  │
├──────────────────────────────────────────────────────┤
│  分类 Tab：All | Crypto | Politics | Sports | Stocks  │
├──────────────────────────────────────────────────────┤
│  市场卡片列表（每行2个）                               │
│  ┌─────────────────────┐  ┌─────────────────────┐   │
│  │ 市场标题            │  │ 市场标题            │   │
│  │ YES 72% ████░░ NO  │  │ YES 45% ████░░ NO  │   │
│  │ Vol: $2.1M | 3天后  │  │ Vol: $890K | 7天后  │   │
│  │ [Bet on MoonX →]    │  │ [Bet on MoonX →]    │   │
│  └─────────────────────┘  └─────────────────────┘   │
├──────────────────────────────────────────────────────┤
│  SEO 描述段落（SSR，200字，覆盖关键词）                 │
├──────────────────────────────────────────────────────┤
│  相关文章推荐（内链到预测学院）                         │
└──────────────────────────────────────────────────────┘
```

**页面元素规格：**

| 页面元素 | 前端功能逻辑 | 数据调用逻辑 | 其他说明 |
|---------|-----------|------------|---------|
| SEO Meta（SSR） | Title: `Top {分类} Prediction Markets Today \| MoonX`<br>Meta: `Discover the most active {分类} prediction markets. Track real-time odds on Polymarket and Kalshi, aggregated by MoonX.` | **SSR** 生成，分类名填入模板 | 每日重新生成，meta 包含当日日期 |
| H1 | `Top {分类} Prediction Markets Today` | 模板+分类，SSR | |
| 分类 Tab | All / Crypto / Politics / Sports / Stocks / Science，点击切换 | URL 参数切换：`/trending/crypto`、`/trending/politics` 等 | 每个分类都是独立 URL，独立 SEO |
| 市场卡片列表 | 显示：市场标题 / YES% 进度条 / Volume / 截止日期 / 来源平台（Polymarket/Kalshi/Manifold 图标）<br>点击跳转 MoonX 市场页 | **SSR**：MoonX 聚合 API 取 Top 20 热门市场，按 Volume 排序<br>CSR 每30分钟更新赔率 | SSR 快照保证 Google 爬到初始数据 |
| SEO 描述段落 | 200字自然语言描述，覆盖页面核心关键词 | **SSR 写死模板**，包含：当前热门市场名称、Volume 数字、分类名 | 每日重新生成 |
| 相关文章内链 | 底部展示 3 篇预测学院相关文章 | CMS API 按 tag 匹配 | |
| 更新时间 | 显示页面数据最后更新时间 | 服务端生成时间戳，SSR | 增加内容新鲜度信号 |

**分类页 URL 列表：**
- `/en/moonx/markets/trending/` （All，主榜单）
- `/en/moonx/markets/trending/crypto`
- `/en/moonx/markets/trending/politics`
- `/en/moonx/markets/trending/sports`
- `/en/moonx/markets/trending/stocks`
- `/en/moonx/markets/trending/science`

---

### 5.3 后台管理工具

#### 5.3.1 【内容 CMS】文章管理

> 覆盖：预测学院 + Meme 学院所有文章的创建/编辑/发布

**文章新增/编辑页字段：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| 文章分类 | 下拉：prediction / meme | 是 | 决定 URL 前缀 |
| Slug（URL） | 文本 | 是 | 字母+连字符，系统自动校验唯一性 |
| H1 标题 | 文本（≤80字符） | 是 | 页面显示标题 |
| SEO Title | 文本（≤60字符） | 是 | `<title>` 标签，字符计数实时显示 |
| Meta Description | 文本（≤155字符） | 是 | 字符计数实时显示 |
| 封面图 | 图片上传 | 是 | 自动生成 alt = H1 标题 |
| 文章正文 | Markdown 富文本编辑器 | 是 | 支持 H2/H3/表格/图片/内链 |
| FAQ 模块 | 动态添加 Q&A 组 | 否 | 用于生成 FAQPage Schema |
| CTA 文案 | 文本 | 是 | 默认：「Compare on MoonX →」 |
| CTA 跳转 URL | 文本 | 是 | 默认：`/en/moonx/markets/trending` |
| 标签（Tags） | 多选/自由输入 | 是 | 用于相关文章推荐匹配 |
| 难度等级 | 下拉：Beginner / Advanced | 是 | |
| 作者 | 文本 | 否 | 默认：MoonX Team |
| 发布状态 | 草稿 / 待审核 / 已发布 | 是 | |
| 发布时间 | 日期选择器 | 是 | 支持定时发布 |
| 关联内链 | 多选已有文章 | 否 | 底部相关推荐强制指定 |

**列表页筛选条件：** 分类（多选）/ 标签（多选）/ 状态（多选）/ 发布时间范围

---

#### 5.3.2 【程序化内容】后台配置工具

> 覆盖：如何参与预测市场 / 如何参与美股 / 如何购买 Meme 的模板配置

**模板配置字段：**

| 字段 | 适用页面 | 说明 |
|------|---------|------|
| 市场准入阈值 | 预测事件指南 | Volume 下限，低于此值不生成页面 |
| 事件简介模板 | 预测事件指南 | Markdown 模板，`{事件名}` 等变量自动替换 |
| 步骤文案模板 | 全部程序化页面 | 固定3步引导，支持多语言 |
| 股票列表维护 | 美股预测页 | 维护支持的股票 TICKER 列表 + 公司简介 |
| Token 准入阈值 | Meme 购买页 | 市值/Volume 门槛配置 |
| FAQ 模板库 | 全部程序化页面 | 预设 FAQ 模板，按页面类型匹配 |
| noindex 规则配置 | 全部程序化页面 | 配置自动加 noindex 的条件 |

---

#### 5.3.3 【运营 Banner】配置工具

复用 BYDFi 现有运营资源位管理工具，新增适用范围：
- 预测学院文章页（`/en/moonx/learn/prediction/*`）
- Meme 学院文章页（`/en/moonx/learn/meme/*`）
- 程序化页面（`/en/moonx/guide/*`、`/en/moonx/markets/stocks/*`）

---

### 5.5 导航入口

#### 5.5.1 bydfi.com 顶部 Nav 新增 Learn 菜单

**改动范围：** bydfi.com MoonX 顶部全局导航栏（`moonx-fixed-theme-header` 组件）

**位置：** Lucky Draw 之后新增 Learn 菜单项

**改后 Nav 顺序：**
```
Exchange | 🔥 MoonX | Pump | Markets | Trade | Copy Trade | Monitor | Portfolio | 🎰 Lucky Draw | Learn ← 新增
```

**Learn 菜单展开项（hover 下拉）：**

| 菜单项 | 跳转 URL | 说明 |
|--------|---------|------|
| Trending Markets | `/en/moonx/markets/trending` | 热门预测市场榜单 |
| Prediction Academy | `/en/moonx/learn/prediction` | 预测市场学习中心 |
| Meme Academy | `/en/moonx/learn/meme` | Meme 币研究学习中心 |

**技术规格：**
- 菜单项使用 `<a href>` 标签（非 JS 跳转），确保 Google 可爬取内链
- 当前页面在 Learn 子页时，Learn 菜单项高亮（active 状态）
- 下拉菜单 hover 触发，样式与现有 Nav 保持一致

**SEO 价值：** 全站每个页面（包括高权重首页、交易页）均向 SEO 内容页传递 PageRank，是整个 SEO 体系中权重传递的核心链路。

---

### 5.4 SEO 技术基础要求

#### Sitemap 结构

```
sitemap_index.xml
├── sitemap_moonx_articles.xml     （预测学院 + Meme 学院文章）
├── sitemap_moonx_guides.xml       （程序化·事件指南 + Meme 购买）
├── sitemap_moonx_stocks.xml       （美股预测页）
├── sitemap_moonx_tokens.xml       （Token 价格页，满足准入条件）
└── sitemap_moonx_trending.xml     （热门榜单页）
```

**Sitemap 更新频率：**
- 文章页：发布/更新后实时更新
- 程序化页：每日凌晨全量重建

#### Schema 类型对应

| 页面类型 | Schema 类型 |
|---------|------------|
| 文章详情页 | Article + FAQPage + BreadcrumbList |
| 程序化·事件指南 | Article + FAQPage + BreadcrumbList |
| 美股预测页 | FinancialProduct + FAQPage + BreadcrumbList |
| Meme 购买页 | Article + FAQPage + BreadcrumbList |
| Token 价格页 | FinancialProduct + BreadcrumbList |
| 热门榜单页 | ItemList + BreadcrumbList |

#### Canonical 规则

| 场景 | Canonical 指向 |
|------|--------------|
| 正常页面 | 指向自身 |
| 有 `?page=2` 参数的列表页 | 指向不带参数的第1页 |
| 已结束市场（noindex） | 保留页面，加 noindex meta |

---

## 六、验收标准

### 6.1 人工内容页面验收

- [ ] 访问 `/en/moonx/learn/prediction/` 列表页，查看源代码，能看到完整 HTML 文章列表（SSR）
- [ ] 访问 `/en/moonx/learn/prediction/polymarket-vs-kalshi` 详情页，查看源代码，能看到完整文章内容（SSR）
- [ ] `<title>` 为「Polymarket vs Kalshi: Full Comparison 2026 | MoonX」
- [ ] `<meta name="description">` 存在且 ≤ 155 字符
- [ ] `<link rel="canonical">` 正确指向当前 URL
- [ ] 页面包含 Article Schema（可用 Google Rich Results Test 验证）
- [ ] 页面包含 FAQPage Schema（有 FAQ 模块时）
- [ ] 页面包含 BreadcrumbList Schema
- [ ] FAQ 模块折叠/展开功能正常
- [ ] 内链正常跳转，无 404
- [ ] GSC URL Inspection 抓取结果：「页面已编入索引」

### 6.2 程序化页面验收

- [ ] 访问 `/en/moonx/markets/stocks/TSLA`，源代码包含 TSLA 相关 Polymarket 市场数据（SSR）
- [ ] Token 页 `/en/moonx/solana/token/{contract}` 源代码包含 Title、Meta、H1
- [ ] sitemap_moonx_tokens.xml 包含满足准入条件的 Token URL
- [ ] 热门榜单页 CSR 赔率数据30分钟内更新一次
- [ ] 已结束市场页面加了 `<meta name="robots" content="noindex">`

### 6.3 后台工具验收

- [ ] CMS 可新增/编辑/发布文章，状态流转正常
- [ ] 发布文章后，前端页面实时更新，无需重新部署
- [ ] SEO Title 超过60字符时，后台有实时警告
- [ ] 文章 Slug 已存在时，后台报错提示
- [ ] 定时发布功能：设置未来时间发布，到时间自动上线

---

### 6.4 导航入口验收

- [ ] bydfi.com 顶部 Nav 在 Lucky Draw 之后出现 Learn 菜单项
- [ ] hover Learn 展开下拉菜单，包含：Trending Markets / Prediction Academy / Meme Academy
- [ ] 三个子项均为 `<a href>` 标签，Googlebot 可直接抓取
- [ ] 当前页面在 `/en/moonx/learn/*` 或 `/en/moonx/markets/trending` 时，Learn 呈高亮状态
- [ ] 查看任意 bydfi.com 页面源代码，能找到 Learn 下拉内三个子链接的 `<a href>`

---

*文档版本：V1.1 · 更新日期：2026-03-13 · 负责人：Kelly · 变更：新增需求12（导航入口）及 §5.5 导航入口规格*
