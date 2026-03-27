# MoonX SSR Meta 注入规格
**给技术团队 | 2026-03-11**
**目的：** 为以下 MoonX 页面提供 SSR 静态内容注入规格，供 Google 爬虫正常索引

---

## 原则

- **静态内容（必须 SSR）：** title、description、H1、功能描述段落、FAQ 文字、Schema JSON-LD
- **动态内容（可 CSR）：** 实时价格、排行列表、K线图、交易量数字
- Schema 必须在服务端注入 `<head>` 内 `<script type="application/ld+json">`，禁止客户端异步写入

---

## 页面 1：MoonX 首页
**URL:** `/en/moonx`

### Meta
```
title: MoonX — Prediction Market Aggregator | Trade Polymarket, Kalshi & More
description: MoonX aggregates prediction markets from Polymarket, Kalshi, and Manifold in one place. Compare odds, track smart money, and trade meme tokens on Solana. Built by BYDFi.
```

### H1（页面显示）
```
The Prediction Market Aggregator
```

### 静态说明段落（SSR 注入，可视觉上放在功能区下方）
```
MoonX combines prediction markets from Polymarket, Kalshi, and Manifold with 
real-time Solana meme token tracking. Access trending markets, follow smart money 
wallets, and trade from a single platform—no multiple accounts needed.
```

### Schema
```json
{
  "@context": "https://schema.org",
  "@type": "WebPage",
  "name": "MoonX — Prediction Market Aggregator",
  "description": "MoonX aggregates prediction markets from Polymarket, Kalshi, and Manifold.",
  "url": "https://www.bydfi.com/en/moonx",
  "provider": {
    "@type": "Organization",
    "name": "MoonX by BYDFi",
    "url": "https://www.bydfi.com/en/moonx"
  }
}
```

---

## 页面 2：Trending Markets
**URL:** `/en/moonx/markets/trending`

### Meta
```
title: Trending Prediction Markets Today — Live Odds | MoonX
description: Track the hottest prediction markets right now. MoonX shows trending markets from Polymarket, Kalshi, and Manifold with live odds and 24h volume. Updated in real time.
```

### H1
```
Trending Prediction Markets
```

### 静态说明段落（页面顶部，爬虫可读）
```
MoonX tracks the most active prediction markets across Polymarket, Kalshi, and 
Manifold in real time. Browse trending markets by category—politics, crypto, 
sports, and more—and compare odds side by side before you trade.
```

### FAQ Schema（3 个问题，SSR 注入）
```json
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {
      "@type": "Question",
      "name": "What are trending prediction markets?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Trending prediction markets are active markets showing the highest trading volume and price movement in the last 24 hours across platforms like Polymarket and Kalshi."
      }
    },
    {
      "@type": "Question",
      "name": "How does MoonX show trending markets?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "MoonX aggregates live data from Polymarket, Kalshi, and Manifold, then ranks markets by 24h volume and activity to surface the most relevant trending opportunities."
      }
    },
    {
      "@type": "Question",
      "name": "Can I trade directly from the trending page?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Yes. Click any market on MoonX to see full details and place a trade directly through the platform."
      }
    }
  ]
}
```

---

## 页面 3：Smart Money Tracker
**URL:** `/en/moonx/monitor/dynamic/hot`

### Meta
```
title: Smart Money Crypto Tracker — Follow Top Wallets | MoonX
description: Track what smart money wallets are buying and selling in real time. MoonX monitors top Solana traders so you can spot trends before they go viral. Free to use.
```

### H1
```
Smart Money Tracker
```

### 静态说明段落
```
Smart money tracking shows you what the most profitable crypto wallets are trading 
right now. MoonX monitors on-chain activity from top Solana traders and surfaces 
their moves in real time—so you can follow the signal, not the noise.
```

### FAQ Schema
```json
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {
      "@type": "Question",
      "name": "What is smart money tracking in crypto?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Smart money tracking monitors wallets belonging to highly profitable traders and funds. By watching their on-chain activity, you can identify which tokens they're accumulating or selling before price moves happen."
      }
    },
    {
      "@type": "Question",
      "name": "How does MoonX track smart money?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "MoonX analyzes on-chain data from the Solana blockchain to identify wallets with consistently high returns. It then displays their recent trades and portfolio changes in a real-time dashboard."
      }
    },
    {
      "@type": "Question",
      "name": "Is smart money tracking free on MoonX?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Yes. MoonX's smart money tracker is free to use. Create an account to access full wallet history and set up trade alerts."
      }
    }
  ]
}
```

---

## 页面 4：Token 详情页（动态模板）
**URL:** `/en/moonx/solana/token/{contract}`

### Meta 模板（服务端动态生成）
```
title: {TOKEN_SYMBOL} Price, Chart & Trading | MoonX
  示例: PEPE Price, Chart & Trading | MoonX
  字符限制: 50-60 字符

description: Trade {TOKEN_SYMBOL} on MoonX. Live price ${PRICE}, 24h {CHANGE_DIRECTION} {CHANGE_PCT}%, volume ${VOLUME_24H}. Track smart money and trade Solana tokens in real time.
  示例: Trade PEPE on MoonX. Live price $0.0000142, 24h up 18.3%, volume $2.4M. Track smart money and trade Solana tokens in real time.
  字符限制: 140-160 字符
```

### H1 模板
```
{TOKEN_SYMBOL} ({TOKEN_NAME}) — Live Price & Trading
```

### 固定静态段落（所有 token 页一致，SSR 注入）
```
Trade {TOKEN_SYMBOL} on MoonX, the Solana meme token trading platform. 
View live price charts, track smart money wallet activity, and execute trades 
directly. MoonX aggregates liquidity across Solana DEXs for best execution.
```

### Schema 模板
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

配合 BreadcrumbList（参考 SEO全站规范 10.4，已有示例）。

### 死币处理
不满足准入条件的 token（LP<$5K / Holders<100 / 上线<48h / Vol<$1K / Tx<50）：
- `<meta name="robots" content="noindex, nofollow" />`
- 不放入 sitemap

---

## 页面 5：Pump 页
**URL:** `/en/moonx/pump`

### Meta
```
title: Pump.fun Token Trading — New Launches on Solana | MoonX
description: Trade the latest pump.fun token launches on Solana. MoonX shows new token launches with real-time price, volume, and smart money activity. Find the next 100x early.
```

### H1
```
Pump.fun Token Trading
```

### 静态说明段落
```
MoonX tracks every new token launched on pump.fun in real time. Filter by 
market cap, holder count, and smart money activity to find early opportunities 
before they trend. Trade directly from MoonX with live Solana DEX liquidity.
```

---

## Sitemap 收录规则（for Token 页）

Token 页加入 sitemap 的前提（准入阈值）：
- LP池 ≥ $5,000 USD
- Holders ≥ 100
- 上线 ≥ 48小时
- 24h Vol ≥ $1,000 USD
- 24h Tx ≥ 50笔

不满足以上任一条件 → noindex，不放 sitemap

Sitemap 文件建议：
`www.bydfi.com/events/cms/sitemap/pseo/moonx-token-summary.xml`
- changefreq: `daily`
- priority: `0.6`
- lastmod: 实际最后更新时间（精确到秒，带 UTC 时区）

---

## 需要技术确认的事项

1. 上述哪些页面的 SSR 改造在当前排期内？
2. Token 页的 token 名称/价格/涨跌幅数据，服务端可以获取吗？（SSR meta 动态注入需要）
3. token sitemap 自动更新机制是否可以接入现有的 sitemap 扫描体系？
4. `/en/moonx/learn/` 路径什么时候可以建好？

