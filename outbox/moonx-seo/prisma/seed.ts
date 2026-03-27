import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

async function main() {
  console.log('🌱 Seeding MoonX SEO database…')

  // ── Articles ────────────────────────────────────────────────────────────

  await prisma.article.upsert({
    where: { slug: 'polymarket-vs-kalshi' },
    update: {},
    create: {
      category: 'prediction',
      slug: 'polymarket-vs-kalshi',
      h1: 'Polymarket vs Kalshi: Which Prediction Market is Better in 2026?',
      seoTitle: 'Polymarket vs Kalshi (2026): Full Comparison | MoonX',
      metaDesc: 'Polymarket vs Kalshi compared: liquidity, fees, market selection, regulation, and who wins for different trader types. Updated 2026.',
      content: `## Polymarket vs Kalshi: The Complete 2026 Guide

Polymarket and Kalshi are the two biggest names in prediction markets — but they serve very different audiences. Here's everything you need to know.

### What is Polymarket?

Polymarket is a decentralized prediction market built on Polygon. It uses USDC as its trading currency and requires no KYC for most markets. With over **$500M in monthly volume**, it's the largest prediction market by liquidity.

**Polymarket pros:**
- No KYC required (pseudonymous trading)
- Highest liquidity and most active markets
- Global access
- Open-source, non-custodial

**Polymarket cons:**
- US users officially restricted (though VPNs are common)
- Smart contract risk
- Some markets have low liquidity

### What is Kalshi?

Kalshi is a CFTC-regulated US prediction market. It requires identity verification and operates under strict US financial regulations. This makes it safer and more legitimate for US traders.

**Kalshi pros:**
- CFTC-regulated — legal for US users
- Bank-level security and fraud protection
- USD deposits (no crypto required)
- Growing market selection

**Kalshi cons:**
- US-only (no international access)
- Lower liquidity than Polymarket on most markets
- Higher fees for smaller traders
- KYC required

### Head-to-Head Comparison

| Feature | Polymarket | Kalshi |
|---------|------------|--------|
| Regulation | Unregulated (Polygon) | CFTC-regulated |
| US access | Restricted | ✅ Fully legal |
| Volume | ~$500M/mo | ~$50M/mo |
| Markets | 500+ | 200+ |
| Currency | USDC | USD |
| KYC | No | Yes |
| Fees | ~1-2% | ~1-3% |

### Which Should You Use?

**Choose Polymarket if:** You want maximum liquidity, global access, and trade crypto naturally.

**Choose Kalshi if:** You're US-based, want regulatory protection, and prefer USD.

**Use MoonX:** MoonX aggregates both — so you always see the best odds across both platforms without logging into each separately.

## How MoonX Helps

MoonX is the first prediction market aggregator. We show you odds from Polymarket, Kalshi, and Manifold side-by-side so you can:

1. **Find the best price** — platforms often disagree on odds
2. **Track smart money** — see which wallets are moving large positions
3. **Discover new markets** — one feed for all platforms

[Start comparing on MoonX →](https://www.bydfi.com/en/moonx/markets/trending)
`,
      author: 'Nate Silver (MoonX Strategy)',
      tags: JSON.stringify(['polymarket', 'kalshi', 'prediction-market', 'comparison']),
      faqs: JSON.stringify([
        { q: 'Is Polymarket legal in the US?', a: 'Polymarket has restricted US users from its platform, though enforcement is limited. Kalshi is the legally compliant alternative for US traders.' },
        { q: 'Which has better liquidity: Polymarket or Kalshi?', a: 'Polymarket has significantly higher liquidity — typically 5-10x more volume on most markets compared to Kalshi.' },
        { q: 'Can I use both Polymarket and Kalshi?', a: 'Yes — many traders use both to find arbitrage opportunities. MoonX makes this easier by showing odds from both platforms in one place.' },
        { q: 'What currency does Polymarket use?', a: 'Polymarket uses USDC (a USD-pegged stablecoin on the Polygon blockchain). Kalshi uses US dollars directly via bank/card.' },
      ]),
      ctaText: 'Compare Polymarket vs Kalshi odds live on MoonX — the prediction market aggregator.',
      ctaUrl: 'https://www.bydfi.com/en/moonx/markets/trending',
      status: 'published',
      publishedAt: new Date('2026-03-01'),
    },
  })

  await prisma.article.upsert({
    where: { slug: 'prediction-market-aggregator' },
    update: {},
    create: {
      category: 'prediction',
      slug: 'prediction-market-aggregator',
      h1: 'What is a Prediction Market Aggregator? (And Why You Need One)',
      seoTitle: 'Prediction Market Aggregator: What It Is & Why It Matters | MoonX',
      metaDesc: 'A prediction market aggregator combines odds from Polymarket, Kalshi, and Manifold in one place. Learn why aggregation gives you better prices and smarter insights.',
      content: `## What is a Prediction Market Aggregator?

A prediction market aggregator is a platform that pulls together odds, liquidity, and data from multiple prediction markets — like Polymarket, Kalshi, and Manifold — and displays them in a unified interface.

Think of it like Google Flights for prediction markets: instead of checking each airline separately, you see all options at once.

### Why Aggregation Matters

**1. Better prices through comparison**

Different platforms price the same event differently. On a given political market, Polymarket might show 55¢ for YES while Kalshi shows 52¢. If you're selling YES shares, Polymarket is clearly better. Without an aggregator, you'd never know.

**2. Liquidity discovery**

Some markets exist only on one platform. An aggregator surfaces these opportunities automatically.

**3. Smart money tracking**

Aggregators can track whale wallets across platforms simultaneously, giving you early signals when sophisticated traders shift positions.

### MoonX: The First Prediction Market Aggregator

MoonX was built specifically to aggregate Polymarket, Kalshi, and Manifold. In addition to live odds comparison, MoonX offers:

- **Smart money feeds** — see where large wallets are moving
- **Meme token prediction markets** — crypto-native markets not available elsewhere
- **SEO-optimized market pages** — research any event before you trade

MoonX is available at [bydfi.com/en/moonx](https://www.bydfi.com/en/moonx/markets/trending).
`,
      author: 'MoonX Research',
      tags: JSON.stringify(['prediction-market', 'aggregator', 'polymarket', 'kalshi']),
      faqs: JSON.stringify([
        { q: 'What is the best prediction market aggregator?', a: 'MoonX is currently the only dedicated prediction market aggregator, combining Polymarket, Kalshi, and Manifold odds in a single interface with smart money tracking.' },
        { q: 'How does a prediction market aggregator make money?', a: 'Aggregators typically earn through referral fees when users trade on connected platforms, or through premium data subscriptions.' },
      ]),
      ctaText: 'Try MoonX — the prediction market aggregator.',
      ctaUrl: 'https://www.bydfi.com/en/moonx/markets/trending',
      status: 'published',
      publishedAt: new Date('2026-02-15'),
    },
  })

  await prisma.article.upsert({
    where: { slug: 'how-to-trade-meme-tokens-solana' },
    update: {},
    create: {
      category: 'meme',
      slug: 'how-to-trade-meme-tokens-solana',
      h1: 'How to Trade Meme Tokens on Solana: A Complete 2026 Guide',
      seoTitle: 'How to Trade Meme Tokens on Solana (2026) | MoonX',
      metaDesc: 'Step-by-step guide to trading Solana meme tokens safely. Learn how to read smart money signals, check contract security, analyze holder distribution, and use prediction market odds.',
      content: `## How to Trade Meme Tokens on Solana

Solana meme tokens are one of the highest risk, highest reward asset classes in crypto. This guide walks you through a systematic approach.

### Step 1: Check Contract Security

Before buying any meme token, run it through a security checker:

- **RugCheck** — scores contracts 0-100 for risk factors
- Look for: mint authority enabled, freeze authority, top holder concentration
- A score below 50 should require extra due diligence

MoonX shows RugCheck scores directly on every token page.

### Step 2: Analyze Holder Distribution

Concentrated supply is a major risk factor. Use the holder distribution chart to identify:

- If top 3 wallets hold >30% — high dump risk
- Dev wallet holding — check for large unlocked supply
- Gradual distribution is healthier than sudden whale accumulation

### Step 3: Read Smart Money Signals

"Smart money" wallets have historically shown strong returns. Track them with:

- MoonX Smart Money Feed — see recent buys/sells from tracked wallets
- GMGN.ai — cross-reference wallet PnL history
- Look for wallets with >200% average PnL buying your target token

### Step 4: Check Token Lifecycle

Token age matters enormously:
- **0-2 hours**: Highest risk, highest potential
- **2-24 hours**: Early adopter window
- **1-7 days**: Trending/established — lower risk but less upside
- **7+ days without fading**: Rare survivor — can be a longer hold

### Step 5: Set Position Sizing

Meme tokens require strict position sizing:
- Never put more than 1-2% of portfolio in a single meme token
- Use the profit calculator to plan exit targets before entering
- Have at minimum a 3x target to justify the binary risk

### Step 6: Monitor Prediction Markets

Some meme tokens have linked prediction markets on Polymarket. MoonX shows these odds directly on token pages — they can give you crowd-sourced signals about whether a token will survive.

## Tools on MoonX

Every token page on MoonX includes all 7 tools you need:
1. Polymarket odds widget
2. RugCheck security score
3. Token age lifecycle indicator
4. Holder distribution chart
5. P&L calculator
6. Community sentiment
7. Smart money feed

[Explore meme tokens on MoonX →](https://www.bydfi.com/en/moonx/markets/trending)
`,
      author: 'Arthur (MoonX Social)',
      tags: JSON.stringify(['meme-token', 'solana', 'trading-guide', 'smart-money']),
      faqs: JSON.stringify([
        { q: 'What is the safest way to trade meme tokens?', a: 'Always check contract security scores (RugCheck), analyze holder distribution for concentration risk, and never invest more than you can afford to lose completely. Use strict position sizing.' },
        { q: 'How do I find smart money wallets for meme tokens?', a: 'MoonX tracks wallets with historically high returns and shows their recent activity on each token page. You can also use GMGN.ai to cross-reference wallet performance.' },
        { q: 'What makes a meme token go viral?', a: 'Strong narrative/meme, low initial market cap, organic community growth, and smart money early involvement are the most common factors. Timing relative to broader crypto sentiment also matters.' },
      ]),
      ctaText: 'Analyze any Solana meme token with MoonX — 7 tools in one page.',
      ctaUrl: 'https://www.bydfi.com/en/moonx/markets/trending',
      status: 'published',
      publishedAt: new Date('2026-03-05'),
    },
  })

  // ── Programmatic Templates ──────────────────────────────────────────────

  await prisma.programmaticTemplate.upsert({
    where: { id: 'tpl-event' },
    update: {},
    create: {
      id: 'tpl-event',
      pageType: 'event',
      titleTpl: '{EVENT} Prediction Market Odds | MoonX',
      descTpl: 'Live prediction market odds for {EVENT}. {TOP_OUTCOME} at {TOP_PROB}% probability. Volume {VOLUME}. Compare Polymarket, Kalshi odds on MoonX.',
      contentTpl: '## {EVENT}\n\n{DESCRIPTION}\n\n### Current Odds\n\n{OUTCOMES_TABLE}\n\n### How to Trade\n\n{GUIDE_CONTENT}',
      faqTpl: JSON.stringify([
        { q: 'What are the current odds for {EVENT}?', a: '{TOP_OUTCOME} is currently at {TOP_PROB}% probability based on aggregated prediction market data.' },
        { q: 'When does this market resolve?', a: 'This market resolves on {END_DATE}.' },
      ]),
      noindexDays: 7,
      volumeMin: 10000,
    },
  })

  await prisma.programmaticTemplate.upsert({
    where: { id: 'tpl-stock' },
    update: {},
    create: {
      id: 'tpl-stock',
      pageType: 'stock',
      titleTpl: 'Will {TICKER} Beat Earnings? Prediction Market Odds | MoonX',
      descTpl: 'Prediction market gives {TICKER} a {BEAT_PCT}% chance of beating next earnings estimate. Live odds from Polymarket and Kalshi aggregated on MoonX.',
      contentTpl: '## {COMPANY} ({TICKER}) Earnings Prediction\n\n{DESCRIPTION}\n\n### Prediction Market Consensus\n\nThe market currently implies **{BEAT_PCT}% probability** {TICKER} beats estimates.\n\n{ANALYSIS}',
      faqTpl: JSON.stringify([
        { q: 'Will {TICKER} beat earnings?', a: 'The prediction market currently gives {TICKER} a {BEAT_PCT}% chance of beating next quarter estimates.' },
        { q: 'When are {TICKER} next earnings?', a: '{COMPANY} is expected to report on {EARNINGS_DATE}.' },
      ]),
      noindexDays: 30,
      volumeMin: 5000,
    },
  })

  await prisma.programmaticTemplate.upsert({
    where: { id: 'tpl-meme-token' },
    update: {},
    create: {
      id: 'tpl-meme-token',
      pageType: 'meme_token',
      titleTpl: '{SYMBOL} Price Today: ${PRICE} ({CHANGE_24H}) | MoonX',
      descTpl: '{NAME} ({SYMBOL}) live price ${PRICE}, {CHANGE_24H} 24h. Volume ${VOLUME}, {HOLDERS} holders. View smart money, security score, and prediction odds on MoonX.',
      contentTpl: '## About {NAME} ({SYMBOL})\n\n{NAME} is a Solana meme token launched {AGE_DAYS} days ago. Current price ${PRICE} with {HOLDERS} holders.\n\n### How to Trade {SYMBOL}\n\n{TRADING_GUIDE}',
      faqTpl: JSON.stringify([
        { q: 'What is {SYMBOL}?', a: '{NAME} ({SYMBOL}) is a Solana-based meme token. It launched {AGE_DAYS} days ago and currently has {HOLDERS} holders.' },
        { q: 'Is {SYMBOL} safe to buy?', a: 'Always check the RugCheck security score and holder distribution before buying any meme token. Past performance does not indicate future results.' },
      ]),
      noindexDays: 2,  // After 48h with no volume, noindex
      volumeMin: 500,
    },
  })

  // ── Banners ─────────────────────────────────────────────────────────────

  await prisma.banner.upsert({
    where: { id: 'banner-main-cta' },
    update: {},
    create: {
      id: 'banner-main-cta',
      pages: JSON.stringify(['prediction_article', 'event_guide', 'trending']),
      imageUrl: 'https://www.bydfi.com/en/moonx/og-banner.png',
      linkUrl: 'https://www.bydfi.com/en/moonx/markets/trending',
      altText: 'MoonX — Trade prediction markets with smart money insights',
      position: 'sidebar_top',
      active: true,
    },
  })

  await prisma.banner.upsert({
    where: { id: 'banner-meme-token' },
    update: {},
    create: {
      id: 'banner-meme-token',
      pages: JSON.stringify(['meme_token']),
      imageUrl: 'https://www.bydfi.com/en/moonx/og-meme.png',
      linkUrl: 'https://www.bydfi.com/en/moonx/markets/trending',
      altText: 'Track meme tokens with smart money signals on MoonX',
      position: 'sidebar_top',
      active: true,
    },
  })

  // ── Sample Token Snapshots ──────────────────────────────────────────────

  const tokens = [
    {
      contract: 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
      symbol: 'BONK', name: 'Bonk',
      price: 0.0000156, priceChange: 12.4, volume24h: 45000000, mcap: 1200000000,
      holders: 850000, lpUsd: 8500000,
      launchedAt: new Date('2022-12-25'),
    },
    {
      contract: '7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr',
      symbol: 'POPCAT', name: 'Popcat',
      price: 0.00231, priceChange: -5.2, volume24h: 8200000, mcap: 230000000,
      holders: 42000, lpUsd: 1200000,
      launchedAt: new Date('2024-01-10'),
    },
    {
      contract: 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm',
      symbol: 'WIF', name: 'dogwifhat',
      price: 0.72, priceChange: 8.9, volume24h: 125000000, mcap: 720000000,
      holders: 180000, lpUsd: 32000000,
      launchedAt: new Date('2023-11-20'),
    },
  ]

  for (const t of tokens) {
    await prisma.tokenSnapshot.upsert({
      where: { contract: t.contract },
      update: { price: t.price, priceChange: t.priceChange, volume24h: t.volume24h },
      create: {
        contract: t.contract, symbol: t.symbol, name: t.name,
        price: t.price, priceChange: t.priceChange,
        volume24h: t.volume24h, mcap: t.mcap,
        holders: t.holders, lpUsd: t.lpUsd,
        launchedAt: t.launchedAt,
        noindex: false,
      },
    })
  }

  console.log('✅ Seed complete.')
  console.log(`   Articles: 3 | Templates: 3 | Banners: 2 | Tokens: ${tokens.length}`)
}

main()
  .catch(e => { console.error(e); process.exit(1) })
  .finally(() => prisma.$disconnect())
