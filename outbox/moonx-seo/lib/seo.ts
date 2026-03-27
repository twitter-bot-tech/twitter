import type { Metadata } from 'next'
import type { TokenData } from './token-data'
import type { PolyMarket } from './polymarket'

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL || 'https://www.bydfi.com'
const SITE_NAME = 'MoonX by BYDFi'

// ── Article pages (prediction / meme learn) ────────────────────────────────

export function articleMeta(article: {
  seoTitle: string
  metaDesc: string
  slug: string
  category: string
  coverImage?: string | null
}): Metadata {
  const path = `/en/moonx/learn/${article.category}/${article.slug}`
  return {
    title: article.seoTitle,
    description: article.metaDesc,
    alternates: { canonical: `${SITE_URL}${path}` },
    openGraph: {
      title: article.seoTitle,
      description: article.metaDesc,
      url: `${SITE_URL}${path}`,
      siteName: SITE_NAME,
      type: 'article',
      images: article.coverImage ? [{ url: article.coverImage }] : [],
    },
  }
}

// ── Meme Token pages ───────────────────────────────────────────────────────

export function memeTokenMeta(token: TokenData): Metadata {
  const path = `/en/moonx/solana/token/${token.contract}`
  const priceFormatted = token.price < 0.001
    ? token.price.toExponential(4)
    : token.price.toFixed(6)
  const changeSign = token.priceChange24h >= 0 ? '+' : ''

  const title = `${token.symbol} Price Today: $${priceFormatted} (${changeSign}${token.priceChange24h.toFixed(1)}%) | MoonX`
  const description = `${token.name} (${token.symbol}) live price $${priceFormatted}, ${changeSign}${token.priceChange24h.toFixed(1)}% 24h. Volume $${formatVolume(token.volume24h)}, ${token.holders.toLocaleString()} holders. View smart money activity, security score, and Polymarket odds on MoonX.`

  return {
    title,
    description,
    alternates: { canonical: `${SITE_URL}${path}` },
    robots: token.isDead
      ? { index: false, follow: false, noarchive: true }
      : { index: true, follow: true },
    openGraph: {
      title,
      description,
      url: `${SITE_URL}${path}`,
      siteName: SITE_NAME,
    },
  }
}

// ── Event guide pages (programmatic) ──────────────────────────────────────

export function eventGuideMeta(market: PolyMarket): Metadata {
  const path = `/en/moonx/guide/prediction/${market.slug}`
  const topOutcome = [...market.outcomes].sort((a, b) => b.probability - a.probability)[0]
  const title = `${market.question} | Prediction Market Odds | MoonX`
  const description = `Live odds: "${topOutcome?.title}" at ${Math.round((topOutcome?.probability || 0) * 100)}% probability. Volume $${formatVolume(market.volume)}, liquidity $${formatVolume(market.liquidity)}. Compare Polymarket, Kalshi odds on MoonX.`

  return {
    title,
    description,
    alternates: { canonical: `${SITE_URL}${path}` },
    openGraph: { title, description, url: `${SITE_URL}${path}`, siteName: SITE_NAME },
  }
}

// ── Stock prediction pages (programmatic) ─────────────────────────────────

export function stockPredictionMeta(ticker: string, companyName: string, odds: number): Metadata {
  const path = `/en/moonx/markets/stocks/${ticker}`
  const title = `Will ${companyName} (${ticker}) Beat Earnings? Prediction Market Odds | MoonX`
  const description = `Prediction market gives ${Math.round(odds * 100)}% chance ${ticker} beats next earnings. Live odds aggregated from Polymarket & Kalshi on MoonX.`

  return {
    title,
    description,
    alternates: { canonical: `${SITE_URL}${path}` },
    openGraph: { title, description, url: `${SITE_URL}${path}`, siteName: SITE_NAME },
  }
}

// ── News page ──────────────────────────────────────────────────────────────

export function newsMeta(category?: string): Metadata {
  const cat = category && category !== 'all' ? ` — ${category.charAt(0).toUpperCase() + category.slice(1)}` : ''
  return {
    title: `Crypto News${cat} — Latest Updates | MoonX`,
    description: 'Stay ahead with the latest crypto news. Bitcoin, Ethereum, Solana, altcoins, and DeFi updates — aggregated and curated by MoonX.',
    alternates: { canonical: `${SITE_URL}/en/moonx/news${category && category !== 'all' ? `?cat=${category}` : ''}` },
    openGraph: {
      title: `Crypto News${cat} | MoonX`,
      description: 'Latest crypto news and market updates on MoonX.',
      url: `${SITE_URL}/en/moonx/news`,
      siteName: SITE_NAME,
    },
  }
}

// ── Trending page ──────────────────────────────────────────────────────────

export function trendingMeta(): Metadata {
  return {
    title: 'Trending Prediction Markets — Live Odds | MoonX',
    description: 'Top 20 trending prediction markets right now. Compare live odds from Polymarket, Kalshi, and Manifold in one place. Track smart money and volume on MoonX.',
    alternates: { canonical: `${SITE_URL}/en/moonx/markets/trending` },
    openGraph: {
      title: 'Trending Prediction Markets — Live Odds | MoonX',
      description: 'Top trending prediction markets with live odds aggregated from Polymarket, Kalshi, and Manifold.',
      url: `${SITE_URL}/en/moonx/markets/trending`,
      siteName: SITE_NAME,
    },
  }
}

// ── Schema.org JSON-LD factories ───────────────────────────────────────────

export function articleSchema(article: {
  h1: string
  metaDesc: string
  author: string
  publishedAt: Date | null
  updatedAt: Date
  slug: string
  category: string
  coverImage?: string | null
}) {
  return {
    '@context': 'https://schema.org',
    '@type': 'Article',
    headline: article.h1,
    description: article.metaDesc,
    author: { '@type': 'Person', name: article.author },
    publisher: {
      '@type': 'Organization',
      name: SITE_NAME,
      logo: { '@type': 'ImageObject', url: `${SITE_URL}/logo.png` },
    },
    datePublished: (article.publishedAt || article.updatedAt).toISOString(),
    dateModified: article.updatedAt.toISOString(),
    url: `${SITE_URL}/en/moonx/learn/${article.category}/${article.slug}`,
    image: article.coverImage || `${SITE_URL}/og-default.png`,
  }
}

export function faqSchema(faqs: Array<{ q: string; a: string }>) {
  return {
    '@context': 'https://schema.org',
    '@type': 'FAQPage',
    mainEntity: faqs.map(({ q, a }) => ({
      '@type': 'Question',
      name: q,
      acceptedAnswer: { '@type': 'Answer', text: a },
    })),
  }
}

export function tokenSchema(token: TokenData) {
  return {
    '@context': 'https://schema.org',
    '@type': 'FinancialProduct',
    name: `${token.name} (${token.symbol})`,
    description: `${token.symbol} is a Solana meme token. Current price $${token.price}, ${token.holders} holders.`,
    url: `${SITE_URL}/en/moonx/solana/token/${token.contract}`,
    offers: {
      '@type': 'Offer',
      price: token.price.toString(),
      priceCurrency: 'USD',
    },
  }
}

export function breadcrumbSchema(items: Array<{ name: string; url: string }>) {
  return {
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: items.map((item, idx) => ({
      '@type': 'ListItem',
      position: idx + 1,
      name: item.name,
      item: item.url,
    })),
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────

function formatVolume(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`
  return n.toFixed(0)
}
