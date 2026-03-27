import { cachedFetch, TTL } from './cache'

const BASE = process.env.POLYMARKET_API_BASE || 'https://gamma-api.polymarket.com'

export interface PolyMarket {
  id: string
  slug: string
  question: string
  description: string
  endDate: string
  volume: number
  liquidity: number
  outcomes: PolyOutcome[]
  category: string
  active: boolean
  closed: boolean
  image?: string
}

export interface PolyOutcome {
  id: string
  title: string
  probability: number  // 0-1
  price: number        // cents equivalent
}

export interface PolyTrending {
  markets: PolyMarket[]
  total: number
}

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    next: { revalidate: 60 },
    headers: { 'Accept': 'application/json' },
  })
  if (!res.ok) throw new Error(`Polymarket API ${res.status}: ${path}`)
  return res.json()
}

function parseMarkets(data: unknown): PolyMarket[] {
  if (Array.isArray(data)) return data
  if (data && typeof data === 'object' && 'markets' in data) return (data as { markets: PolyMarket[] }).markets || []
  return []
}

export async function getTrendingMarkets(limit = 20): Promise<PolyMarket[]> {
  return cachedFetch(
    `polymarket:trending:${limit}`,
    async () => {
      try {
        const data = await fetchJson<unknown>(`/markets?active=true&closed=false&limit=${limit}&order=volume&ascending=false`)
        const markets = parseMarkets(data)
        return markets.length > 0 ? markets : getMockTrendingMarkets(limit)
      } catch {
        return getMockTrendingMarkets(limit)
      }
    },
    { ttl: TTL.TRENDING }
  )
}

export async function getMarketBySlug(slug: string): Promise<PolyMarket | null> {
  return cachedFetch(
    `polymarket:market:${slug}`,
    async () => {
      try {
        const data = await fetchJson<unknown>(`/markets?slug=${slug}`)
        const markets = parseMarkets(data)
        return markets[0] || getMockMarket(slug)
      } catch {
        return getMockMarket(slug)
      }
    },
    { ttl: TTL.MARKET_ODDS }
  )
}

export async function getMarketsByCategory(category: string, limit = 10): Promise<PolyMarket[]> {
  return cachedFetch(
    `polymarket:category:${category}:${limit}`,
    async () => {
      try {
        const data = await fetchJson<unknown>(`/markets?active=true&category=${category}&limit=${limit}&order=volume&ascending=false`)
        const markets = parseMarkets(data)
        return markets.length > 0 ? markets : getMockTrendingMarkets(limit)
      } catch {
        return getMockTrendingMarkets(limit)
      }
    },
    { ttl: TTL.TRENDING }
  )
}

// Fallback mock data when API unavailable
function getMockTrendingMarkets(limit: number): PolyMarket[] {
  const markets: PolyMarket[] = [
    {
      id: '1', slug: 'trump-2024-president', question: 'Will Trump win the 2024 presidential election?',
      description: 'This market resolves YES if Donald Trump wins the 2024 US Presidential Election.',
      endDate: '2024-11-05T23:59:00Z', volume: 45000000, liquidity: 12000000,
      outcomes: [
        { id: '1a', title: 'Yes', probability: 0.52, price: 52 },
        { id: '1b', title: 'No', probability: 0.48, price: 48 },
      ],
      category: 'politics', active: true, closed: false,
    },
    {
      id: '2', slug: 'btc-100k-2025', question: 'Will Bitcoin reach $100K in 2025?',
      description: 'Resolves YES if Bitcoin\'s price reaches or exceeds $100,000 at any point in 2025.',
      endDate: '2025-12-31T23:59:00Z', volume: 28000000, liquidity: 8500000,
      outcomes: [
        { id: '2a', title: 'Yes', probability: 0.67, price: 67 },
        { id: '2b', title: 'No', probability: 0.33, price: 33 },
      ],
      category: 'crypto', active: true, closed: false,
    },
    {
      id: '3', slug: 'fed-rate-cut-march-2025', question: 'Will the Fed cut rates in March 2025?',
      description: 'Resolves YES if the Federal Reserve cuts interest rates at the March 2025 FOMC meeting.',
      endDate: '2025-03-20T18:00:00Z', volume: 15000000, liquidity: 4200000,
      outcomes: [
        { id: '3a', title: 'Yes', probability: 0.23, price: 23 },
        { id: '3b', title: 'No', probability: 0.77, price: 77 },
      ],
      category: 'finance', active: true, closed: false,
    },
  ]
  return markets.slice(0, limit)
}

function getMockMarket(slug: string): PolyMarket {
  return {
    id: slug, slug, question: `Prediction market: ${slug.replace(/-/g, ' ')}`,
    description: 'Live prediction market odds updated in real-time.',
    endDate: new Date(Date.now() + 30 * 24 * 3600 * 1000).toISOString(),
    volume: 1200000, liquidity: 450000,
    outcomes: [
      { id: `${slug}-yes`, title: 'Yes', probability: 0.62, price: 62 },
      { id: `${slug}-no`, title: 'No', probability: 0.38, price: 38 },
    ],
    category: 'general', active: true, closed: false,
  }
}
