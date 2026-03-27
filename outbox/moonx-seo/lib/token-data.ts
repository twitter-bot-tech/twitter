import { cachedFetch, TTL } from './cache'

export interface TokenData {
  contract: string
  symbol: string
  name: string
  price: number
  priceChange24h: number
  volume24h: number
  mcap: number
  holders: number
  lpUsd: number
  launchedAt: Date
  image?: string
  website?: string
  twitter?: string
  telegram?: string
  // Computed
  isHot: boolean   // volume > $1M
  isDead: boolean  // volume < $500 AND age > 48h
  ageHours: number
}

export interface HolderDistribution {
  label: string
  pct: number
  address?: string
}

export interface SmartMoneyTx {
  wallet: string
  action: 'buy' | 'sell'
  amountUsd: number
  timestamp: Date
  pnlPct?: number
}

// Helius API for Solana token data
const HELIUS_BASE = 'https://api.helius.xyz/v0'
const HELIUS_KEY = process.env.HELIUS_API_KEY || ''

export async function getTokenData(contract: string): Promise<TokenData | null> {
  return cachedFetch(
    `token:${contract}`,
    async () => {
      try {
        if (HELIUS_KEY) {
          const res = await fetch(`${HELIUS_BASE}/token-metadata?api-key=${HELIUS_KEY}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mintAccounts: [contract] }),
          })
          if (res.ok) {
            const [meta] = await res.json()
            if (meta) return mapHeliusToToken(contract, meta)
          }
        }
        // Fallback to DexScreener public API
        const dex = await fetch(`https://api.dexscreener.com/latest/dex/tokens/${contract}`, {
          next: { revalidate: 300 },
        })
        if (dex.ok) {
          const data = await dex.json()
          const pair = data.pairs?.[0]
          if (pair) return mapDexScreenerToToken(contract, pair)
        }
        return getMockToken(contract)
      } catch {
        return getMockToken(contract)
      }
    },
    { ttl: TTL.TOKEN }
  )
}

export async function getHolderDistribution(contract: string): Promise<HolderDistribution[]> {
  return cachedFetch(
    `holders:${contract}`,
    async () => {
      try {
        // Solscan public API
        const res = await fetch(`https://public-api.solscan.io/token/holders?tokenAddress=${contract}&limit=10`, {
          next: { revalidate: 300 },
        })
        if (res.ok) {
          const data = await res.json()
          return mapSolscanHolders(data.data || [])
        }
        return getMockHolders()
      } catch {
        return getMockHolders()
      }
    },
    { ttl: TTL.TOKEN }
  )
}

export async function getSmartMoneyFeed(contract: string, limit = 5): Promise<SmartMoneyTx[]> {
  return cachedFetch(
    `smartmoney:${contract}:${limit}`,
    async () => getMockSmartMoney(contract, limit),
    { ttl: TTL.HOT_TOKEN }
  )
}

export async function getRiskScore(contract: string): Promise<{ score: number; risks: string[]; verdict: 'SAFE' | 'MODERATE' | 'RISKY' | 'RUGGED' }> {
  return cachedFetch(
    `risk:${contract}`,
    async () => {
      try {
        const res = await fetch(`https://api.rugcheck.xyz/v1/tokens/${contract}/report/summary`, {
          next: { revalidate: 300 },
        })
        if (res.ok) {
          const data = await res.json()
          return {
            score: data.score || 0,
            risks: (data.risks || []).map((r: { name: string }) => r.name),
            verdict: (data.rugged ? 'RUGGED' : data.score > 70 ? 'SAFE' : data.score > 40 ? 'MODERATE' : 'RISKY') as 'SAFE' | 'MODERATE' | 'RISKY' | 'RUGGED',
          }
        }
      } catch {
        // ignore
      }
      return { score: 72, risks: ['Mint authority enabled'], verdict: 'MODERATE' }
    },
    { ttl: TTL.TOKEN }
  )
}

// Mappers
function mapDexScreenerToToken(contract: string, pair: Record<string, unknown>): TokenData {
  const baseToken = pair.baseToken as Record<string, string>
  const priceUsd = parseFloat((pair.priceUsd as string) || '0')
  const priceChange = (pair.priceChange as Record<string, number>)?.h24 || 0
  const volume = (pair.volume as Record<string, number>)?.h24 || 0
  const mcap = (pair.fdv as number) || 0
  const liquidity = (pair.liquidity as Record<string, number>)?.usd || 0
  const launchedAt = pair.pairCreatedAt ? new Date(pair.pairCreatedAt as number) : new Date(Date.now() - 7 * 86400000)
  const ageHours = (Date.now() - launchedAt.getTime()) / 3600000

  return {
    contract,
    symbol: baseToken?.symbol || 'UNKNOWN',
    name: baseToken?.name || 'Unknown Token',
    price: priceUsd,
    priceChange24h: priceChange,
    volume24h: volume,
    mcap,
    holders: 0,
    lpUsd: liquidity,
    launchedAt,
    isHot: volume > 1_000_000,
    isDead: volume < 500 && ageHours > 48,
    ageHours,
  }
}

function mapHeliusToToken(contract: string, meta: Record<string, unknown>): TokenData {
  return {
    contract,
    symbol: (meta.symbol as string) || 'UNKNOWN',
    name: (meta.name as string) || 'Unknown Token',
    price: 0,
    priceChange24h: 0,
    volume24h: 0,
    mcap: 0,
    holders: 0,
    lpUsd: 0,
    launchedAt: new Date(),
    isHot: false,
    isDead: false,
    ageHours: 0,
  }
}

function mapSolscanHolders(data: Array<{ owner: string; amount: number; decimals: number }>): HolderDistribution[] {
  const total = data.reduce((sum, h) => sum + h.amount / Math.pow(10, h.decimals || 6), 0)
  return data.slice(0, 10).map((h, i) => ({
    label: i < 3 ? `Top ${i + 1}` : `Holder ${i + 1}`,
    pct: total > 0 ? Math.round((h.amount / Math.pow(10, h.decimals || 6) / total) * 10000) / 100 : 0,
    address: h.owner,
  }))
}

// Mock data fallbacks
function getMockToken(contract: string): TokenData {
  const symbols = ['PEPE', 'BONK', 'WIF', 'BOME', 'POPCAT']
  const sym = symbols[Math.abs(contract.charCodeAt(0) % symbols.length)]
  const volume = Math.random() > 0.3 ? 1200000 + Math.random() * 5000000 : 300 + Math.random() * 200
  const ageHours = 24 + Math.random() * 200
  return {
    contract,
    symbol: sym,
    name: `${sym} Token`,
    price: 0.00000123 + Math.random() * 0.00001,
    priceChange24h: (Math.random() - 0.4) * 80,
    volume24h: volume,
    mcap: volume * 15,
    holders: Math.floor(500 + Math.random() * 5000),
    lpUsd: volume * 0.3,
    launchedAt: new Date(Date.now() - ageHours * 3600000),
    isHot: volume > 1_000_000,
    isDead: volume < 500 && ageHours > 48,
    ageHours,
  }
}

function getMockHolders(): HolderDistribution[] {
  return [
    { label: 'Top 1 (Dev)', pct: 12.5 },
    { label: 'Top 2', pct: 8.3 },
    { label: 'Top 3', pct: 6.1 },
    { label: 'Top 4-10', pct: 22.4 },
    { label: 'Others', pct: 50.7 },
  ]
}

function getMockSmartMoney(_contract: string, limit: number): SmartMoneyTx[] {
  const txs: SmartMoneyTx[] = [
    { wallet: '7xKX...9vPQ', action: 'buy', amountUsd: 45000, timestamp: new Date(Date.now() - 1800000), pnlPct: 234 },
    { wallet: 'BmN3...4wRt', action: 'sell', amountUsd: 12000, timestamp: new Date(Date.now() - 3600000), pnlPct: -15 },
    { wallet: 'Ck8P...zXqL', action: 'buy', amountUsd: 8500, timestamp: new Date(Date.now() - 7200000) },
    { wallet: 'FpQ2...nYv6', action: 'buy', amountUsd: 31000, timestamp: new Date(Date.now() - 10800000), pnlPct: 89 },
    { wallet: 'Ht5M...eWbA', action: 'sell', amountUsd: 22000, timestamp: new Date(Date.now() - 14400000), pnlPct: 412 },
  ]
  return txs.slice(0, limit)
}
