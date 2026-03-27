import { NextRequest, NextResponse } from 'next/server'
import { getTokenData, getHolderDistribution, getSmartMoneyFeed, getRiskScore } from '@/lib/token-data'
import { getMarketBySlug } from '@/lib/polymarket'

export const dynamic = 'force-dynamic'

export async function GET(req: NextRequest, { params }: { params: { contract: string } }) {
  const { contract } = params
  const fields = req.nextUrl.searchParams.get('fields')

  try {
    if (fields === 'odds') {
      // Try to find a linked prediction market
      const market = await getMarketBySlug(contract.toLowerCase().slice(0, 10))
      return NextResponse.json({
        odds: market?.outcomes || [],
      })
    }

    if (fields === 'risk') {
      const risk = await getRiskScore(contract)
      return NextResponse.json({ risk })
    }

    if (fields === 'smartmoney') {
      const smartmoney = await getSmartMoneyFeed(contract, 5)
      return NextResponse.json({ smartmoney })
    }

    if (fields === 'holders') {
      const holders = await getHolderDistribution(contract)
      return NextResponse.json({ holders })
    }

    if (fields === 'sentiment') {
      // Stub — in production, aggregate from Twitter/Telegram keyword scoring
      const hash = contract.charCodeAt(0) + contract.charCodeAt(1)
      const score = (hash * 37) % 100
      const sentimentMap: Record<number, string> = { 0: 'bearish', 1: 'neutral', 2: 'bullish', 3: 'hype' }
      const bucket = Math.floor(score / 25)
      return NextResponse.json({
        sentiment: {
          sentiment: sentimentMap[bucket] || 'neutral',
          score,
          keywords: ['moon', 'pump', 'lfg'].slice(0, 3),
        },
      })
    }

    // Full token data
    const [token, holders, smartmoney, risk] = await Promise.all([
      getTokenData(contract),
      getHolderDistribution(contract),
      getSmartMoneyFeed(contract, 5),
      getRiskScore(contract),
    ])

    if (!token) return NextResponse.json({ error: 'Token not found' }, { status: 404 })

    return NextResponse.json({ token, holders, smartmoney, risk }, {
      headers: {
        'Cache-Control': token.isHot ? 's-maxage=60' : 's-maxage=300',
      },
    })
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown error'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
