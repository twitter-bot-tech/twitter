import { NextRequest, NextResponse } from 'next/server'
import { getTrendingMarkets, getMarketBySlug } from '@/lib/polymarket'

export const dynamic = 'force-dynamic'

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url)
  const slug = searchParams.get('slug')
  const limit = parseInt(searchParams.get('limit') || '20')

  try {
    if (slug) {
      const market = await getMarketBySlug(slug)
      return NextResponse.json({ market }, {
        headers: { 'Cache-Control': 's-maxage=60, stale-while-revalidate=30' },
      })
    }

    const markets = await getTrendingMarkets(limit)
    return NextResponse.json({ markets }, {
      headers: { 'Cache-Control': 's-maxage=120, stale-while-revalidate=60' },
    })
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown error'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
