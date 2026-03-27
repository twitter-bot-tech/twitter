import { Metadata } from 'next'
import Link from 'next/link'
import { getTrendingMarkets } from '@/lib/polymarket'
import { trendingMeta, breadcrumbSchema } from '@/lib/seo'
import { SchemaScript } from '@/components/seo/SchemaScript'
import { Breadcrumb } from '@/components/seo/Breadcrumb'
import { TrendingUp, DollarSign, Clock, ArrowRight } from 'lucide-react'

export const dynamic = 'force-dynamic'
export const metadata: Metadata = trendingMeta()

function formatVolume(n: number | string | undefined) {
  const num = Number(n) || 0
  if (num >= 1_000_000) return `$${(num / 1_000_000).toFixed(1)}M`
  if (num >= 1_000) return `$${(num / 1_000).toFixed(0)}K`
  return `$${num.toFixed(0)}`
}

export default async function TrendingPage() {
  const markets = await getTrendingMarkets(20)
  const breadcrumbs = [
    { name: 'MoonX', href: '/en/moonx/markets/trending' },
    { name: 'Trending Markets' },
  ]

  return (
    <>
      <SchemaScript schema={breadcrumbSchema(breadcrumbs.map(b => ({ name: b.name, url: `https://www.bydfi.com${b.href || ''}` })))} />

      <main className="max-w-5xl mx-auto px-4 py-8">
        <Breadcrumb items={breadcrumbs} />

        {/* Header */}
        <div className="flex items-start justify-between mb-6">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <TrendingUp className="w-6 h-6 text-blue-500" />
              <h1 className="text-2xl font-bold">Trending Prediction Markets</h1>
            </div>
            <p className="text-sm text-muted-foreground">
              Live odds aggregated from Polymarket, Kalshi & Manifold. Updated every 2 minutes.
            </p>
          </div>
          <span className="text-xs text-green-600 bg-green-50 border border-green-200 px-2 py-1 rounded-full flex items-center gap-1">
            <span className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
            Live
          </span>
        </div>

        {/* Market links to stock predictions */}
        <div className="flex gap-2 flex-wrap mb-6">
          {['AAPL', 'NVDA', 'TSLA', 'META', 'MSFT'].map(ticker => (
            <Link key={ticker} href={`/en/moonx/markets/stocks/${ticker.toLowerCase()}`}
              className="text-xs border px-2.5 py-1 rounded-full hover:bg-accent transition-colors">
              {ticker} earnings
            </Link>
          ))}
        </div>

        {/* Markets grid */}
        <div className="space-y-3">
          {markets.map((market, idx) => {
            const sortedOutcomes = [...market.outcomes].sort((a, b) => b.probability - a.probability)
            const topOutcome = sortedOutcomes[0]
            const daysLeft = Math.ceil((new Date(market.endDate).getTime() - Date.now()) / 86400000)

            return (
              <Link key={market.id} href={`/en/moonx/guide/prediction/${market.slug}`}>
                <div className="rounded-xl border bg-card p-4 hover:shadow-md transition-all hover:border-blue-200 group">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-[10px] font-medium text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded">
                          #{idx + 1}
                        </span>
                        <span className="text-[10px] text-muted-foreground uppercase">{market.category}</span>
                      </div>
                      <h2 className="text-sm font-semibold leading-snug line-clamp-2 mb-2">{market.question}</h2>

                      {/* Odds bar */}
                      {sortedOutcomes.length >= 2 && (
                        <div className="flex gap-1 h-2 rounded-full overflow-hidden mb-2">
                          {sortedOutcomes.map((o, i) => (
                            <div
                              key={o.id}
                              className={`h-full transition-all ${i === 0 ? 'bg-blue-500' : 'bg-gray-200'}`}
                              style={{ width: `${o.probability * 100}%` }}
                              title={`${o.title}: ${Math.round(o.probability * 100)}%`}
                            />
                          ))}
                        </div>
                      )}

                      <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
                        <span className="flex items-center gap-0.5"><DollarSign className="w-3 h-3" />{formatVolume(market.volume)}</span>
                        <span className="flex items-center gap-0.5"><Clock className="w-3 h-3" />{daysLeft > 0 ? `${daysLeft}d left` : 'Closed'}</span>
                        {topOutcome && (
                          <span className="font-medium text-foreground">{topOutcome.title} {Math.round(topOutcome.probability * 100)}¢</span>
                        )}
                      </div>
                    </div>

                    <ArrowRight className="w-4 h-4 text-muted-foreground group-hover:text-blue-500 transition-colors flex-shrink-0 mt-1" />
                  </div>
                </div>
              </Link>
            )
          })}
        </div>

        {markets.length === 0 && (
          <div className="text-center py-16 text-muted-foreground">
            <TrendingUp className="w-12 h-12 mx-auto mb-3 opacity-20" />
            <p>Loading markets…</p>
          </div>
        )}

        {/* Footer nav */}
        <div className="mt-8 pt-6 border-t flex flex-wrap gap-4 text-sm text-muted-foreground">
          <Link href="/en/moonx/news" className="hover:text-foreground">Crypto News →</Link>
          <Link href="/en/moonx/learn/prediction" className="hover:text-foreground">Prediction Academy →</Link>
          <Link href="/en/moonx/learn/meme" className="hover:text-foreground">Meme Academy →</Link>
        </div>
      </main>
    </>
  )
}
