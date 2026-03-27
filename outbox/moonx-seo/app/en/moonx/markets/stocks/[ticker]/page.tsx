import { Metadata } from 'next'
import { stockPredictionMeta, faqSchema, breadcrumbSchema } from '@/lib/seo'
import { SchemaScript } from '@/components/seo/SchemaScript'
import { Breadcrumb } from '@/components/seo/Breadcrumb'
import { TrendingUp, BarChart3, ExternalLink } from 'lucide-react'

interface Props { params: { ticker: string } }

export const revalidate = 300

// Static dataset for prototype — programmatic generation from this map
const STOCKS: Record<string, { name: string; sector: string; earningsDate: string; beatOdds: number; description: string }> = {
  AAPL: { name: 'Apple', sector: 'Technology', earningsDate: '2026-04-28', beatOdds: 0.71, description: 'Apple Inc. designs and sells consumer electronics, software, and services.' },
  NVDA: { name: 'NVIDIA', sector: 'Semiconductors', earningsDate: '2026-05-21', beatOdds: 0.84, description: 'NVIDIA Corporation designs graphics and compute processors.' },
  TSLA: { name: 'Tesla', sector: 'EV / Clean Energy', earningsDate: '2026-04-22', beatOdds: 0.48, description: 'Tesla Inc. manufactures electric vehicles and energy storage systems.' },
  META: { name: 'Meta Platforms', sector: 'Social Media', earningsDate: '2026-04-29', beatOdds: 0.77, description: 'Meta Platforms operates Facebook, Instagram, and WhatsApp.' },
  MSFT: { name: 'Microsoft', sector: 'Technology', earningsDate: '2026-04-27', beatOdds: 0.79, description: 'Microsoft Corporation develops software, services, and cloud products.' },
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const ticker = params.ticker.toUpperCase()
  const stock = STOCKS[ticker]
  if (!stock) return {}
  return stockPredictionMeta(ticker, stock.name, stock.beatOdds)
}

export async function generateStaticParams() {
  return Object.keys(STOCKS).map(ticker => ({ ticker: ticker.toLowerCase() }))
}

export default function StockPredictionPage({ params }: Props) {
  const ticker = params.ticker.toUpperCase()
  const stock = STOCKS[ticker]

  if (!stock) {
    return (
      <main className="max-w-3xl mx-auto px-4 py-16 text-center">
        <h1 className="text-2xl font-bold mb-2">Stock not found</h1>
        <p className="text-muted-foreground">We don&apos;t have prediction market data for {ticker} yet.</p>
      </main>
    )
  }

  const daysToEarnings = Math.ceil((new Date(stock.earningsDate).getTime() - Date.now()) / 86400000)
  const faqs = [
    { q: `Will ${ticker} beat earnings?`, a: `The prediction market currently gives ${ticker} a ${Math.round(stock.beatOdds * 100)}% chance of beating its next earnings estimate. This is based on aggregated Polymarket and Kalshi odds on MoonX.` },
    { q: 'When are the next earnings?', a: `${stock.name} (${ticker}) is expected to report earnings on ${new Date(stock.earningsDate).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}.` },
    { q: 'How accurate are prediction market earnings forecasts?', a: 'Prediction markets have historically outperformed analyst consensus in forecasting binary outcomes. They aggregate the wisdom of crowds with financial incentives for accuracy.' },
  ]

  const breadcrumbs = [
    { name: 'MoonX', href: '/en/moonx/markets/trending' },
    { name: 'Stock Predictions', href: '/en/moonx/markets/trending' },
    { name: `${ticker} Earnings Prediction` },
  ]

  return (
    <>
      <SchemaScript schema={[
        faqSchema(faqs),
        breadcrumbSchema(breadcrumbs.map(b => ({ name: b.name, url: `https://www.bydfi.com${b.href || ''}` }))),
      ]} />

      <main className="max-w-3xl mx-auto px-4 py-8">
        <Breadcrumb items={breadcrumbs} />

        <header className="mb-6">
          <span className="text-xs font-medium text-purple-600 bg-purple-50 px-2 py-0.5 rounded-full">{stock.sector}</span>
          <h1 className="text-2xl font-bold mt-2 mb-1">
            Will {stock.name} ({ticker}) Beat Earnings?
          </h1>
          <p className="text-sm text-muted-foreground">{stock.description}</p>
        </header>

        {/* Prediction card */}
        <div className="rounded-2xl border bg-card p-6 mb-6 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 className="w-5 h-5 text-blue-500" />
            <span className="font-semibold">Prediction Market Consensus</span>
          </div>

          <div className="mb-4">
            <div className="flex justify-between text-sm mb-1">
              <span className="font-medium text-green-600">Beats Estimates</span>
              <span className="font-bold text-green-600">{Math.round(stock.beatOdds * 100)}¢</span>
            </div>
            <div className="h-4 bg-muted rounded-full overflow-hidden">
              <div className="h-full bg-green-500 rounded-full transition-all duration-700" style={{ width: `${stock.beatOdds * 100}%` }} />
            </div>
            <div className="flex justify-between text-sm mt-2">
              <span className="text-muted-foreground">Misses Estimates</span>
              <span className="text-muted-foreground">{Math.round((1 - stock.beatOdds) * 100)}¢</span>
            </div>
            <div className="h-4 bg-muted rounded-full overflow-hidden mt-1">
              <div className="h-full bg-red-400 rounded-full" style={{ width: `${(1 - stock.beatOdds) * 100}%` }} />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 pt-4 border-t text-center">
            <div>
              <TrendingUp className="w-4 h-4 text-muted-foreground mx-auto mb-0.5" />
              <p className="text-sm font-semibold">{Math.round(stock.beatOdds * 100)}%</p>
              <p className="text-xs text-muted-foreground">Implied probability</p>
            </div>
            <div>
              <BarChart3 className="w-4 h-4 text-muted-foreground mx-auto mb-0.5" />
              <p className="text-sm font-semibold">{daysToEarnings > 0 ? `${daysToEarnings}d` : 'Passed'}</p>
              <p className="text-xs text-muted-foreground">Until earnings</p>
            </div>
          </div>
        </div>

        {/* CTA */}
        <div className="rounded-xl bg-gradient-to-r from-blue-600 to-purple-600 p-5 text-white mb-8">
          <h2 className="font-bold mb-1">Trade {ticker} earnings on MoonX</h2>
          <p className="text-sm text-blue-100 mb-3">Compare odds from multiple prediction markets and trade with edge.</p>
          <a href="https://www.bydfi.com/en/moonx/markets/trending" className="inline-flex items-center gap-2 bg-white text-blue-600 text-sm font-medium px-4 py-2 rounded-lg hover:bg-blue-50 transition-colors">
            Open MoonX <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>

        {/* FAQ */}
        <section>
          <h2 className="text-xl font-bold mb-4">FAQ</h2>
          <div className="space-y-3">
            {faqs.map((faq, i) => (
              <details key={i} className="rounded-xl border p-4">
                <summary className="font-medium cursor-pointer list-none">{faq.q}</summary>
                <p className="mt-2 text-sm text-muted-foreground">{faq.a}</p>
              </details>
            ))}
          </div>
        </section>
      </main>
    </>
  )
}
