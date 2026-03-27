import { Metadata } from 'next'
import { notFound } from 'next/navigation'
import { getMarketBySlug } from '@/lib/polymarket'
import { eventGuideMeta, faqSchema, breadcrumbSchema } from '@/lib/seo'
import { SchemaScript } from '@/components/seo/SchemaScript'
import { Breadcrumb } from '@/components/seo/Breadcrumb'
import { TrendingUp, DollarSign, Clock, ExternalLink } from 'lucide-react'

interface Props { params: { 'event-slug': string } }

export const revalidate = 60  // Re-render every minute for live odds

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const market = await getMarketBySlug(params['event-slug'])
  if (!market) return {}
  return eventGuideMeta(market)
}

function formatVolume(n: number) {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `$${(n / 1_000).toFixed(0)}K`
  return `$${n.toFixed(0)}`
}

export default async function EventGuidePage({ params }: Props) {
  const market = await getMarketBySlug(params['event-slug'])
  if (!market) notFound()

  const sortedOutcomes = [...market.outcomes].sort((a, b) => b.probability - a.probability)
  const daysUntilClose = Math.ceil((new Date(market.endDate).getTime() - Date.now()) / 86400000)

  const faqs = [
    { q: `What are the current odds for "${market.question}"?`, a: `As of the latest update, the market gives ${sortedOutcomes[0].title} a ${Math.round(sortedOutcomes[0].probability * 100)}% probability. You can track live odds on MoonX, which aggregates Polymarket, Kalshi, and Manifold.` },
    { q: 'How do prediction markets work?', a: 'Prediction markets let you buy shares in outcomes. Each share is worth $1 if the outcome occurs, $0 if not. The share price reflects the market\'s implied probability.' },
    { q: 'How is MoonX different from Polymarket?', a: 'MoonX is a prediction market aggregator — we show you odds from Polymarket, Kalshi, and Manifold in one place, so you can find the best prices and track smart money flows.' },
    { q: 'When does this market resolve?', a: `This market is scheduled to resolve on ${new Date(market.endDate).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}, ${daysUntilClose > 0 ? `in approximately ${daysUntilClose} days` : 'which has passed'}.` },
  ]

  const breadcrumbs = [
    { name: 'MoonX', href: '/en/moonx/markets/trending' },
    { name: 'Event Guides', href: '/en/moonx/markets/trending' },
    { name: market.question.length > 50 ? market.question.slice(0, 50) + '…' : market.question },
  ]

  return (
    <>
      <SchemaScript schema={[
        faqSchema(faqs),
        breadcrumbSchema(breadcrumbs.map(b => ({ name: b.name, url: `https://www.bydfi.com${b.href || ''}` }))),
      ]} />

      <main className="max-w-3xl mx-auto px-4 py-8">
        <Breadcrumb items={breadcrumbs} />

        {/* Header */}
        <header className="mb-6">
          <span className="text-xs font-medium text-blue-600 bg-blue-50 px-2 py-0.5 rounded-full">
            {market.category.toUpperCase()}
          </span>
          <h1 className="text-2xl font-bold mt-2 mb-1 leading-tight">{market.question}</h1>
          <p className="text-sm text-muted-foreground">{market.description}</p>
        </header>

        {/* Live Odds Card */}
        <div className="rounded-2xl border bg-card p-6 mb-6 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-blue-500" />
              <span className="font-semibold">Live Prediction Market Odds</span>
            </div>
            <span className="text-xs text-green-600 bg-green-50 px-2 py-0.5 rounded-full">● Live</span>
          </div>

          <div className="space-y-3 mb-4">
            {sortedOutcomes.map((outcome, i) => (
              <div key={outcome.id}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="font-medium">{outcome.title}</span>
                  <span className="font-bold">{Math.round(outcome.probability * 100)}¢</span>
                </div>
                <div className="h-3 bg-muted rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ${i === 0 ? 'bg-blue-500' : 'bg-gray-300'}`}
                    style={{ width: `${outcome.probability * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-3 gap-4 pt-4 border-t text-center">
            <div>
              <DollarSign className="w-4 h-4 text-muted-foreground mx-auto mb-0.5" />
              <p className="text-sm font-semibold">{formatVolume(market.volume)}</p>
              <p className="text-xs text-muted-foreground">Volume</p>
            </div>
            <div>
              <DollarSign className="w-4 h-4 text-muted-foreground mx-auto mb-0.5" />
              <p className="text-sm font-semibold">{formatVolume(market.liquidity)}</p>
              <p className="text-xs text-muted-foreground">Liquidity</p>
            </div>
            <div>
              <Clock className="w-4 h-4 text-muted-foreground mx-auto mb-0.5" />
              <p className="text-sm font-semibold">{daysUntilClose > 0 ? `${daysUntilClose}d` : 'Closed'}</p>
              <p className="text-xs text-muted-foreground">Until close</p>
            </div>
          </div>
        </div>

        {/* Trade CTA */}
        <div className="rounded-xl bg-gradient-to-r from-blue-600 to-purple-600 p-5 text-white mb-8">
          <h2 className="font-bold mb-1">Trade this market on MoonX</h2>
          <p className="text-sm text-blue-100 mb-3">Compare odds from Polymarket, Kalshi, and Manifold — get the best price.</p>
          <a
            href={`https://www.bydfi.com/en/moonx/markets/trending`}
            className="inline-flex items-center gap-2 bg-white text-blue-600 text-sm font-medium px-4 py-2 rounded-lg hover:bg-blue-50 transition-colors"
          >
            Open MoonX <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>

        {/* How to trade section */}
        <section className="mb-8">
          <h2 className="text-xl font-bold mb-4">How to Trade This Market</h2>
          <div className="prose prose-slate max-w-none text-sm">
            <p>
              Prediction markets like this one let you profit from your research and forecasting skills. Here&apos;s how to approach &quot;{market.question}&quot;:
            </p>
            <ol>
              <li><strong>Assess the base rate.</strong> Start with historical data — how often does this type of event occur?</li>
              <li><strong>Check smart money flows.</strong> MoonX tracks which wallets are buying YES vs NO positions.</li>
              <li><strong>Compare platforms.</strong> Polymarket, Kalshi, and Manifold sometimes show different odds for the same event — arbitrage opportunities exist.</li>
              <li><strong>Manage your position size.</strong> With {daysUntilClose > 0 ? `${daysUntilClose} days until resolution` : 'market closed'}, factor in time decay.</li>
            </ol>
          </div>
        </section>

        {/* FAQ */}
        <section>
          <h2 className="text-xl font-bold mb-4">Frequently Asked Questions</h2>
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
