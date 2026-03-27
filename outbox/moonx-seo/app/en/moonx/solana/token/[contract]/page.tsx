import { Metadata } from 'next'
import { getTokenData, getHolderDistribution, getSmartMoneyFeed, getRiskScore } from '@/lib/token-data'
import { memeTokenMeta, tokenSchema, breadcrumbSchema } from '@/lib/seo'
import { SchemaScript } from '@/components/seo/SchemaScript'
import { Breadcrumb } from '@/components/seo/Breadcrumb'
import { OddsWidget } from '@/components/meme-tools/OddsWidget'
import { SecurityScore } from '@/components/meme-tools/SecurityScore'
import { TokenLifecycle } from '@/components/meme-tools/TokenLifecycle'
import { HolderChart } from '@/components/meme-tools/HolderChart'
import { ProfitCalculator } from '@/components/meme-tools/ProfitCalculator'
import { SentimentPill } from '@/components/meme-tools/SentimentPill'
import { SmartMoneyFeed } from '@/components/meme-tools/SmartMoneyFeed'
import { AlertTriangle, ExternalLink, TrendingUp, TrendingDown } from 'lucide-react'

interface Props { params: { contract: string } }

// Hot tokens refresh every 60s, others every 5min
export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const token = await getTokenData(params.contract)
  if (!token) return {}
  return memeTokenMeta(token)
}

export async function generateStaticParams() {
  // Seed contracts for static generation — extend from DB in production
  return [
    { contract: 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263' }, // BONK
    { contract: '7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr' }, // POPCAT
    { contract: 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm' }, // WIF
  ]
}

export default async function MemeTokenPage({ params }: Props) {
  const { contract } = params

  const [token, holders, smartMoney, riskData] = await Promise.all([
    getTokenData(contract),
    getHolderDistribution(contract),
    getSmartMoneyFeed(contract, 5),
    getRiskScore(contract),
  ])

  // Dynamic revalidate based on volume
  // For Next.js 14: set at route segment level
  // We handle hot tokens via route-level config exported below

  if (!token) {
    return (
      <main className="max-w-3xl mx-auto px-4 py-16 text-center">
        <AlertTriangle className="w-12 h-12 mx-auto mb-3 text-yellow-500" />
        <h1 className="text-xl font-bold mb-2">Token not found</h1>
        <p className="text-muted-foreground text-sm">We couldn&apos;t load data for contract {contract.slice(0, 8)}…</p>
      </main>
    )
  }

  const breadcrumbs = [
    { name: 'MoonX', href: '/en/moonx/markets/trending' },
    { name: 'Solana Tokens', href: '/en/moonx/learn/meme' },
    { name: `${token.symbol} (${contract.slice(0, 6)}…)` },
  ]

  const priceFormatted = token.price < 0.001 ? token.price.toExponential(4) : token.price.toFixed(6)
  const changeColor = token.priceChange24h >= 0 ? 'text-green-600' : 'text-red-500'
  const changeSign = token.priceChange24h >= 0 ? '+' : ''
  const isDead = token.isDead

  return (
    <>
      <SchemaScript schema={[
        tokenSchema(token),
        breadcrumbSchema(breadcrumbs.map(b => ({ name: b.name, url: `https://www.bydfi.com${b.href || ''}` }))),
      ]} />

      <main className="max-w-5xl mx-auto px-4 py-6">
        <Breadcrumb items={breadcrumbs} />

        {/* Dead token banner */}
        {isDead && (
          <div className="rounded-xl bg-gray-100 border border-gray-300 p-4 mb-4 flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-gray-500 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-medium text-gray-700 text-sm">This token appears inactive</p>
              <p className="text-xs text-gray-500">Volume under $500 in the past 24 hours. Historical data is preserved for reference.</p>
            </div>
          </div>
        )}

        {/* Token header */}
        <div className="flex items-start justify-between gap-4 mb-6">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <h1 className="text-2xl font-bold">{token.name}</h1>
              <span className="text-sm font-mono text-muted-foreground bg-muted px-1.5 py-0.5 rounded">{token.symbol}</span>
              {token.isHot && (
                <span className="text-[10px] font-bold text-white bg-orange-500 px-2 py-0.5 rounded-full">🔥 HOT</span>
              )}
            </div>
            <p className="text-xs text-muted-foreground font-mono">{contract}</p>
          </div>
          <a
            href={`https://www.bydfi.com/en/moonx/markets/trending`}
            className="flex items-center gap-1.5 text-sm bg-blue-600 text-white px-3 py-1.5 rounded-lg hover:bg-blue-700 transition-colors"
          >
            Trade <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>

        {/* Price stats row */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
          <div className="rounded-xl border bg-card p-3">
            <p className="text-xs text-muted-foreground mb-0.5">Price</p>
            <p className="font-bold text-lg font-mono">${priceFormatted}</p>
            <p className={`text-xs font-medium ${changeColor} flex items-center gap-0.5`}>
              {token.priceChange24h >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
              {changeSign}{token.priceChange24h.toFixed(1)}% (24h)
            </p>
          </div>
          <div className="rounded-xl border bg-card p-3">
            <p className="text-xs text-muted-foreground mb-0.5">Volume 24h</p>
            <p className="font-bold text-lg">
              {token.volume24h >= 1_000_000 ? `$${(token.volume24h / 1_000_000).toFixed(1)}M` :
               token.volume24h >= 1_000 ? `$${(token.volume24h / 1_000).toFixed(0)}K` : `$${token.volume24h.toFixed(0)}`}
            </p>
          </div>
          <div className="rounded-xl border bg-card p-3">
            <p className="text-xs text-muted-foreground mb-0.5">Market Cap</p>
            <p className="font-bold text-lg">
              {token.mcap >= 1_000_000 ? `$${(token.mcap / 1_000_000).toFixed(1)}M` :
               token.mcap >= 1_000 ? `$${(token.mcap / 1_000).toFixed(0)}K` : `$${token.mcap.toFixed(0)}`}
            </p>
          </div>
          <div className="rounded-xl border bg-card p-3">
            <p className="text-xs text-muted-foreground mb-0.5">LP (USD)</p>
            <p className="font-bold text-lg">
              {token.lpUsd >= 1_000_000 ? `$${(token.lpUsd / 1_000_000).toFixed(1)}M` :
               token.lpUsd >= 1_000 ? `$${(token.lpUsd / 1_000).toFixed(0)}K` : `$${token.lpUsd.toFixed(0)}`}
            </p>
          </div>
        </div>

        {/* 7 Tool widgets grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Tool 1: Polymarket Odds */}
          <OddsWidget
            contract={contract}
            initialOdds={undefined}
          />

          {/* Tool 2: Security Score */}
          <SecurityScore contract={contract} initial={riskData} />

          {/* Tool 3: Token Lifecycle */}
          <TokenLifecycle launchedAt={token.launchedAt} volume24h={token.volume24h} />

          {/* Tool 5: Profit Calculator */}
          <ProfitCalculator currentPrice={token.price} symbol={token.symbol} />

          {/* Tool 6: Sentiment */}
          <SentimentPill contract={contract} />

          {/* Tool 4: Holder Chart — spans 2 cols on large screens */}
          <div className="lg:col-span-1">
            <HolderChart holders={holders} totalHolders={token.holders} />
          </div>
        </div>

        {/* Tool 7: Smart Money Feed — full width */}
        <div className="mt-4">
          <SmartMoneyFeed contract={contract} initial={smartMoney} />
        </div>

        {/* SEO content */}
        <section className="mt-8 pt-6 border-t prose prose-slate max-w-none text-sm">
          <h2>About {token.name} ({token.symbol})</h2>
          <p>
            {token.name} ({token.symbol}) is a Solana-based meme token with a current price of ${priceFormatted} and {token.holders.toLocaleString()} holders.
            The token was launched approximately {Math.floor(token.ageHours / 24)} days ago and currently has ${
              token.lpUsd >= 1_000 ? `${(token.lpUsd / 1_000).toFixed(0)}K` : token.lpUsd.toFixed(0)
            } in liquidity.
          </p>
          <p>
            MoonX provides real-time analytics for {token.symbol} including Polymarket odds, contract security scores powered by RugCheck, smart money wallet tracking, and holder distribution analysis — all in one place.
          </p>
          <h3>How to trade {token.symbol}</h3>
          <p>
            To trade {token.symbol}, you can use DEXs on Solana such as Raydium or Jupiter. Always check the security score before trading, and monitor smart money wallets for early signals. The profit calculator above can help you plan entry and exit points.
          </p>
        </section>
      </main>
    </>
  )
}

// Dynamic revalidate — Next.js 14 doesn't support conditional exports,
// so we use a conservative 5-minute default; hot tokens force revalidation
// via on-demand ISR in production (POST /api/revalidate)
export const revalidate = 300
