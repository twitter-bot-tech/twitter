import { Metadata } from 'next'
import Link from 'next/link'
import { notFound } from 'next/navigation'
import { Newspaper, Clock, Share2, TrendingUp, ArrowRight } from 'lucide-react'
import { Breadcrumb } from '@/components/seo/Breadcrumb'
import { SchemaScript } from '@/components/seo/SchemaScript'
import { breadcrumbSchema } from '@/lib/seo'
import { CATEGORIES, CATEGORY_SLUGS, SLUG_TO_CATEGORY, NEWS_ITEMS, readingTime } from '../../data'

export const revalidate = 300

export async function generateStaticParams() {
  return Object.values(CATEGORY_SLUGS).map(cat => ({ cat }))
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ cat: string }>
}): Promise<Metadata> {
  const { cat } = await params
  const categoryName = SLUG_TO_CATEGORY[cat]
  if (!categoryName) return {}

  const descriptions: Record<string, string> = {
    'Bitcoin': 'Latest Bitcoin news: price analysis, ETF inflows, institutional adoption, and on-chain data. Stay informed on BTC with MoonX.',
    'Ethereum': 'Latest Ethereum news: upgrades, Layer 2 growth, DeFi activity, and ETH price analysis. Track Ethereum developments on MoonX.',
    'Solana': 'Latest Solana news: DEX volumes, meme tokens, ecosystem growth, and SOL price updates. Follow Solana with MoonX.',
    'Altcoin': 'Latest altcoin news: XRP, AVAX, and emerging cryptocurrencies. Track altcoin developments and price action on MoonX.',
    'DeFi': 'Latest DeFi news: protocol revenues, TVL milestones, governance votes, and yield opportunities. Follow DeFi with MoonX.',
    'Crypto Headlines': 'Top crypto headlines: regulatory news, ETF approvals, exchange updates, and major market events. Stay current on MoonX.',
  }

  return {
    title: `${categoryName} News — Latest Updates | MoonX`,
    description: descriptions[categoryName] || `Latest ${categoryName} news and market updates on MoonX.`,
    alternates: { canonical: `https://www.bydfi.com/en/moonx/news/category/${cat}` },
    openGraph: {
      title: `${categoryName} News | MoonX`,
      description: descriptions[categoryName] || `Latest ${categoryName} news on MoonX.`,
      url: `https://www.bydfi.com/en/moonx/news/category/${cat}`,
      siteName: 'MoonX by BYDFi',
    },
  }
}

function timeAgo(dateStr: string) {
  const diff = Date.now() - new Date(dateStr).getTime()
  const hours = Math.floor(diff / 3600000)
  if (hours < 1) return 'Just now'
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

export default async function NewsCategoryPage({
  params,
}: {
  params: Promise<{ cat: string }>
}) {
  const { cat } = await params
  const categoryName = SLUG_TO_CATEGORY[cat]
  if (!categoryName) notFound()

  const articles = NEWS_ITEMS.filter(n => n.category === categoryName)

  const breadcrumbs = [
    { name: 'MoonX', href: '/en/moonx/markets/trending' },
    { name: 'Crypto News', href: '/en/moonx/news' },
    { name: categoryName },
  ]

  const categoryDescriptions: Record<string, string> = {
    'Bitcoin': 'Price analysis, ETF inflows, institutional adoption, and on-chain data.',
    'Ethereum': 'Upgrades, Layer 2 growth, DeFi activity, and ETH price analysis.',
    'Solana': 'DEX volumes, meme tokens, ecosystem growth, and SOL price updates.',
    'Altcoin': 'XRP, AVAX, and emerging cryptocurrencies — price action and developments.',
    'DeFi': 'Protocol revenues, TVL milestones, governance votes, and yield opportunities.',
    'Crypto Headlines': 'Regulatory news, ETF approvals, exchange updates, and major market events.',
  }

  return (
    <>
      <SchemaScript
        schema={breadcrumbSchema(
          breadcrumbs.map(b => ({ name: b.name, url: `https://www.bydfi.com${b.href || ''}` }))
        )}
      />

      <main className="max-w-6xl mx-auto px-4 py-8">
        <Breadcrumb items={breadcrumbs} />

        {/* Hero */}
        <div className="rounded-2xl bg-gradient-to-br from-slate-900 to-slate-800 border border-slate-700 p-8 mb-8 relative overflow-hidden">
          <div className="absolute right-8 top-4 w-24 h-24 rounded-full bg-orange-500/10 border border-orange-500/20" />
          <div className="absolute right-20 top-12 w-16 h-16 rounded-full bg-blue-500/10 border border-blue-500/20" />

          <div className="flex items-center gap-3 mb-2">
            <Newspaper className="w-8 h-8 text-orange-500" />
            <h1 className="text-3xl font-bold text-white">{categoryName} News</h1>
          </div>
          <p className="text-slate-400 max-w-xl">
            {categoryDescriptions[categoryName]}
          </p>
          <p className="text-slate-500 text-sm mt-2">{articles.length} articles</p>
        </div>

        {/* Category tabs */}
        <div className="flex gap-1 flex-wrap mb-6 border-b pb-0">
          <Link
            href="/en/moonx/news"
            className="px-4 py-2.5 text-sm font-medium transition-colors border-b-2 -mb-px border-transparent text-muted-foreground hover:text-foreground hover:border-muted-foreground"
          >
            All
          </Link>
          {CATEGORIES.slice(1).map(c => {
            const slug = CATEGORY_SLUGS[c]
            const isActive = c === categoryName
            return (
              <Link
                key={c}
                href={`/en/moonx/news/category/${slug}`}
                className={`px-4 py-2.5 text-sm font-medium transition-colors border-b-2 -mb-px ${
                  isActive
                    ? 'border-orange-500 text-orange-500'
                    : 'border-transparent text-muted-foreground hover:text-foreground hover:border-muted-foreground'
                }`}
              >
                {c}
              </Link>
            )
          })}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Article list */}
          <div className="lg:col-span-2 space-y-0 divide-y">
            {articles.map(article => (
              <Link key={article.id} href={`/en/moonx/news/${article.slug}`} className="block group">
                <article className="py-5 flex gap-4">
                  <div className={`flex-shrink-0 w-28 h-20 rounded-lg bg-gradient-to-br ${article.color} flex items-center justify-center`}>
                    <span className="text-white font-bold text-xs">{article.tag}</span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <h2 className="font-semibold text-sm leading-snug line-clamp-2 mb-1.5 group-hover:text-orange-500 transition-colors">
                      {article.title}
                    </h2>
                    <p className="text-xs text-muted-foreground line-clamp-2 mb-2">{article.excerpt}</p>
                    <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
                      <span className="text-orange-500 font-medium">Author: {article.source}</span>
                      <span className="flex items-center gap-0.5">
                        <Clock className="w-3 h-3" />{timeAgo(article.date)}
                      </span>
                      <span>{readingTime(article.content)} min read</span>
                      <span className="flex items-center gap-0.5 hover:text-foreground transition-colors cursor-pointer ml-auto">
                        <Share2 className="w-3 h-3" /> Share
                      </span>
                    </div>
                  </div>
                </article>
              </Link>
            ))}

            {articles.length === 0 && (
              <div className="text-center py-16 text-muted-foreground">
                <Newspaper className="w-12 h-12 mx-auto mb-3 opacity-20" />
                <p>No {categoryName} news yet.</p>
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-4">
            <div className="rounded-xl bg-gradient-to-br from-orange-500 to-amber-600 p-5 text-white">
              <p className="text-xs font-medium opacity-80 mb-1">Start Trading on MoonX</p>
              <p className="text-2xl font-bold mb-0.5">$50 Bonus</p>
              <p className="text-xs opacity-80 mb-3">For new users. Prediction markets + meme tokens.</p>
              <Link href="https://www.bydfi.com" className="block text-center bg-white text-orange-600 font-semibold text-sm py-2 rounded-lg hover:bg-orange-50 transition-colors">
                Start Now
              </Link>
            </div>

            <div className="rounded-xl border p-4">
              <div className="flex items-center gap-2 mb-3">
                <TrendingUp className="w-4 h-4 text-blue-500" />
                <h3 className="font-semibold text-sm">Trending Markets</h3>
              </div>
              <div className="space-y-2">
                {['Will BTC reach $150K by June?', 'Will ETH flip BTC in 2026?', 'Will Solana ETF launch in Q2?'].map((q, i) => (
                  <Link key={i} href="/en/moonx/markets/trending" className="flex items-start justify-between gap-2 py-2 border-b last:border-0 group">
                    <span className="text-xs text-muted-foreground group-hover:text-foreground transition-colors line-clamp-2">{q}</span>
                    <ArrowRight className="w-3 h-3 flex-shrink-0 text-muted-foreground group-hover:text-blue-500 transition-colors mt-0.5" />
                  </Link>
                ))}
              </div>
              <Link href="/en/moonx/markets/trending" className="text-xs text-blue-500 hover:underline mt-2 inline-block">
                View all markets →
              </Link>
            </div>

            <div className="rounded-xl border p-4">
              <h3 className="font-semibold text-sm mb-3">All Categories</h3>
              <div className="flex flex-col gap-1">
                {CATEGORIES.slice(1).map(c => (
                  <Link
                    key={c}
                    href={`/en/moonx/news/category/${CATEGORY_SLUGS[c]}`}
                    className={`text-sm px-3 py-1.5 rounded-lg transition-colors ${
                      c === categoryName
                        ? 'bg-orange-50 text-orange-600 font-medium'
                        : 'hover:bg-accent text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    {c}
                  </Link>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t flex flex-wrap gap-4 text-sm text-muted-foreground">
          <Link href="/en/moonx/news" className="hover:text-foreground">All News →</Link>
          <Link href="/en/moonx/markets/trending" className="hover:text-foreground">Prediction Markets →</Link>
          <Link href="/en/moonx/learn/meme" className="hover:text-foreground">Meme Academy →</Link>
        </div>
      </main>
    </>
  )
}
