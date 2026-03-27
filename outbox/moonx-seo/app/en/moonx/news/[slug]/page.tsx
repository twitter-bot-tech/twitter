import { Metadata } from 'next'
import Link from 'next/link'
import { notFound } from 'next/navigation'
import { Clock, ArrowLeft, Share2, TrendingUp, ArrowRight, Tag, BookOpen } from 'lucide-react'
import { Breadcrumb } from '@/components/seo/Breadcrumb'
import { SchemaScript } from '@/components/seo/SchemaScript'
import { breadcrumbSchema } from '@/lib/seo'
import { NEWS_ITEMS, CATEGORY_SLUGS, readingTime } from '../data'

export const revalidate = 300

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL || 'https://www.bydfi.com'

export async function generateStaticParams() {
  return NEWS_ITEMS.map(item => ({ slug: item.slug }))
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>
}): Promise<Metadata> {
  const { slug } = await params
  const article = NEWS_ITEMS.find(n => n.slug === slug)
  if (!article) return {}

  const ogImageUrl = `${SITE_URL}/api/og/news/${article.slug}`

  return {
    title: `${article.title} | MoonX`,
    description: article.excerpt,
    keywords: article.keywords,
    alternates: { canonical: `${SITE_URL}/en/moonx/news/${article.slug}` },
    openGraph: {
      title: article.title,
      description: article.excerpt,
      url: `${SITE_URL}/en/moonx/news/${article.slug}`,
      siteName: 'MoonX by BYDFi',
      type: 'article',
      publishedTime: article.date,
      authors: [article.source],
      tags: article.keywords,
      images: [{ url: ogImageUrl, width: 1200, height: 630, alt: article.title }],
    },
    twitter: {
      card: 'summary_large_image',
      title: article.title,
      description: article.excerpt,
      images: [ogImageUrl],
    },
  }
}

function formatDate(dateStr: string) {
  return new Date(dateStr).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function renderContent(content: string) {
  return content.split('\n\n').map((block, i) => {
    if (block.startsWith('**') && block.endsWith('**') && !block.slice(2).includes('**')) {
      return <h3 key={i} className="text-lg font-bold mt-6 mb-2">{block.slice(2, -2)}</h3>
    }
    const parts = block.split(/(\*\*[^*]+\*\*)/)
    return (
      <p key={i} className="text-sm leading-relaxed text-muted-foreground mb-4">
        {parts.map((part, j) =>
          part.startsWith('**') && part.endsWith('**')
            ? <strong key={j} className="text-foreground font-semibold">{part.slice(2, -2)}</strong>
            : part
        )}
      </p>
    )
  })
}

export default async function NewsDetailPage({
  params,
}: {
  params: Promise<{ slug: string }>
}) {
  const { slug } = await params
  const article = NEWS_ITEMS.find(n => n.slug === slug)
  if (!article) notFound()

  const related = NEWS_ITEMS.filter(n => n.slug !== slug && n.category === article.category).slice(0, 3)
  const catSlug = CATEGORY_SLUGS[article.category]
  const mins = readingTime(article.content)

  const breadcrumbs = [
    { name: 'MoonX', href: '/en/moonx/markets/trending' },
    { name: 'Crypto News', href: '/en/moonx/news' },
    { name: article.category, href: `/en/moonx/news/category/${catSlug}` },
    { name: article.title.slice(0, 45) + '…' },
  ]

  // NewsArticle Schema (Google News compliant)
  const newsArticleSchema = {
    '@context': 'https://schema.org',
    '@type': 'NewsArticle',
    headline: article.title,
    description: article.excerpt,
    keywords: article.keywords.join(', '),
    image: `${SITE_URL}/api/og/news/${article.slug}`,
    datePublished: article.date,
    dateModified: article.date,
    author: { '@type': 'Organization', name: article.source, url: SITE_URL },
    publisher: {
      '@type': 'Organization',
      name: 'MoonX by BYDFi',
      url: SITE_URL,
      logo: {
        '@type': 'ImageObject',
        url: `${SITE_URL}/logo.png`,
        width: 200,
        height: 60,
      },
    },
    mainEntityOfPage: {
      '@type': 'WebPage',
      '@id': `${SITE_URL}/en/moonx/news/${article.slug}`,
    },
    articleSection: article.category,
    wordCount: article.content.trim().split(/\s+/).length,
    timeRequired: `PT${mins}M`,
  }

  // FAQ Schema
  const faqSchema = {
    '@context': 'https://schema.org',
    '@type': 'FAQPage',
    mainEntity: article.faqs.map(faq => ({
      '@type': 'Question',
      name: faq.q,
      acceptedAnswer: { '@type': 'Answer', text: faq.a },
    })),
  }

  return (
    <>
      <SchemaScript schema={newsArticleSchema} />
      <SchemaScript schema={faqSchema} />
      <SchemaScript schema={breadcrumbSchema(breadcrumbs.map(b => ({ name: b.name, url: `${SITE_URL}${b.href || ''}` })))} />

      <main className="max-w-6xl mx-auto px-4 py-8">
        <Breadcrumb items={breadcrumbs} />

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Article */}
          <div className="lg:col-span-2">
            <Link
              href="/en/moonx/news"
              className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors mb-6"
            >
              <ArrowLeft className="w-4 h-4" /> Back to News
            </Link>

            {/* Category + tag */}
            <div className="flex items-center gap-2 mb-3">
              <Link
                href={`/en/moonx/news/category/${catSlug}`}
                className="flex items-center gap-1 text-xs text-orange-600 bg-orange-50 px-2 py-0.5 rounded-full hover:bg-orange-100 transition-colors"
              >
                <Tag className="w-3 h-3" /> {article.category}
              </Link>
              <span className={`text-xs font-bold text-white px-2 py-0.5 rounded-full bg-gradient-to-r ${article.color}`}>
                {article.tag}
              </span>
            </div>

            {/* Title */}
            <h1 className="text-2xl font-bold leading-snug mb-4">{article.title}</h1>

            {/* Meta */}
            <div className="flex items-center gap-4 text-xs text-muted-foreground mb-6 pb-6 border-b">
              <span className="text-orange-500 font-medium">{article.source}</span>
              <span className="flex items-center gap-1">
                <Clock className="w-3 h-3" /> {formatDate(article.date)}
              </span>
              <span className="flex items-center gap-1">
                <BookOpen className="w-3 h-3" /> {mins} min read
              </span>
              <button className="flex items-center gap-1 ml-auto hover:text-foreground transition-colors">
                <Share2 className="w-3 h-3" /> Share
              </button>
            </div>

            {/* Hero image */}
            <div className={`w-full h-48 rounded-xl bg-gradient-to-br ${article.color} flex items-center justify-center mb-6`}>
              <span className="text-white font-bold text-4xl opacity-40">{article.tag}</span>
            </div>

            {/* Content */}
            <div className="prose-sm max-w-none">
              {renderContent(article.content)}
            </div>

            {/* Keywords */}
            <div className="mt-6 pt-4 border-t flex flex-wrap gap-2">
              {article.keywords.map(kw => (
                <span key={kw} className="text-xs bg-muted px-2.5 py-1 rounded-full text-muted-foreground">
                  {kw}
                </span>
              ))}
            </div>

            {/* FAQ */}
            <div className="mt-8 pt-6 border-t">
              <h2 className="font-bold text-base mb-4">Frequently Asked Questions</h2>
              <div className="space-y-4">
                {article.faqs.map((faq, i) => (
                  <div key={i} className="rounded-xl border p-4">
                    <h3 className="font-semibold text-sm mb-2">{faq.q}</h3>
                    <p className="text-sm text-muted-foreground leading-relaxed">{faq.a}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Related */}
            {related.length > 0 && (
              <div className="mt-8 pt-6 border-t">
                <h2 className="font-bold text-base mb-4">More {article.category} News</h2>
                <div className="space-y-4">
                  {related.map(r => (
                    <Link key={r.id} href={`/en/moonx/news/${r.slug}`} className="flex gap-3 group">
                      <div className={`flex-shrink-0 w-16 h-12 rounded-lg bg-gradient-to-br ${r.color} flex items-center justify-center`}>
                        <span className="text-white font-bold text-[10px]">{r.tag}</span>
                      </div>
                      <div>
                        <p className="text-sm font-medium leading-snug line-clamp-2 group-hover:text-orange-500 transition-colors">{r.title}</p>
                        <p className="text-[11px] text-muted-foreground mt-0.5">{r.source} · {readingTime(r.content)} min read</p>
                      </div>
                    </Link>
                  ))}
                </div>
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
              <h3 className="font-semibold text-sm mb-3">All News Categories</h3>
              <div className="flex flex-col gap-1">
                {Object.entries(CATEGORY_SLUGS).map(([name, catSlug]) => (
                  <Link
                    key={name}
                    href={`/en/moonx/news/category/${catSlug}`}
                    className={`text-sm px-3 py-1.5 rounded-lg transition-colors ${
                      name === article.category
                        ? 'bg-orange-50 text-orange-600 font-medium'
                        : 'hover:bg-accent text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    {name}
                  </Link>
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>
    </>
  )
}
