import { MetadataRoute } from 'next'
import prisma from '@/lib/prisma'
import { NEWS_ITEMS, CATEGORY_SLUGS } from '@/app/en/moonx/news/data'

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL || 'https://www.bydfi.com'

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const now = new Date()

  // News pages
  const newsListPage: MetadataRoute.Sitemap = [
    { url: `${SITE_URL}/en/moonx/news`, lastModified: now, changeFrequency: 'hourly', priority: 0.9 },
    ...Object.values(CATEGORY_SLUGS).map(slug => ({
      url: `${SITE_URL}/en/moonx/news/category/${slug}`,
      lastModified: now,
      changeFrequency: 'hourly' as const,
      priority: 0.8,
    })),
  ]

  const newsDetailPages: MetadataRoute.Sitemap = NEWS_ITEMS.map(item => ({
    url: `${SITE_URL}/en/moonx/news/${item.slug}`,
    lastModified: new Date(item.date),
    changeFrequency: 'daily' as const,
    priority: 0.8,
  }))

  // Static pages
  const staticPages: MetadataRoute.Sitemap = [
    { url: `${SITE_URL}/en/moonx/markets/trending`, lastModified: now, changeFrequency: 'hourly', priority: 1.0 },
    { url: `${SITE_URL}/en/moonx/learn/prediction`, lastModified: now, changeFrequency: 'weekly', priority: 0.8 },
    { url: `${SITE_URL}/en/moonx/learn/meme`, lastModified: now, changeFrequency: 'weekly', priority: 0.8 },
    // Stock prediction pages
    ...['aapl', 'nvda', 'tsla', 'meta', 'msft'].map(ticker => ({
      url: `${SITE_URL}/en/moonx/markets/stocks/${ticker}`,
      lastModified: now,
      changeFrequency: 'daily' as const,
      priority: 0.7,
    })),
  ]

  // Published articles
  const articles = await prisma.article.findMany({
    where: { status: 'published' },
    select: { slug: true, category: true, updatedAt: true },
  })

  const articlePages: MetadataRoute.Sitemap = articles.map(a => ({
    url: `${SITE_URL}/en/moonx/learn/${a.category}/${a.slug}`,
    lastModified: a.updatedAt,
    changeFrequency: 'weekly',
    priority: 0.8,
  }))

  // Token pages — only non-noindex tokens
  const tokens = await prisma.tokenSnapshot.findMany({
    where: { noindex: false },
    select: { contract: true, updatedAt: true, volume24h: true },
  })

  const tokenPages: MetadataRoute.Sitemap = tokens
    .filter(t => t.volume24h >= 500)  // Only tokens with meaningful volume
    .map(t => ({
      url: `${SITE_URL}/en/moonx/solana/token/${t.contract}`,
      lastModified: t.updatedAt,
      changeFrequency: t.volume24h > 1_000_000 ? ('hourly' as const) : ('daily' as const),
      priority: t.volume24h > 1_000_000 ? 0.9 : 0.6,
    }))

  return [...newsListPage, ...newsDetailPages, ...staticPages, ...articlePages, ...tokenPages]
}
