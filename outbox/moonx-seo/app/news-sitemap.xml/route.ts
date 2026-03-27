import { NextResponse } from 'next/server'
import { NEWS_ITEMS } from '@/app/en/moonx/news/data'

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL || 'https://www.bydfi.com'
const PUBLICATION_NAME = 'MoonX by BYDFi'

// Google News sitemap: only articles published within the last 2 days
// https://developers.google.com/search/docs/crawling-indexing/sitemaps/news-sitemap
export async function GET() {
  const cutoff = Date.now() - 2 * 24 * 60 * 60 * 1000
  const recentItems = NEWS_ITEMS.filter(item => new Date(item.date).getTime() > cutoff)

  const urls = recentItems.map(item => {
    const pubDate = new Date(item.date).toISOString()
    return `
  <url>
    <loc>${SITE_URL}/en/moonx/news/${item.slug}</loc>
    <news:news>
      <news:publication>
        <news:name>${PUBLICATION_NAME}</news:name>
        <news:language>en</news:language>
      </news:publication>
      <news:publication_date>${pubDate}</news:publication_date>
      <news:title><![CDATA[${item.title}]]></news:title>
      <news:keywords>${item.keywords.join(', ')}</news:keywords>
    </news:news>
    <lastmod>${pubDate}</lastmod>
  </url>`
  }).join('')

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset
  xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
  xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">
${urls}
</urlset>`

  return new NextResponse(xml, {
    headers: {
      'Content-Type': 'application/xml',
      'Cache-Control': 's-maxage=300, stale-while-revalidate',
    },
  })
}
