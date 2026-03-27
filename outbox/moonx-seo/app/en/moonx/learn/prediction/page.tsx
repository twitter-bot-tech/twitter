import { Metadata } from 'next'
import Link from 'next/link'
import prisma from '@/lib/prisma'
import { Breadcrumb } from '@/components/seo/Breadcrumb'
import { SchemaScript } from '@/components/seo/SchemaScript'
import { breadcrumbSchema } from '@/lib/seo'
import { BookOpen, Clock, Tag } from 'lucide-react'

export const metadata: Metadata = {
  title: 'Prediction Market Academy — Learn & Trade Smarter | MoonX',
  description: 'Master prediction markets with MoonX Academy. Learn how Polymarket, Kalshi, and Manifold work, compare odds, and find trading opportunities.',
  alternates: { canonical: 'https://www.bydfi.com/en/moonx/learn/prediction' },
}

export const revalidate = 3600

export default async function PredictionLearnPage() {
  const articles = await prisma.article.findMany({
    where: { category: 'prediction', status: 'published' },
    orderBy: { publishedAt: 'desc' },
    select: { id: true, slug: true, h1: true, metaDesc: true, tags: true, publishedAt: true, author: true, coverImage: true },
  })

  const breadcrumbs = [
    { name: 'MoonX', href: '/en/moonx/markets/trending' },
    { name: 'Learn', href: '/en/moonx/learn/prediction' },
    { name: 'Prediction Markets' },
  ]

  return (
    <>
      <SchemaScript schema={breadcrumbSchema(breadcrumbs.map(b => ({ name: b.name, url: `https://www.bydfi.com${b.href || ''}` })))} />

      <main className="max-w-5xl mx-auto px-4 py-8">
        <Breadcrumb items={breadcrumbs} />

        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-3">Prediction Market Academy</h1>
          <p className="text-lg text-muted-foreground">
            Learn how prediction markets work, compare Polymarket vs Kalshi, and find the best trading opportunities.
          </p>
        </div>

        {/* Featured article */}
        {articles[0] && (
          <Link href={`/en/moonx/learn/prediction/${articles[0].slug}`} className="block mb-8">
            <div className="rounded-2xl border bg-gradient-to-br from-blue-50 to-purple-50 p-6 hover:shadow-md transition-shadow">
              <span className="text-xs font-medium text-blue-600 bg-blue-100 px-2 py-0.5 rounded-full mb-3 inline-block">Featured</span>
              <h2 className="text-xl font-bold mb-2">{articles[0].h1}</h2>
              <p className="text-muted-foreground text-sm line-clamp-2">{articles[0].metaDesc}</p>
              <div className="flex items-center gap-3 mt-3 text-xs text-muted-foreground">
                <span className="flex items-center gap-1"><Clock className="w-3 h-3" /> {articles[0].publishedAt ? new Date(articles[0].publishedAt).toLocaleDateString() : 'Recent'}</span>
                <span className="flex items-center gap-1"><BookOpen className="w-3 h-3" /> {articles[0].author}</span>
              </div>
            </div>
          </Link>
        )}

        {/* Article grid */}
        <div className="grid md:grid-cols-2 gap-4">
          {articles.slice(1).map(article => {
            const tags: string[] = JSON.parse(article.tags || '[]')
            return (
              <Link key={article.id} href={`/en/moonx/learn/prediction/${article.slug}`}>
                <article className="rounded-xl border p-5 hover:shadow-md transition-shadow h-full flex flex-col">
                  <div className="flex gap-2 flex-wrap mb-2">
                    {tags.slice(0, 2).map(t => (
                      <span key={t} className="text-[10px] text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded flex items-center gap-0.5">
                        <Tag className="w-2.5 h-2.5" />{t}
                      </span>
                    ))}
                  </div>
                  <h2 className="font-semibold mb-2 flex-1">{article.h1}</h2>
                  <p className="text-sm text-muted-foreground line-clamp-2 mb-3">{article.metaDesc}</p>
                  <p className="text-xs text-muted-foreground">
                    {article.publishedAt ? new Date(article.publishedAt).toLocaleDateString() : ''}
                  </p>
                </article>
              </Link>
            )
          })}
        </div>

        {articles.length === 0 && (
          <div className="text-center py-16 text-muted-foreground">
            <BookOpen className="w-12 h-12 mx-auto mb-3 opacity-20" />
            <p>Articles coming soon. Check back shortly.</p>
          </div>
        )}
      </main>
    </>
  )
}
