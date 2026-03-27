import { Metadata } from 'next'
import Link from 'next/link'
import prisma from '@/lib/prisma'
import { Breadcrumb } from '@/components/seo/Breadcrumb'
import { SchemaScript } from '@/components/seo/SchemaScript'
import { breadcrumbSchema } from '@/lib/seo'
import { Rocket, TrendingUp, BookOpen } from 'lucide-react'

export const metadata: Metadata = {
  title: 'Meme Token Academy — Learn to Trade Meme Coins | MoonX',
  description: 'Level up your meme token game. Learn how to analyze meme coins on Solana, read smart money signals, evaluate security risks, and trade prediction markets.',
  alternates: { canonical: 'https://www.bydfi.com/en/moonx/learn/meme' },
}

export const revalidate = 3600

export default async function MemeLearnPage() {
  const articles = await prisma.article.findMany({
    where: { category: 'meme', status: 'published' },
    orderBy: { publishedAt: 'desc' },
    select: { id: true, slug: true, h1: true, metaDesc: true, tags: true, publishedAt: true },
  })

  const breadcrumbs = [
    { name: 'MoonX', href: '/en/moonx/markets/trending' },
    { name: 'Learn', href: '/en/moonx/learn/prediction' },
    { name: 'Meme Tokens' },
  ]

  return (
    <>
      <SchemaScript schema={breadcrumbSchema(breadcrumbs.map(b => ({ name: b.name, url: `https://www.bydfi.com${b.href || ''}` })))} />

      <main className="max-w-5xl mx-auto px-4 py-8">
        <Breadcrumb items={breadcrumbs} />

        {/* Hero */}
        <div className="rounded-2xl bg-gradient-to-br from-orange-50 to-purple-50 border p-8 mb-8">
          <div className="flex items-center gap-3 mb-3">
            <Rocket className="w-8 h-8 text-orange-500" />
            <h1 className="text-3xl font-bold">Meme Token Academy</h1>
          </div>
          <p className="text-muted-foreground max-w-xl">
            Master the art of meme coin trading. From reading smart money wallets to evaluating contract security — everything you need to trade Solana meme tokens on MoonX.
          </p>
          <div className="flex gap-4 mt-4 text-sm">
            <span className="flex items-center gap-1.5 text-green-600"><TrendingUp className="w-4 h-4" /> Live price tools</span>
            <span className="flex items-center gap-1.5 text-blue-600"><BookOpen className="w-4 h-4" /> {articles.length} guides</span>
          </div>
        </div>

        {/* Articles */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {articles.map(article => {
            const tags: string[] = JSON.parse(article.tags || '[]')
            return (
              <Link key={article.id} href={`/en/moonx/learn/meme/${article.slug}`}>
                <article className="rounded-xl border p-5 hover:shadow-md transition-shadow h-full flex flex-col bg-card">
                  <div className="flex flex-wrap gap-1 mb-2">
                    {tags.slice(0, 2).map(t => (
                      <span key={t} className="text-[10px] text-orange-600 bg-orange-50 px-1.5 py-0.5 rounded">{t}</span>
                    ))}
                  </div>
                  <h2 className="font-semibold mb-2 flex-1 leading-snug">{article.h1}</h2>
                  <p className="text-xs text-muted-foreground line-clamp-2 mb-3">{article.metaDesc}</p>
                  <p className="text-[11px] text-muted-foreground">
                    {article.publishedAt ? new Date(article.publishedAt).toLocaleDateString() : ''}
                  </p>
                </article>
              </Link>
            )
          })}
        </div>

        {articles.length === 0 && (
          <div className="text-center py-16 text-muted-foreground">
            <Rocket className="w-12 h-12 mx-auto mb-3 opacity-20" />
            <p>Meme token guides coming soon.</p>
          </div>
        )}
      </main>
    </>
  )
}
