import { Metadata } from 'next'
import { notFound } from 'next/navigation'
import ReactMarkdown from 'react-markdown'
import prisma from '@/lib/prisma'
import { articleMeta, articleSchema, faqSchema, breadcrumbSchema } from '@/lib/seo'
import { SchemaScript } from '@/components/seo/SchemaScript'
import { Breadcrumb } from '@/components/seo/Breadcrumb'
import { ExternalLink } from 'lucide-react'

interface Props { params: { slug: string } }

export const revalidate = 3600

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const article = await prisma.article.findUnique({ where: { slug: params.slug } })
  if (!article) return {}
  return articleMeta(article)
}

export default async function MemeArticlePage({ params }: Props) {
  const article = await prisma.article.findUnique({ where: { slug: params.slug } })
  if (!article || article.status !== 'published') notFound()

  const faqs: Array<{ q: string; a: string }> = JSON.parse(article.faqs || '[]')
  const breadcrumbs = [
    { name: 'MoonX', href: '/en/moonx/markets/trending' },
    { name: 'Meme Academy', href: '/en/moonx/learn/meme' },
    { name: article.h1 },
  ]
  const schemas = [
    articleSchema(article),
    ...(faqs.length > 0 ? [faqSchema(faqs)] : []),
    breadcrumbSchema(breadcrumbs.map(b => ({ name: b.name, url: `https://www.bydfi.com${b.href || ''}` }))),
  ]

  return (
    <>
      <SchemaScript schema={schemas} />
      <main className="max-w-3xl mx-auto px-4 py-8">
        <Breadcrumb items={breadcrumbs} />
        <h1 className="text-3xl font-bold mb-6 leading-tight">{article.h1}</h1>

        {article.coverImage && (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={article.coverImage} alt={article.h1} className="w-full rounded-2xl mb-8 object-cover" />
        )}

        <article className="prose prose-slate max-w-none mb-8">
          <ReactMarkdown>{article.content}</ReactMarkdown>
        </article>

        {faqs.length > 0 && (
          <section className="mb-8">
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
        )}

        {article.ctaText && article.ctaUrl && (
          <div className="rounded-2xl bg-gradient-to-r from-orange-500 to-pink-600 p-6 text-white text-center">
            <p className="font-semibold mb-3">{article.ctaText}</p>
            <a href={article.ctaUrl} className="inline-flex items-center gap-2 bg-white text-orange-600 font-medium px-5 py-2.5 rounded-lg hover:bg-orange-50 transition-colors">
              Trade on MoonX <ExternalLink className="w-4 h-4" />
            </a>
          </div>
        )}
      </main>
    </>
  )
}
