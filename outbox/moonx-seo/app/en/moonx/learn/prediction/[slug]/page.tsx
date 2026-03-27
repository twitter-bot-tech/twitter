import { Metadata } from 'next'
import { notFound } from 'next/navigation'
import ReactMarkdown from 'react-markdown'
import prisma from '@/lib/prisma'
import { articleMeta, articleSchema, faqSchema, breadcrumbSchema } from '@/lib/seo'
import { SchemaScript } from '@/components/seo/SchemaScript'
import { Breadcrumb } from '@/components/seo/Breadcrumb'
import { Calendar, User, Tag, ExternalLink } from 'lucide-react'

interface Props {
  params: { slug: string }
}

export const revalidate = 3600

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const article = await prisma.article.findUnique({ where: { slug: params.slug } })
  if (!article) return {}
  return articleMeta(article)
}

export async function generateStaticParams() {
  const articles = await prisma.article.findMany({
    where: { category: 'prediction', status: 'published' },
    select: { slug: true },
  })
  return articles.map(a => ({ slug: a.slug }))
}

export default async function PredictionArticlePage({ params }: Props) {
  const article = await prisma.article.findUnique({ where: { slug: params.slug } })
  if (!article || article.status !== 'published') notFound()

  const faqs: Array<{ q: string; a: string }> = JSON.parse(article.faqs || '[]')
  const tags: string[] = JSON.parse(article.tags || '[]')

  const breadcrumbs = [
    { name: 'MoonX', href: '/en/moonx/markets/trending' },
    { name: 'Prediction Academy', href: '/en/moonx/learn/prediction' },
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

        {/* Header */}
        <header className="mb-8">
          <div className="flex flex-wrap gap-2 mb-3">
            {tags.map(tag => (
              <span key={tag} className="text-xs text-blue-600 bg-blue-50 px-2 py-0.5 rounded-full flex items-center gap-1">
                <Tag className="w-2.5 h-2.5" />{tag}
              </span>
            ))}
          </div>
          <h1 className="text-3xl font-bold mb-4 leading-tight">{article.h1}</h1>
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <span className="flex items-center gap-1.5"><User className="w-4 h-4" />{article.author}</span>
            {article.publishedAt && (
              <span className="flex items-center gap-1.5">
                <Calendar className="w-4 h-4" />
                {new Date(article.publishedAt).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
              </span>
            )}
          </div>
        </header>

        {/* Cover image */}
        {article.coverImage && (
          <div className="mb-8 rounded-2xl overflow-hidden">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={article.coverImage} alt={article.h1} className="w-full object-cover" />
          </div>
        )}

        {/* Content */}
        <article className="prose prose-slate max-w-none mb-8">
          <ReactMarkdown>{article.content}</ReactMarkdown>
        </article>

        {/* FAQ Section */}
        {faqs.length > 0 && (
          <section className="mb-8">
            <h2 className="text-xl font-bold mb-4">Frequently Asked Questions</h2>
            <div className="space-y-4">
              {faqs.map((faq, i) => (
                <details key={i} className="rounded-xl border p-4 group">
                  <summary className="font-medium cursor-pointer list-none flex justify-between items-center">
                    {faq.q}
                    <span className="text-muted-foreground text-lg group-open:rotate-180 transition-transform">+</span>
                  </summary>
                  <p className="mt-3 text-sm text-muted-foreground leading-relaxed">{faq.a}</p>
                </details>
              ))}
            </div>
          </section>
        )}

        {/* CTA */}
        {article.ctaText && article.ctaUrl && (
          <div className="rounded-2xl bg-gradient-to-r from-blue-600 to-purple-600 p-6 text-white text-center">
            <p className="font-semibold mb-3">{article.ctaText}</p>
            <a
              href={article.ctaUrl}
              className="inline-flex items-center gap-2 bg-white text-blue-600 font-medium px-5 py-2.5 rounded-lg hover:bg-blue-50 transition-colors"
            >
              Explore MoonX <ExternalLink className="w-4 h-4" />
            </a>
          </div>
        )}
      </main>
    </>
  )
}
