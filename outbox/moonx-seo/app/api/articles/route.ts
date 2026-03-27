import { NextRequest, NextResponse } from 'next/server'
import prisma from '@/lib/prisma'

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url)
  const status = searchParams.get('status')
  const category = searchParams.get('category')

  const where: Record<string, string> = {}
  if (status) where.status = status
  if (category) where.category = category

  const articles = await prisma.article.findMany({
    where,
    orderBy: { updatedAt: 'desc' },
  })
  return NextResponse.json(articles)
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const { category, slug, h1, seoTitle, metaDesc, content, author, tags, faqs, ctaText, ctaUrl, coverImage } = body

    if (!slug || !h1 || !content) {
      return NextResponse.json({ error: 'slug, h1, and content are required' }, { status: 400 })
    }

    const article = await prisma.article.create({
      data: {
        category: category || 'prediction',
        slug,
        h1,
        seoTitle: seoTitle || h1,
        metaDesc: metaDesc || h1,
        content,
        author: author || 'MoonX Team',
        tags: typeof tags === 'string' ? tags : JSON.stringify(tags || []),
        faqs: typeof faqs === 'string' ? faqs : JSON.stringify(faqs || []),
        ctaText: ctaText || null,
        ctaUrl: ctaUrl || null,
        coverImage: coverImage || null,
        status: 'draft',
      },
    })
    return NextResponse.json(article, { status: 201 })
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown error'
    return NextResponse.json({ error: message }, { status: 400 })
  }
}
