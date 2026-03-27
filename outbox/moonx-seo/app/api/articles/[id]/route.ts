import { NextRequest, NextResponse } from 'next/server'
import prisma from '@/lib/prisma'

export async function GET(_req: NextRequest, { params }: { params: { id: string } }) {
  const article = await prisma.article.findUnique({ where: { id: params.id } })
  if (!article) return NextResponse.json({ error: 'Not found' }, { status: 404 })
  return NextResponse.json(article)
}

export async function PUT(req: NextRequest, { params }: { params: { id: string } }) {
  try {
    const body = await req.json()
    const article = await prisma.article.update({
      where: { id: params.id },
      data: {
        category: body.category,
        slug: body.slug,
        h1: body.h1,
        seoTitle: body.seoTitle || body.h1,
        metaDesc: body.metaDesc,
        content: body.content,
        author: body.author,
        tags: typeof body.tags === 'string' ? body.tags : JSON.stringify(body.tags || []),
        faqs: typeof body.faqs === 'string' ? body.faqs : JSON.stringify(body.faqs || []),
        ctaText: body.ctaText || null,
        ctaUrl: body.ctaUrl || null,
        coverImage: body.coverImage || null,
      },
    })
    return NextResponse.json(article)
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown error'
    return NextResponse.json({ error: message }, { status: 400 })
  }
}

export async function DELETE(_req: NextRequest, { params }: { params: { id: string } }) {
  await prisma.article.delete({ where: { id: params.id } })
  return NextResponse.json({ ok: true })
}
