import { NextRequest, NextResponse } from 'next/server'
import prisma from '@/lib/prisma'

// Status machine transitions
const TRANSITIONS: Record<string, Record<string, string>> = {
  draft: { submit: 'in_review' },
  in_review: { approve: 'reviewing', reject: 'rejected' },
  reviewing: { publish: 'published', reject: 'rejected' },
  rejected: { submit: 'in_review' },
  published: {},
}

export async function PATCH(req: NextRequest, { params }: { params: { id: string } }) {
  try {
    const { action, note } = await req.json()
    const article = await prisma.article.findUnique({ where: { id: params.id } })
    if (!article) return NextResponse.json({ error: 'Not found' }, { status: 404 })

    const nextStatus = TRANSITIONS[article.status]?.[action]
    if (!nextStatus) {
      return NextResponse.json(
        { error: `Invalid transition: ${article.status} → ${action}` },
        { status: 400 }
      )
    }

    const updateData: Record<string, string | Date | null> = { status: nextStatus }
    if (nextStatus === 'published') updateData.publishedAt = new Date()
    if (action === 'reject') updateData.reviewNote = note || 'No reason provided'
    if (action === 'submit') updateData.reviewNote = null

    const updated = await prisma.article.update({
      where: { id: params.id },
      data: updateData,
    })

    return NextResponse.json({ status: updated.status, reviewNote: updated.reviewNote })
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown error'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
