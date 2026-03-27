'use client'

import { useEffect, useState } from 'react'
import { useRouter, useParams } from 'next/navigation'
import { Save, Send, CheckCircle, XCircle, Globe, ArrowLeft } from 'lucide-react'
import Link from 'next/link'

interface ArticleForm {
  category: string
  slug: string
  h1: string
  seoTitle: string
  metaDesc: string
  content: string
  author: string
  tags: string
  faqs: string
  ctaText: string
  ctaUrl: string
  coverImage: string
  status: string
  reviewNote: string
}

const EMPTY: ArticleForm = {
  category: 'prediction', slug: '', h1: '', seoTitle: '', metaDesc: '',
  content: '', author: 'MoonX Team', tags: '[]', faqs: '[]',
  ctaText: 'Start trading prediction markets', ctaUrl: 'https://www.bydfi.com/en/moonx/markets/trending',
  coverImage: '', status: 'draft', reviewNote: '',
}

const STATUS_ACTIONS: Record<string, Array<{ label: string; action: string; icon: React.ElementType; color: string }>> = {
  draft: [{ label: 'Submit for Review', action: 'submit', icon: Send, color: 'bg-yellow-500 hover:bg-yellow-600' }],
  in_review: [
    { label: 'Approve', action: 'approve', icon: CheckCircle, color: 'bg-green-600 hover:bg-green-700' },
    { label: 'Reject', action: 'reject', icon: XCircle, color: 'bg-red-500 hover:bg-red-600' },
  ],
  reviewing: [
    { label: 'Publish Now', action: 'publish', icon: Globe, color: 'bg-blue-600 hover:bg-blue-700' },
    { label: 'Reject', action: 'reject', icon: XCircle, color: 'bg-red-500 hover:bg-red-600' },
  ],
  rejected: [{ label: 'Resubmit', action: 'submit', icon: Send, color: 'bg-yellow-500 hover:bg-yellow-600' }],
  published: [],
}

export default function ArticleEditPage() {
  const router = useRouter()
  const params = useParams()
  const id = params.id as string
  const isNew = id === 'new'

  const [form, setForm] = useState<ArticleForm>(EMPTY)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState('')

  useEffect(() => {
    if (!isNew) {
      fetch(`/api/articles/${id}`)
        .then(r => r.json())
        .then(data => {
          if (data) setForm({
            ...data,
            tags: typeof data.tags === 'string' ? data.tags : JSON.stringify(data.tags),
            faqs: typeof data.faqs === 'string' ? data.faqs : JSON.stringify(data.faqs),
            reviewNote: data.reviewNote || '',
            coverImage: data.coverImage || '',
          })
        })
    }
  }, [id, isNew])

  const handleChange = (field: keyof ArticleForm) => (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>
  ) => setForm(prev => ({ ...prev, [field]: e.target.value }))

  const save = async () => {
    setSaving(true)
    setMessage('')
    try {
      const method = isNew ? 'POST' : 'PUT'
      const url = isNew ? '/api/articles' : `/api/articles/${id}`
      const res = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      const data = await res.json()
      if (res.ok) {
        setMessage('Saved ✓')
        if (isNew) router.push(`/admin/articles/${data.id}`)
      } else {
        setMessage(`Error: ${data.error}`)
      }
    } finally {
      setSaving(false)
    }
  }

  const doStatusAction = async (action: string) => {
    const body: Record<string, string> = { action }
    if (action === 'reject') {
      const note = prompt('Rejection reason:')
      if (!note) return
      body.note = note
    }
    const res = await fetch(`/api/articles/${id}/status`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (res.ok) {
      const data = await res.json()
      setForm(prev => ({ ...prev, status: data.status, reviewNote: data.reviewNote || '' }))
      setMessage(`Status updated to: ${data.status}`)
    }
  }

  const actions = STATUS_ACTIONS[form.status] || []

  return (
    <div className="p-6 max-w-3xl">
      <div className="flex items-center gap-3 mb-6">
        <Link href="/admin/articles" className="p-1.5 hover:bg-accent rounded-lg transition-colors">
          <ArrowLeft className="w-4 h-4" />
        </Link>
        <h1 className="text-xl font-bold">{isNew ? 'New Article' : 'Edit Article'}</h1>
        <span className={`ml-auto text-xs font-medium px-2 py-0.5 rounded-full ${
          form.status === 'published' ? 'bg-green-100 text-green-700' :
          form.status === 'in_review' ? 'bg-yellow-100 text-yellow-700' :
          'bg-gray-100 text-gray-600'
        }`}>{form.status}</span>
      </div>

      {form.reviewNote && form.status === 'rejected' && (
        <div className="rounded-xl bg-red-50 border border-red-200 p-3 mb-4 text-sm text-red-700">
          <strong>Rejection note:</strong> {form.reviewNote}
        </div>
      )}

      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs font-medium text-muted-foreground">Category</label>
            <select value={form.category} onChange={handleChange('category')}
              className="w-full mt-1 border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500">
              <option value="prediction">Prediction</option>
              <option value="meme">Meme</option>
            </select>
          </div>
          <div>
            <label className="text-xs font-medium text-muted-foreground">Slug</label>
            <input value={form.slug} onChange={handleChange('slug')} placeholder="polymarket-vs-kalshi"
              className="w-full mt-1 border rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-blue-500" />
          </div>
        </div>

        <div>
          <label className="text-xs font-medium text-muted-foreground">H1 Title</label>
          <input value={form.h1} onChange={handleChange('h1')} placeholder="Polymarket vs Kalshi: Which Prediction Market is Better?"
            className="w-full mt-1 border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500" />
        </div>

        <div>
          <label className="text-xs font-medium text-muted-foreground">SEO Title (50-60 chars) — {form.seoTitle.length}/60</label>
          <input value={form.seoTitle} onChange={handleChange('seoTitle')}
            className="w-full mt-1 border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500" />
        </div>

        <div>
          <label className="text-xs font-medium text-muted-foreground">Meta Description (150-160 chars) — {form.metaDesc.length}/160</label>
          <textarea value={form.metaDesc} onChange={handleChange('metaDesc')} rows={2}
            className="w-full mt-1 border rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:ring-1 focus:ring-blue-500" />
        </div>

        <div>
          <label className="text-xs font-medium text-muted-foreground">Content (Markdown)</label>
          <textarea value={form.content} onChange={handleChange('content')} rows={16}
            className="w-full mt-1 border rounded-lg px-3 py-2 text-sm font-mono resize-y focus:outline-none focus:ring-1 focus:ring-blue-500" />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs font-medium text-muted-foreground">Author</label>
            <input value={form.author} onChange={handleChange('author')}
              className="w-full mt-1 border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500" />
          </div>
          <div>
            <label className="text-xs font-medium text-muted-foreground">Cover Image URL</label>
            <input value={form.coverImage} onChange={handleChange('coverImage')} placeholder="https://..."
              className="w-full mt-1 border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500" />
          </div>
        </div>

        <div>
          <label className="text-xs font-medium text-muted-foreground">Tags (JSON array)</label>
          <input value={form.tags} onChange={handleChange('tags')} placeholder='["polymarket","kalshi","prediction-market"]'
            className="w-full mt-1 border rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-blue-500" />
        </div>

        <div>
          <label className="text-xs font-medium text-muted-foreground">FAQs (JSON array of &#123;q, a&#125;)</label>
          <textarea value={form.faqs} onChange={handleChange('faqs')} rows={4}
            className="w-full mt-1 border rounded-lg px-3 py-2 text-sm font-mono resize-y focus:outline-none focus:ring-1 focus:ring-blue-500" />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs font-medium text-muted-foreground">CTA Text</label>
            <input value={form.ctaText} onChange={handleChange('ctaText')}
              className="w-full mt-1 border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500" />
          </div>
          <div>
            <label className="text-xs font-medium text-muted-foreground">CTA URL</label>
            <input value={form.ctaUrl} onChange={handleChange('ctaUrl')}
              className="w-full mt-1 border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500" />
          </div>
        </div>
      </div>

      {/* Action bar */}
      <div className="flex items-center gap-3 mt-6 pt-4 border-t">
        <button onClick={save} disabled={saving}
          className="flex items-center gap-2 bg-gray-800 text-white text-sm px-4 py-2 rounded-lg hover:bg-gray-900 disabled:opacity-50 transition-colors">
          <Save className="w-4 h-4" />
          {saving ? 'Saving…' : 'Save Draft'}
        </button>

        {!isNew && actions.map(({ label, action, icon: Icon, color }) => (
          <button key={action} onClick={() => doStatusAction(action)}
            className={`flex items-center gap-2 text-white text-sm px-4 py-2 rounded-lg transition-colors ${color}`}>
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}

        {message && <span className="text-sm text-green-600 ml-auto">{message}</span>}
      </div>
    </div>
  )
}
