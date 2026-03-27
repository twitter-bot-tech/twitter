import prisma from '@/lib/prisma'
import Link from 'next/link'
import { Plus, Edit, Eye } from 'lucide-react'

const STATUS_COLORS: Record<string, string> = {
  published: 'bg-green-100 text-green-700',
  in_review: 'bg-yellow-100 text-yellow-700',
  reviewing: 'bg-blue-100 text-blue-700',
  draft: 'bg-gray-100 text-gray-600',
  rejected: 'bg-red-100 text-red-700',
  scheduled: 'bg-purple-100 text-purple-700',
}

export default async function ArticlesListPage({
  searchParams,
}: {
  searchParams: { status?: string; category?: string }
}) {
  const where: Record<string, string> = {}
  if (searchParams.status) where.status = searchParams.status
  if (searchParams.category) where.category = searchParams.category

  const articles = await prisma.article.findMany({
    where,
    orderBy: { updatedAt: 'desc' },
    select: { id: true, h1: true, slug: true, status: true, category: true, author: true, updatedAt: true, publishedAt: true },
  })

  const statuses = ['draft', 'in_review', 'reviewing', 'published', 'rejected', 'scheduled']

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-bold">Articles</h1>
        <Link href="/admin/articles/new" className="flex items-center gap-2 bg-blue-600 text-white text-sm px-3 py-1.5 rounded-lg hover:bg-blue-700 transition-colors">
          <Plus className="w-4 h-4" /> New
        </Link>
      </div>

      {/* Filters */}
      <div className="flex gap-2 flex-wrap mb-4">
        <Link href="/admin/articles" className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${!searchParams.status ? 'bg-blue-600 text-white border-blue-600' : 'hover:bg-accent'}`}>
          All ({articles.length})
        </Link>
        {statuses.map(s => (
          <Link key={s} href={`/admin/articles?status=${s}`}
            className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${searchParams.status === s ? 'bg-blue-600 text-white border-blue-600' : 'hover:bg-accent'}`}>
            {s}
          </Link>
        ))}
        <div className="ml-auto flex gap-2">
          <Link href="/admin/articles?category=prediction" className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${searchParams.category === 'prediction' ? 'bg-purple-600 text-white' : 'hover:bg-accent'}`}>
            Prediction
          </Link>
          <Link href="/admin/articles?category=meme" className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${searchParams.category === 'meme' ? 'bg-orange-500 text-white' : 'hover:bg-accent'}`}>
            Meme
          </Link>
        </div>
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl border overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-gray-50">
              <th className="text-left p-3 text-xs text-muted-foreground font-medium">Title</th>
              <th className="text-left p-3 text-xs text-muted-foreground font-medium">Category</th>
              <th className="text-left p-3 text-xs text-muted-foreground font-medium">Status</th>
              <th className="text-left p-3 text-xs text-muted-foreground font-medium">Updated</th>
              <th className="text-left p-3 text-xs text-muted-foreground font-medium">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {articles.map(article => (
              <tr key={article.id} className="hover:bg-gray-50 transition-colors">
                <td className="p-3">
                  <p className="font-medium line-clamp-1">{article.h1}</p>
                  <p className="text-xs text-muted-foreground">/{article.slug}</p>
                </td>
                <td className="p-3">
                  <span className={`text-[10px] px-1.5 py-0.5 rounded ${article.category === 'prediction' ? 'bg-purple-50 text-purple-700' : 'bg-orange-50 text-orange-700'}`}>
                    {article.category}
                  </span>
                </td>
                <td className="p-3">
                  <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${STATUS_COLORS[article.status] || ''}`}>
                    {article.status}
                  </span>
                </td>
                <td className="p-3 text-xs text-muted-foreground">{new Date(article.updatedAt).toLocaleDateString()}</td>
                <td className="p-3">
                  <div className="flex items-center gap-2">
                    <Link href={`/admin/articles/${article.id}`} className="p-1 hover:bg-accent rounded transition-colors">
                      <Edit className="w-3.5 h-3.5" />
                    </Link>
                    {article.status === 'published' && (
                      <Link href={`/en/moonx/learn/${article.category}/${article.slug}`} target="_blank"
                        className="p-1 hover:bg-accent rounded transition-colors text-green-600">
                        <Eye className="w-3.5 h-3.5" />
                      </Link>
                    )}
                  </div>
                </td>
              </tr>
            ))}
            {articles.length === 0 && (
              <tr><td colSpan={5} className="p-8 text-center text-muted-foreground text-sm">No articles found.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
