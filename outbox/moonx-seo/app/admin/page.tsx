import prisma from '@/lib/prisma'
import Link from 'next/link'
import { FileText, CheckCircle, Clock, AlertCircle, Plus } from 'lucide-react'

export default async function AdminDashboard() {
  const [total, published, inReview, drafts] = await Promise.all([
    prisma.article.count(),
    prisma.article.count({ where: { status: 'published' } }),
    prisma.article.count({ where: { status: { in: ['in_review', 'reviewing'] } } }),
    prisma.article.count({ where: { status: 'draft' } }),
  ])

  const recent = await prisma.article.findMany({
    orderBy: { updatedAt: 'desc' },
    take: 5,
    select: { id: true, h1: true, status: true, category: true, updatedAt: true },
  })

  const stats = [
    { label: 'Total Articles', value: total, icon: FileText, color: 'text-blue-500' },
    { label: 'Published', value: published, icon: CheckCircle, color: 'text-green-500' },
    { label: 'In Review', value: inReview, icon: Clock, color: 'text-yellow-500' },
    { label: 'Drafts', value: drafts, icon: AlertCircle, color: 'text-gray-400' },
  ]

  const statusColors: Record<string, string> = {
    published: 'bg-green-100 text-green-700',
    in_review: 'bg-yellow-100 text-yellow-700',
    reviewing: 'bg-blue-100 text-blue-700',
    draft: 'bg-gray-100 text-gray-600',
    rejected: 'bg-red-100 text-red-700',
    scheduled: 'bg-purple-100 text-purple-700',
  }

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-bold">Dashboard</h1>
        <Link href="/admin/articles/new" className="flex items-center gap-2 bg-blue-600 text-white text-sm px-3 py-1.5 rounded-lg hover:bg-blue-700 transition-colors">
          <Plus className="w-4 h-4" /> New Article
        </Link>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {stats.map(stat => {
          const Icon = stat.icon
          return (
            <div key={stat.label} className="bg-white rounded-xl border p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-muted-foreground">{stat.label}</span>
                <Icon className={`w-4 h-4 ${stat.color}`} />
              </div>
              <p className="text-2xl font-bold">{stat.value}</p>
            </div>
          )
        })}
      </div>

      {/* Recent articles */}
      <div className="bg-white rounded-xl border">
        <div className="p-4 border-b flex items-center justify-between">
          <h2 className="font-semibold text-sm">Recent Articles</h2>
          <Link href="/admin/articles" className="text-xs text-blue-600 hover:underline">View all →</Link>
        </div>
        <div className="divide-y">
          {recent.map(article => (
            <Link key={article.id} href={`/admin/articles/${article.id}`} className="flex items-center gap-3 p-4 hover:bg-gray-50 transition-colors">
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">{article.h1}</p>
                <p className="text-xs text-muted-foreground">
                  {article.category} · {new Date(article.updatedAt).toLocaleDateString()}
                </p>
              </div>
              <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full ${statusColors[article.status] || statusColors.draft}`}>
                {article.status}
              </span>
            </Link>
          ))}
          {recent.length === 0 && (
            <div className="p-8 text-center text-sm text-muted-foreground">
              No articles yet. <Link href="/admin/articles/new" className="text-blue-600 hover:underline">Create one →</Link>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
