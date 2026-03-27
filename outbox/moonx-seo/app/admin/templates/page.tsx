import prisma from '@/lib/prisma'
import { Edit, Info } from 'lucide-react'
import Link from 'next/link'

export default async function TemplatesPage() {
  const templates = await prisma.programmaticTemplate.findMany({
    orderBy: { updatedAt: 'desc' },
  })

  const PAGE_TYPE_LABELS: Record<string, string> = {
    event: 'Event Guide',
    stock: 'Stock Prediction',
    meme_token: 'Meme Token',
    trending: 'Trending List',
  }

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-xl font-bold mb-1">Programmatic Templates</h1>
          <p className="text-sm text-muted-foreground">Configure title/description/content templates for auto-generated pages.</p>
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 mb-6 flex items-start gap-3">
        <Info className="w-4 h-4 text-blue-500 flex-shrink-0 mt-0.5" />
        <div className="text-sm text-blue-700">
          <strong>Template variables:</strong> Use <code className="bg-blue-100 px-1 rounded">&#123;SYMBOL&#125;</code>, <code className="bg-blue-100 px-1 rounded">&#123;EVENT&#125;</code>, <code className="bg-blue-100 px-1 rounded">&#123;TICKER&#125;</code>, <code className="bg-blue-100 px-1 rounded">&#123;PRICE&#125;</code>, <code className="bg-blue-100 px-1 rounded">&#123;VOLUME&#125;</code> in templates. These are replaced at render time with live data.
        </div>
      </div>

      <div className="space-y-3">
        {templates.map(tpl => (
          <div key={tpl.id} className="bg-white rounded-xl border p-4">
            <div className="flex items-start justify-between mb-3">
              <div>
                <span className="text-[10px] font-bold text-purple-600 bg-purple-50 px-2 py-0.5 rounded-full uppercase tracking-wide">
                  {PAGE_TYPE_LABELS[tpl.pageType] || tpl.pageType}
                </span>
              </div>
              <div className="flex items-center gap-3 text-xs text-muted-foreground">
                <span>noindex after <strong>{tpl.noindexDays}d</strong></span>
                <span>min vol <strong>${tpl.volumeMin.toLocaleString()}</strong></span>
                <button className="p-1 hover:bg-accent rounded transition-colors">
                  <Edit className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
            <div className="space-y-2 text-sm">
              <div>
                <span className="text-xs text-muted-foreground font-medium">Title:</span>
                <p className="font-mono text-xs bg-gray-50 rounded px-2 py-1 mt-0.5">{tpl.titleTpl}</p>
              </div>
              <div>
                <span className="text-xs text-muted-foreground font-medium">Description:</span>
                <p className="text-xs bg-gray-50 rounded px-2 py-1 mt-0.5 line-clamp-2">{tpl.descTpl}</p>
              </div>
            </div>
            <p className="text-[10px] text-muted-foreground mt-2">Updated {new Date(tpl.updatedAt).toLocaleDateString()}</p>
          </div>
        ))}
        {templates.length === 0 && (
          <div className="text-center py-12 text-muted-foreground text-sm">
            No templates configured. Run <code className="bg-muted px-1 rounded">npm run db:seed</code> to add defaults.
          </div>
        )}
      </div>
    </div>
  )
}
