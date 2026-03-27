import prisma from '@/lib/prisma'
import { Plus, Eye, EyeOff } from 'lucide-react'

export default async function BannersPage() {
  const banners = await prisma.banner.findMany({ orderBy: { id: 'asc' } })

  const POSITION_LABELS: Record<string, string> = {
    sidebar_top: 'Sidebar Top',
    sidebar_bottom: 'Sidebar Bottom',
    in_article: 'In-Article',
  }

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-bold">Banners & CTAs</h1>
        <button className="flex items-center gap-2 bg-blue-600 text-white text-sm px-3 py-1.5 rounded-lg hover:bg-blue-700 transition-colors">
          <Plus className="w-4 h-4" /> Add Banner
        </button>
      </div>

      <div className="space-y-3">
        {banners.map(banner => {
          const pages: string[] = JSON.parse(banner.pages || '[]')
          const now = new Date()
          const isLive = banner.active &&
            (!banner.startsAt || new Date(banner.startsAt) <= now) &&
            (!banner.endsAt || new Date(banner.endsAt) >= now)

          return (
            <div key={banner.id} className={`bg-white rounded-xl border p-4 ${!isLive ? 'opacity-60' : ''}`}>
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <span className={`w-2 h-2 rounded-full ${isLive ? 'bg-green-500' : 'bg-gray-300'}`} />
                    <span className="text-xs font-medium">{POSITION_LABELS[banner.position] || banner.position}</span>
                    <span className="text-xs text-muted-foreground">→ {banner.linkUrl.slice(0, 40)}…</span>
                  </div>
                  <p className="text-sm font-medium mb-1">{banner.altText}</p>
                  <div className="flex flex-wrap gap-1 mb-2">
                    {pages.map((p, i) => (
                      <span key={i} className="text-[10px] bg-gray-100 px-1.5 py-0.5 rounded">{p}</span>
                    ))}
                  </div>
                  {(banner.startsAt || banner.endsAt) && (
                    <p className="text-xs text-muted-foreground">
                      {banner.startsAt ? `From ${new Date(banner.startsAt).toLocaleDateString()}` : ''}
                      {banner.endsAt ? ` Until ${new Date(banner.endsAt).toLocaleDateString()}` : ''}
                    </p>
                  )}
                </div>
                <div className="flex items-center gap-2 ml-4">
                  <button className="p-1.5 hover:bg-accent rounded transition-colors text-muted-foreground">
                    {isLive ? <Eye className="w-4 h-4 text-green-500" /> : <EyeOff className="w-4 h-4" />}
                  </button>
                </div>
              </div>
            </div>
          )
        })}
        {banners.length === 0 && (
          <div className="text-center py-12 text-muted-foreground text-sm">
            No banners configured.
          </div>
        )}
      </div>
    </div>
  )
}
