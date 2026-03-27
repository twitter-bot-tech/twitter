import Link from 'next/link'
import { LayoutDashboard, FileText, LayoutTemplate, Image, Settings } from 'lucide-react'

const nav = [
  { href: '/admin', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/admin/articles', label: 'Articles', icon: FileText },
  { href: '/admin/templates', label: 'Templates', icon: LayoutTemplate },
  { href: '/admin/banners', label: 'Banners', icon: Image },
]

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className="w-56 bg-white border-r flex flex-col">
        <div className="p-4 border-b">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg moonx-gradient flex items-center justify-center text-white text-xs font-bold">M</div>
            <div>
              <p className="text-sm font-bold leading-none">MoonX CMS</p>
              <p className="text-[10px] text-muted-foreground">Content Manager</p>
            </div>
          </div>
        </div>

        <nav className="flex-1 p-2">
          {nav.map(item => {
            const Icon = item.icon
            return (
              <Link
                key={item.href}
                href={item.href}
                className="flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-muted-foreground hover:bg-accent hover:text-foreground transition-colors mb-0.5"
              >
                <Icon className="w-4 h-4" />
                {item.label}
              </Link>
            )
          })}
        </nav>

        <div className="p-4 border-t">
          <Link href="/en/moonx/markets/trending" className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground">
            <Settings className="w-3.5 h-3.5" />
            View Site
          </Link>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        {children}
      </main>
    </div>
  )
}
