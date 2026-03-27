'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useState } from 'react'
import { ChevronDown } from 'lucide-react'

const NAV_ITEMS = [
  { label: 'Exchange', href: 'https://www.bydfi.com/en/trade', external: true },
  { label: '🔥 MoonX', href: 'https://www.bydfi.com/en/moonx/pump', external: true },
  { label: 'Pump', href: 'https://www.bydfi.com/en/moonx/pump', external: true },
  { label: 'Markets', href: 'https://www.bydfi.com/en/moonx/markets', external: true },
  { label: 'Trade', href: 'https://www.bydfi.com/en/moonx/trade', external: true },
  { label: 'Copy Trade', href: 'https://www.bydfi.com/en/moonx/copy', external: true },
  { label: 'Monitor', href: 'https://www.bydfi.com/en/moonx/monitor', external: true },
  { label: 'Portfolio', href: 'https://www.bydfi.com/en/moonx/portfolio', external: true },
  { label: '🎰 Lucky Draw', href: 'https://www.bydfi.com/en/moonx/lucky', external: true },
  {
    label: 'Learn',
    href: '/en/moonx/markets/trending',
    children: [
      { label: 'Trending Markets', href: '/en/moonx/markets/trending', desc: 'Live prediction market odds' },
      { label: 'Prediction Academy', href: '/en/moonx/learn/prediction', desc: 'How prediction markets work' },
      { label: 'Meme Academy', href: '/en/moonx/learn/meme', desc: 'Research & buy meme coins' },
    ],
  },
]

export function SiteNav() {
  const pathname = usePathname()
  const [learnOpen, setLearnOpen] = useState(false)

  const isLearnActive = pathname.startsWith('/en/moonx/learn') || pathname.startsWith('/en/moonx/markets')

  return (
    <nav className="sticky top-0 z-50 bg-[#0d0e12] border-b border-white/[0.06]">
      <div className="max-w-[1440px] mx-auto px-4 flex items-center h-12 gap-1">
        {/* Logo */}
        <Link href="https://www.bydfi.com" className="flex items-center gap-1.5 mr-4 shrink-0">
          <span className="text-sm font-bold text-white">BYDFi</span>
          <span className="text-[10px] text-[#ffd30f] font-semibold bg-[#ffd30f]/10 px-1.5 py-0.5 rounded">MOONX</span>
        </Link>

        {/* Nav items (scrollable, no dropdown here) */}
        <div className="flex items-center gap-0.5 overflow-x-auto scrollbar-hide">
          {NAV_ITEMS.filter(item => !item.children).map((item) => (
            <a
              key={item.label}
              href={item.href}
              target={item.external ? '_blank' : undefined}
              rel={item.external ? 'noopener noreferrer' : undefined}
              className="px-3 py-1.5 rounded text-xs font-medium whitespace-nowrap text-white/50 hover:text-white/80 hover:bg-white/[0.04] transition-colors"
            >
              {item.label}
            </a>
          ))}
        </div>

        {/* Learn dropdown — outside overflow container so it's not clipped */}
        <div className="relative shrink-0" onMouseEnter={() => setLearnOpen(true)} onMouseLeave={() => setLearnOpen(false)}>
          <button
            className={`flex items-center gap-1 px-3 py-1.5 rounded text-xs font-medium whitespace-nowrap transition-colors ${
              isLearnActive
                ? 'text-[#ffd30f] bg-[#ffd30f]/10'
                : 'text-white/70 hover:text-white hover:bg-white/[0.06]'
            }`}
          >
            Learn
            <ChevronDown className={`w-3 h-3 transition-transform ${learnOpen ? 'rotate-180' : ''}`} />
          </button>

          {learnOpen && (
            <div className="absolute top-full right-0 mt-1 w-52 bg-[#16171d] border border-white/[0.08] rounded-xl shadow-2xl py-1.5 z-50">
              {NAV_ITEMS.find(i => i.children)!.children!.map((child) => (
                <Link
                  key={child.href}
                  href={child.href}
                  className="flex flex-col px-4 py-2.5 hover:bg-white/[0.05] transition-colors group"
                >
                  <span className={`text-xs font-medium ${pathname === child.href ? 'text-[#ffd30f]' : 'text-white/90 group-hover:text-white'}`}>
                    {child.label}
                  </span>
                  <span className="text-[10px] text-white/40 mt-0.5">{child.desc}</span>
                </Link>
              ))}
            </div>
          )}
        </div>

        {/* Right side */}
        <div className="ml-auto flex items-center gap-2 shrink-0">
          <a href="https://www.bydfi.com/en/login" target="_blank" rel="noopener noreferrer"
            className="text-xs text-white/60 hover:text-white px-3 py-1.5 rounded hover:bg-white/[0.06] transition-colors">
            Log In
          </a>
          <a href="https://www.bydfi.com/en/register" target="_blank" rel="noopener noreferrer"
            className="text-xs bg-[#ffd30f] text-black font-semibold px-3 py-1.5 rounded-lg hover:bg-[#f5c800] transition-colors">
            Sign Up
          </a>
        </div>
      </div>
    </nav>
  )
}
