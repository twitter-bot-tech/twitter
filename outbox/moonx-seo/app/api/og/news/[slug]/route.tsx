import { ImageResponse } from 'next/og'
import { NextRequest } from 'next/server'
import { NEWS_ITEMS } from '@/app/en/moonx/news/data'

export const runtime = 'edge'

const CATEGORY_COLORS: Record<string, [string, string]> = {
  'Bitcoin':          ['#f97316', '#eab308'],
  'Ethereum':         ['#3b82f6', '#6366f1'],
  'Solana':           ['#a855f7', '#ec4899'],
  'Altcoin':          ['#06b6d4', '#3b82f6'],
  'DeFi':             ['#ec4899', '#f43f5e'],
  'Crypto Headlines': ['#22c55e', '#14b8a6'],
}

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ slug: string }> }
) {
  const { slug } = await params
  const article = NEWS_ITEMS.find(n => n.slug === slug)

  if (!article) {
    // Fallback OG for unknown slugs
    return new ImageResponse(
      (
        <div
          style={{
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: '#0d0e12',
          }}
        >
          <span style={{ color: '#fff', fontSize: 48, fontWeight: 700 }}>MoonX News</span>
        </div>
      ),
      { width: 1200, height: 630 }
    )
  }

  const [colorFrom, colorTo] = CATEGORY_COLORS[article.category] ?? ['#f97316', '#eab308']

  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          background: '#0d0e12',
          padding: '60px',
          fontFamily: 'system-ui, sans-serif',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {/* Background accent blob */}
        <div
          style={{
            position: 'absolute',
            top: -80,
            right: -80,
            width: 400,
            height: 400,
            borderRadius: '50%',
            background: `radial-gradient(circle, ${colorFrom}22 0%, transparent 70%)`,
          }}
        />

        {/* Top bar: site + category */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 40 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div
              style={{
                background: `linear-gradient(135deg, ${colorFrom}, ${colorTo})`,
                borderRadius: 8,
                padding: '6px 14px',
                color: '#fff',
                fontSize: 14,
                fontWeight: 700,
              }}
            >
              MoonX
            </div>
            <span style={{ color: '#64748b', fontSize: 14 }}>Crypto News</span>
          </div>
          <div
            style={{
              background: `linear-gradient(135deg, ${colorFrom}, ${colorTo})`,
              color: '#fff',
              fontSize: 13,
              fontWeight: 700,
              padding: '4px 12px',
              borderRadius: 20,
            }}
          >
            {article.tag}
          </div>
        </div>

        {/* Category label */}
        <div
          style={{
            color: colorFrom,
            fontSize: 14,
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: 2,
            marginBottom: 16,
          }}
        >
          {article.category}
        </div>

        {/* Title */}
        <div
          style={{
            color: '#f1f5f9',
            fontSize: 42,
            fontWeight: 800,
            lineHeight: 1.2,
            flex: 1,
            maxWidth: 900,
          }}
        >
          {article.title}
        </div>

        {/* Bottom: source + date + reading time */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 24,
            paddingTop: 32,
            borderTop: '1px solid #1e293b',
            marginTop: 'auto',
          }}
        >
          <span style={{ color: colorFrom, fontSize: 14, fontWeight: 600 }}>{article.source}</span>
          <span style={{ color: '#475569', fontSize: 14 }}>
            {new Date(article.date).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}
          </span>
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ color: '#475569', fontSize: 13 }}>bydfi.com/moonx</span>
          </div>
        </div>
      </div>
    ),
    { width: 1200, height: 630 }
  )
}
