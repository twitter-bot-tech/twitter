'use client'

import { useEffect, useState } from 'react'
import { MessageCircle } from 'lucide-react'

type Sentiment = 'bullish' | 'bearish' | 'neutral' | 'hype'

interface SentimentData {
  sentiment: Sentiment
  score: number  // 0-100 bullish
  keywords: string[]
}

const SENTIMENT_CONFIG: Record<Sentiment, { label: string; color: string; bg: string; emoji: string }> = {
  bullish: { label: 'Bullish', color: 'text-green-700', bg: 'bg-green-100', emoji: '🐂' },
  bearish: { label: 'Bearish', color: 'text-red-700', bg: 'bg-red-100', emoji: '🐻' },
  neutral: { label: 'Neutral', color: 'text-gray-700', bg: 'bg-gray-100', emoji: '😐' },
  hype: { label: 'Extreme Hype', color: 'text-purple-700', bg: 'bg-purple-100', emoji: '🚀' },
}

function getMockSentiment(contract: string): SentimentData {
  const hash = contract.charCodeAt(0) + contract.charCodeAt(1)
  const score = (hash * 37) % 100
  const sentiment: Sentiment = score > 75 ? 'hype' : score > 55 ? 'bullish' : score > 40 ? 'neutral' : 'bearish'
  const keywords = ['moon', 'pump', 'lfg', 'wagmi', 'gm'].slice(0, 3)
  return { sentiment, score, keywords }
}

export function SentimentPill({ contract }: { contract: string }) {
  const [data, setData] = useState<SentimentData>(getMockSentiment(contract))
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const interval = setInterval(async () => {
      setLoading(true)
      try {
        const res = await fetch(`/api/tokens/${contract}?fields=sentiment`)
        if (res.ok) {
          const d = await res.json()
          if (d.sentiment) setData(d.sentiment)
        }
      } catch {
        // keep mock
      } finally {
        setLoading(false)
      }
    }, 300_000)  // 5 min refresh
    return () => clearInterval(interval)
  }, [contract])

  const config = SENTIMENT_CONFIG[data.sentiment]

  return (
    <div className="rounded-xl border p-4">
      <div className="flex items-center gap-2 mb-3">
        <MessageCircle className="w-4 h-4 text-muted-foreground" />
        <span className="font-semibold text-sm">Community Sentiment</span>
        {loading && <span className="text-[10px] text-muted-foreground animate-pulse">updating…</span>}
      </div>

      <div className="flex items-center gap-2 mb-3">
        <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-sm font-medium ${config.bg} ${config.color}`}>
          {config.emoji} {config.label}
        </span>
        <span className="text-xs text-muted-foreground">{data.score}% bullish</span>
      </div>

      {/* Sentiment bar */}
      <div className="flex gap-0.5 h-2 rounded-full overflow-hidden mb-2">
        <div className="bg-green-400" style={{ width: `${data.score}%` }} />
        <div className="bg-red-400 flex-1" />
      </div>

      <div className="flex flex-wrap gap-1 mt-2">
        {data.keywords.map((kw, i) => (
          <span key={i} className="text-[10px] px-1.5 py-0.5 bg-muted rounded-md text-muted-foreground">
            #{kw}
          </span>
        ))}
      </div>
    </div>
  )
}
