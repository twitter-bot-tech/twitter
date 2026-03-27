'use client'

import { useEffect, useState } from 'react'
import { TrendingUp, RefreshCw } from 'lucide-react'

interface Outcome {
  title: string
  probability: number
}

interface OddsWidgetProps {
  contract: string
  initialOdds?: Outcome[]
}

export function OddsWidget({ contract, initialOdds }: OddsWidgetProps) {
  const [odds, setOdds] = useState<Outcome[]>(initialOdds || [])
  const [loading, setLoading] = useState(!initialOdds)
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date())

  const fetchOdds = async () => {
    try {
      const res = await fetch(`/api/tokens/${contract}?fields=odds`)
      if (res.ok) {
        const data = await res.json()
        if (data.odds) {
          setOdds(data.odds)
          setLastUpdated(new Date())
        }
      }
    } catch {
      // keep existing
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!initialOdds) fetchOdds()
    const interval = setInterval(fetchOdds, 60_000)
    return () => clearInterval(interval)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [contract])

  if (!odds.length) {
    return (
      <div className="rounded-xl border p-4">
        <div className="flex items-center gap-2 mb-2">
          <TrendingUp className="w-4 h-4 text-blue-500" />
          <span className="font-semibold text-sm">Polymarket Odds</span>
        </div>
        <p className="text-xs text-muted-foreground">No linked prediction market found for this token.</p>
      </div>
    )
  }

  return (
    <div className="rounded-xl border p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-blue-500" />
          <span className="font-semibold text-sm">Polymarket Odds</span>
        </div>
        <button onClick={fetchOdds} className="text-muted-foreground hover:text-foreground">
          <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      <div className="space-y-2">
        {odds.map((o, i) => (
          <div key={i}>
            <div className="flex justify-between text-sm mb-1">
              <span>{o.title}</span>
              <span className="font-bold">{Math.round(o.probability * 100)}¢</span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${i === 0 ? 'bg-green-500' : 'bg-red-400'}`}
                style={{ width: `${o.probability * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      <p className="text-[10px] text-muted-foreground mt-2">
        Updated {lastUpdated.toLocaleTimeString()} · Powered by MoonX
      </p>
    </div>
  )
}
