'use client'

import { Shield, ShieldAlert, ShieldCheck, ShieldX } from 'lucide-react'
import { useEffect, useState } from 'react'

interface RiskData {
  score: number
  risks: string[]
  verdict: 'SAFE' | 'MODERATE' | 'RISKY' | 'RUGGED'
}

interface SecurityScoreProps {
  contract: string
  initial?: RiskData
}

const VERDICT_CONFIG = {
  SAFE: { icon: ShieldCheck, color: 'text-green-500', bg: 'bg-green-50', label: 'Safe' },
  MODERATE: { icon: Shield, color: 'text-yellow-500', bg: 'bg-yellow-50', label: 'Moderate Risk' },
  RISKY: { icon: ShieldAlert, color: 'text-orange-500', bg: 'bg-orange-50', label: 'High Risk' },
  RUGGED: { icon: ShieldX, color: 'text-red-600', bg: 'bg-red-50', label: 'RUGGED' },
}

export function SecurityScore({ contract, initial }: SecurityScoreProps) {
  const [data, setData] = useState<RiskData | null>(initial || null)

  useEffect(() => {
    if (initial) return
    fetch(`/api/tokens/${contract}?fields=risk`)
      .then(r => r.ok ? r.json() : null)
      .then(d => d?.risk && setData(d.risk))
      .catch(() => {})
  }, [contract, initial])

  if (!data) {
    return (
      <div className="rounded-xl border p-4 animate-pulse">
        <div className="h-4 bg-muted rounded w-1/2 mb-2" />
        <div className="h-3 bg-muted rounded w-3/4" />
      </div>
    )
  }

  const config = VERDICT_CONFIG[data.verdict] || VERDICT_CONFIG.MODERATE
  const Icon = config.icon

  return (
    <div className={`rounded-xl border p-4 ${config.bg}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Icon className={`w-4 h-4 ${config.color}`} />
          <span className="font-semibold text-sm">Security Score</span>
        </div>
        <span className={`font-bold text-lg ${config.color}`}>{data.score}/100</span>
      </div>

      {/* Score bar */}
      <div className="h-2 bg-white/60 rounded-full overflow-hidden mb-2">
        <div
          className={`h-full rounded-full transition-all duration-700 ${
            data.score > 70 ? 'bg-green-500' : data.score > 40 ? 'bg-yellow-500' : 'bg-red-500'
          }`}
          style={{ width: `${data.score}%` }}
        />
      </div>

      <p className={`text-xs font-medium ${config.color} mb-1`}>{config.label}</p>

      {data.risks.length > 0 && (
        <ul className="text-xs text-muted-foreground space-y-0.5">
          {data.risks.slice(0, 3).map((r, i) => (
            <li key={i} className="flex items-start gap-1">
              <span className="text-red-400 mt-0.5">•</span>
              <span>{r}</span>
            </li>
          ))}
        </ul>
      )}

      <p className="text-[10px] text-muted-foreground mt-2">Via RugCheck API</p>
    </div>
  )
}
