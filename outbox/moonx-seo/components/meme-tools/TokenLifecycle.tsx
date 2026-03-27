'use client'

import { Clock } from 'lucide-react'
import { useEffect, useState } from 'react'

interface TokenLifecycleProps {
  launchedAt: Date | string
  volume24h: number
}

function getLifecycleStage(ageHours: number, volume: number): {
  label: string
  color: string
  emoji: string
  description: string
} {
  if (volume < 500 && ageHours > 48) {
    return { label: 'Inactive', color: 'text-gray-400', emoji: '💀', description: 'Very low activity' }
  }
  if (ageHours < 2) {
    return { label: 'Just Launched', color: 'text-green-500', emoji: '🚀', description: 'Brand new token' }
  }
  if (ageHours < 24) {
    return { label: 'Early', color: 'text-blue-500', emoji: '🌱', description: 'Under 24 hours old' }
  }
  if (ageHours < 72) {
    return { label: 'Trending', color: 'text-purple-500', emoji: '🔥', description: '1-3 days old' }
  }
  if (ageHours < 168) {
    return { label: 'Established', color: 'text-orange-500', emoji: '⭐', description: '3-7 days old' }
  }
  return { label: 'Veteran', color: 'text-yellow-600', emoji: '💎', description: 'Over 1 week old' }
}

function formatAge(hours: number): string {
  if (hours < 1) return `${Math.floor(hours * 60)}m`
  if (hours < 24) return `${Math.floor(hours)}h`
  return `${Math.floor(hours / 24)}d ${Math.floor(hours % 24)}h`
}

export function TokenLifecycle({ launchedAt, volume24h }: TokenLifecycleProps) {
  const [ageHours, setAgeHours] = useState(0)

  useEffect(() => {
    const launch = new Date(launchedAt)
    const calc = () => setAgeHours((Date.now() - launch.getTime()) / 3600000)
    calc()
    const interval = setInterval(calc, 60_000)
    return () => clearInterval(interval)
  }, [launchedAt])

  const stage = getLifecycleStage(ageHours, volume24h)

  return (
    <div className="rounded-xl border p-4">
      <div className="flex items-center gap-2 mb-3">
        <Clock className="w-4 h-4 text-muted-foreground" />
        <span className="font-semibold text-sm">Token Age</span>
      </div>

      <div className="flex items-center justify-between">
        <div>
          <p className="text-2xl font-bold">{formatAge(ageHours)}</p>
          <p className="text-xs text-muted-foreground">since launch</p>
        </div>
        <div className="text-right">
          <p className={`text-sm font-semibold ${stage.color}`}>
            {stage.emoji} {stage.label}
          </p>
          <p className="text-xs text-muted-foreground">{stage.description}</p>
        </div>
      </div>

      {/* Lifecycle progress bar */}
      <div className="mt-3 flex gap-1">
        {['Just Launched', 'Early', 'Trending', 'Established', 'Veteran'].map((s, i) => {
          const thresholds = [2, 24, 72, 168, Infinity]
          const active = ageHours < thresholds[i] && (i === 0 || ageHours >= thresholds[i - 1])
          const past = ageHours >= thresholds[i]
          return (
            <div
              key={s}
              className={`h-1.5 flex-1 rounded-full transition-all ${
                active ? 'bg-blue-500' : past ? 'bg-blue-200' : 'bg-muted'
              }`}
              title={s}
            />
          )
        })}
      </div>
    </div>
  )
}
