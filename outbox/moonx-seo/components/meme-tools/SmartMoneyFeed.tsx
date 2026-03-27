'use client'

import { useEffect, useState } from 'react'
import { ArrowUpRight, ArrowDownLeft, Zap } from 'lucide-react'

interface SmartMoneyTx {
  wallet: string
  action: 'buy' | 'sell'
  amountUsd: number
  timestamp: Date | string
  pnlPct?: number
}

interface SmartMoneyFeedProps {
  contract: string
  initial?: SmartMoneyTx[]
}

function timeAgo(ts: Date | string): string {
  const diff = (Date.now() - new Date(ts).getTime()) / 1000
  if (diff < 60) return `${Math.floor(diff)}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

function formatUsd(n: number): string {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `$${(n / 1_000).toFixed(0)}K`
  return `$${n.toFixed(0)}`
}

export function SmartMoneyFeed({ contract, initial }: SmartMoneyFeedProps) {
  const [txs, setTxs] = useState<SmartMoneyTx[]>(initial || [])
  const [loading, setLoading] = useState(!initial)

  const fetchTxs = async () => {
    try {
      const res = await fetch(`/api/tokens/${contract}?fields=smartmoney`)
      if (res.ok) {
        const data = await res.json()
        if (data.smartmoney) setTxs(data.smartmoney)
      }
    } catch {
      // keep existing
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!initial) fetchTxs()
    const interval = setInterval(fetchTxs, 120_000)  // 2 min refresh
    return () => clearInterval(interval)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [contract])

  return (
    <div className="rounded-xl border p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4 text-yellow-500" />
          <span className="font-semibold text-sm">Smart Money Activity</span>
        </div>
        {loading && <span className="text-[10px] text-muted-foreground animate-pulse">loading…</span>}
      </div>

      {txs.length === 0 ? (
        <p className="text-xs text-muted-foreground">No smart money activity detected recently.</p>
      ) : (
        <div className="space-y-2">
          {txs.map((tx, i) => (
            <div key={i} className="flex items-center gap-2 text-xs">
              {tx.action === 'buy' ? (
                <ArrowUpRight className="w-3.5 h-3.5 text-green-500 flex-shrink-0" />
              ) : (
                <ArrowDownLeft className="w-3.5 h-3.5 text-red-500 flex-shrink-0" />
              )}
              <span className="font-mono text-muted-foreground">{tx.wallet}</span>
              <span className={`font-semibold ${tx.action === 'buy' ? 'text-green-600' : 'text-red-500'}`}>
                {tx.action === 'buy' ? 'BOT' : 'SOLD'} {formatUsd(tx.amountUsd)}
              </span>
              {tx.pnlPct !== undefined && (
                <span className={`text-[10px] px-1 py-0.5 rounded ${tx.pnlPct >= 0 ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                  {tx.pnlPct >= 0 ? '+' : ''}{tx.pnlPct.toFixed(0)}% PnL
                </span>
              )}
              <span className="text-muted-foreground ml-auto">{timeAgo(tx.timestamp)}</span>
            </div>
          ))}
        </div>
      )}

      <p className="text-[10px] text-muted-foreground mt-2">Smart money wallets tracked by MoonX · Refreshes every 2m</p>
    </div>
  )
}
