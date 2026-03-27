'use client'

import { useState } from 'react'
import { Calculator } from 'lucide-react'

interface ProfitCalculatorProps {
  currentPrice: number
  symbol: string
}

export function ProfitCalculator({ currentPrice, symbol }: ProfitCalculatorProps) {
  const [investment, setInvestment] = useState('100')
  const [targetPrice, setTargetPrice] = useState('')
  const [buyPrice, setBuyPrice] = useState(currentPrice.toExponential(4))

  const investAmt = parseFloat(investment) || 0
  const buyPriceAmt = parseFloat(buyPrice) || currentPrice
  const targetPriceAmt = parseFloat(targetPrice) || 0

  const tokensOwned = buyPriceAmt > 0 ? investAmt / buyPriceAmt : 0
  const currentValue = tokensOwned * currentPrice
  const targetValue = targetPriceAmt > 0 ? tokensOwned * targetPriceAmt : 0
  const targetPnl = targetValue - investAmt
  const targetPnlPct = investAmt > 0 ? (targetPnl / investAmt) * 100 : 0
  const currentPnl = currentValue - investAmt
  const currentPnlPct = investAmt > 0 ? (currentPnl / investAmt) * 100 : 0

  return (
    <div className="rounded-xl border p-4">
      <div className="flex items-center gap-2 mb-3">
        <Calculator className="w-4 h-4 text-blue-500" />
        <span className="font-semibold text-sm">P&L Calculator</span>
      </div>

      <div className="space-y-2">
        <div>
          <label className="text-xs text-muted-foreground">Investment (USD)</label>
          <input
            type="number"
            value={investment}
            onChange={e => setInvestment(e.target.value)}
            className="w-full border rounded-lg px-3 py-1.5 text-sm mt-0.5 focus:outline-none focus:ring-1 focus:ring-blue-500"
            placeholder="100"
          />
        </div>
        <div>
          <label className="text-xs text-muted-foreground">Buy Price (USD)</label>
          <input
            type="text"
            value={buyPrice}
            onChange={e => setBuyPrice(e.target.value)}
            className="w-full border rounded-lg px-3 py-1.5 text-sm mt-0.5 font-mono focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>
        <div>
          <label className="text-xs text-muted-foreground">Target Price (USD)</label>
          <input
            type="text"
            value={targetPrice}
            onChange={e => setTargetPrice(e.target.value)}
            className="w-full border rounded-lg px-3 py-1.5 text-sm mt-0.5 font-mono focus:outline-none focus:ring-1 focus:ring-blue-500"
            placeholder="Enter target price"
          />
        </div>
      </div>

      <div className="mt-3 pt-3 border-t space-y-1.5">
        <div className="flex justify-between text-xs">
          <span className="text-muted-foreground">Tokens owned</span>
          <span className="font-mono">{tokensOwned > 0 ? tokensOwned.toLocaleString(undefined, { maximumFractionDigits: 0 }) : '—'} {symbol}</span>
        </div>
        <div className="flex justify-between text-xs">
          <span className="text-muted-foreground">Current value</span>
          <span className={`font-mono font-medium ${currentPnl >= 0 ? 'text-green-600' : 'text-red-500'}`}>
            ${currentValue.toFixed(2)} ({currentPnlPct >= 0 ? '+' : ''}{currentPnlPct.toFixed(1)}%)
          </span>
        </div>
        {targetPriceAmt > 0 && (
          <div className="flex justify-between text-xs">
            <span className="text-muted-foreground">At target</span>
            <span className={`font-mono font-semibold ${targetPnl >= 0 ? 'text-green-600' : 'text-red-500'}`}>
              ${targetValue.toFixed(2)} ({targetPnlPct >= 0 ? '+' : ''}{targetPnlPct.toFixed(1)}%)
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
