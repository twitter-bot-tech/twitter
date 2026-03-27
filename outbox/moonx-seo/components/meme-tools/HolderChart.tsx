'use client'

import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts'

interface HolderDistribution {
  label: string
  pct: number
  address?: string
}

interface HolderChartProps {
  holders: HolderDistribution[]
  totalHolders: number
}

const COLORS = ['#EF4444', '#F97316', '#EAB308', '#3B82F6', '#8B5CF6', '#10B981']

const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ name: string; value: number }> }) => {
  if (active && payload?.[0]) {
    return (
      <div className="bg-card border rounded-lg p-2 text-xs shadow-lg">
        <p className="font-medium">{payload[0].name}</p>
        <p className="text-muted-foreground">{payload[0].value.toFixed(1)}% of supply</p>
      </div>
    )
  }
  return null
}

export function HolderChart({ holders, totalHolders }: HolderChartProps) {
  const data = holders.map(h => ({ name: h.label, value: h.pct }))
  const top3Pct = holders.slice(0, 3).reduce((sum, h) => sum + h.pct, 0)

  return (
    <div className="rounded-xl border p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="font-semibold text-sm">Holder Distribution</span>
        <span className="text-xs text-muted-foreground">{totalHolders.toLocaleString()} total</span>
      </div>

      {/* Concentration warning */}
      {top3Pct > 30 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-2 mb-3 text-xs text-yellow-700">
          ⚠️ Top 3 wallets hold {top3Pct.toFixed(1)}% — concentrated supply
        </div>
      )}

      <ResponsiveContainer width="100%" height={180}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={45}
            outerRadius={70}
            paddingAngle={2}
            dataKey="value"
          >
            {data.map((_entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
          <Legend
            iconType="circle"
            iconSize={8}
            formatter={(value: string) => <span className="text-xs">{value}</span>}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}
