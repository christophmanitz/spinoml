import type { NodeActivation } from './client'
import { Heatmap, VectorBars } from './primitives'

// Compact, glanceable preview rendered inside a canvas node when Explain mode
// is on. Picks a representation from the activation preview kind.
export default function MiniViz({ act, hex }: { act: NodeActivation; hex: string }) {
  const p = act.preview
  const sparsity = act.stats.frac_zero

  return (
    <div className="flex flex-col gap-1">
      {p?.kind === 'maps' && (
        <div className="flex flex-wrap gap-0.5">
          {p.maps.slice(0, 6).map((g, i) => (
            <Heatmap key={i} grid={g} cell={3} gap={0} mode="mono" hex={hex} />
          ))}
          {p.channels > 6 && <span className="text-[8px] text-[#5b6168]">+{p.channels - 6}</span>}
        </div>
      )}
      {p?.kind === 'vector' && <VectorBars values={p.values.slice(0, 64)} width={140} height={26} hex={hex} />}
      {p?.kind === 'matrix' && <Heatmap grid={p.grid} cell={3} gap={0} mode="diverging" maxW={140} />}
      {p?.kind === 'tokens' && (() => {
        // Prefer interpretable ESPF substructure labels over raw token ids.
        const subs = p.labels?.filter((l) => l !== '<pad>').map((l) => (l === '<unk>' ? '∅' : l))
        const items = subs && subs.length ? subs : p.values.map(String)
        const cap = subs && subs.length ? 6 : 12
        return (
          <div className="font-mono text-[9px] text-[#9aa1a8]">
            {items.slice(0, cap).join(' ')}{items.length > cap ? ' …' : ''}
          </div>
        )
      })()}
      {p?.kind === 'scalar' && <div className="font-mono text-[10px] text-[#9aa1a8]">{p.value}</div>}

      {sparsity > 0.001 && (
        <div className="flex items-center gap-1" title={`${(sparsity * 100).toFixed(0)}% Nullen (Sparsity)`}>
          <div className="h-1 flex-1 overflow-hidden rounded bg-[#2a2f36]">
            <div className="h-full" style={{ width: `${sparsity * 100}%`, background: hex }} />
          </div>
          <span className="text-[8px] text-[#5b6168]">{(sparsity * 100).toFixed(0)}%</span>
        </div>
      )}
    </div>
  )
}
