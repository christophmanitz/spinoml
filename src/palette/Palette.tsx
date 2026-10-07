import { useState } from 'react'
import { LAYER_GROUPS } from '../layers/registry'
import { colorForCategory } from '../layers/categories'
import CategoryIcon from '../layers/CategoryIcon'

const STORAGE_KEY = 'spinoml.palette.collapsed.v2'

function loadCollapsed(): Set<string> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    // No saved preference yet → start with every group collapsed.
    if (raw == null) return new Set(LAYER_GROUPS.map((g) => g.name))
    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) return new Set(LAYER_GROUPS.map((g) => g.name))
    return new Set(parsed.filter((s): s is string => typeof s === 'string'))
  } catch { return new Set(LAYER_GROUPS.map((g) => g.name)) }
}

export default function Palette() {
  const [collapsed, setCollapsed] = useState<Set<string>>(loadCollapsed)

  const toggle = (name: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify([...next])) }
      catch {
        // Private mode/quota: only cross-reload persistence of the collapse
        // state is lost; the live collapsed groups are still applied.
      }
      return next
    })
  }

  return (
    <aside className="h-full overflow-y-auto p-2 text-sm">
      <div className="mb-0.5 text-xs uppercase tracking-wide text-[#6f767e]">Layers</div>
      <div className="mb-2 text-[10px] text-[#6f767e]">auf den Canvas ziehen</div>
      {LAYER_GROUPS.map((group) => {
        const color = colorForCategory(group.name)
        const isOpen = !collapsed.has(group.name)
        return (
          <div key={group.name} className="mb-1.5">
            <button
              onClick={() => toggle(group.name)}
              className="flex w-full items-center gap-1.5 rounded px-0.5 py-0.5 text-xs font-medium hover:bg-[#13171b]"
              style={{ color }}
            >
              <span className="inline-block w-3 text-center text-[10px] opacity-70">
                {isOpen ? '▾' : '▸'}
              </span>
              <span
                aria-hidden
                className="inline-flex h-4 w-4 items-center justify-center rounded"
                style={{ background: `${color}22`, color }}
              >
                <CategoryIcon cat={group.name} />
              </span>
              <span className="flex-1 text-left">{group.name}</span>
              <span className="rounded bg-[#1a1e22] px-1 text-[9px] text-[#6f767e]">{group.layers.length}</span>
            </button>
            {isOpen && (
              <div className="mt-1 flex flex-col gap-1 pl-[18px]">
                {group.layers.map((layer) => (
                  <div
                    key={layer}
                    className="cursor-grab rounded border border-[#1f2429] bg-[#13171b] px-2 py-1 text-xs hover:border-[#3a4148] hover:bg-[#181d22] active:cursor-grabbing"
                    draggable
                    onDragStart={(e) => {
                      e.dataTransfer.setData('application/spinoml-layer', layer)
                      e.dataTransfer.effectAllowed = 'copy'
                    }}
                  >
                    {layer}
                  </div>
                ))}
              </div>
            )}
          </div>
        )
      })}
    </aside>
  )
}
