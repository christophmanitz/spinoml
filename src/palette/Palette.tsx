import { LAYER_GROUPS } from '../layers/registry'
import { iconForCategory, colorForCategory } from '../layers/categories'

export default function Palette() {
  return (
    <aside className="h-full overflow-y-auto p-2 text-sm">
      <div className="mb-2 text-xs uppercase tracking-wide text-[#7a8088]">Layers</div>
      <div className="mb-3 text-[10px] text-[#7a8088]">drag onto canvas</div>
      {LAYER_GROUPS.map((group) => {
        const color = colorForCategory(group.name)
        const icon = iconForCategory(group.name)
        return (
          <div key={group.name} className="mb-3">
            <div className="mb-1 flex items-center gap-1.5 text-xs font-medium" style={{ color }}>
              <span
                aria-hidden
                className="inline-flex h-4 w-4 items-center justify-center rounded text-[11px] leading-none"
                style={{ background: `${color}22`, color }}
              >
                {icon}
              </span>
              <span>{group.name}</span>
            </div>
            <div className="flex flex-col gap-1">
              {group.layers.map((layer) => (
                <div
                  key={layer}
                  className="cursor-grab rounded border border-[#1f2429] bg-[#13171b] px-2 py-1 text-xs hover:border-[#3a4148] hover:bg-[#181d22] active:cursor-grabbing"
                  draggable
                  onDragStart={(e) => {
                    e.dataTransfer.setData('application/mlforge-layer', layer)
                    e.dataTransfer.effectAllowed = 'copy'
                  }}
                >
                  {layer}
                </div>
              ))}
            </div>
          </div>
        )
      })}
    </aside>
  )
}
