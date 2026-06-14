import { TRAINING_GROUPS } from './registry'
import { colorForTrainingCategory, iconForTrainingCategory } from './theme'

export const TRAINING_DRAG_MIME = 'application/mlforge-training'

export default function TrainingPalette() {
  return (
    <aside className="h-full overflow-y-auto p-2 text-sm">
      <div className="mb-2 text-xs uppercase tracking-wide text-[#7a8088]">Training</div>
      <div className="mb-3 text-[10px] text-[#7a8088]">drag onto canvas</div>
      {TRAINING_GROUPS.map((group) => {
        const color = colorForTrainingCategory(group.name)
        const icon = iconForTrainingCategory(group.name)
        return (
          <div key={group.name} className="mb-3">
            <div className="mb-1 flex items-center gap-1.5 text-xs font-medium" style={{ color }}>
              <span
                className="inline-flex h-4 w-4 items-center justify-center rounded text-[11px] leading-none"
                style={{ background: `${color}33`, color }}
              >
                {icon}
              </span>
              <span>{group.name}</span>
            </div>
            <div className="flex flex-col gap-1">
              {group.nodes.map((node) => (
                <div
                  key={node}
                  className="cursor-grab rounded border border-[#1f2429] bg-[#13171b] px-2 py-1 text-xs hover:border-[#3a4148] hover:bg-[#181d22] active:cursor-grabbing"
                  draggable
                  onDragStart={(e) => {
                    e.dataTransfer.setData(TRAINING_DRAG_MIME, node)
                    e.dataTransfer.effectAllowed = 'copy'
                  }}
                >
                  {node}
                </div>
              ))}
            </div>
          </div>
        )
      })}
    </aside>
  )
}
