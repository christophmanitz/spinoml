import { useViewModeStore, type ViewMode } from './viewMode'

const TABS: { mode: ViewMode; label: string }[] = [
  { mode: 'architecture', label: 'Architektur' },
  { mode: 'training', label: 'Training' },
]

export default function ModeToggle() {
  const mode = useViewModeStore((s) => s.mode)
  const setMode = useViewModeStore((s) => s.setMode)
  return (
    <div className="flex items-center rounded border border-[#1f2429] bg-[#0e1115] p-0.5 text-[11px]">
      {TABS.map((t) => (
        <button
          key={t.mode}
          onClick={() => setMode(t.mode)}
          className={`rounded px-2 py-0.5 ${
            mode === t.mode ? 'bg-[#13344f] text-[#6ab7ff]' : 'text-[#7a8088] hover:text-[#cfd3d8]'
          }`}
        >
          {t.label}
        </button>
      ))}
    </div>
  )
}
