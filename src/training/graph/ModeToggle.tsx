import { useViewModeStore, type ViewMode } from './viewMode'

const TABS: { mode: ViewMode; label: string }[] = [
  { mode: 'architecture', label: 'Architektur' },
  { mode: 'data', label: 'Data' },
  { mode: 'training', label: 'Training' },
]

export default function ModeToggle() {
  const mode = useViewModeStore((s) => s.mode)
  const setMode = useViewModeStore((s) => s.setMode)
  return (
    <div className="flex items-center rounded border border-[#1f2429] bg-[#0e1216] p-0.5 text-[11px]">
      {TABS.map((t) => (
        <button
          key={t.mode}
          onClick={() => setMode(t.mode)}
          className={`rounded px-2 py-0.5 ${
            mode === t.mode ? 'bg-[var(--accent-sel)] text-[var(--accent)]' : 'text-[#6f767e] hover:text-[#cfd3d8]'
          }`}
        >
          {t.label}
        </button>
      ))}
    </div>
  )
}
