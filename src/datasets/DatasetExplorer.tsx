import { useEffect } from 'react'
import { isTauri } from '../workspace/tauri-fs'
import { useDatasetsStore } from './store'
import { iconFor, colorFor, guessKindFromName, formatSize } from './icons'

export default function DatasetExplorer() {
  const entries = useDatasetsStore((s) => s.entries)
  const loading = useDatasetsStore((s) => s.listLoading)
  const error = useDatasetsStore((s) => s.listError)
  const selectedRel = useDatasetsStore((s) => s.selectedRel)
  const refresh = useDatasetsStore((s) => s.refresh)
  const select = useDatasetsStore((s) => s.select)
  const inspects = useDatasetsStore((s) => s.inspects)

  useEffect(() => {
    if (isTauri()) void refresh()
  }, [refresh])

  if (!isTauri()) {
    return (
      <div className="flex h-full items-center justify-center px-4 text-center text-xs text-[#7a8088]">
        Datensätze brauchen einen echten Workspace-Ordner — öffne SpinoML in Tauri.
      </div>
    )
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center justify-between border-b border-[#1f2429] px-3 py-2 text-xs uppercase tracking-wider text-[#7a8088]">
        <span>Datasets</span>
        <button
          onClick={() => void refresh()}
          className="rounded px-1.5 py-0.5 text-[#9aa1a8] hover:bg-[#1a1e22]"
          title="reload datasets/ folder"
        >
          ↻
        </button>
      </div>
      {error && (
        <div className="border-b border-[#1f2429] bg-rose-900/20 px-3 py-2 text-xs text-rose-300">
          {error}
        </div>
      )}
      <div className="min-h-0 flex-1 overflow-auto">
        {loading && <div className="px-3 py-4 text-xs text-[#7a8088]">loading…</div>}
        {!loading && entries.length === 0 && (
          <div className="px-3 py-4 text-xs leading-relaxed text-[#7a8088]">
            Leg Dateien in <code className="text-[#9aa1a8]">datasets/</code> deines
            Workspace-Ordners ab — CSV, .pt, ImageFolder, .pdb, .smi oder
            eine <code>name.hf</code>-Textdatei mit <code>hf:dataset_name</code>.
          </div>
        )}
        {entries.map((entry) => {
          const cached = inspects[entry.relpath]
          const realKind = cached?.data?.kind
          const kind = realKind ?? guessKindFromName(entry.name, entry.is_dir)
          const isSelected = entry.relpath === selectedRel
          return (
            <button
              key={entry.relpath}
              onClick={() => select(entry.relpath)}
              className={`flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs ${
                isSelected ? 'bg-[#1a2030]' : 'hover:bg-[#1a1e22]'
              }`}
            >
              <span className={`rounded px-1 font-mono text-[10px] ${colorFor(kind)}`}>
                {iconFor(kind)}
              </span>
              <span className="flex-1 truncate text-[#e6e8eb]">{entry.name}</span>
              <span className="text-[10px] text-[#7a8088]">{formatSize(entry.size_bytes)}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
