import { useEffect, useState } from 'react'

import { isTauri } from '../../workspace/tauri-fs'
import { useTrainingGraphStore } from './store'
import { listTrainingGraphs, saveTrainingGraph, loadTrainingGraph } from './files'

const BTN = 'rounded border border-[#1f2429] bg-[#13171b] px-2 py-1 text-[11px] text-[#9aa1a8] hover:border-[#3a4148] hover:bg-[#1a1f24] hover:text-[#e6e8eb]'

export default function TrainingGraphBar() {
  const nodeCount = useTrainingGraphStore((s) => s.nodes.length)
  const resetGraph = useTrainingGraphStore((s) => s.resetGraph)
  const [name, setName] = useState('training')
  const [graphs, setGraphs] = useState<string[]>([])
  const [msg, setMsg] = useState<string | null>(null)

  const refreshList = () => { if (isTauri()) void listTrainingGraphs().then(setGraphs).catch(() => {}) }
  useEffect(refreshList, [])

  async function onSave() {
    setMsg(null)
    try {
      const rel = await saveTrainingGraph(name)
      setMsg(`gespeichert: ${rel.split('/').pop()}`)
      refreshList()
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e))
    }
  }

  async function onLoad(rel: string) {
    if (!rel) return
    setMsg(null)
    try {
      await loadTrainingGraph(rel)
      setName(rel.split('/').pop()!.replace(/\.mltrain$/i, ''))
      setMsg(`geladen: ${rel.split('/').pop()}`)
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="flex flex-col gap-1 rounded border border-[#1f2429] bg-[#0e1115]/90 p-1.5 backdrop-blur">
      <div className="flex items-center gap-1">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-28 rounded border border-[#1f2429] bg-[#0b0e11] px-1.5 py-1 text-[11px] text-[#e6e8eb] focus:border-[#6ab7ff] focus:outline-none"
          placeholder="name"
        />
        <button onClick={() => void onSave()} className={BTN} disabled={!isTauri()}>Speichern</button>
        <select onChange={(e) => { void onLoad(e.target.value); e.target.value = '' }} className={BTN} defaultValue="" title="Graph laden">
          <option value="">Laden…</option>
          {graphs.map((g) => <option key={g} value={g}>{g.split('/').pop()}</option>)}
        </select>
        <button onClick={() => { resetGraph(); setMsg(null) }} className={BTN}>Neu</button>
      </div>
      {nodeCount === 0 && (
        <div className="max-w-xs text-[10px] text-[#7a8088]">
          Leerer Graph — zieh Knoten aus der Palette: DatasetSource, ModelSource, Loss, Optimizer, TrainLoop.
        </div>
      )}
      {msg && <div className="text-[10px] text-[#6ab7ff]">{msg}</div>}
    </div>
  )
}
