import { useMemo, useState } from 'react'

import { useDataGraphStore } from './store'
import { compileDataGraph } from '../../codegen/dataGenerator'
import { generateDataCode } from '../../codegen/dataCodegen'
import { useChatStore } from '../../chat/store'
import { runWorkspaceScript, type RunScriptResult } from '../../datasets/client'
import { getCurrentConnection } from '../../connections/store'
import { useWorkspaceStore } from '../../workspace/store'

const BTN = 'rounded border border-[#1f2429] bg-[#13171b] px-2 py-1 text-[11px] text-[#9aa1a8] hover:border-[#3a4148] hover:bg-[#1a1f24] hover:text-[#e6e8eb] disabled:opacity-40'

type RunResult = RunScriptResult & { offline?: boolean }

// The workspace root as the torch sidecar sees it: the remote root for an SSH
// workspace (the HPC sidecar runs there), else the local workspace folder.
function rootForSidecar(): string | null {
  const c = getCurrentConnection()
  if (c.kind === 'remote-ssh') return c.root
  return useWorkspaceStore.getState().workspaceRoot
}

export default function DataGraphBar() {
  const nodes = useDataGraphStore((s) => s.nodes)
  const edges = useDataGraphStore((s) => s.edges)
  const resetGraph = useDataGraphStore((s) => s.resetGraph)
  const chatOnline = useChatStore((s) => s.online)
  const [msg, setMsg] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [output, setOutput] = useState<RunResult | null>(null)

  const compile = useMemo(
    () => compileDataGraph({
      nodes: nodes.map((n) => ({ id: n.id, dataType: n.data.dataType, params: n.data.params })),
      edges: edges.map((e) => ({ source: e.source, target: e.target })),
    }),
    [nodes, edges],
  )

  // Direct run — NO chatbot. The torch sidecar writes the compiled pipeline to
  // agent/data_pipeline.py and runs it where the workspace lives.
  async function runDirect(mode: 'shell' | 'slurm') {
    const root = rootForSidecar()
    if (!root) { setOutput({ ok: false, error: 'Kein Workspace geöffnet — Verzeichnis öffnen.' }); return }
    setRunning(true); setOutput(null); setMsg(null)
    try {
      const res = await runWorkspaceScript(root, 'agent/data_pipeline.py', generateDataCode(compile.plan), mode)
      setOutput(res as RunResult)
    } finally { setRunning(false) }
  }

  // Optional: let the chatbot write + run it (so it can adapt the script).
  function runViaChat() {
    const message = [
      'Führe diese Daten-Pipeline aus dem Data-Canvas aus.',
      'Schreibe das folgende Skript mit write_file nach agent/data_pipeline.py und führe es dann mit',
      'run_script aus (mode "shell", bei schwerer Arbeit auf HPC mode "slurm"). Wenn ein Schritt',
      'fehlschlägt, korrigiere das Skript und führe erneut aus. Halte den Lauf mit record_step fest.',
      '', '```python', generateDataCode(compile.plan), '```',
    ].join('\n')
    void useChatStore.getState().send(message)
    setMsg('an den Chatbot geschickt — Bestätigung im Chat')
  }

  return (
    <div className="flex flex-col gap-1 rounded border border-[#1f2429] bg-[#0e1216]/90 p-1.5 backdrop-blur">
      <div className="flex items-center gap-1">
        <button
          onClick={() => void runDirect('shell')}
          className="rounded bg-[var(--accent-sel)] px-2 py-1 text-[11px] text-[var(--accent)] hover:bg-[var(--accent-sel-hover)] disabled:opacity-40"
          disabled={!compile.plan || running}
          title="Pipeline jetzt ausführen (ohne Chatbot)"
        >{running ? '… läuft' : '▶ Pipeline ausführen'}</button>
        <button
          onClick={() => void runDirect('slurm')}
          className={BTN}
          disabled={!compile.plan || running}
          title="Als SLURM-Job (sbatch) — für schwere Pipelines auf dem HPC. Das Skript muss #SBATCH-Direktiven enthalten (oder ein CustomScript-Wrapper)."
        >SLURM</button>
        <button onClick={() => { resetGraph(); setMsg(null) }} className={BTN} title="Alle Knoten entfernen">Leeren</button>
      </div>
      <div className="flex items-center gap-1">
        <button
          onClick={runViaChat}
          className={BTN}
          disabled={!compile.plan || chatOnline === false}
          title="Vom Chatbot schreiben + ausführen lassen (kann das Skript anpassen / SLURM wählen)"
        >via Chat</button>
      </div>
      {nodes.length === 0 && (
        <div className="max-w-xs text-[10px] text-[#6f767e]">
          Leere Pipeline — zieh Knoten aus der Palette: TableSource → Transform/Fetch/Graph → WriteDataset.
        </div>
      )}
      {msg && <div className="text-[10px] text-[var(--accent)]">{msg}</div>}
      {output && <RunOutput output={output} onClose={() => setOutput(null)} />}
    </div>
  )
}

function RunOutput({ output, onClose }: { output: RunResult; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6" onClick={onClose}>
      <div
        className="flex h-[70vh] w-[760px] max-w-full flex-col overflow-hidden rounded-lg border border-[#262c33] bg-[#0e1216] shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-center justify-between border-b border-[#1f2429] px-4 py-2.5">
          <span className="text-sm font-medium text-[#e6e8eb]">
            Pipeline-Lauf {output.ok ? <span className="text-emerald-400">✓</span> : <span className="text-rose-400">✗</span>}
            {output.mode ? <span className="ml-2 text-[11px] text-[#6f767e]">{output.mode}</span> : null}
          </span>
          <button className="rounded px-2 py-0.5 text-xs text-[#9aa1a8] hover:bg-[#1f2429] hover:text-[#e6e8eb]" onClick={onClose}>schließen ✕</button>
        </div>
        <div className="min-h-0 flex-1 space-y-2 overflow-auto p-3 font-mono text-[11px] leading-snug">
          {output.error && <div className="text-rose-300">{output.error}</div>}
          {output.offline && <div className="text-amber-300/80">Torch-Sidecar nicht erreichbar — läuft er? (lokal: npm run sidecar:torch · remote: HPC-Sidecar verbinden)</div>}
          {output.job_id && <div className="text-emerald-300">SLURM-Job {output.job_id} abgeschickt — Status via squeue oder Terminal-Tab.</div>}
          {output.stdout && (<div><div className="mb-0.5 text-[10px] uppercase tracking-wide text-[#6f767e]">stdout</div><pre className="whitespace-pre-wrap text-[#b8c0c8]">{output.stdout}</pre></div>)}
          {output.stderr && (<div><div className="mb-0.5 text-[10px] uppercase tracking-wide text-[#6f767e]">stderr</div><pre className="whitespace-pre-wrap text-amber-300/80">{output.stderr}</pre></div>)}
          {!output.error && !output.stdout && !output.stderr && !output.job_id && <div className="text-[#6f767e]">(keine Ausgabe)</div>}
        </div>
      </div>
    </div>
  )
}
