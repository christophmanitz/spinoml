import { useEffect, useState } from 'react'

import { training } from './backend'
import { useTrainingStore } from './store'
import LineChart, { type Series } from './charts/LineChart'
import { parseEventLines, epochEnds, CHART_COLORS } from './charts/series'
import type { TrainingEvent } from './types'

type Loaded = {
  runId: string
  label: string
  color: string
  events: TrainingEvent[]
  config: Record<string, unknown> | null
  /** final/best metrics pulled from events */
  final: { val_loss: number | null; val_acc: number | null; train_loss: number | null; n_params: number | null; seconds: number | null }
}

function pickFinal(events: TrainingEvent[]): Loaded['final'] {
  const ep = epochEnds(events)
  const last = ep[ep.length - 1]
  const done = events.find((e) => e.kind === 'run.done')
  const built = events.find((e) => e.kind === 'model.built')
  const bestVal = ep.reduce<number | null>((b, e) => {
    const v = typeof e.val_loss === 'number' ? e.val_loss : null
    return v != null && (b == null || v < b) ? v : b
  }, null)
  return {
    val_loss: bestVal ?? (typeof done?.best_val_loss === 'number' ? done.best_val_loss : null),
    val_acc: typeof last?.val_acc === 'number' ? last.val_acc : null,
    train_loss: typeof last?.train_loss === 'number' ? last.train_loss : null,
    n_params: typeof built?.n_params === 'number' ? built.n_params : null,
    seconds: typeof done?.total_seconds === 'number' ? done.total_seconds : null,
  }
}

export default function CompareModal() {
  const ids = useTrainingStore((s) => s.compareIds)
  const close = useTrainingStore((s) => s.closeCompare)
  const runs = useTrainingStore((s) => s.runs)
  const [loaded, setLoaded] = useState<Loaded[]>([])
  const [lossLog, setLossLog] = useState(false)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      const out = await Promise.all(ids.map(async (runId, i): Promise<Loaded> => {
        const summary = runs.find((r) => r.run_id === runId)
        let events: TrainingEvent[] = []
        let config: Record<string, unknown> | null = null
        try { events = parseEventLines(await training.readFile(runId, 'events.jsonl')) } catch { /* none yet */ }
        try { config = JSON.parse(await training.readFile(runId, 'run.json')) } catch { /* none */ }
        return {
          runId,
          label: summary?.run_label || runId,
          color: CHART_COLORS[i % CHART_COLORS.length],
          events,
          config,
          final: pickFinal(events),
        }
      }))
      if (!cancelled) setLoaded(out)
    })()
    return () => { cancelled = true }
  }, [ids, runs])

  useEffect(() => {
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape') close() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [close])

  // overlay val_loss (falling back to train_loss) — one line per run
  const lossSeries: Series[] = loaded.map((l) => {
    const ep = epochEnds(l.events)
    const useVal = ep.some((e) => typeof e.val_loss === 'number')
    return {
      label: l.label,
      color: l.color,
      points: ep.map((e) => ({
        x: (e.epoch as number) + 1,
        y: typeof (useVal ? e.val_loss : e.train_loss) === 'number' ? (useVal ? e.val_loss : e.train_loss) as number : null,
      })),
    }
  })

  const diff = configDiff(loaded.map((l) => flattenTraining(l.config)))

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6"
      onClick={(e) => { if (e.target === e.currentTarget) close() }}
    >
      <div className="flex h-full max-h-[92vh] w-full max-w-5xl flex-col overflow-hidden rounded-lg border border-[#1f2429] bg-[#0e1115] shadow-2xl">
        <div className="flex items-center gap-2 border-b border-[#1f2429] px-4 py-3">
          <span className="text-sm text-[#e6e8eb]">Vergleich · {loaded.length} Runs</span>
          <button
            onClick={() => exportCsv(loaded, diff)}
            disabled={loaded.length === 0}
            className="ml-auto rounded border border-[#1f2429] bg-[#13171b] px-2 py-0.5 text-[11px] text-[#9aa1a8] hover:border-[#3a4148] hover:text-[#e6e8eb] disabled:opacity-40"
            title="Metriken + Config-Diff als CSV exportieren"
          >
            CSV
          </button>
          <button onClick={close} className="rounded px-2 py-0.5 text-[#7a8088] hover:bg-[#1a1e22] hover:text-[#e6e8eb]">×</button>
        </div>

        <div className="min-h-0 flex-1 space-y-5 overflow-auto p-4 text-[12px] text-[#cfd3d8]">
          {/* loss overlay */}
          <div className="rounded border border-[#1f2429] bg-[#0a0d10] p-3">
            <div className="mb-2 flex items-center">
              <span className="text-[11px] text-[#9aa1a8]">Loss (val, sonst train)</span>
              <button
                onClick={() => setLossLog((v) => !v)}
                className={`ml-auto rounded px-1.5 py-0.5 text-[10px] ${lossLog ? 'bg-[#13344f] text-[#6ab7ff]' : 'text-[#7a8088] hover:bg-[#1a1e22]'}`}
              >
                log
              </button>
            </div>
            <LineChart series={lossSeries} yLog={lossLog} height={260} xLabel="epoch" />
          </div>

          {/* final metrics table */}
          <div>
            <div className="mb-1 text-[11px] text-[#7a8088]">Finale Metriken</div>
            <table className="w-full text-left text-[11px]">
              <thead className="text-[#7a8088]">
                <tr>
                  <th className="py-1 pr-3">Run</th>
                  <th className="pr-3">best val</th>
                  <th className="pr-3">val acc</th>
                  <th className="pr-3">train loss</th>
                  <th className="pr-3">params</th>
                  <th>Dauer</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {loaded.map((l) => (
                  <tr key={l.runId} className="border-t border-[#171b1f]">
                    <td className="py-0.5 pr-3">
                      <span className="mr-1.5 inline-block h-2 w-2 rounded-sm align-middle" style={{ background: l.color }} />
                      <span className="font-sans text-[#e6e8eb]">{l.label}</span>
                    </td>
                    <td className="pr-3 text-[#5fd39a]">{fmtNum(l.final.val_loss, 4)}</td>
                    <td className="pr-3">{fmtNum(l.final.val_acc, 4)}</td>
                    <td className="pr-3">{fmtNum(l.final.train_loss, 4)}</td>
                    <td className="pr-3">{l.final.n_params != null ? l.final.n_params.toLocaleString() : '—'}</td>
                    <td>{l.final.seconds != null ? `${Math.round(l.final.seconds)}s` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* config diff */}
          <div>
            <div className="mb-1 text-[11px] text-[#7a8088]">Config-Unterschiede</div>
            {diff.length === 0 ? (
              <div className="text-[11px] text-[#7a8088]">Identische Trainings-Configs.</div>
            ) : (
              <table className="w-full text-left text-[11px]">
                <thead className="text-[#7a8088]">
                  <tr>
                    <th className="py-1 pr-3">Feld</th>
                    {loaded.map((l) => <th key={l.runId} className="pr-3">{l.label}</th>)}
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {diff.map((row) => (
                    <tr key={row.key} className="border-t border-[#171b1f]">
                      <td className="py-0.5 pr-3 text-[#9aa1a8]">{row.key}</td>
                      {row.values.map((v, i) => (
                        <td key={i} className="pr-3 text-[#e6c34a]">{v}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function fmtNum(v: number | null, digits: number): string {
  return v == null ? '—' : Number(v.toFixed(digits)).toString()
}

function csvCell(v: string | number | null): string {
  const s = v == null ? '' : String(v)
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

/** One row per run: fixed metric columns + every differing config field. */
function exportCsv(loaded: Loaded[], diff: DiffRow[]) {
  const metricCols = ['best_val_loss', 'val_acc', 'train_loss', 'n_params', 'seconds'] as const
  const header = ['run_id', 'label', ...metricCols, ...diff.map((d) => d.key)]
  const rows = loaded.map((l, i) => [
    l.runId,
    l.label,
    l.final.val_loss, l.final.val_acc, l.final.train_loss, l.final.n_params, l.final.seconds,
    ...diff.map((d) => d.values[i]),
  ])
  const csv = [header, ...rows].map((r) => r.map(csvCell).join(',')).join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'mlforge-compare.csv'
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

/** Flatten the training section of run.json into dotted scalar keys. */
function flattenTraining(config: Record<string, unknown> | null): Record<string, string> {
  const out: Record<string, string> = {}
  const t = (config?.training ?? {}) as Record<string, unknown>
  const walk = (obj: Record<string, unknown>, prefix: string) => {
    for (const [k, v] of Object.entries(obj)) {
      const key = prefix ? `${prefix}.${k}` : k
      if (v != null && typeof v === 'object' && !Array.isArray(v)) walk(v as Record<string, unknown>, key)
      else out[key] = Array.isArray(v) ? JSON.stringify(v) : String(v)
    }
  }
  walk(t, '')
  const model = config?.model_path
  if (typeof model === 'string') out['model'] = model.split('/').pop() ?? model
  return out
}

type DiffRow = { key: string; values: string[] }

function configDiff(maps: Record<string, string>[]): DiffRow[] {
  if (maps.length < 2) return []
  const keys = new Set<string>()
  for (const m of maps) for (const k of Object.keys(m)) keys.add(k)
  const rows: DiffRow[] = []
  for (const key of Array.from(keys).sort()) {
    const values = maps.map((m) => m[key] ?? '—')
    if (new Set(values).size > 1) rows.push({ key, values })
  }
  return rows
}
