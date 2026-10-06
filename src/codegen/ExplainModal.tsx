import { useEffect, useMemo, useState } from 'react'
import { useGraphStore } from '../canvas/GraphStore'
import { llmHealthState } from '../chat/client'
import { explainModel } from './explain'
import { fetchModelExplanation } from './modelIntent'
import Markdown from '../ui/Markdown'

function Shape({ s }: { s?: number[] }) {
  if (!s || !s.length) return <span className="text-[#5b6168]">?</span>
  return <span className="font-mono text-[var(--accent)]">[{s.join(', ')}]</span>
}

export default function ExplainModal({ onClose }: { onClose: () => void }) {
  const nodes = useGraphStore((s) => s.nodes)
  const edges = useGraphStore((s) => s.edges)
  const exp = useMemo(() => explainModel(nodes, edges), [nodes, edges])

  const [intent, setIntent] = useState('')
  const [intentState, setIntentState] = useState<'loading' | 'ok' | 'offline' | 'auth' | 'error'>('loading')
  const [authMessage, setAuthMessage] = useState('')

  useEffect(() => {
    const ctrl = new AbortController()
    let alive = true
    ;(async () => {
      const h = await llmHealthState()
      if (h.state === 'auth-failed') {
        if (alive) { setAuthMessage(h.message ?? 'Sidecar-Authentifizierung fehlgeschlagen.'); setIntentState('auth') }
        return
      }
      if (h.state !== 'online') { if (alive) setIntentState('offline'); return }
      try {
        const s = await fetchModelExplanation(exp, ctrl.signal)
        if (!alive) return
        if (s) { setIntent(s); setIntentState('ok') } else { setIntentState('error') }
      } catch {
        if (alive) setIntentState('error')
      }
    })()
    return () => { alive = false; ctrl.abort() }
    // exp.flowText changes whenever the structure/shapes do — re-explain then.
  }, [exp])

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6"
      onClick={onClose}
    >
      <div
        className="flex max-h-[80vh] w-[640px] max-w-full flex-col overflow-hidden rounded-lg border border-[#262c33] bg-[#0e1216] shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-center justify-between border-b border-[#1f2429] px-4 py-2.5">
          <span className="text-sm font-medium text-[#e6e8eb]">Modell-Erklärung</span>
          <button
            className="rounded px-2 py-0.5 text-xs text-[#9aa1a8] hover:bg-[#1f2429] hover:text-[#e6e8eb]"
            onClick={onClose}
          >
            schließen ✕
          </button>
        </div>

        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-3 text-sm leading-relaxed text-[#d4d7db]">
          {/* What it does + how — LLM explanation (markdown) */}
          <section>
            <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-[#6f767e]">Erklärung</h3>
            {intentState === 'loading' && <p className="text-[#6f767e]">… wird erklärt (kann einen Moment dauern)</p>}
            {intentState === 'ok' && <div className="text-[#d4d7db]"><Markdown>{intent}</Markdown></div>}
            {intentState === 'offline' && (
              <p className="text-[#6f767e]">
                LLM offline — nur der Datenfluss unten (deterministisch). Starte den LLM-Sidecar für eine Klartext-Zusammenfassung.
              </p>
            )}
            {intentState === 'auth' && (
              <p className="text-rose-300/90">{authMessage}</p>
            )}
            {intentState === 'error' && (
              <p className="text-[#6f767e]">Konnte keine LLM-Zusammenfassung holen — der Datenfluss unten beschreibt das Modell.</p>
            )}
          </section>

          {/* Data flow — deterministic */}
          <section>
            <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-[#6f767e]">Datenfluss</h3>
            {exp.steps.length === 0 ? (
              <p className="text-[#6f767e]">Noch keine Schichten — füge Layer hinzu.</p>
            ) : (
              <ol className="space-y-1.5">
                {exp.inputs.map((i) => (
                  <li key={`in-${i.name}`} className="flex items-baseline gap-2">
                    <span className="rounded bg-[#15324a] px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-[var(--accent)]">in</span>
                    <span className="text-[#e6e8eb]">{i.name}{i.isGraph ? ' (Graph)' : ''}</span>
                    <Shape s={i.shape} />
                  </li>
                ))}
                {exp.steps.map((s) => (
                  <li key={s.id} className="flex items-baseline gap-2 border-l border-[#1f2429] pl-3">
                    <span className="font-mono text-[#cdd3da]">{s.module}</span>
                    {s.summary && <span className="text-[#9aa1a8]">{s.summary}</span>}
                    {s.outShape && <span className="ml-auto"><Shape s={s.outShape} /></span>}
                  </li>
                ))}
                {exp.output && (
                  <li className="flex items-baseline gap-2">
                    <span className="rounded bg-[#1c3a2a] px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-[#5fd39a]">out</span>
                    <Shape s={exp.output} />
                  </li>
                )}
              </ol>
            )}
          </section>

          {exp.issues.length > 0 && (
            <section>
              <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-amber-500/80">Hinweise</h3>
              <ul className="ml-4 list-disc space-y-0.5 text-[13px] text-amber-300/90">
                {exp.issues.map((m, i) => <li key={i}>{m}</li>)}
              </ul>
            </section>
          )}
        </div>
      </div>
    </div>
  )
}
