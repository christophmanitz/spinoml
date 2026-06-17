import { useEffect, useState } from 'react'
import { useGraphStore } from '../canvas/GraphStore'
import { LAYERS } from '../layers/registry'
import { colorForCategory } from '../layers/categories'
import { isTauri } from '../workspace/tauri-fs'
import { getCurrentConnection } from '../connections/store'
import { training } from '../training/backend'
import type { RunSummary } from '../training/types'
import { useVizStore } from './store'
import type { NodeActivation, Weights, Preview } from './client'

type EdgesPreview = Extract<Preview, { kind: 'edges' }>
import { Heatmap, VectorBars, Histogram, TensorShape, NodeLinkGraph, Legend, LEGEND_DIVERGING, legendMono } from './primitives'
import { explainFor } from './explainText'

// The big "what happens in this layer" panel. Replaces the Inspector while
// Explain mode is on. Category-driven (3Blue1Brown style).
export default function LayerExplain() {
  const selectedId = useGraphStore((s) => s.selectedNodeId)
  const nodes = useGraphStore((s) => s.nodes)
  const node = nodes.find((n) => n.id === selectedId)
  const edges = useGraphStore((s) => s.edges)
  const byNode = useVizStore((s) => s.byNode)
  const weightsByNode = useVizStore((s) => s.weightsByNode)
  const running = useVizStore((s) => s.running)
  const error = useVizStore((s) => s.error)
  const source = useVizStore((s) => s.source)
  const sampleNote = useVizStore((s) => s.sampleNote)
  const ran = Object.keys(byNode).length > 0

  return (
    <div className="flex h-full min-h-0 flex-col p-3 text-sm">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-xs uppercase tracking-wide text-[#7a8088]">Explain · Datenfluss</div>
        {source && (
          <span className="text-[10px] text-[#5b6168]">
            {source === 'dataset' ? `Sample: ${sampleNote ?? 'Dataset'}` : 'synthetische Eingabe'}
          </span>
        )}
      </div>

      <WeightsBar />

      {error && <div className="mb-2 rounded bg-rose-950/40 px-2 py-1 font-mono text-[10px] text-rose-300">{error}</div>}
      {!ran && !error && (
        <div className="flex flex-col gap-2 text-xs text-[#9aa1a8]">
          {running ? (
            <div className="text-[#7a8088]">schicke ein Beispiel durch…</div>
          ) : (
            <>
              <p className="text-[#e6e8eb]">Was passiert eigentlich im Modell?</p>
              <p>
                Drück oben <span className="text-[#e6e8eb]">„▶ Beispiel"</span>. Dann schicke ich ein
                echtes Datenbeispiel durch dein Netz und zeige dir <em>Layer für Layer</em>, welche
                Zahlen herauskommen — als Bilder, Balken und leuchtende Neuronen.
              </p>
              <p>
                Danach: einen Node anklicken → hier erscheint in Klartext, <span className="text-[#e6e8eb]">was
                dieser Layer macht</span> und <span className="text-[#e6e8eb]">wie du das Bild liest</span>.
              </p>
            </>
          )}
        </div>
      )}

      {ran && !node && <div className="text-xs text-[#7a8088]">Wähle einen Node, um zu sehen, was dort passiert.</div>}

      {ran && node && (() => {
        const act = byNode[node.id]
        const spec = LAYERS[node.data.layerType]
        const cat = spec?.category ?? 'IO'
        const hex = colorForCategory(cat)
        const predIds = edges.filter((e) => e.target === node.id).map((e) => e.source)
        const predAct = predIds[0] ? byNode[predIds[0]] : undefined
        const predActs = predIds.map((id) => byNode[id]).filter(Boolean) as NodeActivation[]
        const weights = weightsByNode[node.id]
        const ex = explainFor(node.data.layerType, cat)
        const inShape = node.data.inferredInputShape
        const outShape = node.data.inferredOutputShape
        const isInput = node.data.layerType === 'Input'
        // Graph structure: find the edge_index input so GNN views can draw it.
        const edgeNode = nodes.find((n) => n.data.layerType === 'Input' && byNode[n.id]?.preview?.kind === 'edges')
        const edgePrev: EdgesPreview | undefined =
          edgeNode && byNode[edgeNode.id]?.preview?.kind === 'edges'
            ? (byNode[edgeNode.id].preview as EdgesPreview)
            : undefined

        // Pick a colour legend matching what the chosen view encodes.
        const pk = act?.preview?.kind
        const legendItems =
          (cat === 'Linear' && node.data.layerType === 'Linear') || cat === 'Conv' || pk === 'matrix'
            ? LEGEND_DIVERGING
            : pk === 'maps'
              ? legendMono(hex)
              : null

        return (
          <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto">
            <div>
              <div className="flex items-center gap-1.5 text-[13px]" style={{ color: hex }}>
                <span>{node.data.layerType}</span>
                <span className="text-[10px] text-[#5b6168]">{cat}</span>
              </div>
              {act && <StatsLine act={act} />}
              <p className="mt-1.5 text-[11px] leading-snug text-[#cbd2d9]">{ex.does}</p>
            </div>

            {!act && <div className="text-xs text-[#7a8088]">Für diesen Node liegen keine Aktivierungen vor.</div>}

            {act && (inShape || outShape) && (
              <div className="flex flex-col gap-1.5 rounded border border-[#1f2429] bg-[#0e1216] px-2 py-2">
                <div className="text-[10px] uppercase tracking-wide text-[#5b6168]">Form (Dimensionen)</div>
                {isInput ? (
                  <div className="flex items-center gap-2">
                    <TensorShape shape={outShape ?? inShape ?? []} face={faceOf(act)} hex={hex} />
                    <span className="font-mono text-[10px] text-[#9aa1a8]">[{(outShape ?? inShape ?? []).join(', ')}]</span>
                  </div>
                ) : (
                  <div className="flex flex-wrap items-end gap-3">
                    <div className="flex flex-col items-center gap-0.5">
                      <TensorShape shape={inShape ?? []} face={faceOf(predAct)} hex="#7a8088" />
                      <span className="font-mono text-[9px] text-[#5b6168]">[{(inShape ?? []).join(', ')}]</span>
                    </div>
                    <span className="pb-3 text-base" style={{ color: hex }}>→</span>
                    <div className="flex flex-col items-center gap-0.5">
                      <TensorShape shape={outShape ?? []} face={faceOf(act)} hex={hex} />
                      <span className="font-mono text-[9px] text-[#5b6168]">[{(outShape ?? []).join(', ')}]</span>
                    </div>
                  </div>
                )}
                {(() => {
                  const sh = isInput ? (outShape ?? inShape) : outShape
                  if (!sh) return null
                  const bn = batchNote(sh)
                  return (
                    <div className="mt-0.5 flex flex-col gap-0.5">
                      <p className="text-[11px] leading-snug text-[#cbd2d9]">
                        <span className="font-mono text-[#9aa1a8]">[{sh.join(', ')}]</span> = {
                          cat === 'Graph' && sh.length === 2
                            ? `${sh[0]} Knoten, je ein ${sh[1]}-dim Merkmalsvektor — ein GRAPH, kein Bild.`
                            : describeShape(sh)
                        }
                      </p>
                      {cat === 'Graph'
                        ? <p className="text-[10px] leading-snug text-[#7a8088]">Keine Batch-Dimension: die erste Zahl zählt Knoten, nicht Beispiele.</p>
                        : bn && <p className="text-[10px] leading-snug text-[#7a8088]">{bn}</p>}
                    </div>
                  )
                })()}
              </div>
            )}

            {act && node.data.layerType === 'Input' && <PreviewView act={act} hex={hex} title="Eingabe" />}
            {act && node.data.layerType === 'Output' && <ProbBars act={act} hex={hex} />}
            {act && cat === 'IO' && !['Input', 'Output'].includes(node.data.layerType) && <PreviewView act={act} hex={hex} />}
            {act && cat === 'Linear' && node.data.layerType === 'Linear' && (
              <LinearView inAct={predAct} outAct={act} weights={weights} hex={hex} />
            )}
            {act && cat === 'Linear' && node.data.layerType !== 'Linear' && <PreviewView act={act} hex={hex} />}
            {act && cat === 'Activation' && (
              <ActivationView type={node.data.layerType} inAct={predAct} outAct={act} hex={hex} />
            )}
            {act && cat === 'Conv' && <ConvView act={act} weights={weights} hex={hex} />}
            {act && cat === 'Pool' && <PoolView before={predAct} after={act} hex={hex} />}
            {act && cat === 'Norm' && <NormView before={predAct} after={act} hex={hex} />}
            {act && cat === 'Reshape' && (
              <ReshapeView before={predAct} after={act} hex={hex}
                inShape={node.data.inferredInputShape} outShape={node.data.inferredOutputShape} />
            )}
            {act && cat === 'Merge' && <MergeView inputs={predActs} after={act} type={node.data.layerType} hex={hex} />}
            {act && cat === 'Graph' && <GraphView act={act} edges={edgePrev} hex={hex} />}
            {act && !['IO', 'Linear', 'Activation', 'Conv', 'Pool', 'Norm', 'Reshape', 'Merge', 'Graph'].includes(cat) && (
              <PreviewView act={act} hex={hex} />
            )}

            {act && (
              <div className="mt-1 flex flex-col gap-1.5 rounded border border-[#1f2429] bg-[#0e1216] px-2 py-1.5">
                <div className="text-[10px] uppercase tracking-wide text-[#5b6168]">Wie lese ich das?</div>
                <p className="text-[11px] leading-snug text-[#9aa1a8]">{ex.read}</p>
                {legendItems && <Legend items={legendItems} />}
              </div>
            )}
          </div>
        )
      })()}
    </div>
  )
}

// Choose which weights flow through: random init (untrained) or a trained run
// checkpoint. Trained weights only work on a local workspace (the local sidecar
// can't read a remote HPC checkpoint).
function WeightsBar() {
  const weightsRunId = useVizStore((s) => s.weightsRunId)
  const weightsSource = useVizStore((s) => s.weightsSource)
  const weightsNote = useVizStore((s) => s.weightsNote)
  const setWeightsRun = useVizStore((s) => s.setWeightsRun)
  const [runs, setRuns] = useState<RunSummary[]>([])

  const local = isTauri() && getCurrentConnection().kind === 'local'
  useEffect(() => {
    if (!local) return
    let alive = true
    training.list().then((list) => { if (alive) setRuns(list.filter((r) => r.has_checkpoint)) }).catch(() => {})
    return () => { alive = false }
  }, [local])

  if (!local) return null

  return (
    <div className="mb-2 flex flex-wrap items-center gap-2 rounded border border-[#1f2429] bg-[#0e1216] px-2 py-1.5">
      <span className="text-[10px] uppercase tracking-wide text-[#5b6168]">Gewichte</span>
      {runs.length === 0 ? (
        <span className="text-[10px] text-[#7a8088]">zufällig — noch kein trainierter Run mit Checkpoint</span>
      ) : (
        <select
          className="rounded border border-[#1f2429] bg-[#0b0e11] px-1.5 py-0.5 text-[11px] text-[#e6e8eb] outline-none focus:border-[#6ab7ff]"
          value={weightsRunId ?? ''}
          onChange={(e) => setWeightsRun(e.target.value || null)}
        >
          <option value="">Zufällig (untrainiert)</option>
          {runs.map((r) => (
            <option key={r.run_id} value={r.run_id}>{r.run_label || r.run_id}</option>
          ))}
        </select>
      )}
      {weightsSource && (
        <span
          className={`rounded px-1.5 py-0.5 text-[10px] ${weightsSource === 'trained' ? 'bg-emerald-900/40 text-emerald-300' : 'bg-[#1f2429] text-[#7a8088]'}`}
          title={weightsNote ?? undefined}
        >
          {weightsSource === 'trained' ? '✓ trainiert' : 'zufällig'}
        </span>
      )}
      {weightsNote && weightsSource === 'random' && weightsRunId && (
        <span className="text-[10px] text-amber-400/80" title={weightsNote}>⚠ passt nicht</span>
      )}
    </div>
  )
}

function StatsLine({ act }: { act: NodeActivation }) {
  const s = act.stats
  return (
    <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[10px] text-[#7a8088]">
      <span>min {s.min}</span>
      <span>max {s.max}</span>
      <span>μ {s.mean}</span>
      <span>σ {s.std}</span>
      {s.frac_zero > 0.001 && <span style={{ color: '#4dd0a8' }}>{(s.frac_zero * 100).toFixed(0)}% Null (Sparsity)</span>}
    </div>
  )
}

// Generic preview for any activation: maps grid / vector bars / matrix / tokens.
function PreviewView({ act, hex, title }: { act: NodeActivation; hex: string; title?: string }) {
  const p = act.preview
  if (!p) return null
  return (
    <div className="flex flex-col gap-1">
      {title && <div className="text-[11px] text-[#9aa1a8]">{title}</div>}
      {p.kind === 'vector' && <VectorBars values={p.values} width={260} height={70} hex={hex} />}
      {p.kind === 'matrix' && <Heatmap grid={p.grid} cell={8} mode="diverging" maxW={260} />}
      {p.kind === 'tokens' && (
        <div className="font-mono text-[11px] text-[#9aa1a8]">{p.values.join(' · ')}</div>
      )}
      {p.kind === 'scalar' && <div className="font-mono text-sm" style={{ color: hex }}>{p.value}</div>}
      {p.kind === 'maps' && (
        <div className="flex flex-wrap gap-1">
          {p.maps.map((g, i) => <Heatmap key={i} grid={g} cell={7} gap={0} mode="mono" hex={hex} />)}
          {p.channels > p.shown && <span className="self-end text-[10px] text-[#5b6168]">+{p.channels - p.shown} Kanäle</span>}
        </div>
      )}
      {p.kind === 'edges' && (
        <>
          <NodeLinkGraph edges={p.edges} nNodes={p.n_nodes} hex="#34d399" />
          <div className="text-[10px] text-[#5b6168]">{p.n_nodes} Knoten · {p.n_edges} Kanten — das ist die Graphstruktur (edge_index)</div>
        </>
      )}
    </div>
  )
}

// GNN layers: show the actual graph (nodes coloured by activation) + the
// node-feature matrix, and explain message passing.
function GraphView({ act, edges, hex }: { act: NodeActivation; edges?: EdgesPreview; hex: string }) {
  const grid = act.preview?.kind === 'matrix' ? act.preview.grid : null
  const nodeMag = grid ? grid.map((row) => row.reduce((a, b) => a + Math.abs(b), 0) / Math.max(1, row.length)) : undefined
  return (
    <div className="flex flex-col gap-2">
      {edges ? (
        <>
          <div className="text-[11px] text-[#9aa1a8]">Der Graph — Helligkeit eines Knotens = wie stark er nach diesem Layer aktiviert ist.</div>
          <NodeLinkGraph edges={edges.edges} nNodes={edges.n_nodes} activations={nodeMag} hex={hex} />
          <div className="text-[10px] text-[#5b6168]">{edges.n_nodes} Knoten · {edges.n_edges} Kanten</div>
        </>
      ) : (
        <div className="text-[11px] text-[#7a8088]">Kein edge_index-Input gefunden — die Graphstruktur ist unbekannt.</div>
      )}
      {grid && (
        <div>
          <div className="mb-1 text-[11px] text-[#9aa1a8]">Knoten-Merkmale (Zeile = Knoten ↓, Spalte = Merkmal →)</div>
          <Heatmap grid={grid} cell={8} mode="diverging" maxW={240} />
        </div>
      )}
    </div>
  )
}

// Linear: the 3B1B neuron network — input neurons → output neurons, brightness
// = activation, edges coloured by weight sign (blue −, red +).
function LinearView({
  inAct, outAct, weights, hex,
}: {
  inAct?: NodeActivation
  outAct: NodeActivation
  weights?: Weights
  hex: string
}) {
  const inVals = inAct?.preview?.kind === 'vector' ? inAct.preview.values : []
  const outVals = outAct.preview?.kind === 'vector' ? outAct.preview.values : []
  const NIN = Math.min(inVals.length, 12)
  const NOUT = Math.min(outVals.length, 8)
  const W = 280
  const H = Math.max(140, Math.max(NIN, NOUT) * 16)
  const xIn = 40
  const xOut = W - 40
  const yFor = (i: number, n: number) => (H / (n + 1)) * (i + 1)
  const inAbs = Math.max(1e-6, ...inVals.map((v) => Math.abs(v)))
  const outAbs = Math.max(1e-6, ...outVals.map((v) => Math.abs(v)))
  const wgrid = weights?.kind === 'matrix' ? weights.grid : null
  const wAbs = wgrid ? Math.max(1e-6, ...wgrid.flat().map((v) => Math.abs(v))) : 1

  return (
    <div className="flex flex-col gap-2">
      <div className="flex justify-between text-[10px] text-[#5b6168]">
        <span>Eingang ({inVals.length})</span>
        <span>Ausgang ({outVals.length})</span>
      </div>
      <svg width={W} height={H} style={{ display: 'block' }}>
        {wgrid && Array.from({ length: NIN }).map((_, i) =>
          Array.from({ length: NOUT }).map((__, j) => {
            const wi = Math.floor((i / Math.max(1, NIN)) * wgrid[0].length)
            const wj = Math.floor((j / Math.max(1, NOUT)) * wgrid.length)
            const w = wgrid[wj]?.[wi] ?? 0
            const t = Math.abs(w) / wAbs
            if (t < 0.05) return null
            return (
              <line
                key={`${i}-${j}`}
                x1={xIn} y1={yFor(i, NIN)} x2={xOut} y2={yFor(j, NOUT)}
                stroke={w < 0 ? '#6ab7ff' : '#ff6b6b'}
                strokeWidth={0.5 + 2 * t}
                strokeOpacity={(0.1 + 0.6 * t).toFixed(2)}
              />
            )
          }),
        )}
        {Array.from({ length: NIN }).map((_, i) => (
          <circle key={`in${i}`} cx={xIn} cy={yFor(i, NIN)} r={5}
            fill={hex} fillOpacity={(0.15 + 0.85 * Math.abs(inVals[i] ?? 0) / inAbs).toFixed(2)}
            stroke="#2a2f36" />
        ))}
        {Array.from({ length: NOUT }).map((_, j) => (
          <circle key={`out${j}`} cx={xOut} cy={yFor(j, NOUT)} r={5}
            fill={hex} fillOpacity={(0.15 + 0.85 * Math.abs(outVals[j] ?? 0) / outAbs).toFixed(2)}
            stroke="#2a2f36" />
        ))}
      </svg>
      {wgrid && (
        <div>
          <div className="mb-1 text-[11px] text-[#9aa1a8]">Gewichtsmatrix [{weights?.kind === 'matrix' ? weights.shape.join('×') : ''}]</div>
          <Heatmap grid={wgrid} cell={4} gap={0} mode="diverging" maxW={260} />
        </div>
      )}
    </div>
  )
}

const ACT_FNS: Record<string, (x: number) => number> = {
  ReLU: (x) => Math.max(0, x),
  Sigmoid: (x) => 1 / (1 + Math.exp(-x)),
  Tanh: (x) => Math.tanh(x),
  SiLU: (x) => x / (1 + Math.exp(-x)),
  GELU: (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))),
}

// Activation: plot the function curve and drop the REAL incoming values onto it.
function ActivationView({
  type, inAct, outAct, hex,
}: {
  type: string
  inAct?: NodeActivation
  outAct: NodeActivation
  hex: string
}) {
  const fn = ACT_FNS[type]
  if (!fn) {
    // Softmax / LogSoftmax etc. — not pointwise; show the distribution.
    return <ProbBars act={outAct} hex={hex} />
  }
  const xs = inAct?.preview?.kind === 'vector' ? inAct.preview.values : []
  const lo = Math.min(-4, ...xs)
  const hi = Math.max(4, ...xs)
  const W = 260
  const H = 160
  const pad = 18
  const yLo = Math.min(0, ...xs.map(fn), -1)
  const yHi = Math.max(1, ...xs.map(fn))
  const sx = (x: number) => pad + ((x - lo) / (hi - lo)) * (W - 2 * pad)
  const sy = (y: number) => H - pad - ((y - yLo) / (yHi - yLo)) * (H - 2 * pad)
  const curve = Array.from({ length: 80 }, (_, i) => lo + (i / 79) * (hi - lo))
    .map((x) => `${sx(x).toFixed(1)},${sy(fn(x)).toFixed(1)}`).join(' ')

  return (
    <div className="flex flex-col gap-1">
      <div className="text-[11px] text-[#9aa1a8]">{type}: jeder Punkt = ein echter Eingabewert → Ausgabe auf der Kurve</div>
      <svg width={W} height={H} style={{ display: 'block' }}>
        <line x1={pad} y1={sy(0)} x2={W - pad} y2={sy(0)} stroke="#2a2f36" />
        <line x1={sx(0)} y1={pad} x2={sx(0)} y2={H - pad} stroke="#2a2f36" />
        <polyline points={curve} fill="none" stroke={hex} strokeWidth={2} />
        {xs.slice(0, 200).map((x, i) => (
          <circle key={i} cx={sx(x)} cy={sy(fn(x))} r={2.2} fill={hex} fillOpacity={0.5} />
        ))}
        <text x={W - pad} y={H - 4} textAnchor="end" fontSize={9} fill="#5b6168">Eingabe →</text>
        <text x={4} y={pad} fontSize={9} fill="#5b6168">↑ Ausgabe</text>
      </svg>
    </div>
  )
}

// Plain-language decode of a tensor shape, matching the glyph decomposition.
function describeShape(shape: number[]): string {
  const d = shape.length
  if (d === 0) return 'ein einzelner Skalar (eine Zahl).'
  if (d === 1) return `ein Vektor mit ${shape[0]} ${shape[0] === 1 ? 'Wert' : 'Werten'}.`
  if (d === 2) {
    const [a, b] = shape
    if (a === 1) return `1 Beispiel im Batch, dann ein Vektor mit ${b} Werten.`
    return `eine ${a}×${b}-Matrix (${a} Zeilen, ${b} Spalten).`
  }
  if (d === 3) {
    const [c, h, w] = shape
    return `${c} ${c === 1 ? 'Kanal' : 'Kanäle'}, jeweils eine ${h}×${w}-Matrix. (Kein Batch.)`
  }
  const b = shape[0], h = shape[d - 2], w = shape[d - 1]
  const c = shape.slice(1, d - 2).reduce((x, y) => x * y, 1)
  return `${b} ${b === 1 ? 'Beispiel' : 'Beispiele'} im Batch · ${c} ${c === 1 ? 'Kanal' : 'Kanäle'} · je eine ${h}×${w}-Matrix.`
}

// Note explaining the leading "batch" number where it applies.
function batchNote(shape: number[]): string | null {
  const d = shape.length
  if (d >= 4 || (d === 2 && shape[0] === 1)) {
    return `Die erste Zahl (${shape[0]}) ist die Batch-Größe — so viele Beispiele laufen gleichzeitig durchs Netz. `
      + `Beim Erklären schicken wir genau 1 durch; beim echten Training sind es oft 32–256.`
  }
  return null
}

// A 2D grid to paint on a TensorShape's front face (real values, if any).
function faceOf(act?: NodeActivation): number[][] | undefined {
  const p = act?.preview
  if (p?.kind === 'maps') return p.maps[0]
  if (p?.kind === 'matrix') return p.grid
  return undefined
}

// Flatten any preview into a flat list of values (for histograms / quick reads).
function previewValues(act?: NodeActivation): number[] {
  const p = act?.preview
  if (!p) return []
  if (p.kind === 'vector') return p.values
  if (p.kind === 'tokens') return p.values
  if (p.kind === 'scalar') return [p.value]
  if (p.kind === 'matrix') return p.grid.flat()
  if (p.kind === 'maps') return p.maps.flat(2)
  return []
}

// Compact preview used inside the before/after + merge layouts.
function SmallPreview({ act, hex }: { act?: NodeActivation; hex: string }) {
  const p = act?.preview
  if (!p) return <span className="text-[10px] text-[#5b6168]">—</span>
  if (p.kind === 'maps') {
    return (
      <div className="flex flex-wrap gap-0.5">
        {p.maps.slice(0, 8).map((g, i) => <Heatmap key={i} grid={g} cell={5} gap={0} mode="mono" hex={hex} />)}
      </div>
    )
  }
  if (p.kind === 'matrix') return <Heatmap grid={p.grid} cell={4} gap={0} mode="diverging" maxW={200} />
  if (p.kind === 'vector') return <VectorBars values={p.values} width={200} height={40} hex={hex} />
  if (p.kind === 'tokens') return <div className="font-mono text-[10px] text-[#9aa1a8]">{p.values.join(' · ')}</div>
  if (p.kind === 'edges') return <NodeLinkGraph edges={p.edges} nNodes={p.n_nodes} hex={hex} size={120} />
  if (p.kind === 'scalar') return <div className="font-mono text-[11px]" style={{ color: hex }}>{p.value}</div>
  return null
}

// Output / Softmax: horizontal probability (or score) bars, argmax highlighted.
function ProbBars({ act, hex }: { act: NodeActivation; hex: string }) {
  const vals = act.preview?.kind === 'vector' ? act.preview.values : null
  if (!vals || vals.length > 40) return <PreviewView act={act} hex={hex} title="Ausgabe" />
  const sum = vals.reduce((a, b) => a + b, 0)
  const isProb = vals.every((v) => v >= -0.001 && v <= 1.001) && Math.abs(sum - 1) < 0.05
  const maxAbs = Math.max(1e-6, ...vals.map((v) => Math.abs(v)))
  const maxI = vals.indexOf(Math.max(...vals))
  return (
    <div className="flex flex-col gap-0.5">
      {!isProb && <div className="text-[10px] text-[#5b6168]">Roh-Scores (noch keine Wahrscheinlichkeiten)</div>}
      {vals.map((v, i) => (
        <div key={i} className="flex items-center gap-2 text-[10px]">
          <span className="w-5 text-right text-[#7a8088]">{i}</span>
          <div className="h-3 flex-1 overflow-hidden rounded bg-[#0b0e11]">
            <div className="h-full rounded" style={{ width: `${(Math.abs(v) / maxAbs) * 100}%`, background: i === maxI ? hex : '#3a4148' }} />
          </div>
          <span className="w-12 text-right font-mono text-[#9aa1a8]">{isProb ? `${(v * 100).toFixed(0)}%` : v.toFixed(2)}</span>
        </div>
      ))}
    </div>
  )
}

// Pool: feature maps before → after (smaller/coarser).
function PoolView({ before, after, hex }: { before?: NodeActivation; after: NodeActivation; hex: string }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="text-[10px] text-[#5b6168]">vorher</div>
      <SmallPreview act={before} hex={hex} />
      <div className="text-center text-[11px] text-[#7a8088]">↓ zusammengefasst &amp; verkleinert</div>
      <div className="text-[10px] text-[#5b6168]">nachher</div>
      <SmallPreview act={after} hex={hex} />
    </div>
  )
}

// Norm: value distribution before vs after (recentred + rescaled).
function NormView({ before, after, hex }: { before?: NodeActivation; after: NodeActivation; hex: string }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="text-[10px] text-[#5b6168]">Verteilung vorher (μ {before?.stats.mean ?? '–'}, σ {before?.stats.std ?? '–'})</div>
      <Histogram values={previewValues(before)} hex="#9aa1a8" />
      <div className="text-[10px] text-[#5b6168]">Verteilung nachher (μ {after.stats.mean}, σ {after.stats.std})</div>
      <Histogram values={previewValues(after)} hex={hex} />
    </div>
  )
}

// Reshape: same numbers, new shape (no computation).
function ReshapeView({
  before, after, inShape, outShape, hex,
}: {
  before?: NodeActivation
  after: NodeActivation
  inShape?: number[]
  outShape?: number[]
  hex: string
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="font-mono text-[10px] text-[#5b6168]">vorher [{inShape?.join(', ') ?? '?'}]</div>
      <SmallPreview act={before} hex={hex} />
      <div className="text-center text-[11px] text-[#7a8088]">↓ nur umsortiert — gleiche Zahlen</div>
      <div className="font-mono text-[10px] text-[#5b6168]">nachher [{outShape?.join(', ') ?? '?'}]</div>
      <SmallPreview act={after} hex={hex} />
    </div>
  )
}

// Merge: several inputs combine into one output.
const MERGE_OP: Record<string, string> = { Add: '+', Multiply: '×', Concat: '⊕', Stack: '▦' }
function MergeView({ inputs, after, type, hex }: { inputs: NodeActivation[]; after: NodeActivation; type: string; hex: string }) {
  const op = MERGE_OP[type] ?? '⋈'
  return (
    <div className="flex flex-col gap-2">
      <div className="text-[10px] text-[#5b6168]">Eingänge</div>
      {inputs.map((a, i) => (
        <div key={i} className="flex items-center gap-2">
          {i > 0 && <span className="text-sm" style={{ color: hex }}>{op}</span>}
          <SmallPreview act={a} hex={hex} />
        </div>
      ))}
      <div className="text-center text-[11px] text-[#7a8088]">= Ergebnis</div>
      <SmallPreview act={after} hex={hex} />
    </div>
  )
}

// Conv: output feature maps + learned kernels.
function ConvView({ act, weights, hex }: { act: NodeActivation; weights?: Weights; hex: string }) {
  return (
    <div className="flex flex-col gap-3">
      {act.preview?.kind === 'maps' && (
        <div>
          <div className="mb-1 text-[11px] text-[#9aa1a8]">Feature-Maps ({act.preview.channels} Kanäle, jeder ein Filter-Ergebnis)</div>
          <div className="flex flex-wrap gap-1">
            {act.preview.maps.map((g, i) => <Heatmap key={i} grid={g} cell={9} gap={0} mode="mono" hex={hex} />)}
          </div>
        </div>
      )}
      {weights?.kind === 'kernels' && (
        <div>
          <div className="mb-1 text-[11px] text-[#9aa1a8]">Kernel ({weights.out_channels}×{weights.in_channels}, gemittelt über Eingangskanäle)</div>
          <div className="flex flex-wrap gap-1">
            {weights.kernels.map((k, i) => <Heatmap key={i} grid={k} cell={10} gap={1} mode="diverging" />)}
          </div>
        </div>
      )}
    </div>
  )
}
