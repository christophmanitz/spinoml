// Data-graph → ordered pipeline PLAN. PURE (same snapshot → same plan, no I/O),
// like trainingGenerator.ts. Topologically sorts the DAG so the emitted script
// (dataCodegen.ts) runs steps in dependency order, and reports issues/warnings
// for the inspector. Kept decoupled from the data store so it can be reused by
// the code panel, the inspector and (later) the chatbot.

import type { DataGraphSnapshot } from '../data/graph/store'
import { DATA_NODES } from '../data/graph/registry'

export type DataPlanNode = { id: string; dataType: string; params: Record<string, unknown> }

export type DataPlan = {
  /** Nodes in topological (run) order. */
  order: DataPlanNode[]
  hasSource: boolean
  hasSink: boolean
}

export type DataCompile = {
  ok: boolean
  issues: string[]
  warnings: string[]
  plan: DataPlan | null
}

type SnapNode = DataGraphSnapshot['nodes'][number]

const SOURCE_TYPES = new Set(['TableSource'])
const SINK_TYPES = new Set(['WriteDataset', 'SmilesToGraph', 'StructureToGraph'])

export function compileDataGraph(snapshot: DataGraphSnapshot): DataCompile {
  const issues: string[] = []
  const warnings: string[] = []
  const nodes = snapshot.nodes
  const edges = snapshot.edges

  if (nodes.length === 0) {
    return { ok: false, issues: ['Leerer Graph — zieh einen TableSource-Knoten auf den Canvas.'], warnings, plan: null }
  }

  for (const n of nodes) {
    if (!DATA_NODES[n.dataType]) issues.push(`Unbekannter Knotentyp: ${n.dataType} (${n.id}).`)
  }

  // Topological sort (Kahn). A cycle leaves some nodes unscheduled → issue.
  const byId = new Map<string, SnapNode>(nodes.map((n) => [n.id, n]))
  const indeg = new Map<string, number>(nodes.map((n) => [n.id, 0]))
  const adj = new Map<string, string[]>(nodes.map((n) => [n.id, []]))
  for (const e of edges) {
    if (!byId.has(e.source) || !byId.has(e.target)) continue
    adj.get(e.source)!.push(e.target)
    indeg.set(e.target, (indeg.get(e.target) ?? 0) + 1)
  }
  const queue = nodes.filter((n) => (indeg.get(n.id) ?? 0) === 0).map((n) => n.id)
  // Stable order: roots in their snapshot order.
  const order: DataPlanNode[] = []
  const localIndeg = new Map(indeg)
  while (queue.length) {
    const id = queue.shift()!
    const n = byId.get(id)!
    order.push({ id: n.id, dataType: n.dataType, params: n.params })
    for (const nx of adj.get(id) ?? []) {
      localIndeg.set(nx, (localIndeg.get(nx) ?? 0) - 1)
      if ((localIndeg.get(nx) ?? 0) === 0) queue.push(nx)
    }
  }
  if (order.length < nodes.length) {
    issues.push('Der Graph enthält einen Zyklus — Daten-Pipelines müssen azyklisch sein.')
  }

  // The emitted script threads ONE shared pandas DataFrame `df`, so the graph must
  // be a single LINEAR chain: exactly one root, no branching, and the first node
  // must establish df (TableSource, or a CustomScript that loads it). Otherwise
  // independent chains would silently clobber df / df would be None.
  const roots = nodes.filter((n) => (indeg.get(n.id) ?? 0) === 0)
  if (roots.length > 1) {
    issues.push('Mehrere Quell-/Wurzelknoten — eine Pipeline muss EINE lineare Kette durch einen gemeinsamen DataFrame sein. Baue getrennte Pipelines, oder führe Zweige in einem CustomScript zusammen.')
  }
  const branch = nodes.find((n) => (adj.get(n.id)?.length ?? 0) > 1)
  if (branch) {
    issues.push(`Knoten ${branch.id} verzweigt zu mehreren Folgeknoten — die Pipeline muss linear sein (ein df).`)
  }
  const first = order[0]
  if (first && first.dataType !== 'TableSource' && first.dataType !== 'CustomScript') {
    issues.push('Der erste Knoten muss ein TableSource sein (oder ein CustomScript, das df lädt) — sonst ist df leer.')
  }

  const hasSource = nodes.some((n) => SOURCE_TYPES.has(n.dataType))
  const hasSink = nodes.some((n) => SINK_TYPES.has(n.dataType))
  if (!hasSource) warnings.push('Kein TableSource — die Pipeline beginnt ohne geladene Tabelle (ein CustomScript kann eigene Daten laden).')
  if (!hasSink) warnings.push('Kein Ausgabeknoten (WriteDataset / SmilesToGraph / StructureToGraph) — es wird kein Datensatz geschrieben.')

  // Disconnected nodes (no edges at all) when there is more than one node.
  if (nodes.length > 1) {
    const connected = new Set<string>()
    for (const e of edges) { connected.add(e.source); connected.add(e.target) }
    const loose = nodes.filter((n) => !connected.has(n.id))
    if (loose.length) warnings.push(`Nicht verbunden: ${loose.map((n) => n.id).join(', ')} — verdrahte die Schritte zu einer Kette.`)
  }

  const ok = issues.length === 0
  const plan: DataPlan | null = ok ? { order, hasSource, hasSink } : null
  return { ok, issues, warnings, plan }
}
