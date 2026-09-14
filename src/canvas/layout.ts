// Shared layered ("Sugiyama-lite") graph layout used by every canvas
// (architecture / training / data). Replaces the old DFS column-packers that
// overlapped nodes and placed them by walk-order rather than true depth.
//
// How it works:
//  1. RANK each node by longest path from the sources (nodes with no incoming
//     edge). An edge always points from a lower rank to a higher one, so when we
//     map rank → the flow axis the edges ALWAYS run forward (top→bottom for TB,
//     left→right for LR) — no backward/sideways edges, which is what made the
//     old layout look wrong against the flow-direction handles.
//  2. ORDER nodes within each rank by the barycenter (mean order) of their
//     neighbours, a couple of up/down sweeps — the standard crossing-reduction
//     heuristic, so connected nodes line up and edges don't criss-cross.
//  3. PLACE: rank → position along the flow axis, order → position across it.
//     Each rank gets distinct cross slots, so nodes never overlap. Layers are
//     centred against the widest one for a tidy, symmetric look.

import type { XYPosition } from '@xyflow/react'

export type FlowDir = 'TB' | 'LR'
export type LayoutNode = { id: string }
export type LayoutEdge = { source: string; target: string }

export type LayoutOpts = {
  direction?: FlowDir
  /** gap between consecutive layers, along the flow axis */
  rankGap?: number
  /** gap between siblings, across the flow axis */
  crossGap?: number
  originX?: number
  originY?: number
}

function stableSortBy<T>(arr: T[], key: (x: T) => number): T[] {
  return arr
    .map((item, i) => ({ item, i, k: key(item) }))
    .sort((a, b) => (a.k - b.k) || (a.i - b.i))
    .map((x) => x.item)
}

export function layeredLayout(
  nodes: LayoutNode[],
  edges: LayoutEdge[],
  opts: LayoutOpts = {},
): Map<string, XYPosition> {
  const dir = opts.direction ?? 'TB'
  const rankGap = opts.rankGap ?? (dir === 'LR' ? 250 : 130)
  const crossGap = opts.crossGap ?? (dir === 'LR' ? 120 : 230)
  const originX = opts.originX ?? 80
  const originY = opts.originY ?? 60

  const out = new Map<string, XYPosition>()
  if (nodes.length === 0) return out

  const ids = new Set(nodes.map((n) => n.id))
  const succ = new Map<string, string[]>()
  const pred = new Map<string, string[]>()
  for (const n of nodes) { succ.set(n.id, []); pred.set(n.id, []) }
  for (const e of edges) {
    if (e.source === e.target || !ids.has(e.source) || !ids.has(e.target)) continue
    succ.get(e.source)!.push(e.target)
    pred.get(e.target)!.push(e.source)
  }

  // ── 1. ranks via longest path (Kahn topological order) ──
  const indeg = new Map<string, number>()
  for (const n of nodes) indeg.set(n.id, pred.get(n.id)!.length)
  const rank = new Map<string, number>()
  const queue: string[] = []
  for (const n of nodes) if (indeg.get(n.id) === 0) { rank.set(n.id, 0); queue.push(n.id) }
  for (let qi = 0; qi < queue.length; qi++) {
    const u = queue[qi]
    const ru = rank.get(u) ?? 0
    for (const v of succ.get(u)!) {
      if ((rank.get(v) ?? -1) < ru + 1) rank.set(v, ru + 1)
      indeg.set(v, indeg.get(v)! - 1)
      if (indeg.get(v) === 0) queue.push(v)
    }
  }
  // Cycle fallback: any node a topo-walk never reached (part of a cycle) gets a
  // rank just past its highest-ranked predecessor, so it's still placed sanely.
  for (const n of nodes) {
    if (rank.has(n.id)) continue
    const prs = pred.get(n.id)!.map((p) => rank.get(p) ?? 0)
    rank.set(n.id, prs.length ? Math.max(...prs) + 1 : 0)
  }

  // ── 2. group into layers (input order = stable initial within-layer order) ──
  const maxRank = Math.max(...nodes.map((n) => rank.get(n.id)!))
  const layers: string[][] = Array.from({ length: maxRank + 1 }, () => [])
  for (const n of nodes) layers[rank.get(n.id)!].push(n.id)

  const orderIndex = new Map<string, number>()
  const reindex = () => { for (const layer of layers) layer.forEach((id, i) => orderIndex.set(id, i)) }
  reindex()

  const barycenter = (id: string, neighbours: Map<string, string[]>): number => {
    const ns = neighbours.get(id)!.filter((x) => orderIndex.has(x))
    if (ns.length === 0) return orderIndex.get(id) ?? 0
    return ns.reduce((a, x) => a + orderIndex.get(x)!, 0) / ns.length
  }

  // alternating down (order by predecessors) / up (order by successors) sweeps
  for (let sweep = 0; sweep < 4; sweep++) {
    if (sweep % 2 === 0) {
      for (let r = 1; r < layers.length; r++) {
        layers[r] = stableSortBy(layers[r], (id) => barycenter(id, pred))
        reindex()
      }
    } else {
      for (let r = layers.length - 2; r >= 0; r--) {
        layers[r] = stableSortBy(layers[r], (id) => barycenter(id, succ))
        reindex()
      }
    }
  }

  // ── 3. place, centring each layer against the widest one ──
  const maxSize = Math.max(...layers.map((l) => l.length))
  layers.forEach((layer, r) => {
    const start = (maxSize - layer.length) / 2
    layer.forEach((id, i) => {
      const cross = (start + i) * crossGap
      const along = r * rankGap
      out.set(id, dir === 'LR'
        ? { x: originX + along, y: originY + cross }
        : { x: originX + cross, y: originY + along })
    })
  })
  return out
}
