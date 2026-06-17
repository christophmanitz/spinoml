import { create } from 'zustand'
import { useGraphStore, captureSnapshot, type GraphSnapshot } from './GraphStore'
import { useHistoryStore, suspendHistory } from '../history/store'
import { DEFAULT_SUBGRAPH } from '../layers/registry'

// Subcanvas navigation. GraphStore always holds the CURRENTLY-FOCUSED graph.
// Each frame remembers the parent graph (as a snapshot) we came from, plus which
// Group node we descended into. On exit we fold the edited subgraph back into
// that group's `subgraph` param. Persistence folds the whole stack to the root
// (see captureRootSnapshot) so what's saved is always the full nested model.

type Frame = { groupId: string; label: string; parent: GraphSnapshot }

type ScopeState = {
  stack: Frame[]
  enterGroup: (groupId: string) => void
  /** Pop frames until the stack has `depth` entries (0 = root). */
  exitTo: (depth: number) => void
  /** Drop scope WITHOUT touching GraphStore — call when a new root graph is
   *  loaded externally (file open, template, new). */
  reset: () => void
}

function setSubgraph(snap: GraphSnapshot, groupId: string, sub: GraphSnapshot): GraphSnapshot {
  return {
    nodes: snap.nodes.map((n) =>
      n.id === groupId ? { ...n, params: { ...n.params, subgraph: sub } } : n,
    ),
    edges: snap.edges,
  }
}

export const useScopeStore = create<ScopeState>((set, get) => ({
  stack: [],

  enterGroup: (groupId) => {
    const node = useGraphStore.getState().nodes.find((n) => n.id === groupId)
    if (!node || node.data.layerType !== 'Subgraph') return
    const parent = captureSnapshot(useGraphStore.getState())
    const sub = (node.data.params.subgraph as GraphSnapshot | undefined) ?? (DEFAULT_SUBGRAPH as GraphSnapshot)
    const label = String(node.data.params.class_name ?? 'SubModule') || 'SubModule'
    set({ stack: [...get().stack, { groupId, label, parent }] })
    suspendHistory(() => useGraphStore.getState().loadSnapshot(sub))
    useHistoryStore.getState().clear()
  },

  exitTo: (depth) => {
    const stack = [...get().stack]
    while (stack.length > depth) {
      const inner = captureSnapshot(useGraphStore.getState())
      const frame = stack.pop()!
      const parent = setSubgraph(frame.parent, frame.groupId, inner)
      suspendHistory(() => useGraphStore.getState().loadSnapshot(parent))
      useHistoryStore.getState().clear()
    }
    set({ stack })
  },

  reset: () => set({ stack: [] }),
}))

/** The full nested model: fold the focused graph up through every frame. When
 *  at root this is just the current graph. */
export function captureRootSnapshot(): GraphSnapshot {
  const { stack } = useScopeStore.getState()
  let inner = captureSnapshot(useGraphStore.getState())
  for (let i = stack.length - 1; i >= 0; i--) {
    inner = setSubgraph(stack[i].parent, stack[i].groupId, inner)
  }
  return inner
}
