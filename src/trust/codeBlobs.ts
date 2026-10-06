import { sha256Hex } from '../training/snapshot'

/** The four places in the graph where a node legitimately carries arbitrary
 *  Python. Kept as a closed union so a trust record can never be replayed
 *  across kinds (the kind is part of the hash domain). `custom-init-args` is
 *  executable too: codegen emits `Cls(<init_args>)`. */
export type CodeKind =
  | 'custom-layer'
  | 'custom-init-args'
  | 'dataop-script'
  | 'data-custom-script'

/** A piece of executable code found in a graph. `path` is the node id for a
 *  top-level node, or `outerId/innerId/...` for a node nested inside Subgraph
 *  nodes, so two same-named inner nodes in different groups stay distinct. */
export type CodeBlob = {
  kind: CodeKind
  nodeId: string
  path: string
  source: string
}

/** Live React-Flow architecture node shape (canvas/GraphStore). */
export type ArchNodeInput = {
  id: string
  data?: { layerType?: string; params?: Record<string, unknown> }
}

/** Live data-canvas node (`data.dataType`) or a serialized data snapshot
 *  (`dataType` flat). Both are accepted so a file and a live graph agree. */
export type DataNodeInput = {
  id: string
  dataType?: string
  params?: Record<string, unknown>
  data?: { dataType?: string; params?: Record<string, unknown> }
}

const MAX_DEPTH = 16

type NormalizedNode = {
  id: string
  layerType: string | undefined
  params: Record<string, unknown> | undefined
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function stringId(value: unknown): string {
  return typeof value === 'string' ? value : String(value ?? '')
}

/** Non-string values are stringified (never dropped): a source of `5` becomes
 *  `'5'`, an object becomes `'[object Object]'` — so an LLM cannot hide code by
 *  handing a non-string. Only empty/whitespace means "nothing executes". */
function sourceOf(params: Record<string, unknown> | undefined, key: string): string {
  if (!params) return ''
  const value = params[key]
  if (typeof value === 'string') return value
  if (value === undefined || value === null) return ''
  return String(value)
}

/** Nested graph content lives in `params.subgraph = { nodes, edges }` (the
 *  GraphSnapshot shape). Any object with a `nodes` array is traversed, even if
 *  the node's layerType is unknown — defensive so a forged node cannot hide a
 *  nested graph behind a wrong type label. */
function subgraphNodes(params: Record<string, unknown> | undefined): ReadonlyArray<unknown> {
  if (!params) return []
  const sub = params.subgraph
  if (!isRecord(sub)) return []
  const nodes = sub.nodes
  return Array.isArray(nodes) ? nodes : []
}

function pushBlob(out: CodeBlob[], kind: CodeKind, nodeId: string, path: string, source: string): void {
  if (source.trim() === '') return
  out.push({ kind, nodeId, path, source })
}

function normalizeArchNode(raw: unknown): NormalizedNode | null {
  if (!isRecord(raw)) return null
  const id = stringId(raw.id)
  const data = isRecord(raw.data) ? raw.data : undefined
  let layerType: string | undefined
  let params: Record<string, unknown> | undefined
  if (data) {
    if (typeof data.layerType === 'string') layerType = data.layerType
    if (isRecord(data.params)) params = data.params
  }
  if (layerType === undefined && typeof raw.layerType === 'string') layerType = raw.layerType
  if (params === undefined && isRecord(raw.params)) params = raw.params
  return { id, layerType, params }
}

function walkArch(nodes: ReadonlyArray<unknown>, prefix: string, depth: number, out: CodeBlob[]): void {
  for (const raw of nodes) {
    const node = normalizeArchNode(raw)
    if (!node) continue
    const path = prefix ? `${prefix}/${node.id}` : node.id
    if (node.layerType === 'Custom') {
      pushBlob(out, 'custom-layer', node.id, path, sourceOf(node.params, 'source'))
      // `init_args` is emitted verbatim as `Cls(<init_args>)` in the generated
      // model, so it is arbitrary executable Python too. Push it directly after
      // the source blob to keep the order deterministic (source, then args).
      pushBlob(out, 'custom-init-args', node.id, path + '#init_args', sourceOf(node.params, 'init_args'))
    } else if (node.layerType === 'DataOp') {
      pushBlob(out, 'dataop-script', node.id, path, sourceOf(node.params, 'script'))
    }
    const nested = subgraphNodes(node.params)
    if (nested.length === 0) continue
    if (depth + 1 > MAX_DEPTH) {
      // Too deep to walk reliably: fold the whole remainder into ONE synthetic
      // blob so nothing past the cap can ever be skipped silently.
      pushBlob(
        out,
        'custom-layer',
        node.id,
        path,
        JSON.stringify({ id: node.id, layerType: node.layerType, params: node.params }),
      )
    } else {
      walkArch(nested, path, depth + 1, out)
    }
  }
}

/** Every intentional-arbitrary-code blob in an architecture graph, depth-first
 *  in array order, including nodes nested inside Subgraph nodes to any depth. */
export function collectCodeBlobs(nodes: ReadonlyArray<ArchNodeInput>): CodeBlob[] {
  const out: CodeBlob[] = []
  walkArch(nodes, '', 0, out)
  return out
}

function normalizeDataNode(raw: unknown): NormalizedNode | null {
  if (!isRecord(raw)) return null
  const id = stringId(raw.id)
  const data = isRecord(raw.data) ? raw.data : undefined
  let layerType: string | undefined
  let params: Record<string, unknown> | undefined
  if (data) {
    if (typeof data.dataType === 'string') layerType = data.dataType
    if (isRecord(data.params)) params = data.params
  }
  if (layerType === undefined && typeof raw.dataType === 'string') layerType = raw.dataType
  if (params === undefined && isRecord(raw.params)) params = raw.params
  return { id, layerType, params }
}

/** Every CustomScript code blob in a data graph (only non-empty bodies). */
export function collectDataCodeBlobs(nodes: ReadonlyArray<DataNodeInput>): CodeBlob[] {
  const out: CodeBlob[] = []
  for (const raw of nodes) {
    const node = normalizeDataNode(raw)
    if (!node) continue
    if (node.layerType === 'CustomScript') {
      pushBlob(out, 'data-custom-script', node.id, node.id, sourceOf(node.params, 'code'))
    }
  }
  return out
}

/** Lowercase hex SHA-256 of `kind + '\0' + source` (UTF-8). The NUL separator
 *  plus the kind prefix make collisions across kinds impossible. */
export async function hashBlob(kind: CodeKind, source: string): Promise<string> {
  return sha256Hex(kind + '\0' + source)
}
