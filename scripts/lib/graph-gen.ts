// Property-based graph generator + INDEPENDENT shape/parameter oracle, plus the
// mutation operators consumed by `scripts/test-fuzz.ts`.
//
// This module is the whole independent side of the property test: it builds
// graphs CORRECT BY CONSTRUCTION and tracks every tensor's shape and every
// module's parameter count by hand. It deliberately does NOT call the app's
// shape inference (`src/inference/*`) — if the generator and the app agree, the
// app's shape inference is correct for a whole family of random architectures,
// not just for the three hand-written reference graphs.
//
// Determinism: a seeded PRNG (mulberry32) drives everything, and node ids and
// positions are derived from the creation sequence, so the same (seed, index)
// yields byte-identical snapshots. `test-property.ts` asserts this.
//
// Node ids / positions are also kept stable under mutation so a failed mutant
// can be replayed.

import type { LayerNode, GraphSnapshot } from '../../src/canvas/GraphStore'
import type { Edge } from '@xyflow/react'
import { LAYERS, defaultParamsFor } from '../../src/layers/registry'

// ─────────────────────────────────────────────────────────────────────────────
// Seeded PRNG
// ─────────────────────────────────────────────────────────────────────────────

/** mulberry32 — small, fast, deterministic 32-bit PRNG in [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function randInt(rng: () => number, lo: number, hi: number): number {
  return lo + Math.floor(rng() * (hi - lo + 1))
}

function pick<T>(rng: () => number, arr: readonly T[]): T {
  return arr[Math.floor(rng() * arr.length)]
}

function chance(rng: () => number, p: number): boolean {
  return rng() < p
}

// ─────────────────────────────────────────────────────────────────────────────
// Public shapes
// ─────────────────────────────────────────────────────────────────────────────

export type GenFamily = 'mlp' | 'cnn' | 'residual' | 'branch' | 'multi-input' | 'sequence'

export const FAMILIES: readonly GenFamily[] = ['mlp', 'cnn', 'residual', 'branch', 'multi-input', 'sequence']

export type GenInput = { name: string; shape: number[]; dtype: string }

export type GeneratedGraph = {
  family: GenFamily
  index: number
  snapshot: GraphSnapshot
  /** forward inputs, shape INCLUDES the batch dimension (always 1 here) */
  inputs: GenInput[]
  /** expected output shape WITHOUT the batch dimension */
  expectedOutputShape: number[]
  /** independent parameter-count expectation (trainable == total here) */
  expectedParams: number
  /** independent per-module output shapes, keyed by node id (batch included) */
  expectedModuleShapes: Record<string, number[]>
}

// ─────────────────────────────────────────────────────────────────────────────
// Parameter-count oracle (independent formulas)
// ─────────────────────────────────────────────────────────────────────────────

export function linearParams(inFeatures: number, outFeatures: number, bias = true): number {
  return inFeatures * outFeatures + (bias ? outFeatures : 0)
}

export function conv2dParams(inChannels: number, outChannels: number, kernel: number, bias = true): number {
  return outChannels * inChannels * kernel * kernel + (bias ? outChannels : 0)
}

/** BatchNorm affine weight + bias. Running mean/var are buffers, not params. */
export function batchNormParams(numFeatures: number): number {
  return 2 * numFeatures
}

export function embeddingParams(numEmbeddings: number, embeddingDim: number): number {
  return numEmbeddings * embeddingDim
}

// ─────────────────────────────────────────────────────────────────────────────
// Builder — tracks shapes + params as it wires the graph
// ─────────────────────────────────────────────────────────────────────────────

const ACTIVATIONS: readonly { type: string; params: Record<string, unknown> }[] = [
  { type: 'ReLU', params: { inplace: false } },
  { type: 'Tanh', params: {} },
  { type: 'GELU', params: {} },
  { type: 'Sigmoid', params: {} },
]

class Builder {
  readonly nodes: GraphSnapshot['nodes'] = []
  readonly edges: { source: string; target: string }[] = []
  readonly moduleShapes: Record<string, number[]> = {}
  readonly inputs: GenInput[] = []
  params = 0
  private seq = 0

  /** Add a node, merging the registry defaults with `overrides` so the stored
   *  snapshot is already fully-coerced and `validateGraphState` sees no
   *  missing-parameter warnings. Records the module output shape when the layer
   *  compiles to an `nn.Module` attribute. */
  add(layerType: string, overrides: Record<string, unknown>, outputShape: number[]): string {
    this.seq += 1
    const id = `n${this.seq}`
    const params = { ...defaultParamsFor(layerType), ...overrides }
    this.nodes.push({
      id,
      layerType,
      params,
      position: { x: 60 * (this.seq % 8), y: 70 * Math.floor(this.seq / 8) },
    })
    const spec = LAYERS[layerType]
    if (spec?.pytorchModule) this.moduleShapes[id] = [...outputShape]
    if (spec?.kind === 'input') {
      this.inputs.push({
        name: String(params.name ?? `x${this.seq}`),
        shape: [...((params.shape as number[]) ?? outputShape)],
        dtype: String(params.dtype ?? 'float32'),
      })
    }
    return id
  }

  link(source: string, target: string): void {
    this.edges.push({ source, target })
  }

  addParams(n: number): void {
    this.params += n
  }
}

function addActivation(b: Builder, rng: () => number, from: string, shape: number[]): string {
  const act = pick(rng, ACTIVATIONS)
  const id = b.add(act.type, act.params, shape)
  b.link(from, id)
  return id
}

// ─────────────────────────────────────────────────────────────────────────────
// Families
// ─────────────────────────────────────────────────────────────────────────────

function familyMLP(b: Builder, rng: () => number): number[] {
  const inDim = randInt(rng, 1, 64)
  const x = b.add('Input', { name: 'x', shape: [1, inDim], dtype: 'float32' }, [1, inDim])
  let cur = x
  let curShape = [1, inDim]
  const layers = randInt(rng, 1, 5)
  for (let i = 0; i < layers; i++) {
    const outDim = randInt(rng, 1, 64)
    const lin = b.add('Linear', { in_features: curShape[1], out_features: outDim, bias: true }, [curShape[0], outDim])
    b.addParams(linearParams(curShape[1], outDim))
    b.link(cur, lin)
    cur = lin
    curShape = [curShape[0], outDim]
    if (chance(rng, 0.4)) {
      const bn = b.add('BatchNorm1d', { num_features: outDim, eps: 1e-5, momentum: 0.1 }, curShape)
      b.addParams(batchNormParams(outDim))
      b.link(cur, bn)
      cur = bn
    }
    cur = addActivation(b, rng, cur, curShape)
    if (chance(rng, 0.3)) {
      const d = b.add('Dropout', { p: pick(rng, [0.1, 0.2, 0.5]) }, curShape)
      b.link(cur, d)
      cur = d
    }
  }
  // A trailing Reshape exercises the functional reshape path (and gives the
  // fuzz suite a reliable "wrong element count" target).
  const rsh = b.add('Reshape', { shape: [curShape[1], 1] }, [curShape[0], curShape[1], 1])
  b.link(cur, rsh)
  cur = rsh
  curShape = [curShape[0], curShape[1], 1]
  const out = b.add('Output', { name: 'out' }, curShape)
  b.link(cur, out)
  return [curShape[1], 1]
}

function convOutDim(size: number, k: number, s: number, p: number): number {
  return Math.floor((size + 2 * p - k) / s) + 1
}

function familyCNN(b: Builder, rng: () => number): number[] {
  let c = randInt(rng, 1, 4)
  let h = pick(rng, [8, 12, 16])
  let w = pick(rng, [8, 12, 16])
  const x = b.add('Input', { name: 'x', shape: [1, c, h, w], dtype: 'float32' }, [1, c, h, w])
  let cur = x
  let curShape!: number[]
  const blocks = randInt(rng, 1, 3)
  for (let i = 0; i < blocks; i++) {
    const k = pick(rng, [1, 3, 5])
    const s = pick(rng, [1, 2])
    const p = Math.floor(k / 2)
    const ho = convOutDim(h, k, s, p)
    const wo = convOutDim(w, k, s, p)
    const outC = randInt(rng, 1, 16)
    const conv = b.add(
      'Conv2d',
      { in_channels: c, out_channels: outC, kernel_size: [k, k], stride: [s, s], padding: [p, p], bias: true },
      [1, outC, ho, wo],
    )
    b.addParams(conv2dParams(c, outC, k))
    b.link(cur, conv)
    cur = conv
    curShape = [1, outC, ho, wo]
    c = outC
    h = ho
    w = wo
    if (chance(rng, 0.4)) {
      const bn = b.add('BatchNorm2d', { num_features: c, eps: 1e-5, momentum: 0.1 }, curShape)
      b.addParams(batchNormParams(c))
      b.link(cur, bn)
      cur = bn
    }
    cur = addActivation(b, rng, cur, curShape)
    if (chance(rng, 0.5) && h >= 2 && w >= 2) {
      const h2 = Math.floor(h / 2)
      const w2 = Math.floor(w / 2)
      const pool = b.add('MaxPool2d', { kernel_size: [2, 2], stride: [2, 2], padding: [0, 0] }, [1, c, h2, w2])
      b.link(cur, pool)
      cur = pool
      h = h2
      w = w2
    }
  }
  const flat = b.add('Flatten', { start_dim: 1, end_dim: -1 }, [1, c * h * w])
  b.link(cur, flat)
  cur = flat
  const outDim = randInt(rng, 1, 32)
  const lin = b.add('Linear', { in_features: c * h * w, out_features: outDim, bias: true }, [1, outDim])
  b.addParams(linearParams(c * h * w, outDim))
  b.link(cur, lin)
  cur = lin
  curShape = [1, outDim]
  const out = b.add('Output', { name: 'out' }, curShape)
  b.link(cur, out)
  return [outDim]
}

function familyResidual(b: Builder, rng: () => number): number[] {
  const d = randInt(rng, 2, 32)
  const x = b.add('Input', { name: 'x', shape: [1, d], dtype: 'float32' }, [1, d])
  const l1 = b.add('Linear', { in_features: d, out_features: d, bias: true }, [1, d])
  b.addParams(linearParams(d, d))
  b.link(x, l1)
  const a1 = addActivation(b, rng, l1, [1, d])
  const l2 = b.add('Linear', { in_features: d, out_features: d, bias: true }, [1, d])
  b.addParams(linearParams(d, d))
  b.link(a1, l2)
  // Skip connection: the Add merges the original input with the second Linear.
  const add = b.add('Add', {}, [1, d])
  b.link(x, add)
  b.link(l2, add)
  const a2 = addActivation(b, rng, add, [1, d])
  const out = b.add('Output', { name: 'out' }, [1, d])
  b.link(a2, out)
  return [d]
}

function familyBranch(b: Builder, rng: () => number): number[] {
  const inDim = randInt(rng, 2, 32)
  const x = b.add('Input', { name: 'x', shape: [1, inDim], dtype: 'float32' }, [1, inDim])
  const nBranches = randInt(rng, 2, 3)
  let sum = 0
  const heads: string[] = []
  for (let i = 0; i < nBranches; i++) {
    const w = randInt(rng, 2, 24)
    const lin = b.add('Linear', { in_features: inDim, out_features: w, bias: true }, [1, w])
    b.addParams(linearParams(inDim, w))
    b.link(x, lin)
    const head = chance(rng, 0.5) ? addActivation(b, rng, lin, [1, w]) : lin
    heads.push(head)
    sum += w
  }
  const cat = b.add('Concat', { dim: 1 }, [1, sum])
  for (const hd of heads) b.link(hd, cat)
  const outDim = randInt(rng, 1, 16)
  const head = b.add('Linear', { in_features: sum, out_features: outDim, bias: true }, [1, outDim])
  b.addParams(linearParams(sum, outDim))
  b.link(cat, head)
  const out = b.add('Output', { name: 'out' }, [1, outDim])
  b.link(head, out)
  return [outDim]
}

function familyMultiInput(b: Builder, rng: () => number): number[] {
  const count = randInt(rng, 2, 3)
  const mode = pick(rng, ['concat', 'add'] as const)
  const names = ['x', 'y', 'z']
  const widths: number[] = []
  for (let i = 0; i < count; i++) widths.push(randInt(rng, 2, 24))
  if (mode === 'add') {
    const w = widths[0]
    for (let i = 1; i < count; i++) widths[i] = w
  }
  const heads: string[] = []
  for (let i = 0; i < count; i++) {
    const inDim = randInt(rng, 2, 24)
    const x = b.add('Input', { name: names[i], shape: [1, inDim], dtype: 'float32' }, [1, inDim])
    const lin = b.add('Linear', { in_features: inDim, out_features: widths[i], bias: true }, [1, widths[i]])
    b.addParams(linearParams(inDim, widths[i]))
    b.link(x, lin)
    heads.push(chance(rng, 0.5) ? addActivation(b, rng, lin, [1, widths[i]]) : lin)
  }
  const headIn = mode === 'concat' ? widths.reduce((a, v) => a + v, 0) : widths[0]
  const mergeLayer = mode === 'concat' ? 'Concat' : 'Add'
  const mergeParams = mode === 'concat' ? { dim: 1 } : {}
  const merge = b.add(mergeLayer, mergeParams, [1, headIn])
  for (const hd of heads) b.link(hd, merge)
  const outDim = randInt(rng, 1, 16)
  const head = b.add('Linear', { in_features: headIn, out_features: outDim, bias: true }, [1, outDim])
  b.addParams(linearParams(headIn, outDim))
  b.link(merge, head)
  const out = b.add('Output', { name: 'out' }, [1, outDim])
  b.link(head, out)
  return [outDim]
}

function familySequence(b: Builder, rng: () => number): number[] {
  const vocab = randInt(rng, 50, 500)
  const dim = randInt(rng, 4, 32)
  const length = randInt(rng, 3, 12)
  const seq = b.add('Input', { name: 'seq', shape: [1, length], dtype: 'int64' }, [1, length])
  const emb = b.add('Embedding', { num_embeddings: vocab, embedding_dim: dim }, [1, length, dim])
  b.addParams(embeddingParams(vocab, dim))
  b.link(seq, emb)
  const flat = b.add('Flatten', { start_dim: 1, end_dim: -1 }, [1, length * dim])
  b.link(emb, flat)
  const outDim = randInt(rng, 1, 16)
  const lin = b.add('Linear', { in_features: length * dim, out_features: outDim, bias: true }, [1, outDim])
  b.addParams(linearParams(length * dim, outDim))
  b.link(flat, lin)
  const out = b.add('Output', { name: 'out' }, [1, outDim])
  b.link(lin, out)
  return [outDim]
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level generation
// ─────────────────────────────────────────────────────────────────────────────

/** Build one graph at `index` for `seed`. Deterministic in both. */
export function graphAt(index: number, seed: number): GeneratedGraph {
  const rng = mulberry32((seed ^ Math.imul(index + 1, 0x9e3779b9)) >>> 0)
  const family = FAMILIES[index % FAMILIES.length]
  const b = new Builder()
  let outShape: number[]
  switch (family) {
    case 'mlp':
      outShape = familyMLP(b, rng)
      break
    case 'cnn':
      outShape = familyCNN(b, rng)
      break
    case 'residual':
      outShape = familyResidual(b, rng)
      break
    case 'branch':
      outShape = familyBranch(b, rng)
      break
    case 'multi-input':
      outShape = familyMultiInput(b, rng)
      break
    case 'sequence':
      outShape = familySequence(b, rng)
      break
  }
  return {
    family,
    index,
    snapshot: { nodes: b.nodes, edges: b.edges },
    inputs: b.inputs,
    expectedOutputShape: outShape,
    expectedParams: b.params,
    expectedModuleShapes: b.moduleShapes,
  }
}

export function generatePropertyGraphs(seed: number, n: number): GeneratedGraph[] {
  const out: GeneratedGraph[] = []
  for (let i = 0; i < n; i++) out.push(graphAt(i, seed))
  return out
}

export function resolveSeed(): number {
  const raw = process.env.PROPERTY_SEED
  if (raw === undefined || raw.trim() === '') return 1234
  const n = Number(raw)
  return Number.isFinite(n) ? Math.trunc(n) : 1234
}

export function resolveN(): number {
  const raw = process.env.PROPERTY_N
  if (raw === undefined || raw.trim() === '') return 200
  const n = Number(raw)
  return Number.isFinite(n) && n > 0 ? Math.trunc(n) : 200
}

export function resolveOnly(): number | null {
  const raw = process.env.PROPERTY_ONLY
  if (raw === undefined || raw.trim() === '') return null
  const n = Number(raw)
  return Number.isFinite(n) && n >= 0 ? Math.trunc(n) : null
}

// ─────────────────────────────────────────────────────────────────────────────
// Structural conversion helpers (validator path needs ids; snapshot has none)
// ─────────────────────────────────────────────────────────────────────────────

export function snapshotToLayerNodes(snapshot: GraphSnapshot): LayerNode[] {
  return snapshot.nodes.map((n) => ({
    id: n.id,
    type: 'layer' as const,
    position: n.position ?? { x: 0, y: 0 },
    data: { layerType: n.layerType, params: n.params },
  }))
}

export function snapshotToEdges(snapshot: GraphSnapshot): Edge[] {
  return snapshot.edges.map((e, i) => ({ id: `e${i + 1}`, source: e.source, target: e.target }))
}

/** Snapshot an array of LayerNode/Edge back into the GraphSnapshot shape. */
export function layerGraphToSnapshot(nodes: LayerNode[], edges: Edge[]): GraphSnapshot {
  return {
    nodes: nodes.map((n) => ({
      id: n.id,
      layerType: n.data.layerType,
      params: n.data.params,
      position: { x: n.position.x, y: n.position.y },
    })),
    edges: edges.map((e) => ({ source: e.source, target: e.target })),
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mutation operators (fuzz suite)
// ─────────────────────────────────────────────────────────────────────────────

export type MutantKind = 'structural' | 'coerced' | 'load-reject' | 'semantic'

export type Mutant = {
  operator: string
  kind: MutantKind
  /** the mutated graph as a snapshot (fed to loadSnapshot + generate) */
  snapshot: GraphSnapshot
  /** validateGraphState(snapshot) must report an ERROR-severity issue */
  expectValidatorError: boolean
  /** loadSnapshot(snapshot) must return false and leave the store untouched */
  expectLoadReject: boolean
  /** generate + real /infer must answer ok:false classified `invalid` */
  expectSidecarError: boolean
  note: string
}

function copy(snapshot: GraphSnapshot): GraphSnapshot {
  return structuredClone(snapshot)
}

function nodeIndexOf(snapshot: GraphSnapshot, layerType: string): number {
  return snapshot.nodes.findIndex((n) => n.layerType === layerType)
}

function firstIncoming(snapshot: GraphSnapshot, target: string): { source: string } | undefined {
  return snapshot.edges.find((e) => e.target === target)
}

function outgoingCount(snapshot: GraphSnapshot, id: string): number {
  return snapshot.edges.filter((e) => e.source === id).length
}

/**
 * All mutants derived from one valid graph. Operators that need a node the
 * family does not contain (e.g. a Conv2d in an MLP) are skipped for that graph.
 * The expectation flags are what the fuzz harness asserts; they were set by
 * reading `validateGraphState` / `coerceParams` / the sidecar's `/infer`, not by
 * observing the app's answers to these very mutants.
 */
export function mutantsFor(base: GeneratedGraph): Mutant[] {
  const out: Mutant[] = []
  const src = base.snapshot
  const linear = nodeIndexOf(src, 'Linear')
  const conv = nodeIndexOf(src, 'Conv2d')
  const concat = nodeIndexOf(src, 'Concat')
  const add = nodeIndexOf(src, 'Add')
  const reshape = nodeIndexOf(src, 'Reshape')
  const inputNode = nodeIndexOf(src, 'Input')

  // 1. Deleting a node while keeping its incident edges → dangling edge.
  const withEdges = src.nodes.findIndex((n) => outgoingCount(src, n.id) > 0)
  if (withEdges >= 0) {
    const s = copy(src)
    s.nodes.splice(withEdges, 1)
    out.push({
      operator: 'dangling-node',
      kind: 'structural',
      snapshot: s,
      expectValidatorError: true,
      expectLoadReject: true,
      expectSidecarError: false,
      note: 'a node removed but its edges kept',
    })
  }

  if (src.edges.length > 0) {
    const s = copy(src)
    s.edges[s.edges.length - 1].target = 'ghost_node'
    out.push({
      operator: 'edge-target-unknown',
      kind: 'structural',
      snapshot: s,
      expectValidatorError: true,
      expectLoadReject: true,
      expectSidecarError: false,
      note: 'edge points at a non-existent target',
    })

    const s2 = copy(src)
    s2.edges[s2.edges.length - 1].source = 'ghost_node'
    out.push({
      operator: 'edge-source-unknown',
      kind: 'structural',
      snapshot: s2,
      expectValidatorError: true,
      expectLoadReject: true,
      expectSidecarError: false,
      note: 'edge points at a non-existent source',
    })
  }

  // 2. Duplicate node id.
  {
    const s = copy(src)
    const victim = s.nodes[Math.min(1, s.nodes.length - 1)]
    s.nodes.push({ ...structuredClone(victim) })
    out.push({
      operator: 'duplicate-node-id',
      kind: 'structural',
      snapshot: s,
      expectValidatorError: true,
      expectLoadReject: true,
      expectSidecarError: false,
      note: 'two nodes share one id',
    })
  }

  // 3. Unknown layer type.
  {
    const s = copy(src)
    s.nodes[1].layerType = 'TotallyBogusLayer'
    out.push({
      operator: 'unknown-layer-type',
      kind: 'structural',
      snapshot: s,
      expectValidatorError: true,
      expectLoadReject: true,
      expectSidecarError: false,
      note: 'layer type not in the registry',
    })
  }

  // 4. Self-loop.
  if (src.nodes.length > 1) {
    const s = copy(src)
    s.edges.push({ source: 'n2', target: 'n2' })
    out.push({
      operator: 'self-loop',
      kind: 'structural',
      snapshot: s,
      expectValidatorError: true,
      expectLoadReject: true,
      expectSidecarError: false,
      note: 'edge from a node to itself',
    })
  }

  // 5. Directed cycle (sink → Input).
  {
    const sinks = src.nodes.filter((n) => outgoingCount(src, n.id) === 0)
    const sink = sinks.length > 0 ? sinks[sinks.length - 1] : src.nodes[src.nodes.length - 1]
    const s = copy(src)
    s.edges.push({ source: sink.id, target: src.nodes[0].id })
    out.push({
      operator: 'cycle',
      kind: 'structural',
      snapshot: s,
      expectValidatorError: true,
      expectLoadReject: true,
      expectSidecarError: false,
      note: 'sink wired back to the first node closes a loop',
    })
  }

  // 6. Out-of-range numeric params (coercion keeps finite numbers, validator
  //    then rejects them — loadSnapshot rejects too).
  if (linear >= 0) {
    const s = copy(src)
    s.nodes[linear].params.in_features = 0
    out.push({
      operator: 'zero-dim',
      kind: 'structural',
      snapshot: s,
      expectValidatorError: true,
      expectLoadReject: true,
      expectSidecarError: false,
      note: 'Linear.in_features = 0 (min 1)',
    })

    const s2 = copy(src)
    s2.nodes[linear].params.out_features = -5
    out.push({
      operator: 'negative-dim',
      kind: 'structural',
      snapshot: s2,
      expectValidatorError: true,
      expectLoadReject: true,
      expectSidecarError: false,
      note: 'Linear.out_features = -5 (min 1)',
    })
  }

  // 7. Malformed params that coercion HEALS: the raw validator sees a type
  //    error, but loadSnapshot coerces back to the default and accepts.
  if (linear >= 0) {
    const s = copy(src)
    s.nodes[linear].params.in_features = NaN
    out.push({
      operator: 'nan-dim',
      kind: 'coerced',
      snapshot: s,
      expectValidatorError: true,
      expectLoadReject: false,
      expectSidecarError: false,
      note: 'NaN int → validator param-type; coerce restores default',
    })

    const s2 = copy(src)
    s2.nodes[linear].params.in_features = Infinity
    out.push({
      operator: 'inf-dim',
      kind: 'coerced',
      snapshot: s2,
      expectValidatorError: true,
      expectLoadReject: false,
      expectSidecarError: false,
      note: 'Infinity int → validator param-type; coerce restores default',
    })

    const s3 = copy(src)
    s3.nodes[linear].params.in_features = 'not-a-number'
    out.push({
      operator: 'string-for-int',
      kind: 'coerced',
      snapshot: s3,
      expectValidatorError: true,
      expectLoadReject: false,
      expectSidecarError: false,
      note: 'string where an int is expected',
    })
  }

  if (inputNode >= 0) {
    const s = copy(src)
    s.nodes[inputNode].params.shape = { bogus: true }
    out.push({
      operator: 'object-for-shape',
      kind: 'coerced',
      snapshot: s,
      expectValidatorError: true,
      expectLoadReject: false,
      expectSidecarError: false,
      note: 'object where an int list is expected',
    })

    const s2 = copy(src)
    s2.nodes[inputNode].params.dtype = 'float64'
    out.push({
      operator: 'select-out-of-range',
      kind: 'coerced',
      snapshot: s2,
      expectValidatorError: true,
      expectLoadReject: false,
      expectSidecarError: false,
      note: 'dtype outside the select options',
    })
  }

  // 8. null in a min-bounded int: the raw validator only WARNS (missing), but
  //    coercion turns null into 0 → loadSnapshot rejects on param-range.
  if (linear >= 0) {
    const s = copy(src)
    s.nodes[linear].params.in_features = null
    out.push({
      operator: 'null-in-features',
      kind: 'load-reject',
      snapshot: s,
      expectValidatorError: false,
      expectLoadReject: true,
      expectSidecarError: false,
      note: 'null → coerce 0 → rejected by loadSnapshot',
    })
  }

  // ── SEMANTIC mutants: structurally valid, shape-incompatible ──────────────
  if (linear >= 0) {
    const s = copy(src)
    const cur = Number(s.nodes[linear].params.in_features)
    s.nodes[linear].params.in_features = cur + 7
    out.push({
      operator: 'wrong-in-features',
      kind: 'semantic',
      snapshot: s,
      expectValidatorError: false,
      expectLoadReject: false,
      expectSidecarError: true,
      note: 'Linear in_features off by +7 vs upstream',
    })

    const s2 = copy(src)
    s2.nodes[linear].params.in_features = 1e12
    out.push({
      operator: 'huge-dim',
      kind: 'semantic',
      snapshot: s2,
      expectValidatorError: false,
      expectLoadReject: false,
      expectSidecarError: true,
      note: 'Linear in_features = 1e12 (construct fails cleanly)',
    })
  }

  if (conv >= 0) {
    const s = copy(src)
    s.nodes[conv].params.kernel_size = [33, 33]
    s.nodes[conv].params.padding = [0, 0]
    s.nodes[conv].params.stride = [1, 1]
    out.push({
      operator: 'conv-kernel-too-big',
      kind: 'semantic',
      snapshot: s,
      expectValidatorError: false,
      expectLoadReject: false,
      expectSidecarError: true,
      note: 'conv kernel 33 > spatial input',
    })
  }

  if (concat >= 0) {
    const s = copy(src)
    s.nodes[concat].params.dim = 2
    out.push({
      operator: 'concat-wrong-dim',
      kind: 'semantic',
      snapshot: s,
      expectValidatorError: false,
      expectLoadReject: false,
      expectSidecarError: true,
      note: 'Concat dim=2 on rank-2 tensors',
    })
  }

  if (add >= 0) {
    const inc = firstIncoming(src, src.nodes[add].id)
    const pred = inc ? src.nodes.find((n) => n.id === inc.source) : undefined
    if (pred && pred.layerType === 'Linear') {
      const s = copy(src)
      const p = s.nodes.find((n) => n.id === pred.id)
      if (p) {
        p.params.out_features = Number(p.params.out_features) + 7
        out.push({
          operator: 'add-width-mismatch',
          kind: 'semantic',
          snapshot: s,
          expectValidatorError: false,
          expectLoadReject: false,
          expectSidecarError: true,
          note: 'Add of two tensors with different widths',
        })
      }
    }
  }

  if (reshape >= 0) {
    const s = copy(src)
    const shape = s.nodes[reshape].params.shape
    if (Array.isArray(shape) && shape.length > 0) {
      const next = [...(shape as number[])]
      next[0] = Number(next[0]) + 1
      s.nodes[reshape].params.shape = next
      out.push({
        operator: 'reshape-wrong-count',
        kind: 'semantic',
        snapshot: s,
        expectValidatorError: false,
        expectLoadReject: false,
        expectSidecarError: true,
        note: 'Reshape to a different element count',
      })
    }
  }

  return out
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit structural mutants that need edge LISTS (a GraphSnapshot cannot
// express duplicate edge ids / weird node ids, but validateGraphState can).
// ─────────────────────────────────────────────────────────────────────────────

export type RawMutant = {
  operator: string
  nodes: LayerNode[]
  edges: Edge[]
  expectValidatorError: boolean
  note: string
}

export function rawStructuralMutants(): RawMutant[] {
  const out: RawMutant[] = []

  // Duplicate edge id: a minimal valid 3-node graph with two edges sharing id.
  {
    const nodes = snapshotToLayerNodes({
      nodes: [
        { id: 'n1', layerType: 'Input', params: { ...defaultParamsFor('Input'), name: 'x', shape: [1, 1] } },
        { id: 'n2', layerType: 'Linear', params: { ...defaultParamsFor('Linear'), in_features: 1, out_features: 1 } },
        { id: 'n3', layerType: 'Linear', params: { ...defaultParamsFor('Linear'), in_features: 1, out_features: 1 } },
      ],
      edges: [],
    })
    out.push({
      operator: 'duplicate-edge-id',
      nodes,
      edges: [
        { id: 'e1', source: 'n1', target: 'n2' },
        { id: 'e1', source: 'n1', target: 'n3' },
      ],
      expectValidatorError: true,
      note: 'two edges share one id',
    })
  }

  // A minimal graph whose Input node carries an unusual id. The id is used
  // consistently in the edge, so topology stays valid — the point is that
  // validation (and map lookups) must not crash or corrupt on the id itself.
  const oddIds: [string, string][] = [
    ['empty-node-id', ''],
    ['proto-node-id', '__proto__'],
    ['constructor-node-id', 'constructor'],
    ['unicode-node-id', 'nä⇥节点'],
    ['long-node-id', 'n' + 'x'.repeat(4096)],
  ]
  for (const [op, id] of oddIds) {
    out.push({
      operator: op,
      nodes: snapshotToLayerNodes({
        nodes: [
          { id, layerType: 'Input', params: { ...defaultParamsFor('Input'), name: 'x', shape: [1, 1] } },
          { id: 'out', layerType: 'Output', params: { name: 'out' } },
        ],
        edges: [],
      }),
      edges: [{ id: 'e1', source: id, target: 'out' }],
      expectValidatorError: false,
      note: `Input node id ${JSON.stringify(id.slice(0, 24))} must not crash/corrupt`,
    })
  }

  return out
}

/** A deep Linear chain for the performance / stack test. Its param wiring is
 *  intentionally homogeneous (1→1) so the only thing under test is scale. */
export function chainGraph(depth: number): GraphSnapshot {
  const nodes: GraphSnapshot['nodes'] = []
  const edges: { source: string; target: string }[] = []
  for (let i = 0; i < depth; i++) {
    const isFirst = i === 0
    const id = `n${i + 1}`
    nodes.push({
      id,
      layerType: isFirst ? 'Input' : 'Linear',
      params: isFirst
        ? { ...defaultParamsFor('Input'), name: 'x', shape: [1, 1], dtype: 'float32' }
        : { ...defaultParamsFor('Linear'), in_features: 1, out_features: 1 },
      position: { x: (i % 50) * 10, y: Math.floor(i / 50) * 10 },
    })
    if (i > 0) edges.push({ source: `n${i}`, target: `n${i + 1}` })
  }
  nodes.push({ id: `n${depth + 1}`, layerType: 'Output', params: { name: 'out' }, position: { x: 0, y: 0 } })
  edges.push({ source: `n${depth}`, target: `n${depth + 1}` })
  return { nodes, edges }
}
