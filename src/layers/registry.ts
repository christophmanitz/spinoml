export type FieldSpec =
  | { name: string; type: 'int'; min?: number; max?: number; step?: number; default: number }
  | { name: string; type: 'float'; min?: number; max?: number; step?: number; default: number }
  | { name: string; type: 'bool'; default: boolean }
  | { name: string; type: 'select'; options: string[]; default: string }
  | { name: string; type: 'tuple-int'; arity: 2 | 3; default: number[] }
  | { name: string; type: 'shape'; default: number[] }
  /** Runtime-populated dropdown of dataset relpaths from datasetsStore. Stored as string. */
  | { name: string; type: 'dataset-ref'; default: string }
  /** Runtime-populated multi-select of column names from the node's bound dataset
   *  (looks up its sibling 'dataset' param + datasetsStore.inspects). */
  | { name: string; type: 'columns-multi'; default: string[] }
  /** Single-select column. Same lookup as columns-multi, single string. */
  | { name: string; type: 'column-single'; default: string }

export type LayerKind = 'module' | 'input' | 'output' | 'merge'

export type LayerSpec = {
  type: string
  category: string
  /** PyTorch nn.Module path (e.g. "nn.Conv2d"). Empty for non-module kinds. */
  pytorchModule: string
  /** What kind of node this is. Controls codegen. Defaults to 'module' if absent. */
  kind?: LayerKind
  /** For kind='merge': how to emit the forward expression given input variable names.
   *  e.g. (xs) => `torch.cat([${xs.join(', ')}], dim=${dim})`. */
  forwardExpr?: (inputVars: string[], params: Record<string, unknown>) => string
  fields: FieldSpec[]
  summary: (params: Record<string, unknown>) => string
}

const f = {
  int: (name: string, def: number, opts: { min?: number; max?: number; step?: number } = {}): FieldSpec => ({
    name, type: 'int', default: def, ...opts,
  }),
  float: (name: string, def: number, opts: { min?: number; max?: number; step?: number } = {}): FieldSpec => ({
    name, type: 'float', default: def, ...opts,
  }),
  bool: (name: string, def: boolean): FieldSpec => ({ name, type: 'bool', default: def }),
  select: (name: string, options: string[], def: string): FieldSpec => ({
    name, type: 'select', options, default: def,
  }),
  tuple2: (name: string, def: number[]): FieldSpec => ({ name, type: 'tuple-int', arity: 2, default: def }),
  shape: (name: string, def: number[]): FieldSpec => ({ name, type: 'shape', default: def }),
}

const get = <T>(p: Record<string, unknown>, k: string, fallback: T): T =>
  (p[k] as T) ?? fallback

export const LAYERS: Record<string, LayerSpec> = {
  Input: {
    type: 'Input', category: 'IO', pytorchModule: '', kind: 'input',
    fields: [
      { name: 'name', type: 'select', options: ['x', 'x1', 'x2', 'x3', 'q', 'k', 'v', 'cond'], default: 'x' } as FieldSpec,
      f.shape('shape', [1, 3, 224, 224]),
      { name: 'dataset', type: 'dataset-ref', default: '' } as FieldSpec,
      { name: 'features', type: 'columns-multi', default: [] } as FieldSpec,
      { name: 'target', type: 'column-single', default: '' } as FieldSpec,
    ],
    summary: (p) => {
      const ds = String(get(p, 'dataset', ''))
      const dsTag = ds ? ` ← ${ds.split('/').pop()}` : ''
      const feats = get(p, 'features', []) as string[]
      const featTag = ds && feats.length ? ` · ${feats.length} feat` : ''
      return `${get(p, 'name', 'x')} ∈ ${JSON.stringify(get(p, 'shape', [1, 3, 224, 224]))}${dsTag}${featTag}`
    },
  },
  Output: {
    type: 'Output', category: 'IO', pytorchModule: '', kind: 'output',
    fields: [
      { name: 'name', type: 'select', options: ['out', 'logits', 'embedding', 'mu', 'sigma', 'aux'], default: 'out' } as FieldSpec,
    ],
    summary: (p) => `→ ${get(p, 'name', 'out')}`,
  },

  Conv2d: {
    type: 'Conv2d', category: 'Conv', pytorchModule: 'nn.Conv2d',
    fields: [
      f.int('in_channels', 3, { min: 1 }),
      f.int('out_channels', 64, { min: 1 }),
      f.tuple2('kernel_size', [3, 3]),
      f.tuple2('stride', [1, 1]),
      f.tuple2('padding', [1, 1]),
      f.bool('bias', true),
    ],
    summary: (p) => `${get(p, 'in_channels', 3)}→${get(p, 'out_channels', 64)} k${(get(p, 'kernel_size', [3, 3]) as number[]).join('x')}`,
  },
  Conv1d: {
    type: 'Conv1d', category: 'Conv', pytorchModule: 'nn.Conv1d',
    fields: [
      f.int('in_channels', 1, { min: 1 }),
      f.int('out_channels', 16, { min: 1 }),
      f.int('kernel_size', 3, { min: 1 }),
      f.int('stride', 1, { min: 1 }),
      f.int('padding', 1, { min: 0 }),
      f.bool('bias', true),
    ],
    summary: (p) => `${get(p, 'in_channels', 1)}→${get(p, 'out_channels', 16)} k${get(p, 'kernel_size', 3)}`,
  },
  ConvTranspose2d: {
    type: 'ConvTranspose2d', category: 'Conv', pytorchModule: 'nn.ConvTranspose2d',
    fields: [
      f.int('in_channels', 64, { min: 1 }),
      f.int('out_channels', 32, { min: 1 }),
      f.tuple2('kernel_size', [3, 3]),
      f.tuple2('stride', [2, 2]),
      f.tuple2('padding', [1, 1]),
    ],
    summary: (p) => `${get(p, 'in_channels', 64)}→${get(p, 'out_channels', 32)} k${(get(p, 'kernel_size', [3, 3]) as number[]).join('x')} ↑`,
  },

  Linear: {
    type: 'Linear', category: 'Linear', pytorchModule: 'nn.Linear',
    fields: [
      f.int('in_features', 512, { min: 1 }),
      f.int('out_features', 128, { min: 1 }),
      f.bool('bias', true),
    ],
    summary: (p) => `${get(p, 'in_features', 512)}→${get(p, 'out_features', 128)}`,
  },
  Flatten: {
    type: 'Flatten', category: 'Linear', pytorchModule: 'nn.Flatten',
    fields: [
      f.int('start_dim', 1, { min: 0 }),
      f.int('end_dim', -1),
    ],
    summary: (p) => `dims ${get(p, 'start_dim', 1)}…${get(p, 'end_dim', -1)}`,
  },

  BatchNorm2d: {
    type: 'BatchNorm2d', category: 'Norm', pytorchModule: 'nn.BatchNorm2d',
    fields: [
      f.int('num_features', 64, { min: 1 }),
      f.float('eps', 1e-5, { step: 1e-6 }),
      f.float('momentum', 0.1, { min: 0, max: 1, step: 0.01 }),
    ],
    summary: (p) => `bn ${get(p, 'num_features', 64)}`,
  },
  LayerNorm: {
    type: 'LayerNorm', category: 'Norm', pytorchModule: 'nn.LayerNorm',
    fields: [
      f.shape('normalized_shape', [512]),
      f.float('eps', 1e-5, { step: 1e-6 }),
    ],
    summary: (p) => `ln ${JSON.stringify(get(p, 'normalized_shape', [512]))}`,
  },
  GroupNorm: {
    type: 'GroupNorm', category: 'Norm', pytorchModule: 'nn.GroupNorm',
    fields: [
      f.int('num_groups', 8, { min: 1 }),
      f.int('num_channels', 64, { min: 1 }),
    ],
    summary: (p) => `gn ${get(p, 'num_groups', 8)}g·${get(p, 'num_channels', 64)}c`,
  },

  ReLU: { type: 'ReLU', category: 'Activation', pytorchModule: 'nn.ReLU', fields: [f.bool('inplace', false)], summary: () => 'relu' },
  GELU: { type: 'GELU', category: 'Activation', pytorchModule: 'nn.GELU', fields: [], summary: () => 'gelu' },
  SiLU: { type: 'SiLU', category: 'Activation', pytorchModule: 'nn.SiLU', fields: [f.bool('inplace', false)], summary: () => 'silu' },
  Sigmoid: { type: 'Sigmoid', category: 'Activation', pytorchModule: 'nn.Sigmoid', fields: [], summary: () => 'sigmoid' },
  Tanh: { type: 'Tanh', category: 'Activation', pytorchModule: 'nn.Tanh', fields: [], summary: () => 'tanh' },

  MaxPool2d: {
    type: 'MaxPool2d', category: 'Pool', pytorchModule: 'nn.MaxPool2d',
    fields: [
      f.tuple2('kernel_size', [2, 2]),
      f.tuple2('stride', [2, 2]),
      f.tuple2('padding', [0, 0]),
    ],
    summary: (p) => `max k${(get(p, 'kernel_size', [2, 2]) as number[]).join('x')}`,
  },
  AvgPool2d: {
    type: 'AvgPool2d', category: 'Pool', pytorchModule: 'nn.AvgPool2d',
    fields: [
      f.tuple2('kernel_size', [2, 2]),
      f.tuple2('stride', [2, 2]),
      f.tuple2('padding', [0, 0]),
    ],
    summary: (p) => `avg k${(get(p, 'kernel_size', [2, 2]) as number[]).join('x')}`,
  },
  AdaptiveAvgPool2d: {
    type: 'AdaptiveAvgPool2d', category: 'Pool', pytorchModule: 'nn.AdaptiveAvgPool2d',
    fields: [f.tuple2('output_size', [1, 1])],
    summary: (p) => `adaptive ${(get(p, 'output_size', [1, 1]) as number[]).join('x')}`,
  },

  Dropout: {
    type: 'Dropout', category: 'Regularize', pytorchModule: 'nn.Dropout',
    fields: [f.float('p', 0.5, { min: 0, max: 1, step: 0.05 })],
    summary: (p) => `p=${get(p, 'p', 0.5)}`,
  },
  Dropout2d: {
    type: 'Dropout2d', category: 'Regularize', pytorchModule: 'nn.Dropout2d',
    fields: [f.float('p', 0.5, { min: 0, max: 1, step: 0.05 })],
    summary: (p) => `2d p=${get(p, 'p', 0.5)}`,
  },

  MultiheadAttention: {
    type: 'MultiheadAttention', category: 'Attention', pytorchModule: 'nn.MultiheadAttention',
    fields: [
      f.int('embed_dim', 512, { min: 1 }),
      f.int('num_heads', 8, { min: 1 }),
      f.float('dropout', 0.0, { min: 0, max: 1, step: 0.05 }),
      f.bool('batch_first', true),
    ],
    summary: (p) => `mha d=${get(p, 'embed_dim', 512)} h=${get(p, 'num_heads', 8)}`,
  },
  TransformerEncoderLayer: {
    type: 'TransformerEncoderLayer', category: 'Attention', pytorchModule: 'nn.TransformerEncoderLayer',
    fields: [
      f.int('d_model', 512, { min: 1 }),
      f.int('nhead', 8, { min: 1 }),
      f.int('dim_feedforward', 2048, { min: 1 }),
      f.float('dropout', 0.1, { min: 0, max: 1, step: 0.05 }),
      f.select('activation', ['relu', 'gelu'], 'relu'),
      f.bool('batch_first', true),
    ],
    summary: (p) => `enc d=${get(p, 'd_model', 512)} h=${get(p, 'nhead', 8)}`,
  },

  // ─── Merge (functional, no nn.Module) ──────────────────────────────────
  Concat: {
    type: 'Concat', category: 'Merge', pytorchModule: '', kind: 'merge',
    fields: [f.int('dim', 1)],
    forwardExpr: (xs, p) => `torch.cat([${xs.join(', ')}], dim=${get(p, 'dim', 1)})`,
    summary: (p) => `cat dim=${get(p, 'dim', 1)}`,
  },
  Add: {
    type: 'Add', category: 'Merge', pytorchModule: '', kind: 'merge',
    fields: [],
    forwardExpr: (xs) => xs.length <= 1 ? (xs[0] ?? '0')
      : xs.reduce((acc, v) => `${acc} + ${v}`),
    summary: () => 'a + b',
  },
  Multiply: {
    type: 'Multiply', category: 'Merge', pytorchModule: '', kind: 'merge',
    fields: [],
    forwardExpr: (xs) => xs.length <= 1 ? (xs[0] ?? '1')
      : xs.reduce((acc, v) => `${acc} * ${v}`),
    summary: () => 'a · b',
  },
  Stack: {
    type: 'Stack', category: 'Merge', pytorchModule: '', kind: 'merge',
    fields: [f.int('dim', 0)],
    forwardExpr: (xs, p) => `torch.stack([${xs.join(', ')}], dim=${get(p, 'dim', 0)})`,
    summary: (p) => `stack dim=${get(p, 'dim', 0)}`,
  },
}

export function defaultParamsFor(layerType: string): Record<string, unknown> {
  const spec = LAYERS[layerType]
  if (!spec) return {}
  const params: Record<string, unknown> = {}
  for (const field of spec.fields) {
    params[field.name] = field.default
  }
  return params
}

/** Coerce a raw param map (e.g. from the LLM) to types matching the field schema.
 *  - tuple-int with arity N: number → [n, n, …], array of wrong length is padded/truncated
 *  - shape: number → [n], array kept (validated as positive ints)
 *  - int/float: strings parsed
 *  - select: only kept if in options
 *  Unknown keys pass through unchanged so future params don't get dropped silently.
 */
export function coerceParams(
  layerType: string,
  raw: Record<string, unknown>,
): Record<string, unknown> {
  const spec = LAYERS[layerType]
  if (!spec) return raw
  const out: Record<string, unknown> = { ...raw }
  for (const field of spec.fields) {
    if (!(field.name in raw)) continue
    out[field.name] = coerceField(field, raw[field.name])
  }
  return out
}

function coerceField(field: FieldSpec, value: unknown): unknown {
  switch (field.type) {
    case 'int': {
      const n = typeof value === 'string' ? parseInt(value, 10) : Number(value)
      return Number.isFinite(n) ? Math.trunc(n) : field.default
    }
    case 'float': {
      const n = typeof value === 'string' ? parseFloat(value) : Number(value)
      return Number.isFinite(n) ? n : field.default
    }
    case 'bool':
      if (typeof value === 'boolean') return value
      if (value === 'true') return true
      if (value === 'false') return false
      return Boolean(value)
    case 'select': {
      const s = String(value)
      return field.options.includes(s) ? s : field.default
    }
    case 'tuple-int': {
      const arr = toIntArray(value)
      if (arr.length === 0) return field.default
      if (arr.length === field.arity) return arr
      if (arr.length === 1) return Array(field.arity).fill(arr[0])
      if (arr.length > field.arity) return arr.slice(0, field.arity)
      const padded = [...arr]
      while (padded.length < field.arity) padded.push(arr[arr.length - 1])
      return padded
    }
    case 'shape': {
      const arr = toIntArray(value)
      return arr.length ? arr : field.default
    }
    case 'dataset-ref':
      return typeof value === 'string' ? value : field.default
    case 'columns-multi':
      return Array.isArray(value) ? value.filter((v) => typeof v === 'string') : field.default
    case 'column-single':
      return typeof value === 'string' ? value : field.default
  }
}

function toIntArray(value: unknown): number[] {
  if (typeof value === 'number' && Number.isFinite(value)) return [Math.trunc(value)]
  if (typeof value === 'string') {
    const parts = value.split(/[,\s\[\]]+/).filter(Boolean)
    return parts
      .map((p) => parseInt(p, 10))
      .filter((n) => Number.isFinite(n))
  }
  if (Array.isArray(value)) {
    const out: number[] = []
    for (const v of value) {
      const n = typeof v === 'string' ? parseInt(v, 10) : Number(v)
      if (Number.isFinite(n)) out.push(Math.trunc(n))
    }
    return out
  }
  return []
}

export const LAYER_GROUPS: { name: string; layers: string[] }[] = (() => {
  const byCategory: Record<string, string[]> = {}
  for (const spec of Object.values(LAYERS)) {
    if (!byCategory[spec.category]) byCategory[spec.category] = []
    byCategory[spec.category].push(spec.type)
  }
  const order = ['IO', 'Conv', 'Linear', 'Norm', 'Activation', 'Pool', 'Regularize', 'Attention', 'Merge']
  return order.filter((c) => byCategory[c]).map((c) => ({ name: c, layers: byCategory[c] }))
})()
