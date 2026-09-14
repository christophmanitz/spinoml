// THE training-node registry — the Phase-14 analogue of layers/registry.ts.
// Adding a training-node type = editing this one file (spec + default + coerce
// is uniform). Kept fully decoupled from the layer registry: training fields
// are simple (selects + numbers + refs), no shape inference, no PyTorch module
// emission. The codegen for these lives in codegen/trainingGenerator.ts.

export type TrainingFieldSpec =
  | { name: string; type: 'int'; min?: number; max?: number; step?: number; default: number }
  | { name: string; type: 'float'; min?: number; max?: number; step?: number; default: number }
  | { name: string; type: 'bool'; default: boolean }
  | { name: string; type: 'select'; options: string[]; default: string }
  /** Runtime-populated dropdown of dataset relpaths (datasetsStore). */
  | { name: string; type: 'dataset-ref'; default: string }
  /** Runtime-populated dropdown of models/*.spinoml relpaths (fs.list). */
  | { name: string; type: 'model-ref'; default: string }
  /** Single column from the graph's bound DatasetSource. */
  | { name: string; type: 'column-single'; default: string }
  /** Multiple columns from the graph's bound DatasetSource. */
  | { name: string; type: 'columns-multi'; default: string[] }

/** Connection role — drives palette grouping + edge validation in the editor. */
export type TrainingCategory =
  | 'Data'
  | 'Model'
  | 'Objective'
  | 'Schedule'
  | 'Metric'
  | 'Callback'
  | 'Loop'

export type TrainingNodeSpec = {
  type: string
  category: TrainingCategory
  /** May appear more than once in one graph (Metric + every Callback). */
  multi?: boolean
  fields: TrainingFieldSpec[]
  summary: (params: Record<string, unknown>) => string
}

const get = <T>(p: Record<string, unknown>, k: string, fallback: T): T =>
  (p[k] as T) ?? fallback

const f = {
  int: (name: string, def: number, opts: { min?: number; max?: number; step?: number } = {}): TrainingFieldSpec => ({ name, type: 'int', default: def, ...opts }),
  float: (name: string, def: number, opts: { min?: number; max?: number; step?: number } = {}): TrainingFieldSpec => ({ name, type: 'float', default: def, ...opts }),
  bool: (name: string, def: boolean): TrainingFieldSpec => ({ name, type: 'bool', default: def }),
  select: (name: string, options: string[], def: string): TrainingFieldSpec => ({ name, type: 'select', options, default: def }),
}

export const TRAINING_NODES: Record<string, TrainingNodeSpec> = {
  DatasetSource: {
    type: 'DatasetSource', category: 'Data',
    fields: [
      { name: 'dataset', type: 'dataset-ref', default: '' },
      { name: 'target', type: 'column-single', default: '' },
      { name: 'features', type: 'columns-multi', default: [] },
    ],
    summary: (p) => {
      const ds = String(get(p, 'dataset', ''))
      const t = String(get(p, 'target', ''))
      return ds ? `${ds.split('/').pop()}${t ? ` → ${t}` : ''}` : '(kein Datensatz)'
    },
  },
  Split: {
    type: 'Split', category: 'Data',
    fields: [
      f.select('strategy', ['random', 'stratified', 'grouped', 'time-based', 'predefined'], 'random'),
      f.float('val_ratio', 0.2, { min: 0, max: 0.9, step: 0.05 }),
      f.int('seed', 42, { min: 0 }),
    ],
    summary: (p) => {
      const s = String(get(p, 'strategy', 'random'))
      const v = Math.round(get(p, 'val_ratio', 0.2) * 100)
      return `${s} · val ${v}% · seed ${get(p, 'seed', 42)}`
    },
  },
  DataLoader: {
    type: 'DataLoader', category: 'Data',
    fields: [
      f.int('batch_size', 32, { min: 1 }),
      f.bool('shuffle', true),
      f.int('num_workers', 0, { min: 0 }),
      f.bool('drop_last', false),
    ],
    summary: (p) => `batch ${get(p, 'batch_size', 32)}${get(p, 'shuffle', true) ? ' · shuffle' : ''}`,
  },
  ModelSource: {
    type: 'ModelSource', category: 'Model',
    fields: [{ name: 'model', type: 'model-ref', default: '' }],
    summary: (p) => {
      const m = String(get(p, 'model', ''))
      return m ? (m.split('/').pop() ?? m) : '(kein Modell)'
    },
  },
  Loss: {
    type: 'Loss', category: 'Objective',
    fields: [
      f.select('kind', ['CrossEntropyLoss', 'BCEWithLogitsLoss', 'MSELoss', 'L1Loss'], 'CrossEntropyLoss'),
      f.float('label_smoothing', 0, { min: 0, max: 0.9, step: 0.01 }),
    ],
    summary: (p) => String(get(p, 'kind', 'CrossEntropyLoss')),
  },
  Head: {
    type: 'Head', category: 'Objective', multi: true,
    fields: [
      // `output` matches a model Output node's name (the dict key forward()
      // returns). Same option set as the architecture Output node.
      f.select('output', ['out', 'logits', 'embedding', 'mu', 'sigma', 'aux'], 'out'),
      { name: 'target', type: 'column-single', default: '' },
      f.select('loss', ['CrossEntropyLoss', 'BCEWithLogitsLoss', 'MSELoss', 'L1Loss'], 'CrossEntropyLoss'),
      f.float('weight', 1, { min: 0, step: 0.1 }),
      f.float('label_smoothing', 0, { min: 0, max: 0.9, step: 0.01 }),
    ],
    summary: (p) => `${get(p, 'output', 'out')} → ${get(p, 'target', '?') || '?'} · ${get(p, 'loss', 'CrossEntropyLoss')}`,
  },
  Optimizer: {
    type: 'Optimizer', category: 'Objective',
    fields: [
      f.select('kind', ['Adam', 'AdamW', 'SGD', 'RMSprop'], 'Adam'),
      f.float('lr', 1e-3, { min: 0, step: 1e-4 }),
      f.float('weight_decay', 0, { min: 0, step: 1e-4 }),
      f.float('momentum', 0.9, { min: 0, max: 1, step: 0.01 }),
    ],
    summary: (p) => `${get(p, 'kind', 'Adam')} lr=${get(p, 'lr', 1e-3)}`,
  },
  Scheduler: {
    type: 'Scheduler', category: 'Schedule',
    fields: [
      f.select('kind', ['none', 'StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'], 'none'),
      f.int('step_size', 30, { min: 1 }),
      f.float('gamma', 0.1, { min: 0, max: 1, step: 0.01 }),
      f.int('patience', 10, { min: 1 }),
    ],
    summary: (p) => String(get(p, 'kind', 'none')),
  },
  Metric: {
    type: 'Metric', category: 'Metric', multi: true,
    fields: [f.select('kind', ['accuracy', 'f1', 'precision', 'recall', 'mse', 'mae', 'r2'], 'accuracy')],
    summary: (p) => String(get(p, 'kind', 'accuracy')),
  },
  EarlyStopping: {
    type: 'EarlyStopping', category: 'Callback', multi: true,
    fields: [
      f.select('monitor', ['val_loss', 'val_acc', 'train_loss'], 'val_loss'),
      f.int('patience', 20, { min: 1 }),
      f.select('mode', ['min', 'max'], 'min'),
    ],
    summary: (p) => `${get(p, 'monitor', 'val_loss')} · patience ${get(p, 'patience', 20)}`,
  },
  GradientClipping: {
    type: 'GradientClipping', category: 'Callback', multi: true,
    fields: [f.float('max_norm', 1.0, { min: 0, step: 0.1 })],
    summary: (p) => `max_norm ${get(p, 'max_norm', 1.0)}`,
  },
  MixedPrecision: {
    type: 'MixedPrecision', category: 'Callback', multi: true,
    fields: [f.select('dtype', ['fp16', 'bf16'], 'bf16')],
    summary: (p) => `AMP ${get(p, 'dtype', 'bf16')}`,
  },
  TrainLoop: {
    type: 'TrainLoop', category: 'Loop',
    fields: [
      f.int('epochs', 50, { min: 1 }),
      f.int('seed', 42, { min: 0 }),
      f.int('log_every_n_steps', 10, { min: 1 }),
      f.int('val_every_n_epochs', 1, { min: 1 }),
      f.int('gradient_accumulation_steps', 1, { min: 1 }),
    ],
    summary: (p) => `${get(p, 'epochs', 50)} epochs · seed ${get(p, 'seed', 42)}`,
  },
}

export function defaultTrainingParams(nodeType: string): Record<string, unknown> {
  const spec = TRAINING_NODES[nodeType]
  if (!spec) return {}
  const params: Record<string, unknown> = {}
  for (const field of spec.fields) params[field.name] = field.default
  return params
}

function coerceField(field: TrainingFieldSpec, value: unknown): unknown {
  switch (field.type) {
    case 'int': {
      const n = Math.trunc(Number(value))
      return Number.isFinite(n) ? n : field.default
    }
    case 'float': {
      const n = Number(value)
      return Number.isFinite(n) ? n : field.default
    }
    case 'bool':
      return Boolean(value)
    case 'select':
      return field.options.includes(String(value)) ? String(value) : field.default
    case 'dataset-ref':
    case 'model-ref':
    case 'column-single':
      return typeof value === 'string' ? value : field.default
    case 'columns-multi':
      return Array.isArray(value) ? value.map((v) => String(v)) : field.default
  }
}

export function coerceTrainingParams(
  nodeType: string,
  raw: Record<string, unknown>,
): Record<string, unknown> {
  const spec = TRAINING_NODES[nodeType]
  if (!spec) return raw
  const out: Record<string, unknown> = { ...raw }
  for (const field of spec.fields) {
    if (!(field.name in raw)) continue
    out[field.name] = coerceField(field, raw[field.name])
  }
  return out
}

export const TRAINING_GROUPS: { name: TrainingCategory; nodes: string[] }[] = (() => {
  const byCat: Record<string, string[]> = {}
  for (const spec of Object.values(TRAINING_NODES)) {
    ;(byCat[spec.category] ??= []).push(spec.type)
  }
  const order: TrainingCategory[] = ['Data', 'Model', 'Objective', 'Schedule', 'Metric', 'Callback', 'Loop']
  return order.filter((c) => byCat[c]).map((c) => ({ name: c, nodes: byCat[c] }))
})()
