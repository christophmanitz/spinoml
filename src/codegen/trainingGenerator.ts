// Training-graph → run-plan compiler — the Phase-14 analogue of
// codegen/generator.ts. PURE: same snapshot → same plan, no I/O, no Date.now()
// (CLAUDE.md "Code-Generation-Konsistenz" invariant). It does NOT resolve the
// dataset absolute path or assign a run id — those are run-launch concerns; the
// compiler only turns the visual graph into the config the Phase-13 trainer
// consumes (RunConfig.training + dataset selection).

import type { TrainingGraphSnapshot } from '../training/graph/store'
import { defaultTrainingParams } from '../training/graph/registry'
import {
  type TrainingConfig,
  type LossKind,
  type OptimizerKind,
  type SchedulerKind,
  type SplitStrategy,
  type CallbackConfig,
  type Head,
  defaultTrainingConfig,
} from '../training/types'

export type TrainingPlan = {
  modelRelpath: string
  datasetRelpath: string
  /** Single-task target column. Empty in multitask mode (targets live in
   *  training.heads, one per output). */
  target: string
  features: string[] | null
  training: TrainingConfig
}

export type TrainingCompile = {
  ok: boolean
  issues: string[]
  /** Non-blocking advisories — e.g. a node not wired into the TrainLoop. The
   *  trainer is node-driven so these don't stop a launch, but they catch a graph
   *  the user *thinks* is connected but isn't. */
  warnings: string[]
  plan: TrainingPlan | null
}

type SnapNode = TrainingGraphSnapshot['nodes'][number]

function paramsOf(n: SnapNode): Record<string, unknown> {
  return { ...defaultTrainingParams(n.trainingType), ...n.params }
}

function num(p: Record<string, unknown>, k: string, d: number): number {
  const v = Number(p[k])
  return Number.isFinite(v) ? v : d
}

export function compileTrainingGraph(snapshot: TrainingGraphSnapshot): TrainingCompile {
  const issues: string[] = []
  const byType = (t: string) => snapshot.nodes.filter((n) => n.trainingType === t)
  const one = (t: string, required = true): SnapNode | null => {
    const hits = byType(t)
    if (hits.length === 0) {
      if (required) issues.push(`Es fehlt ein ${t}-Knoten.`)
      return null
    }
    if (hits.length > 1) issues.push(`Mehrere ${t}-Knoten — nur der erste wird verwendet.`)
    return hits[0]
  }

  // Multitask: each Head node binds a model output → target column + loss +
  // weight. When ≥1 Head node is present the run is multitask and the single
  // Loss node + DatasetSource.target become optional (heads carry both).
  const headNodes = byType('Head')
  const multitask = headNodes.length > 0

  const loop = one('TrainLoop')
  const dataset = one('DatasetSource')
  const model = one('ModelSource')
  const loss = one('Loss', !multitask)
  const optimizer = one('Optimizer')
  const split = one('Split', false)
  const loader = one('DataLoader', false)
  const scheduler = one('Scheduler', false)

  // Build training config from the available nodes (defaults fill the rest).
  const base = defaultTrainingConfig()
  const lp = loop ? paramsOf(loop) : {}
  const op = optimizer ? paramsOf(optimizer) : {}
  const lo = loss ? paramsOf(loss) : {}
  const sp = split ? paramsOf(split) : {}
  const dl = loader ? paramsOf(loader) : {}
  const sc = scheduler ? paramsOf(scheduler) : {}

  const schedulerKind = (sc.kind as SchedulerKind) ?? 'none'
  const schedulerCfg: TrainingConfig['scheduler'] = { kind: schedulerKind }
  if (schedulerKind === 'StepLR') {
    schedulerCfg.step_size = num(sc, 'step_size', 30)
    schedulerCfg.gamma = num(sc, 'gamma', 0.1)
  } else if (schedulerKind === 'ReduceLROnPlateau') {
    schedulerCfg.patience = num(sc, 'patience', 10)
  }

  // Build the heads array (multitask). Each head needs a target column; the
  // output name defaults to 'out' (the model's sole/default output key).
  const heads: Head[] = headNodes.map((n) => {
    const p = paramsOf(n)
    return {
      output: String(p.output ?? 'out'),
      target: String(p.target ?? ''),
      loss: (p.loss as LossKind) ?? 'CrossEntropyLoss',
      weight: num(p, 'weight', 1),
      label_smoothing: num(p, 'label_smoothing', 0),
    }
  })

  const metrics = byType('Metric').map((n) => String(paramsOf(n).kind)).filter(Boolean)

  const callbacks: CallbackConfig[] = []
  for (const n of byType('EarlyStopping')) {
    const p = paramsOf(n)
    callbacks.push({ kind: 'EarlyStopping', monitor: String(p.monitor), patience: num(p, 'patience', 20), mode: String(p.mode) })
  }
  for (const n of byType('GradientClipping')) {
    callbacks.push({ kind: 'GradientClipping', max_norm: num(paramsOf(n), 'max_norm', 1.0) })
  }
  for (const n of byType('MixedPrecision')) {
    callbacks.push({ kind: 'MixedPrecision', dtype: String(paramsOf(n).dtype) })
  }

  const training: TrainingConfig = {
    epochs: Math.trunc(num(lp, 'epochs', base.epochs)),
    batch_size: Math.trunc(num(dl, 'batch_size', base.batch_size)),
    val_split: num(sp, 'val_ratio', base.val_split),
    split_strategy: String(sp.strategy ?? base.split_strategy) as SplitStrategy,
    seed: Math.trunc(num(lp, 'seed', base.seed)),
    log_every_n_steps: Math.trunc(num(lp, 'log_every_n_steps', base.log_every_n_steps)),
    val_every_n_epochs: Math.max(1, Math.trunc(num(lp, 'val_every_n_epochs', 1))),
    gradient_accumulation_steps: Math.max(1, Math.trunc(num(lp, 'gradient_accumulation_steps', 1))),
    // DataLoader knobs only when a DataLoader node is present (else sidecar defaults).
    ...(loader ? {
      shuffle: Boolean(dl.shuffle),
      num_workers: Math.max(0, Math.trunc(num(dl, 'num_workers', 0))),
      drop_last: Boolean(dl.drop_last),
    } : {}),
    optimizer: {
      kind: (op.kind as OptimizerKind) ?? base.optimizer.kind,
      lr: num(op, 'lr', base.optimizer.lr),
      weight_decay: num(op, 'weight_decay', 0),
      momentum: num(op, 'momentum', 0.9),
    },
    loss: { kind: (lo.kind as LossKind) ?? base.loss.kind, ...(loss ? { label_smoothing: num(lo, 'label_smoothing', 0) } : {}) },
    ...(multitask ? { heads } : {}),
    scheduler: schedulerCfg,
    metrics: metrics.length ? metrics : undefined,
    callbacks: callbacks.length ? callbacks : undefined,
  }

  const dsParams = dataset ? paramsOf(dataset) : {}
  const datasetRelpath = String(dsParams.dataset ?? '')
  const target = String(dsParams.target ?? '')
  const featuresRaw = (dsParams.features as string[] | undefined) ?? []
  const features = featuresRaw.length ? featuresRaw : null
  const modelRelpath = model ? String(paramsOf(model).model ?? '') : ''

  // A .manifest (paired graph dataset) carries its own target → no column needed.
  const isManifest = datasetRelpath.toLowerCase().endsWith('.manifest')
  if (dataset && !datasetRelpath) issues.push('DatasetSource hat keinen Datensatz gewählt.')
  // Single-task needs the DatasetSource target; multitask gets targets from heads.
  if (dataset && !isManifest && !multitask && !target) issues.push('DatasetSource braucht eine Ziel-Spalte (target).')
  if (model && !modelRelpath) issues.push('ModelSource hat kein Modell gewählt.')
  // Each head needs a target column (the output name defaults to 'out').
  if (multitask) {
    if (heads.some((h) => !h.target)) issues.push('Jeder Head-Knoten braucht eine Ziel-Spalte (target).')
    const dupOut = heads.map((h) => h.output).filter((o, i, a) => a.indexOf(o) !== i)
    if (dupOut.length) issues.push(`Doppelter Head-Output „${dupOut[0]}" — Output-Namen müssen eindeutig sein.`)
  }

  // Edge validation (advisory): each core component should reach the TrainLoop
  // along the graph's edges. The compiler assembles the config from node types,
  // so an unwired graph still runs — but unconnected nodes usually signal the
  // user forgot a link, so we surface them as warnings.
  const warnings: string[] = []
  if (loop) {
    const adj = new Map<string, string[]>()
    for (const e of snapshot.edges) {
      const list = adj.get(e.source) ?? []
      list.push(e.target)
      adj.set(e.source, list)
    }
    const reaches = (startId: string): boolean => {
      const seen = new Set<string>()
      const stack = [startId]
      while (stack.length) {
        const cur = stack.pop()!
        if (cur === loop.id) return true
        if (seen.has(cur)) continue
        seen.add(cur)
        for (const nx of adj.get(cur) ?? []) stack.push(nx)
      }
      return false
    }
    const wired = (n: SnapNode | null, label: string) => {
      if (n && n.id !== loop.id && !reaches(n.id)) {
        warnings.push(`${label} ist nicht mit dem TrainLoop verbunden.`)
      }
    }
    wired(dataset, 'DatasetSource')
    wired(model, 'ModelSource')
    wired(loss, 'Loss')
    for (const h of headNodes) wired(h, 'Head')
    wired(optimizer, 'Optimizer')
    wired(scheduler, 'Scheduler')
    for (const m of byType('Metric')) wired(m, 'Metric')
    for (const t of ['EarlyStopping', 'GradientClipping', 'MixedPrecision']) {
      for (const c of byType(t)) wired(c, t)
    }
  }

  const ok = issues.every((m) => m.includes('Mehrere')) &&
    !!loop && !!dataset && !!model && !!optimizer && !!datasetRelpath && !!modelRelpath &&
    (multitask
      ? heads.length > 0 && heads.every((h) => !!h.target)
      : !!loss && (isManifest || !!target))

  // Multitask carries targets in training.heads; the single-task `target` slot
  // is then empty so nothing downstream mistakes one head's column for THE target.
  const plan: TrainingPlan | null = ok
    ? { modelRelpath, datasetRelpath, target: multitask ? '' : target, features, training }
    : null

  return { ok, issues, warnings, plan }
}
