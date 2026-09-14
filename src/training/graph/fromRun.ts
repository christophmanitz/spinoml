// Reconstruct a visual training graph from a saved run's run.json. A run always
// carries its full frozen config (dataset, model, optimizer, loss, scheduler,
// metrics, callbacks, loop) even when it was started manually (NewRunModal) or
// the originating .spinotrain was never saved. This rebuilds an equivalent graph so
// any past run can be reopened on the training canvas. Inverse of
// codegen/trainingGenerator.ts (graph → run.json).

import type { RunConfig } from '../types'
import type { TrainingGraphSnapshot } from './store'

type SnapNode = TrainingGraphSnapshot['nodes'][number]

export function runConfigToTrainingSnapshot(config: RunConfig): TrainingGraphSnapshot {
  const nodes: SnapNode[] = []
  const edges: { source: string; target: string }[] = []
  let seq = 0
  const add = (trainingType: string, params: Record<string, unknown>): string => {
    const id = `r${++seq}`
    // omit position → store.loadSnapshot auto-lays-out
    nodes.push({ id, trainingType, params })
    return id
  }

  const t = config.training
  const ds = config.dataset

  const datasetId = add('DatasetSource', {
    dataset: ds.relpath,
    target: ds.target_column,
    features: ds.feature_columns ?? [],
  })
  const splitId = add('Split', { strategy: t.split_strategy, val_ratio: t.val_split, seed: t.seed })
  const loaderId = add('DataLoader', { batch_size: t.batch_size })
  const modelId = add('ModelSource', { model: config.model_path })
  // Multitask: one Head node per output. Single-task: the legacy Loss node.
  const objectiveIds: string[] = t.heads && t.heads.length
    ? t.heads.map((h) => add('Head', {
        output: h.output,
        target: h.target,
        loss: h.loss,
        weight: h.weight,
        ...(h.label_smoothing != null ? { label_smoothing: h.label_smoothing } : {}),
      }))
    : [add('Loss', { kind: t.loss.kind })]
  const optId = add('Optimizer', {
    kind: t.optimizer.kind,
    lr: t.optimizer.lr,
    weight_decay: t.optimizer.weight_decay,
    ...(t.optimizer.momentum != null ? { momentum: t.optimizer.momentum } : {}),
  })

  const loopId = add('TrainLoop', {
    epochs: t.epochs,
    seed: t.seed,
    log_every_n_steps: t.log_every_n_steps,
  })

  // data pipeline
  edges.push({ source: datasetId, target: splitId })
  edges.push({ source: splitId, target: loaderId })
  edges.push({ source: loaderId, target: loopId })
  // components → loop
  edges.push({ source: modelId, target: loopId })
  for (const id of objectiveIds) edges.push({ source: id, target: loopId })
  edges.push({ source: optId, target: loopId })

  if (t.scheduler && t.scheduler.kind !== 'none') {
    const schedId = add('Scheduler', { ...t.scheduler })
    edges.push({ source: schedId, target: loopId })
  }

  for (const m of t.metrics ?? []) {
    const id = add('Metric', { kind: m })
    edges.push({ source: id, target: loopId })
  }

  for (const cb of t.callbacks ?? []) {
    const { kind, ...rest } = cb
    if (kind !== 'EarlyStopping' && kind !== 'GradientClipping' && kind !== 'MixedPrecision') continue
    const id = add(kind, rest as Record<string, unknown>)
    edges.push({ source: id, target: loopId })
  }

  return { nodes, edges }
}
