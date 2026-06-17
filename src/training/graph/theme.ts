// Colors + icons for training-node categories, mirroring the layer-category
// theming used by the architecture canvas.

import type { TrainingCategory } from './registry'

const COLORS: Record<TrainingCategory, string> = {
  Data: '#6ab7ff',
  Model: '#b48ead',
  Objective: '#5fd39a',
  Schedule: '#e6c34a',
  Metric: '#88c0d0',
  Callback: '#d08770',
  Loop: '#ff7a85',
}

const ICONS: Record<TrainingCategory, string> = {
  Data: '▤',
  Model: '◈',
  Objective: '◎',
  Schedule: '◷',
  Metric: '▣',
  Callback: '⎔',
  Loop: '↻',
}

export function colorForTrainingCategory(cat: TrainingCategory): string {
  return COLORS[cat] ?? '#9aa1a8'
}

export function iconForTrainingCategory(cat: TrainingCategory): string {
  return ICONS[cat] ?? '?'
}
