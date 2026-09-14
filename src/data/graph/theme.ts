// Colors + icons for data-node categories, mirroring the training-canvas theming.

import type { DataCategory } from './registry'

const COLORS: Record<DataCategory, string> = {
  Source: '#6ab7ff',
  Fetch: '#e6c34a',
  Transform: '#5fd39a',
  Graph: '#b48ead',
  Custom: '#d08770',
  Sink: '#ff7a85',
}

const ICONS: Record<DataCategory, string> = {
  Source: '▤',
  Fetch: '⭳',
  Transform: '∿',
  Graph: '◈',
  Custom: '⎔',
  Sink: '⇲',
}

export function colorForDataCategory(cat: DataCategory): string {
  return COLORS[cat] ?? '#9aa1a8'
}

export function iconForDataCategory(cat: DataCategory): string {
  return ICONS[cat] ?? '?'
}
