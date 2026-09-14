// Categories drive both palette grouping and visual node decoration.
// Add new categories here and the canvas/palette will pick them up.

export const CATEGORY_COLORS: Record<string, string> = {
  Conv: 'var(--accent)',
  Linear: '#ffb84d',
  Norm: '#b39dff',
  Activation: '#4dd0a8',
  Pool: '#ff8a65',
  Regularize: '#9aa1a8',
  Attention: '#ff6b9d',
  Recurrent: '#ffd166',
  Graph: '#34d399',
  Merge: '#ec4899',
  Reshape: '#a78bfa',
  IO: '#e6e8eb',
  Custom: '#a3e635',
}

// A short glyph rendered in the node header. Kept ASCII-safe so it survives
// the monospace context cleanly; also legible at small sizes.
export const CATEGORY_ICON: Record<string, string> = {
  Conv: '▦',         // grid → kernel sweep
  Linear: '═',       // matrix multiplication
  Norm: 'μ',         // normalize
  Activation: '∿',   // wave → non-linearity
  Pool: '▽',         // downsample
  Regularize: '✱',   // mask
  Attention: '◈',    // attend
  Recurrent: '↻',    // loop
  Graph: '⬡',        // node graph
  Merge: '⋈',        // join
  Reshape: '⤧',      // rearrange
  IO: '◉',           // terminal
  Custom: '{}',      // free-form code
}

export function iconForCategory(cat: string | undefined): string {
  if (!cat) return '·'
  return CATEGORY_ICON[cat] ?? '·'
}

export function colorForCategory(cat: string | undefined): string {
  if (!cat) return '#9aa1a8'
  return CATEGORY_COLORS[cat] ?? '#9aa1a8'
}
