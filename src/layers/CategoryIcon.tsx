// SVG category icons. Replaces the old Unicode glyphs (μ, ═, ∿, …) whose font
// metrics never centered cleanly inside their chip. Each icon draws on a
// 16×16 viewBox so it sits dead-center in the h-4/w-4 chip, and uses
// currentColor so the chip's text color drives the stroke.

import type { JSX } from 'react'

const COMMON = {
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.4,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
}

const PATHS: Record<string, JSX.Element> = {
  // grid → kernel sweep
  Conv: (
    <g {...COMMON}>
      <rect x="3" y="3" width="10" height="10" rx="1.2" />
      <line x1="6.3" y1="3" x2="6.3" y2="13" />
      <line x1="9.7" y1="3" x2="9.7" y2="13" />
      <line x1="3" y1="6.3" x2="13" y2="6.3" />
      <line x1="3" y1="9.7" x2="13" y2="9.7" />
    </g>
  ),
  // stacked weight rows → matmul / dense
  Linear: (
    <g {...COMMON}>
      <line x1="4" y1="5" x2="12" y2="5" />
      <line x1="4" y1="8" x2="12" y2="8" />
      <line x1="4" y1="11" x2="12" y2="11" />
    </g>
  ),
  // bell curve → normalize
  Norm: (
    <g {...COMMON}>
      <path d="M2.5 12.5 C5.5 12.5 5.5 4 8 4 C10.5 4 10.5 12.5 13.5 12.5" />
    </g>
  ),
  // sine wave → non-linearity
  Activation: (
    <g {...COMMON}>
      <path d="M2.5 8.5 Q5 3 8 8 T13.5 7.5" />
    </g>
  ),
  // funnel → downsample
  Pool: (
    <g {...COMMON}>
      <path d="M3 4 H13 L9.5 8.5 V12.5 L6.5 11 V8.5 Z" />
    </g>
  ),
  // scattered dots, one dropped → regularize / dropout
  Regularize: (
    <g fill="currentColor" stroke="none">
      <circle cx="5" cy="5" r="1.3" />
      <circle cx="11" cy="5" r="1.3" />
      <circle cx="8" cy="8" r="1.3" />
      <circle cx="5" cy="11" r="1.3" />
      <circle cx="11" cy="11" r="1.3" opacity="0.3" />
    </g>
  ),
  // diamond + focus point → attention
  Attention: (
    <g {...COMMON}>
      <path d="M8 2.5 L13.5 8 L8 13.5 L2.5 8 Z" />
      <circle cx="8" cy="8" r="1.4" fill="currentColor" stroke="none" />
    </g>
  ),
  // circular arrow → recurrent loop
  Recurrent: (
    <g {...COMMON}>
      <path d="M12 6.2 A4.5 4.5 0 1 0 12.8 9.2" />
      <path d="M12 3.2 L12 6.4 L8.9 6.1" />
    </g>
  ),
  // nodes + edges → graph
  Graph: (
    <g {...COMMON}>
      <line x1="4.5" y1="11.5" x2="8" y2="4.5" />
      <line x1="8" y1="4.5" x2="11.5" y2="11.5" />
      <line x1="4.5" y1="11.5" x2="11.5" y2="11.5" />
      <circle cx="8" cy="4.5" r="1.8" fill="currentColor" stroke="none" />
      <circle cx="4.5" cy="11.5" r="1.8" fill="currentColor" stroke="none" />
      <circle cx="11.5" cy="11.5" r="1.8" fill="currentColor" stroke="none" />
    </g>
  ),
  // two streams join → merge
  Merge: (
    <g {...COMMON}>
      <path d="M3 3.5 C7 3.5 7 8 9 8" />
      <path d="M3 12.5 C7 12.5 7 8 9 8" />
      <line x1="9" y1="8" x2="13" y2="8" />
    </g>
  ),
  // box with re-arrange arrows → reshape
  Reshape: (
    <g {...COMMON}>
      <rect x="3" y="3" width="6.5" height="6.5" rx="1" />
      <path d="M9 12.5 H13 V8.5" />
      <path d="M13 12.5 L9.5 9" />
    </g>
  ),
  // port ring → IO terminal
  IO: (
    <g {...COMMON}>
      <circle cx="8" cy="8" r="5" />
      <circle cx="8" cy="8" r="1.6" fill="currentColor" stroke="none" />
    </g>
  ),
  // code brackets → custom / free-form
  Custom: (
    <g {...COMMON}>
      <path d="M6 4 L2.5 8 L6 12" />
      <path d="M10 4 L13.5 8 L10 12" />
    </g>
  ),
}

const FALLBACK = (
  <g fill="currentColor" stroke="none">
    <circle cx="8" cy="8" r="1.6" />
  </g>
)

export default function CategoryIcon({ cat, size = 12 }: { cat: string | undefined; size?: number }) {
  return (
    <svg viewBox="0 0 16 16" width={size} height={size} aria-hidden focusable="false">
      {(cat && PATHS[cat]) || FALLBACK}
    </svg>
  )
}
