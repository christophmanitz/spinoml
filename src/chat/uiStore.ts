import { create } from 'zustand'

// Small UI-only store for the chat panel: a font scale the user controls with
// the A−/A+ buttons in the header. Persisted to localStorage so it survives
// reloads. Kept out of GraphStore — this is purely presentational chat state.

const KEY = 'spinoml.chat.fontScale.v1'
const AUTO_KEY = 'spinoml.chat.autoMode.v1'
const DOC_KEY = 'spinoml.chat.docMode.v1'
const MIN = 0.8
const MAX = 2.0
const BASE_PX = 14 // matches the old hardcoded `text-sm` base

// FEAT-4 — how aggressively the chatbot documents to notes/lab-notebook.md.
export type DocMode = 'off' | 'compact' | 'verbose'

function clamp(f: number): number {
  if (!Number.isFinite(f)) return 1
  return Math.min(MAX, Math.max(MIN, Math.round(f * 100) / 100))
}

function load(): number {
  try {
    const v = Number(localStorage.getItem(KEY))
    return v ? clamp(v) : 1
  } catch {
    return 1
  }
}
function loadAuto(): boolean {
  try { return localStorage.getItem(AUTO_KEY) === '1' } catch { return false }
}
function loadDoc(): DocMode {
  try {
    const v = localStorage.getItem(DOC_KEY)
    return v === 'off' || v === 'compact' ? v : 'verbose'
  } catch { return 'verbose' }
}

type ChatUiState = {
  fontScale: number
  /** Base font-size in px to apply to the message area + input. */
  basePx: number
  setFontScale: (f: number) => void
  bumpFontScale: (delta: number) => void
  /** FEAT-3 — Auto-Modus: shell run_script auto-approves (SLURM still confirms). */
  autoMode: boolean
  setAutoMode: (v: boolean) => void
  /** FEAT-4 — documentation verbosity passed to the chatbot. */
  docMode: DocMode
  setDocMode: (m: DocMode) => void
}

export const useChatUi = create<ChatUiState>((set, get) => ({
  fontScale: load(),
  basePx: BASE_PX * load(),
  setFontScale: (f) => {
    const c = clamp(f)
    try { localStorage.setItem(KEY, String(c)) } catch { /* private mode */ }
    set({ fontScale: c, basePx: BASE_PX * c })
  },
  bumpFontScale: (delta) => get().setFontScale(get().fontScale + delta),
  autoMode: loadAuto(),
  setAutoMode: (v) => {
    try { localStorage.setItem(AUTO_KEY, v ? '1' : '0') } catch { /* private mode */ }
    set({ autoMode: v })
  },
  docMode: loadDoc(),
  setDocMode: (m) => {
    try { localStorage.setItem(DOC_KEY, m) } catch { /* private mode */ }
    set({ docMode: m })
  },
}))
