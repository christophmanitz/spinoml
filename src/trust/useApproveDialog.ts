import { create } from 'zustand'

import type { UntrustedBlob } from './gate'

/** Phase 43 — UI state for the one approval surface. Any code path that wants
 *  to run generated code calls `openFor(untrusted)` instead of executing; the
 *  dialog (mounted once in App) reads the list and is the ONLY place that
 *  approves with the user-approval origin. Kept out of every store so no
 *  store can accidentally approve as a side effect. */
type ApproveDialogState = {
  open: boolean
  blobs: UntrustedBlob[]
  /** Show the dialog for exactly these blobs. An empty list is a no-op. */
  openFor: (blobs: UntrustedBlob[]) => void
  /** Drop one already-approved blob; close when nothing is left. */
  dismiss: (sha256: string) => void
  close: () => void
}

export const useApproveDialog = create<ApproveDialogState>((set) => ({
  open: false,
  blobs: [],
  openFor: (blobs) => set({ open: blobs.length > 0, blobs }),
  dismiss: (sha256) => set((s) => {
    const blobs = s.blobs.filter((b) => b.sha256 !== sha256)
    return { blobs, open: blobs.length > 0 }
  }),
  close: () => set({ open: false, blobs: [] }),
}))
