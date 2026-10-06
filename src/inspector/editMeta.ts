// Phase 43 (fix round 1) — pure edit-origin predicates. These make "the user
// actually changed something" explicit so a mirrored/programmatic value can
// never be mistaken for a human edit and auto-approved. No React, no Monaco,
// no I/O: the verify harness unit-tests every branch directly.
export type EditMeta = { userEdited: boolean }

/** True when a text buffer differs from what the store holds — i.e. there is
 *  something to commit at all. An unchanged mirror (H1) must not re-commit. */
export function shouldCommitText(draft: string, stored: string): boolean {
  return draft !== stored
}

/** True when a Monaco buffer should be written back: the user actually edited
 *  it AND the settled text differs from the store. An unedited stale buffer
 *  (the store changed under an uncontrolled editor, H2/H3) must never flush. */
export function shouldFlushCode(edited: boolean, latest: string, stored: string): boolean {
  return edited && latest !== stored
}

/** True when the fullscreen modal produced a user change. It seeds from the
 *  inline text, so closing without typing yields next === seed (H3). */
export function modalEditedByUser(seed: string, next: string): boolean {
  return next !== seed
}
