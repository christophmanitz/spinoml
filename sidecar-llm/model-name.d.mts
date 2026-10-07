// Hand-written type declarations for sidecar-llm/model-name.mjs.
//
// Mirrors the runtime exports exactly (no `any`). The validator returns the
// same discriminated `{ ok: true, name } | { ok: false, error }` shape that
// shell-safety.mjs uses for its argv/URL checks, so call sites already
// destructuring that pattern work here too.

export const MODEL_NAME_MAX_LEN: number
export const MODEL_NAME_RE: RegExp

export type ValidateModelNameResult =
  | { ok: true; name: string }
  | { ok: false; error: string }

export function validateModelName(name: unknown): ValidateModelNameResult
export function validateJsonModelName(name: unknown): ValidateModelNameResult
