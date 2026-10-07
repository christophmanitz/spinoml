// Pure model-name validator for the opencode provider path.
//
// The model string reaches `spawn(OPENCODE_BIN, ['run', '--format', 'json',
// '--model', model, '--pure', prompt])` UNVALIDATED today — a value such as
// `--print-logs`, `-h`, `--dangerously-…` or a string with spaces / control
// characters can be parsed by opencode's option parser as an OPTION instead of
// the model. This module is the SINGLE place that decides whether a value may
// be used as an opencode model id.
//
// Allowed: provider/model ids (`opencode/big-pickle`,
// `tud-ai/deepseek-ai/DeepSeek-V4.1-Flash`, `anthropic/claude-opus-4-8`),
// vendor-prefixed names (`claude-opus-4-8`), namespace tokens joined by `:`.
// Rejected: leading `-`, whitespace, control characters, quote/backtick/
// `=`-prefix tricks, paths (`../x`), and anything > 200 chars.
//
// `validateModelName(name)` returns the discriminated `{ ok: true, name } |
// { ok: false, error }` shape that `shell-safety.mjs` uses for its argv/URL
// checks — the same call sites already destructure that shape, so adding
// this validator there is mechanical. The empty/absent case is intentionally
// NOT an error: callers fall back to `OPENCODE_DEFAULT_MODEL` themselves.

export const MODEL_NAME_MAX_LEN = 200
export const MODEL_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._+:/@-]{0,199}$/

export function validateModelName(name) {
  if (name == null) return { ok: true, name: '' }
  if (typeof name !== 'string') return { ok: false, error: `model must be a string, got ${typeof name}` }
  if (name.length === 0) return { ok: true, name: '' }
  if (name.length > MODEL_NAME_MAX_LEN) return { ok: false, error: `model name too long (${name.length} > ${MODEL_NAME_MAX_LEN})` }
  if (!MODEL_NAME_RE.test(name)) {
    return { ok: false, error: `invalid model name ${JSON.stringify(name)} (expected /^[A-Za-z0-9][A-Za-z0-9._+:/@-]{0,199}$/)` }
  }
  return { ok: true, name }
}

// `validateJsonModelName(name)` is the lighter validator for the JSON-body
// providers (anthropic / openai-compat). A model there is data, not an argv
// token, so it may legitimately contain `-` or `_` etc — but a non-string
// MUST be rejected explicitly instead of crashing the SDK on `typeof` checks.
export function validateJsonModelName(name) {
  if (name == null) return { ok: true, name: '' }
  if (typeof name !== 'string') return { ok: false, error: `model must be a string, got ${typeof name}` }
  if (name.length === 0) return { ok: true, name: '' }
  if (name.length > MODEL_NAME_MAX_LEN) return { ok: false, error: `model name too long (${name.length} > ${MODEL_NAME_MAX_LEN})` }
  // Reject the obvious control / NUL / whitespace traps; provider/model IDs
  // never legitimately contain them.
  if (/[\x00-\x1f\x7f\s]/.test(name)) return { ok: false, error: `model name contains control/whitespace characters` }
  return { ok: true, name }
}
