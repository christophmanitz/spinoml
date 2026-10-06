// Pure Python-literal helpers — turn untrusted (user/LLM) values into inert
// Python source fragments. No I/O, no Date/random (deterministic), so the
// codegen invariance in CLAUDE.md holds. This is the single source of truth for
// escaping across generator.ts / dataCodegen.ts / trainingCodegen.ts.

// Python keywords + soft keywords that cannot be used as bare identifiers.
const PY_KEYWORDS = new Set([
  'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break',
  'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally',
  'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal',
  'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield',
])

/** A single-quoted Python string literal that round-trips through
 *  `ast.literal_eval` exactly. Non-strings go through String(); undefined/null
 *  become ''. Control chars (U+0000–U+001F, U+007F–U+009F) and the line/para
 *  separators U+2028/U+2029 become escapes so no line terminator or NUL can
 *  survive; all other characters (umlauts, CJK, …) are left as-is. */
export function pyStr(s: unknown): string {
  const str = s === undefined || s === null ? '' : String(s)
  let out = "'"
  for (const ch of str) {
    const cp = ch.codePointAt(0)!
    if (ch === '\\') out += '\\\\'
    else if (ch === "'") out += "\\'"
    else if (ch === '\n') out += '\\n'
    else if (ch === '\r') out += '\\r'
    else if (ch === '\t') out += '\\t'
    else if (cp <= 0x1f || (cp >= 0x7f && cp <= 0x9f)) {
      out += '\\x' + cp.toString(16).padStart(2, '0')
    } else if (cp === 0x2028) out += '\\u2028'
    else if (cp === 0x2029) out += '\\u2029'
    else out += ch
  }
  return out + "'"
}

/** Text safe to append after `# `: every control char, NUL, U+2028/U+2029 is
 *  replaced by a space, trailing spaces are trimmed, and the result is capped
 *  at 200 chars (with a trailing ellipsis). */
export function pyComment(s: unknown): string {
  const str = s === undefined || s === null ? '' : String(s)
  let out = ''
  for (const ch of str) {
    const cp = ch.codePointAt(0)!
    if (cp <= 0x1f || (cp >= 0x7f && cp <= 0x9f) || cp === 0x2028 || cp === 0x2029) out += ' '
    else out += ch
  }
  out = out.replace(/ +$/, '')
  if (out.length > 200) out = out.slice(0, 200) + '…'
  return out
}

/** `[<pyStr(a)>, <pyStr(b)>, …]`. */
export function pyList(items: unknown[]): string {
  return '[' + items.map(pyStr).join(', ') + ']'
}

/** A valid Python identifier derived from arbitrary text: chars outside
 *  `[A-Za-z0-9_]` become `_`, a leading digit gets a `_` prefix, empty falls
 *  back, and keywords get a trailing `_`. */
export function pyIdent(s: unknown, fallback = '_x'): string {
  const raw = s === undefined || s === null ? '' : String(s)
  let id = raw.replace(/[^A-Za-z0-9_]/g, '_')
  if (id === '') id = fallback
  if (/^[0-9]/.test(id)) id = '_' + id
  if (PY_KEYWORDS.has(id)) id += '_'
  return id
}

function formatFloat(v: number): string {
  if (v === 0) return '0.0'
  const abs = Math.abs(v)
  if (abs < 1e-3 || abs >= 1e6) return v.toExponential()
  return Number.isInteger(v) ? `${v}.0` : String(v)
}

/** Only finite numbers are emitted (using the historical `formatFloat` logic so
 *  benign output is byte-identical); anything else (NaN, ±Infinity, string,
 *  null, object) yields the fallback formatted the same way. */
export function pyFloat(v: unknown, fallback: number): string {
  if (typeof v === 'number' && Number.isFinite(v)) return formatFloat(v)
  return formatFloat(fallback)
}

/** Only finite numbers are emitted (`Math.trunc`); anything else yields the
 *  fallback. */
export function pyInt(v: unknown, fallback: number): string {
  if (typeof v === 'number' && Number.isFinite(v)) return String(Math.trunc(v))
  return String(Math.trunc(fallback))
}

/** `[a, b, …]` of finite integers: if `v` is not an array whose every element
 *  is a finite number the fallback array is used. */
export function pyIntList(v: unknown, fallback: number[]): string {
  const arr = Array.isArray(v) && v.every((x) => typeof x === 'number' && Number.isFinite(x))
    ? (v as number[]).map((x) => Math.trunc(x))
    : fallback.map((x) => Math.trunc(x))
  return '[' + arr.join(', ') + ']'
}
