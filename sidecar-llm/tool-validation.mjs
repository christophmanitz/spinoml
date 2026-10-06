// Provider-independent validation for LLM tool calls.
//
// Provider output is UNTRUSTED input. This module is the sidecar's defence in
// depth: even though the frontend validates too, an invalid tool call must be
// rejected here with an explicit error, without mutating state and without
// emitting an action. It is pure ESM with no I/O beyond reading the committed
// registry snapshot (layer-catalog.generated.json) once at import.
//
// `validateNodeParams` mirrors the frontend's `coerceParams`/`validateParams`
// semantics closely enough that it accepts EXACTLY the values the frontend
// would store UNCHANGED (see scripts/test-llm-validation-parity.ts). It is
// deliberately at least as strict: it also rejects unknown/prototype keys and
// absurd magnitudes (>1e9) the frontend would happily persist.

import { readFileSync } from 'node:fs'

const CATALOG = JSON.parse(readFileSync(new URL('./layer-catalog.generated.json', import.meta.url), 'utf8'))

const FORBIDDEN_KEYS = new Set(['__proto__', 'constructor', 'prototype'])
const MAX_MAGNITUDE = 1e9

function isPlainObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isKnownRegistry(registryKey) {
  return Object.prototype.hasOwnProperty.call(CATALOG, registryKey)
}

/** The architectural kind of a node ('input' | 'output' | 'module' | …), or null. */
export function nodeKind(registryKey, nodeType) {
  const reg = CATALOG[registryKey]
  if (!reg || typeof nodeType !== 'string') return null
  const spec = reg[nodeType]
  return spec ? (spec.kind ?? null) : null
}

function closeMatches(all, name) {
  const needle = String(name).toLowerCase()
  const matches = []
  for (const candidate of all) {
    const lower = candidate.toLowerCase()
    if (lower.startsWith(needle) || lower.includes(needle) || needle.includes(lower)) matches.push(candidate)
    if (matches.length >= 8) break
  }
  return matches
}

function parseNumber(value) {
  if (typeof value === 'number') return value
  if (typeof value === 'string') {
    if (value.trim() === '') return Number.NaN
    return Number(value)
  }
  return Number.NaN
}

function describe(value) {
  try {
    const s = JSON.stringify(value)
    return s === undefined ? String(value) : s
  } catch {
    return String(value)
  }
}

function checkField(field, value, opts = {}) {
  const fail = (message) => ({ ok: false, error: `${field.name}: ${message}` })
  // The layer registry's bool coercion maps "true"/"false" faithfully, but the
  // training/data registries use `Boolean(value)` — where Boolean("false") is
  // TRUE. Accept the string form only where the frontend agrees on the value.
  const boolStrings = opts.registryKey === 'layers'
  switch (field.type) {
    case 'int': {
      const n = parseNumber(value)
      if (!Number.isFinite(n) || !Number.isInteger(n)) return fail(`expected an integer, got ${describe(value)}`)
      if (Math.abs(n) > MAX_MAGNITUDE) return fail(`absolute value ${n} exceeds ${MAX_MAGNITUDE}`)
      if (field.min !== undefined && n < field.min) return fail(`${n} < min ${field.min}`)
      if (field.max !== undefined && n > field.max) return fail(`${n} > max ${field.max}`)
      return { ok: true, value: n }
    }
    case 'float': {
      const n = parseNumber(value)
      if (!Number.isFinite(n)) return fail(`expected a finite number, got ${describe(value)}`)
      if (field.min !== undefined && n < field.min) return fail(`${n} < min ${field.min}`)
      if (field.max !== undefined && n > field.max) return fail(`${n} > max ${field.max}`)
      return { ok: true, value: n }
    }
    case 'bool': {
      if (typeof value === 'boolean') return { ok: true, value }
      if (value === 'true') return { ok: true, value: true }
      if (value === 'false' && boolStrings) return { ok: true, value: false }
      return fail(`expected a boolean${boolStrings ? ' or "true"/"false"' : ''}, got ${describe(value)}`)
    }
    case 'select': {
      if (typeof value !== 'string' || !field.options.includes(value)) {
        return fail(`${describe(value)} is not one of ${JSON.stringify(field.options)}`)
      }
      return { ok: true, value }
    }
    case 'tuple-int': {
      const ints = []
      const one = (v) => {
        const n = parseNumber(v)
        if (!Number.isFinite(n) || !Number.isInteger(n)) return null
        if (Math.abs(n) > MAX_MAGNITUDE) return null
        if (field.min !== undefined && n < field.min) return null
        if (field.max !== undefined && n > field.max) return null
        return n
      }
      if (Array.isArray(value)) {
        if (value.length !== field.arity) return fail(`expected ${field.arity} ints, got ${describe(value)}`)
        for (const v of value) {
          const n = one(v)
          if (n === null) return fail(`every entry must be a valid int, got ${describe(value)}`)
          ints.push(n)
        }
      } else if (typeof value === 'number' || (typeof value === 'string' && value.trim() !== '')) {
        const n = one(value)
        if (n === null) return fail(`expected ${field.arity} ints or one int, got ${describe(value)}`)
        for (let i = 0; i < field.arity; i++) ints.push(n)
      } else {
        return fail(`expected ${field.arity} ints or one int, got ${describe(value)}`)
      }
      return { ok: true, value: ints }
    }
    case 'int-list':
    case 'shape': {
      if (!Array.isArray(value) || value.length === 0) {
        return fail(`expected a non-empty array of ints, got ${describe(value)}`)
      }
      const out = []
      for (const v of value) {
        const n = parseNumber(v)
        if (!Number.isFinite(n) || !Number.isInteger(n)) return fail(`expected a non-empty array of ints, got ${describe(value)}`)
        if (Math.abs(n) > MAX_MAGNITUDE) return fail(`absolute value ${n} exceeds ${MAX_MAGNITUDE}`)
        out.push(n)
      }
      return { ok: true, value: out }
    }
    case 'columns-multi': {
      if (!Array.isArray(value) || !value.every((v) => typeof v === 'string')) {
        return fail(`expected an array of strings, got ${describe(value)}`)
      }
      return { ok: true, value: value.slice() }
    }
    case 'dataset-ref':
    case 'model-ref':
    case 'column-single':
    case 'text':
    case 'code': {
      if (typeof value !== 'string') return fail(`expected a string, got ${describe(value)}`)
      return { ok: true, value }
    }
    default:
      // Unknown field kind in the generated catalog: fail closed rather than
      // let an unvalidated value reach the graph.
      return fail(`unsupported field type "${field.type}"`)
  }
}

/**
 * Validate a node's params against its registry entry.
 *
 * @param {'layers'|'training'|'data'} registryKey
 * @param {string} nodeType
 * @param {unknown} params   The supplied param object.
 * @param {{ partial?: boolean }} [opts]  `partial` (update_params): only the
 *   supplied keys are validated; the same rules apply to each.
 * @returns {{ ok: true, params: Record<string, unknown> } | { ok: false, error: string }}
 */
export function validateNodeParams(registryKey, nodeType, params, opts = {}) {
  const reg = CATALOG[registryKey]
  if (!isKnownRegistry(registryKey)) return { ok: false, error: `unknown registry "${registryKey}"` }
  if (typeof nodeType !== 'string' || !reg[nodeType]) {
    const matches = closeMatches(Object.keys(reg), nodeType)
    const hint = matches.length
      ? `close matches: ${matches.join(', ')}`
      : 'see the tool description for supported types'
    return { ok: false, error: `unknown node type ${describe(nodeType)} (${hint})` }
  }
  if (!isPlainObject(params)) {
    return { ok: false, error: `params must be a plain object, got ${describe(params)}` }
  }
  const fields = new Map(reg[nodeType].fields.map((f) => [f.name, f]))
  // Params that a frontend PANEL writes (and coerceParams stores unchanged) but
  // that are not FieldSpecs in the registry: the Input binding panel's
  // `bind_field` ('<branch>.<field>' slot). Allowing it explicitly keeps a
  // documented binding workflow from failing on an invisible rule.
  if (registryKey === 'layers' && nodeType === 'Input' && !fields.has('bind_field')) {
    fields.set('bind_field', { name: 'bind_field', type: 'text', default: '' })
  }
  const validKeys = [...fields.keys()]

  for (const key of Object.keys(params)) {
    if (FORBIDDEN_KEYS.has(key)) return { ok: false, error: `parameter name "${key}" is not allowed` }
    if (!fields.has(key)) {
      const hint = validKeys.length ? `valid keys: ${validKeys.join(', ')}` : 'this node takes no parameters'
      return { ok: false, error: `unknown parameter "${key}" for ${nodeType} (${hint})` }
    }
  }

  const out = {}
  for (const [key, value] of Object.entries(params)) {
    const check = checkField(fields.get(key), value, { registryKey })
    if (!check.ok) return { ok: false, error: `${nodeType}.${check.error}` }
    out[key] = check.value
  }
  return { ok: true, params: out }
}

/**
 * Mirror of the frontend's `wouldCreateCycle` (src/canvas/invariants.ts).
 * `edges` may be a Map of edge_key → {source,target} or an array of them.
 */
export function wouldCreateCycle(edges, source, target) {
  if (source === target) return true
  const list = edges instanceof Map ? [...edges.values()] : Array.isArray(edges) ? edges : []
  const adj = new Map()
  for (const e of list) {
    if (!adj.has(e.source)) adj.set(e.source, [])
    adj.get(e.source).push(e.target)
  }
  const seen = new Set([target])
  let frontier = [target]
  while (frontier.length > 0) {
    const next = []
    for (const id of frontier) {
      for (const t of adj.get(id) ?? []) {
        if (t === source) return true
        if (!seen.has(t)) { seen.add(t); next.push(t) }
      }
    }
    frontier = next
  }
  return false
}

// Generic secret shapes, independent of any known key. Applied AFTER the exact
// per-turn secrets so a leaked key is masked even when only its shape is known.
const GENERIC_SECRET_PATTERNS = [
  /sk-[A-Za-z0-9_-]{8,}/g,
  /Bearer\s+[A-Za-z0-9._-]{8,}/gi,
  /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g,
  /:\/\/[^/\s:@]+:[^/\s@]+@/g,
]

/** Replace every known secret and every generic secret shape with [REDACTED]. */
export function redactSecrets(text, secrets) {
  let out = String(text ?? '')
  for (const secret of secrets ?? []) {
    if (typeof secret !== 'string' || secret.length === 0) continue
    out = out.split(secret).join('[REDACTED]')
  }
  for (const pattern of GENERIC_SECRET_PATTERNS) out = out.replace(pattern, '[REDACTED]')
  return out
}
