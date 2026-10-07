// Trust boundary: parse + narrow an untrusted JSON document into a RunConfig.
// `run.json` is written by the training sidecar (a Rust process whose version
// the user can't see). The historical code did `JSON.parse(raw) as RunConfig`
// everywhere — a schema drift would silently mis-type the value, then crash
// later in a confusing place. The narrow check below stops a fundamentally
// malformed file; deeper drift is caught the first time the field is read.

import type { RunConfig } from './types'

function isObj(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

export function parseRunConfig(raw: unknown): RunConfig {
  if (!isObj(raw)) throw new Error('run.json ist beschädigt: kein Objekt')
  const ds = isObj(raw.dataset) ? raw.dataset : null
  if (!ds || typeof ds.relpath !== 'string') {
    throw new Error('run.json ist beschädigt: dataset.relpath fehlt')
  }
  return raw as unknown as RunConfig
}
