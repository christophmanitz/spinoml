// Generate `sidecar-llm/layer-catalog.generated.json` from the three frontend
// registries (architecture layers, training nodes, data nodes).
//
// Run: `npm run gen:layer-catalog`
//
// The committed file is the sidecar's copy of the registries, used by
// `sidecar-llm/tool-validation.mjs` to validate LLM-supplied node types and
// parameters without importing React/TypeScript. Output is deterministic:
// node-type keys sorted, 2-space JSON, trailing newline — so a drift test can
// compare a fresh generation byte-for-byte with the committed file.

import { writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import { LAYERS } from '../src/layers/registry'
import { TRAINING_NODES } from '../src/training/graph/registry'
import { DATA_NODES } from '../src/data/graph/registry'

type AnyField = {
  name: string
  type: string
  default?: unknown
  min?: number
  max?: number
  step?: number
  options?: string[]
  arity?: number
  placeholder?: string
  datalist?: string[]
}

const FIELD_ORDER = ['name', 'type', 'options', 'arity', 'default', 'min', 'max', 'step', 'placeholder', 'datalist'] as const

function fieldToJson(field: AnyField, out: Record<string, unknown>): void {
  for (const key of FIELD_ORDER) {
    const value = (field as Record<string, unknown>)[key]
    if (value !== undefined) out[key] = value
  }
}

function specToJson(spec: { kind?: string; category: string; fields: AnyField[] }): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  if (spec.kind !== undefined) out.kind = spec.kind
  out.category = spec.category
  out.fields = spec.fields.map((field) => {
    const fo: Record<string, unknown> = {}
    fieldToJson(field, fo)
    return fo
  })
  return out
}

function sortedRegistry(reg: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const key of Object.keys(reg).sort()) out[key] = reg[key]
  return out
}

export function buildCatalog(): Record<string, unknown> {
  const layers: Record<string, unknown> = {}
  for (const [type, spec] of Object.entries(LAYERS)) {
    layers[type] = specToJson({
      // LayerSpec.kind defaults to 'module' when omitted.
      kind: spec.kind ?? 'module',
      category: spec.category,
      fields: spec.fields as unknown as AnyField[],
    })
  }
  const training: Record<string, unknown> = {}
  for (const [type, spec] of Object.entries(TRAINING_NODES)) {
    training[type] = specToJson({ category: spec.category, fields: spec.fields as unknown as AnyField[] })
  }
  const data: Record<string, unknown> = {}
  for (const [type, spec] of Object.entries(DATA_NODES)) {
    data[type] = specToJson({ category: spec.category, fields: spec.fields as unknown as AnyField[] })
  }
  return {
    layers: sortedRegistry(layers),
    training: sortedRegistry(training),
    data: sortedRegistry(data),
  }
}

export function generateCatalogJson(): string {
  return JSON.stringify(buildCatalog(), null, 2) + '\n'
}

const isMain = (() => {
  const arg = process.argv[1]
  if (!arg) return false
  try {
    return fileURLToPath(import.meta.url) === path.resolve(arg)
  } catch {
    return false
  }
})()

if (isMain) {
  const target = new URL('../sidecar-llm/layer-catalog.generated.json', import.meta.url)
  writeFileSync(target, generateCatalogJson(), 'utf8')
  console.log(`[gen:layer-catalog] wrote ${fileURLToPath(target)}`)
}
