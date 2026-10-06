// Hand-written type declarations for sidecar-llm/tool-validation.mjs.
// Mirrors the module's runtime exports exactly (no `any`): validation returns
// a discriminated union, and the graph helpers accept the two edge containers
// the callers use.

export type RegistryKey = 'layers' | 'training' | 'data'

export function nodeKind(registryKey: string, nodeType: unknown): string | null

export type ValidateNodeParamsResult =
  | { ok: true; params: Record<string, unknown> }
  | { ok: false; error: string }

export function validateNodeParams(
  registryKey: RegistryKey,
  nodeType: string,
  params: unknown,
  opts?: { partial?: boolean },
): ValidateNodeParamsResult

export interface CycleEdge {
  source: string
  target: string
}

export function wouldCreateCycle(
  edges: Map<string, CycleEdge> | readonly CycleEdge[],
  source: string,
  target: string,
): boolean

export function redactSecrets(
  text: unknown,
  secrets: readonly string[] | null | undefined,
): string
