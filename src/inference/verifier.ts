// Phase 9 (TODO §10): shape failure must fail closed — training must not start
// on a model that is not KNOWN to be valid. This module verifies the FROZEN
// .spinoml model file (the exact one training would use) via the torch sidecar
// before launch, and classifies the result VALID / INVALID / UNKNOWN.
import { inferShapes, type InferResult } from './client'
import { generateFromSnapshot } from '../codegen/generator'
import { parseFile } from '../persistence/file'
import { assertTrusted, UntrustedCodeError } from '../trust/guard'

export type ModelVerification =
  | { status: 'valid' }
  | { status: 'invalid'; stage: string; error: string }
  | { status: 'unknown'; reason: string }

/** Classify a raw /infer result into the three fail-closed states:
 *  - ok      → VALID
 *  - reply ok:false with a real compile/construct/input/forward error → INVALID
 *  - offline / sidecar unreachable → UNKNOWN (not verified, not proven bad)      */
export function verificationFromInferResult(r: InferResult | { ok: false; error: string; offline: true; shapes: Record<string, number[]> }): ModelVerification {
  if (r.ok) return { status: 'valid' }
  if ('offline' in r && r.offline) return { status: 'unknown', reason: r.error }
  const rr = r as InferResult
  if (!rr.ok) return { status: 'invalid', stage: rr.stage ?? 'sidecar', error: rr.error }
  return { status: 'valid' }
}

/** Verify the model file that WILL be trained, exactly as training generates it:
 *  parse → codegen → run the generated forward pass on the torch sidecar.       */
export async function verifyModelForTraining(modelSpinoml: string): Promise<ModelVerification> {
  const snap = parseFile(modelSpinoml)
  // Phase 43 — fail closed before any sidecar call: unapproved Custom/DataOp
  // code is UNVERIFIABLE (→ 'unknown', which NewRunModal already blocks).
  try {
    await assertTrusted(snap.nodes)
  } catch (e) {
    if (e instanceof UntrustedCodeError) return { status: 'unknown', reason: e.message }
    throw e
  }
  const { code, inputs } = generateFromSnapshot(snap)
  const inputShapes = inputs.map((i) => i.shape)
  const inputDtypes = inputs.map((i) => i.dtype)
  const result = await inferShapes(code, inputShapes, inputDtypes)
  return verificationFromInferResult(result)
}