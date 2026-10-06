// Phase 9 (TODO §10): verifyModelForTraining / verificationFromInferResult — the
// fail-closed model-validation gate that blocks training on known-invalid models.
//
// Section 1: pure decision logic (no sidecar required)
// Section 2: real sidecar round-trip (skipped when sidecar unreachable)
import { useGraphStore } from '../src/canvas/GraphStore'
import { serializeCurrent } from '../src/persistence/file'
import { verificationFromInferResult, type ModelVerification } from '../src/inference/verifier'
import type { InferResult } from '../src/inference/client'

let passed = 0
let failed = 0

function check(name: string, cond: boolean): void {
  if (cond) { passed++; process.stdout.write(`  ✓ ${name}\n`) }
  else      { failed++; process.stdout.write(`  ✗ ${name}\n`) }
}

console.log('section 1: verificationFromInferResult decision matrix')
{
  const ok: InferResult = { ok: true, shapes: { relu: [1, 3, 3] }, n_params: 5 }
  const vOk = verificationFromInferResult(ok)
  check('ok result → status valid', vOk.status === 'valid')

  const compileErr: InferResult = { ok: false, stage: 'compile', error: 'SyntaxError: ...', shapes: {} }
  const vCompile = verificationFromInferResult(compileErr)
  check('compile error → status invalid, stage compile', vCompile.status === 'invalid' && vCompile.stage === 'compile')

  const forwardErr: InferResult = { ok: false, stage: 'forward', error: 'RuntimeError: shape mismatch', shapes: { relu: [1, 3, 3] } }
  const vFwd = verificationFromInferResult(forwardErr)
  check('forward error → status invalid, stage forward', vFwd.status === 'invalid' && vFwd.stage === 'forward')

  const constructErr: InferResult = { ok: false, stage: 'construct', error: 'missing required arg', shapes: {} }
  check('construct error → invalid', verificationFromInferResult(constructErr).status === 'invalid')

  const inputErr: InferResult = { ok: false, stage: 'input', error: 'unexpected dtype', shapes: {} }
  check('input error → invalid', verificationFromInferResult(inputErr).status === 'invalid')

  // offline as InferResult has no 'offline' field, so InferErr shape with 'sidecar' stage
  const sidecarErr: InferResult = { ok: false, stage: 'sidecar', error: 'sidecar unreachable', shapes: {} }
  const vSC = verificationFromInferResult(sidecarErr)
  check('sidecar-stage error → invalid (real attempt, real error)', vSC.status === 'invalid' && vSC.stage === 'sidecar')

  // offline returned by inferShapes client when sidecar is down — passed as the
  // general shape, not typed as InferResult, so we test the branch via a cast.
  const offline = { ok: false as const, error: 'sidecar unreachable: fetch failed', offline: true as const, shapes: {} as Record<string, number[]> }
  const vOff = verificationFromInferResult(offline)
  check('offline (fetch failed) → status unknown', vOff.status === 'unknown' && vOff.reason.includes('sidecar unreachable'))
}

console.log('section 2: verifyModelForTraining end-to-end (sidecar required)')
{
  // dynamically import so the module's sidecar URL import resolves
  import('../src/inference/verifier').then(async ({ verifyModelForTraining }) => {
    try {
      // valid model: input [1,3,224,224] → Conv2d → Flatten
      const g = useGraphStore.getState()
      g.resetGraph()
      g.updateNodeParams('input', { shape: [1, 3, 224, 224] })
      const conv = g.addLayer('Conv2d', { x: 100, y: 100 }, { params: { in_channels: 3, out_channels: 8, kernel_size: 3, padding: 1 } })
      g.connectNodes('input', conv)
      const flat = g.addLayer('Flatten', { x: 200, y: 200 }, { params: {} })
      g.connectNodes(conv, flat)
      const v = await verifyModelForTraining(serializeCurrent())
      switch (v.status) {
        case 'valid': {
          check('Conv2d+Flatten model → status valid', true)
          // broken model: Conv2d fed a 1-D input → RuntimeError at forward
          const g2 = useGraphStore.getState()
          g2.resetGraph()
          g2.updateNodeParams('input', { shape: [3] })
          const conv2 = g2.addLayer('Conv2d', { x: 100, y: 100 }, { params: { in_channels: 1, out_channels: 1, kernel_size: 3, padding: 0 } })
          g2.connectNodes('input', conv2)
          const v2 = await verifyModelForTraining(serializeCurrent())
          check('broken Conv2d-on-1D model → status invalid (forward error)', v2.status === 'invalid')
          check('broken model carries a stage', (v2 as { stage?: string }).stage === 'forward')
          break
        }
        case 'unknown':
          // Sidecar offline: the gate still behaves correctly (fail-closed) —
          // training is refused with a reason. The two live-sidecar checks are
          // SKIPPED and reported as such (Rule 6: never claim a pass).
          check('sidecar offline → verify returns unknown (fail-closed)', v.reason.length > 0)
          process.stdout.write('  SKIPPED: valid model → status valid (torch sidecar offline)\n')
          process.stdout.write('  SKIPPED: broken model → status invalid (torch sidecar offline)\n')
          break
      }
    } catch (e) {
      // the gate must never throw — an exception here is a real failure
      check(`verifyModelForTraining did not throw: ${String((e as Error).message).slice(0, 60)}`, false)
    }
  })
}

// The async test runner in section 2 resolves after the promise settles via process
// event-loop drain. Give it 2s, then print results and exit.
setTimeout(() => {
  console.log(`\n${failed === 0 ? '✓ all verifier checks passed' : `✗ ${failed} check(s) failed`} (${passed} passed)`)
  process.exit(failed === 0 ? 0 : 1)
}, 2000)
