#!/usr/bin/env tsx
// Phase 50 — regression harness for the silent-exception fixes.
//
// Each HIDDEN-FAILURE site that was converted into an explicit error/unknown
// state gets at least one assertion here. Most fixes live in Zustand stores /
// React components that need a DOM or Tauri runtime, so the checks are
// source-text assertions (the same style as verify-ui-state /
// verify-frontend-errors); the guard `verify:silent-catch` covers the AST
// side.

import { readFileSync, existsSync } from 'node:fs'
import { join } from 'node:path'

let failures = 0
function check(name: string, cond: boolean) {
  if (cond) console.log(`  ✓ ${name}`)
  else { failures++; console.log(`  ✗ ${name}`) }
}

const src = (p: string) => readFileSync(join(process.cwd(), 'src', p), 'utf8')

console.log('phase 50: silent-exception fixes')

// 1. workspace .py twin — write/rename/remove failures become visible
{
  const ws = src('workspace/store.ts')
  check('.py twin errors collected via pyTwinError state', ws.includes('pyTwinError') && ws.includes('writeTwin'))
  check('.py twin write no longer ignored', !ws.includes("catch { /* ignore .py write errors */ }"))
  check('.py twin rename/remove/move check presence', ws.includes('twinPresent') && ws.includes('hadTwin'))
  const fe = src('workspace/FileExplorer.tsx')
  check('FileExplorer renders the pyTwinError banner', fe.includes('pyTwinError') && fe.includes('pyTwinError &&'))
  check('FileExplorer openGraphOnCanvas surfaces failure', fe.includes('konnte nicht geöffnet werden'))
}

// 2. training/data canvas bind failures become canvas errors
{
  check('training ensureBound sets error status', src('training/graph/doc.ts').includes("setStatus('training', 'error'"))
  check('data ensureBound sets error status', src('data/graph/doc.ts').includes("setStatus('data', 'error'"))
}

// 3. RunDetailModal — run.json/metrics/events/GPU failures are explicit
{
  const m = src('training/RunDetailModal.tsx')
  check('run.json corruption surfaced (cfgError)', m.includes('cfgError') && m.includes('run.json beschädigt'))
  check('file read failures surfaced (readError)', m.includes('readError') && m.includes('readRunFile'))
  check('events read failure surfaced (eventsError)', m.includes('eventsError'))
  check('gpu probe failure surfaced (gpuError)', m.includes('gpuError'))
  check('openOnCanvas failure surfaced (actionError)', m.includes('actionError'))
  check('no blind JSON.parse catch remains', !m.includes("catch { /* run.json not ready / malformed */ }"))
}

// 4. CompareModal — per-run load errors
{
  const c = src('training/CompareModal.tsx')
  check('CompareModal tracks configError + eventsError', c.includes('configError') && c.includes('eventsError'))
  check('CompareModal renders per-run warnings', c.includes('run.json') && c.includes('events'))
}

// 5. EvalRunModal + NewRunModal — remote caps are UNKNOWN, not local
{
  const e = src('training/EvalRunModal.tsx')
  check('EvalRunModal caps probe failure explicit', e.includes('capsError'))
  check('EvalRunModal manifest corruption throws (not fake default)', e.includes('ist beschädigt') && !e.includes('setSrcManifest({ pairs: {}, target: {} })'))
  check('EvalRunModal source-inspect warning surfaced', e.includes('srcWarn'))
  const n = src('training/NewRunModal.tsx')
  check('NewRunModal caps error state exists', n.includes('capsError'))
  check('NewRunModal does not fake local on failed caps probe', !n.includes("setBackendKind('local') } })") && n.includes("Fähigkeiten von"))
  check('NewRunModal blocks the start while the capability probe failed', /const canSubmit =[\s\S]{0,300}&& !capsError/.test(n))
}

// 6. remote sidecar refresh — no stale status
{
  check('remoteSidecar refresh sets explicit unknown on failure',
    src('sidecars/remoteSidecar.ts').includes('Sidecar-Status unbekannt'))
}

// 7. canvas file gate / reload — load failures visible
{
  const g = src('canvasdoc/CanvasFileGate.tsx')
  check('CanvasFileGate bound-load error panel', g.includes('loadError') && g.includes('konnte nicht geladen werden'))
  check('CanvasFileGate list error distinct from empty', g.includes('listErr'))
  check('ReloadCanvasButton surfaces reload failure', src('canvasdoc/ReloadCanvasButton.tsx').includes('Neu laden fehlgeschlagen'))
}

// 8. inspector / explain dropdowns — list failures not shown as "none"
{
  check('TrainingInspector ModelRef list error surfaced',
    src('training/graph/TrainingInspector.tsx').includes('listErr'))
  check('LayerExplain runs list error surfaced',
    src('visualization/LayerExplain.tsx').includes('runsErr'))
}

// 9. chat SSE — malformed frame aborts instead of silent truncation
{
  const c = src('chat/client.ts')
  check('chat/client throws on malformed SSE frame', c.includes('bad SSE chunk from sidecar') && !c.includes("console.warn('bad SSE chunk'"))
}

// 10. datasets history read error
{
  const d = src('datasets/store.ts')
  check('datasets loadHistory error state', d.includes('historyError'))
  check('DatasetDetail renders history error', src('datasets/DatasetDetail.tsx').includes('historyError'))
}

// 11. the guard and the allow-list exist
{
  check('guard script exists', existsSync(join(process.cwd(), 'scripts/verify-silent-catch.ts')))
  check('allow-list document exists', existsSync(join(process.cwd(), 'docs/engineering/SILENT_EXCEPTIONS.md')))
  check('evidence of pre-fix scan exists', existsSync(join(process.cwd(), 'docs/engineering/evidence/phase50-ts-before.txt')))
}

if (failures > 0) {
  console.error(`\n${failures} check(s) failed`)
  process.exit(1)
}
console.log('\n✓ all silent-exception fix checks passed')
