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
import { execFileSync } from 'node:child_process'
import { join } from 'node:path'

let failures = 0
function check(name: string, cond: boolean) {
  if (cond) console.log(`  ✓ ${name}`)
  else { failures++; console.log(`  ✗ ${name}`) }
}

const src = (p: string) => readFileSync(join(process.cwd(), 'src', p), 'utf8')
const node = (p: string) => readFileSync(join(process.cwd(), 'sidecar-llm', p), 'utf8')

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

// 12. Node sidecar (sidecar-llm/*.mjs) — the same audit, functional where a
//     pure helper can be exercised, source-text otherwise (main.mjs is a live
//     HTTP server, so importing it in a check would bind a port / need network).
{
  const main = node('main.mjs')
  const bridge = node('mcp-bridge.mjs')

  // fixed site: notesList local readdir falsely empty
  check('readdirOptional distinguishes ENOENT from a read failure',
    main.includes('async function readdirOptional(') && main.includes("if (e && e.code === 'ENOENT') return []") && main.includes('const items = await readdirOptional(dir)'))
  // fixed site: notesList stat invented size 0
  check('failed note stat reports size null + stat_error (rendered "(size unknown)")',
    main.includes('size: null, mtime: null, stat_error: e.message') && main.includes("'(size unknown)'"))
  // fixed site: wsListDir falsely empty (local .catch, remote `|| true`)
  check('wsListDir uses readdirOptional', /async function wsListDir\([\s\S]{0,500}readdirOptional\(abs\)/.test(main))
  check('wsListDir remote branch surfaces a listing failure',
    main.includes('if [ ! -d ${d} ]; then exit 0; fi; ls -1 ${d}') && !main.includes("ls -1 ${shellQuotePath(`${ws.root}/${reldir}`)} 2>/dev/null || true"))
  // fixed site: wsReadFile swallowed unreadable as ''
  check('wsReadFile throws on non-ENOENT and probes existence remotely',
    main.includes('if [ -e ${p} ]; then cat ${p}; fi') && /async function wsReadFile\([\s\S]{0,800}if \(e && e.code === 'ENOENT'\) return ''/.test(main))
  check('no bare readFile(...).catch(() => \'\') remains', !main.includes("fs.readFile(abs, 'utf8').catch(() => '')"))
  // fixed site: wsListDirDetailed falsely empty
  check('wsListDirDetailed uses readdirOptional with dirents', main.includes('readdirOptional(abs, { withFileTypes: true })'))
  check('wsListDirDetailed remote branch surfaces a listing failure',
    main.includes('if [ ! -d ${shellQuotePath(target)} ]; then exit 0; fi; cd ${shellQuotePath(target)} && ls -1Ap') && !main.includes('ls -1Ap 2>/dev/null || true'))
  // fixed site: readSummaryEvents silent empty summary
  check('readSummaryEvents returns an explicit { text, error }',
    main.includes("return { text: await runSsh(ws.sshTarget, cmd), error: null }") && !main.includes("} catch { return '' }"))
  // fixed site: runsList run.json/metrics.json skipped
  check('runsList reads via readRunJson and surfaces a per-run warning',
    main.includes('async function readRunJson(')
    && main.includes("state: 'missing'")
    && main.includes('warning: warnings.length ? warnings.join')
    && main.includes("readRunJson(ws, `${RUNS_DIR}/${id}/run.json`)"))
  check('no run.json/metrics.json `catch { skip }` remains', !main.includes('catch { /* skip */ }'))
  // fixed site: runRead run.json/metrics.json skipped + events swallowed
  check('runRead reports readRunJson state as warnings',
    /async function runRead\([\s\S]{0,700}warnings/.test(main) && main.includes('warnings, n_params: nParams'))
  // fixed site: downloadToDatasets invented byte count
  check('downloadToDatasets rejects an undeterminable size',
    main.includes('!Number.isFinite(bytes) || bytes < 0') && !main.includes('parseInt(out.trim(), 10) || 0'))
  // fixed site: listOpenCodeModels dropped exit/timeout -> empty model list
  check('listOpenCodeModels throws on timeout/abort/non-zero exit',
    main.includes('timed out') && main.includes('if (r.code !== 0) {') && main.includes('models\\` exited'))
  // fixed site: slurmStatus dropped transport failure -> UNKNOWN
  check('slurmStatus throws on timeout/abort/ssh-255',
    main.includes('slurm_status probe timed out') && main.includes('if (r.code === 255) {'))
  // fixed site: mcp-bridge silently dropped a rejected handler
  check('mcp-bridge replies with an explicit JSON-RPC INTERNAL_ERROR',
    bridge.includes("code: 'INTERNAL_ERROR'") && !bridge.includes("handle(msg).catch(() => { /* the async handlers resolve their own errors */ })"))

  // functional: both edited files are syntactically valid JS
  for (const f of ['main.mjs', 'mcp-bridge.mjs']) {
    let ok = true
    try { execFileSync(process.execPath, ['--check', join(process.cwd(), 'sidecar-llm', f)], { stdio: 'ignore' }) }
    catch { ok = false }
    check(`node --check sidecar-llm/${f}`, ok)
  }
  check('node allow-list documented', readFileSync(join(process.cwd(), 'docs/engineering/SILENT_EXCEPTIONS.md'), 'utf8').includes('## Allow-list — Node sidecar'))
  check('node pre-fix evidence exists', existsSync(join(process.cwd(), 'docs/engineering/evidence/phase50-node-before.txt')))
}

if (failures > 0) {
  console.error(`\n${failures} check(s) failed`)
  process.exit(1)
}
console.log('\n✓ all silent-exception fix checks passed')
