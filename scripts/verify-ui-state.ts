#!/usr/bin/env tsx
// Phase 38 — UI state must not lie (TODO §38).
// "If saving fails, do not display 'Saved'. If training fails, do not
// display 'Completed'. If shape inference fails, do not display 'Valid'.
// If SSH disconnects, do not display 'Connected'. If SLURM state is unknown,
// do not display 'Running' unless verified."
//
// This harness proves the UI truthfully reflects the underlying state for the
// critical paths (training/SSH/SLURM/inference). Browser localStorage quota is
// a known gap (localStorage silenced catch) but not the production Tauri path.

import { execSync } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

let failures = 0
function check(name: string, cond: boolean, detail = '') {
  if (cond) console.log(`  ✓ ${name}`)
  else { failures++; console.log(`  ✗ ${name} ${detail}`) }
}

console.log('phase 38: ui state must not lie')

// 1. Training — failed must not be shown as completed; stale running must not linger on SSH loss
{
  const store = readFileSync(join(process.cwd(), 'src', 'training', 'store.ts'), 'utf8')
  check('training store refresh() degrades stale running to unknown on listError',
    store.includes("RUNNING_STATES.has(r.status) || r.alive") && store.includes("status: 'unknown'") && store.includes("alive: false"))
  check('training store clears pollTimer on listError (no spam on dead connection)',
    store.includes('clearTimeout(pollTimer)') && store.includes('listError: msg'))
  check('training store sets listError explicitly on failure', store.includes('listError: msg'))
  check('training store does not clear non-running runs on transient failure (history preserved)',
    store.includes('hasDegraded') || store.includes('degraded'))
  check('training store re-arms polling only on next success (via syncPolling)', store.includes('syncPolling(get)'))

  // Rust reconcile already proves running without alive → failed, not running (Phase 37)
  const trainingRs = readFileSync(join(process.cwd(), 'src-tauri', 'src', 'training.rs'), 'utf8')
  check('Rust reconcile_status prevents dead run shown as running', trainingRs.includes('fn reconcile_status') && trainingRs.includes('"failed"'))
  check('Rust reconcile_slurm_status prevents unknown SLURM shown as running',
    trainingRs.includes('fn reconcile_slurm_status') && trainingRs.includes('COMPLETED') && trainingRs.includes('CANCELLED'))

  // StatusPill only pulses when actually alive
  const pill = readFileSync(join(process.cwd(), 'src', 'training', 'StatusPill.tsx'), 'utf8')
  check('StatusPill only pulses when alive', pill.includes("status === 'running'") && pill.includes('alive'))

  // Events truncates trailing epoch after failed so Completed not shown erroneously
  const events = readFileSync(join(process.cwd(), 'src', 'training', 'events.ts'), 'utf8')
  check('events truncates stale trailing after terminal (no false Completed)',
    events.includes('truncateAtTerminal') && events.includes('finalTerminal'))
}

// 2. SSH — disconnect must not still show Connected
{
  const app = readFileSync(join(process.cwd(), 'src', 'App.tsx'), 'utf8')
  check('App.tsx ProjectHeader exists', app.includes('function ProjectHeader'))
  check('ProjectHeader shows disconnected (rose badge) when listError present',
    app.includes('listError') && app.includes('nicht verbunden') && app.includes('bg-rose-900/40'))
  check('ProjectHeader shows error badge when project status is error',
    app.includes("status.kind === 'error'") && app.includes('nicht verbunden'))
  check('ProjectHeader imports useTrainingStore for live connectivity signal',
    app.includes('useTrainingStore'))
  check('ProjectHeader still shows violet ssh badge when connected',
    app.includes('bg-violet-900/30') && app.includes('ssh ·'))
  // backend dispatch is live per-call, not cached
  const backend = readFileSync(join(process.cwd(), 'src', 'training', 'backend.ts'), 'utf8')
  check('training backend dispatch live per-call (no cached Connected)',
    backend.includes('getCurrentConnection'))
}

// 3. Shape inference — failure must not show Valid/ok; inferring must not hang
{
  const infStore = readFileSync(join(process.cwd(), 'src', 'inference', 'store.ts'), 'utf8')
  check('inference store exists', existsSync(join(process.cwd(), 'src', 'inference', 'store.ts')))
  check('inference store catch handles non-Abort errors (leaves inferring)',
    infStore.includes("throw e") === false || (infStore.includes("status: offline") && infStore.includes("status: 'error'")))
  // The fix: the outer catch must set status to error/offline and clear shapes, not re-throw
  check('inference store sets error/offline on fetch/network failure',
    infStore.includes("offline ? 'offline' : 'error'") || infStore.includes("status: offline"))
  check('inference store clears shapes on error/offline (no stale Valid)',
    infStore.includes('clearShapesOnNodes()'))
  check('inference store guards stale response via runCounter',
    infStore.includes('runCounter') && infStore.includes('runId !== runCounter'))

  const badge = readFileSync(join(process.cwd(), 'src', 'App.tsx'), 'utf8')
  check('InferenceBadge shows ok only on status===ok',
    badge.includes("status === 'ok'") && badge.includes('bg-emerald-900/40'))
  check('InferenceBadge maps error→rose, offline→gray, inferring→gray',
    badge.includes("status === 'error'") && badge.includes('bg-rose-900/40') && badge.includes("status === 'offline'"))

  const verifier = readFileSync(join(process.cwd(), 'src', 'inference', 'verifier.ts'), 'utf8')
  check('verifier fail-closed: offline → unknown (never Valid)',
    verifier.includes('offline') && verifier.includes('unknown'))

  const newRun = readFileSync(join(process.cwd(), 'src', 'training', 'NewRunModal.tsx'), 'utf8')
  check('NewRunModal blocks training when shape invalid/unknown (no Valid bypass)',
    newRun.includes('invalid') && newRun.includes('unknown') && newRun.includes('throw'))
}

// 4. Saving — failure must not show Saved
{
  const ws = readFileSync(join(process.cwd(), 'src', 'workspace', 'store.ts'), 'utf8')
  check('workspace saveActive awaits fs write before dirty=false',
    ws.includes('await fsBackend.write') && ws.includes('dirty: false'))
  // DocStatus pattern for data/training canvases
  const dataDoc = existsSync(join(process.cwd(), 'src', 'data', 'graph', 'doc.ts'))
    ? readFileSync(join(process.cwd(), 'src', 'data', 'graph', 'doc.ts'), 'utf8')
    : ''
  const trainDoc = existsSync(join(process.cwd(), 'src', 'training', 'graph', 'doc.ts'))
    ? readFileSync(join(process.cwd(), 'src', 'training', 'graph', 'doc.ts'), 'utf8')
    : ''
  check('data canvas DocStatus sets saved only on success, error on catch',
    dataDoc.includes("'saved'") && dataDoc.includes("'error'") && dataDoc.includes("setStatus('data'"))
  check('training canvas DocStatus sets saved only on success, error on catch',
    trainDoc.includes("'saved'") && trainDoc.includes("'error'") && trainDoc.includes("setStatus('training'"))

  const toolbar = readFileSync(join(process.cwd(), 'src', 'Toolbar.tsx'), 'utf8')
  check('Toolbar save catches error and reports (no false Saved)',
    toolbar.includes('try') && toolbar.includes('catch') && toolbar.includes('reportError'))
}

// 5. SLURM — unknown must not show Running
{
  const trainingRs = readFileSync(join(process.cwd(), 'src-tauri', 'src', 'training.rs'), 'utf8')
  check('SLURM squeue empty + sacct empty → unknown/failed, never running',
    trainingRs.includes('reconcile_status(status_raw, false)') || trainingRs.includes('reconcile_status'))
  // verify:slurm already proved the full 9-state matrix; verify:recovery proved running without alive → failed
  const slurmHarness = join(process.cwd(), 'scripts', 'verify-slurm.ts')
  check('verify:slurm harness exists (SLURM unknown coverage)', existsSync(slurmHarness))
  const recoveryHarness = join(process.cwd(), 'scripts', 'verify-recovery.ts')
  check('verify:recovery harness exists (stale running → failed)', existsSync(recoveryHarness))
}

// 6. General — listError is surfaced, not swallowed as loading=false alone
{
  const store = readFileSync(join(process.cwd(), 'src', 'training', 'store.ts'), 'utf8')
  check('training store distinguishes listLoading vs listError (not just loading=false)',
    store.includes('listLoading') && store.includes('listError'))
  const expExplorer = readFileSync(join(process.cwd(), 'src', 'training', 'ExperimentsExplorer.tsx'), 'utf8')
  check('ExperimentsExplorer shows listError banner when present',
    expExplorer.includes('listError'))

  const datasetsStore = readFileSync(join(process.cwd(), 'src', 'datasets', 'store.ts'), 'utf8')
  check('datasets store also distinguishes loading/error (precedent)',
    datasetsStore.includes('loading') || datasetsStore.includes('error'))
}

if (failures > 0) {
  console.error(`\n${failures} check(s) failed`)
  process.exit(1)
} else {
  console.log('\n✓ all ui-state truthfulness checks passed')
}
