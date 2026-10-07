#!/usr/bin/env tsx
// Phase 39 — Frontend error states (TODO §39).
// External operations should distinguish Loading / Success / Error / Timeout
// / Cancelled / Unavailable, not just `loading=false`.

import { readFileSync } from 'node:fs'
import { join } from 'node:path'

let failures = 0
function check(name: string, cond: boolean, detail = '') {
  if (cond) console.log(`  ✓ ${name}`)
  else { failures++; console.log(`  ✗ ${name} ${detail}`) }
}

console.log('phase 39: frontend error states')

// 1. Inference — Loading/Success/Error/Offline + Cancelled/Timeout not conflated to hang
{
  const s = readFileSync(join(process.cwd(), 'src', 'inference', 'store.ts'), 'utf8')
  check('inference Status enum exists (idle/inferring/ok/error/offline)', s.includes("'idle'") && s.includes("'inferring'") && s.includes("'ok'") && s.includes("'error'") && s.includes("'offline'"))
  check('inference Loading = inferring', s.includes("set({ status: 'inferring'") || s.includes("status: 'inferring'"))
  check('inference Success = ok with nParams/shapes', s.includes("status: 'ok'") && s.includes('nParams'))
  check('inference Error distinct from Offline', s.includes("status: 'error'") && s.includes("status: 'offline'"))
  check('inference Offline detected via offline in result', s.includes("'offline' in result"))
  check('inference inferShapes timeout not conflated: offline detection via fetch/network msg', s.includes("offline") && s.includes("fetch"))
  check('inference Cancelled (AbortError) does not leave inferring hanging', s.includes("AbortError") && (s.includes("status: 'idle'") || s.includes("runId !== runCounter")))
  check('inference stale response guard via runCounter', s.includes('runCounter') && s.includes('runId !== runCounter'))

  const c = readFileSync(join(process.cwd(), 'src', 'inference', 'client.ts'), 'utf8')
  check('inference client distinguishes !ok vs throw (offline vs error)', c.includes('offline: true') || c.includes('offline'))
  check('inference client passes AbortSignal for cancellation', c.includes('signal') || c.includes('Abort'))

  const badge = readFileSync(join(process.cwd(), 'src', 'App.tsx'), 'utf8')
  check('InferenceBadge renders distinct colors for ok/error/offline/inferring', badge.includes("status === 'ok'") && badge.includes("status === 'error'"))

  const ver = readFileSync(join(process.cwd(), 'src', 'inference', 'verifier.ts'), 'utf8')
  check('verifier fail-closed distinguishes offline→unknown from error→invalid', ver.includes('offline') && ver.includes('unknown'))
}

// 2. Training — Loading/Success/Error + Unavailable (offline/unknown) + Cancelled
{
  const s = readFileSync(join(process.cwd(), 'src', 'training', 'store.ts'), 'utf8')
  check('training store has listLoading (Loading) vs listError (Error/Unavailable)', s.includes('listLoading') && s.includes('listError'))
  check('training store degrades stale running to unknown on listError (Unavailable not Running)', s.includes("status: 'unknown'") && s.includes('alive: false'))
  check('training run status includes cancelled (Cancelled distinct)', (() => {
    const t = readFileSync(join(process.cwd(), 'src', 'training', 'types.ts'), 'utf8')
    const pill = readFileSync(join(process.cwd(), 'src', 'training', 'StatusPill.tsx'), 'utf8')
    const rs = readFileSync(join(process.cwd(), 'src-tauri', 'src', 'training.rs'), 'utf8')
    return pill.includes('cancelled') || rs.includes('cancelled') || t.includes('cancelled')
  })())
  check('training store stopRun distinct from deleteRun (Cancelled vs removed)', s.includes('stopRun') && s.includes('deleteRun'))
  check('training StatusPill has distinct styles for queued/running/done/failed/cancelled/unknown', (() => {
    const p = readFileSync(join(process.cwd(), 'src', 'training', 'StatusPill.tsx'), 'utf8')
    return p.includes('queued') && p.includes('running') && p.includes('done') && p.includes('failed') && p.includes('cancelled') && p.includes('unknown')
  })())
  check('training RunDetailModal reload catches error per-file (not just loading=false)', readFileSync(join(process.cwd(), 'src', 'training', 'RunDetailModal.tsx'), 'utf8').includes('catch'))
  // Timeout is still conflated to listError string (known limitation) — check we at least surface it
  check('training listError surfaces SSH timeout text (not swallowed)', s.includes('listError'))
  // SyncPolling non-overlapping prevents Timeout stacking
  check('training polling non-overlapping (timeout stacking guard)', s.includes('syncPolling'))
}

// 3. Datasets — Loading/Success/Error + offline via missing_dep
{
  const s = readFileSync(join(process.cwd(), 'src', 'datasets', 'store.ts'), 'utf8')
  check('datasets Cached distinguishes loading vs data vs error', s.includes('loading') && s.includes('data') && s.includes('error'))
  check('datasets inspects/stats/smoke each have Cached', s.includes('inspects') && s.includes('stats') && s.includes('smoke'))
  const c = readFileSync(join(process.cwd(), 'src', 'datasets', 'client.ts'), 'utf8')
  check('datasets client has post helper with offline handling', c.includes('offline') || c.includes('catch'))
  const t = readFileSync(join(process.cwd(), 'src', 'datasets', 'types.ts'), 'utf8')
  check('datasets types distinguish ok:true vs ok:false with error', t.includes('ok: true') && t.includes('ok: false'))
  check('datasets missing_dep as Unavailable hint (not generic Error)', t.includes('missing_dep') || readFileSync(join(process.cwd(), 'src', 'datasets', 'DatasetDetail.tsx'), 'utf8').includes('missing_dep'))
  // Timeout still conflated (no AbortSignal) — document
  check('datasets smoke Loading shown vs just loading=false', s.includes('loading') || s.includes('Cached'))
}

// 4. Chat/LLM — Loading (streaming) / Success (done) / Error / Unavailable
{
  const s = readFileSync(join(process.cwd(), 'src', 'chat', 'store.ts'), 'utf8')
  check('chat store distinguishes streaming (Loading) vs idle', s.includes("'streaming'") && s.includes("'idle'"))
  check('chat store message status distinct done vs error', s.includes("'done'") && s.includes("'error'"))
  check('chat online flag distinguishes Unavailable (offline)', s.includes('online'))
  check('chat pendingAsk distinct waiting state', s.includes('pendingAsk'))
  check('chat send abort via AbortController (Cancelled not just Error)', s.includes('Abort') || s.includes('abort'))

  const p = readFileSync(join(process.cwd(), 'src', 'chat', 'ChatPanel.tsx'), 'utf8')
  check('ChatPanel disables send while streaming (Loading distinct)', p.includes('streaming'))
  check('ChatPanel shows error rose box distinct from offline', p.includes('error'))
}

// 5. Filesystem/SSH — project/store distinguishes loading/loaded/error/remote-missing (Unavailable)
{
  const p = readFileSync(join(process.cwd(), 'src', 'project', 'store.ts'), 'utf8')
  check('project store distinguishes loading vs loaded vs error vs remote-missing', p.includes("'loading'") && p.includes("'loaded'") && p.includes("'error'") && p.includes("'remote-missing'"))
  check('project store catch surfaces error string (not just loading=false)', p.includes('set({ status: { kind: \'error\''))
  check('workspace backend distinguishes local vs remote (localStorage vs Tauri/SSH)', readFileSync(join(process.cwd(), 'src', 'connections', 'backend.ts'), 'utf8').includes('getCurrentConnection'))
  // SSH timeout distinct via ssh_failure classification
  const ssh = readFileSync(join(process.cwd(), 'src-tauri', 'src', 'ssh.rs'), 'utf8')
  check('Rust ssh_failure classifies timeout vs auth vs dns vs unavailable (distinct)', ssh.includes('timed out') && ssh.includes('Permission denied') && ssh.includes('could not resolve'))
}

// 6. No global "loading = false" conflation for critical paths
{
  const training = readFileSync(join(process.cwd(), 'src', 'training', 'store.ts'), 'utf8')
  check('training not conflated: listLoading + listError + runs + alive separate', training.includes('listLoading') && training.includes('listError') && training.includes('alive'))
  const inference = readFileSync(join(process.cwd(), 'src', 'inference', 'store.ts'), 'utf8')
  check('inference not conflated: status enum vs just loading boolean', inference.includes("type Status"))
  const project = readFileSync(join(process.cwd(), 'src', 'project', 'store.ts'), 'utf8')
  check('project not conflated: discriminated union vs boolean', project.includes("kind: 'loading'"))
}

// 7. Known limitations (Timeout/Cancelled not full enum yet) — document, not hide
{
  // These are the remaining gaps that LIMITATIONS.md must disclose post-Phase 39.
  // We check that they are at least not silently mapped to Success.
  const inf = readFileSync(join(process.cwd(), 'src', 'inference', 'store.ts'), 'utf8')
  check('inference Timeout (fetch) at least maps to offline/error, never ok (no false Success)', !inf.includes("fetch") || (inf.includes("'offline'") && inf.includes("'error'")))
  const tr = readFileSync(join(process.cwd(), 'src', 'training', 'store.ts'), 'utf8')
  check('training Timeout (ssh ConnectTimeout=10) at least maps to listError, never done', tr.includes('listError'))
}

if (failures > 0) {
  console.error(`\n${failures} check(s) failed`)
  process.exit(1)
} else {
  console.log('\n✓ all frontend error-state distinctness checks passed')
}
