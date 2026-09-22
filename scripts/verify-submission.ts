#!/usr/bin/env tsx
// Phase 33 — remote submission idempotency.
// The SSH launch claim is one atomic mkdir: concurrent clients cannot both own a
// run directory; a retry only succeeds after the durable pid has been recorded.

import { spawn, spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

let failures = 0
function check(name: string, condition: boolean, detail = '') {
  if (condition) console.log(`  ✓ ${name}`)
  else { failures++; console.log(`  ✗ ${name} ${detail}`) }
}

const CLAIM = 'if ! mkdir -p "$1"; then echo MLF_CLAIM_FAILED; elif mkdir "$2" 2>/dev/null; then echo MLF_CREATED; elif [ -s "$2/pid" ]; then echo MLF_ALREADY_LAUNCHED; else echo MLF_EXISTS_INCOMPLETE; fi'

function claim(runs: string, dir: string): string {
  const result = spawnSync('sh', ['-c', CLAIM, 'claim', runs, dir], { encoding: 'utf8' })
  return result.stdout.trim()
}

function claimAsync(runs: string, dir: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn('sh', ['-c', CLAIM, 'claim', runs, dir], { stdio: ['ignore', 'pipe', 'pipe'] })
    let output = ''
    child.stdout.on('data', (chunk) => { output += chunk })
    child.on('error', reject)
    child.on('close', (code) => code === 0 ? resolve(output.trim()) : reject(new Error(`claim exit=${code}`)))
  })
}

console.log('phase 33: submission idempotency')
{
  const root = mkdtempSync(join(tmpdir(), 'spinoml-submit-'))
  const runs = join(root, 'experiments', 'runs')
  const dir = join(runs, 'one')

  check('first submission atomically claims the run', claim(runs, dir) === 'MLF_CREATED')
  check('retry before pid is an explicit incomplete error', claim(runs, dir) === 'MLF_EXISTS_INCOMPLETE')
  writeFileSync(join(dir, 'pid'), '12345\n')
  check('retry after a lost response recognizes the recorded submission',
    claim(runs, dir) === 'MLF_ALREADY_LAUNCHED')

  const concurrentDir = join(runs, 'concurrent')
  const claims = await Promise.all(Array.from({ length: 16 }, () => claimAsync(runs, concurrentDir)))
  check('exactly one concurrent client owns the submission',
    claims.filter((out) => out === 'MLF_CREATED').length === 1, JSON.stringify(claims))
  check('every other concurrent client is blocked before launch',
    claims.filter((out) => out === 'MLF_EXISTS_INCOMPLETE').length === 15, JSON.stringify(claims))

  // The marker written by the owner is what turns later retries into success.
  writeFileSync(join(concurrentDir, 'pid'), '67890\n')
  check('post-launch retry is idempotent', claim(runs, concurrentDir) === 'MLF_ALREADY_LAUNCHED')

  const marker = readFileSync(join(concurrentDir, 'pid'), 'utf8').trim()
  check('submission marker remains durable', marker === '67890')
}

console.log(failures === 0 ? '\n✓ all submission-idempotency checks passed' : `\n✗ ${failures} check(s) failed`)
process.exit(failures === 0 ? 0 : 1)
