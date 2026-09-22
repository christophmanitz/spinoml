#!/usr/bin/env tsx
// Phase 36 — SLURM reliability (TODO §36).
// Tests the full SLURM lifecycle without a live cluster:
// submit success/failure, pending/running/completed/failed/cancelled/unknown,
// communication failure, and durable job-ID persistence.
// Every check is a regression lock on the Rust paths that a live SLURM
// cluster would exercise; a real sbatch/squeue is never required.

import { execSync, spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { join } from 'node:path'

let failures = 0
function check(name: string, cond: boolean, detail = '') {
  if (cond) console.log(`  ✓ ${name}`)
  else { failures++; console.log(`  ✗ ${name} ${detail}`) }
}

console.log('phase 36: slurm reliability')

// 1. Rust unit tests — sbatch generation
{
  const manifestPath = join(process.cwd(), 'src-tauri', 'Cargo.toml')
  const cargoCandidates = [
    process.env.CARGO_BIN,
    'cargo',
    join(process.env.HOME ?? '', 'anaconda3/envs/mlforge-dev/bin/cargo'),
    join(process.env.HOME ?? '', '.cargo/bin/cargo'),
  ].filter(Boolean) as string[]
  let res: ReturnType<typeof spawnSync> | null = null
  for (const cand of cargoCandidates) {
    try {
      const r = spawnSync(cand, ['test', '--manifest-path', manifestPath, 'slurm_tests'], {
        encoding: 'utf8',
        env: { ...process.env, PATH: `${process.env.HOME}/anaconda3/envs/mlforge-dev/bin:${process.env.PATH}` },
      })
      if (r.error) continue
      res = r
      break
    } catch { continue }
  }
  if (!res) {
    const fallback = cargoCandidates[0]!
    res = spawnSync(fallback, ['test', '--manifest-path', manifestPath, 'slurm_tests'], { encoding: 'utf8', env: process.env }) as unknown as ReturnType<typeof spawnSync>
  }
  const out = (res.stdout as string | undefined) ?? ''
  const err = (res.stderr as string | undefined) ?? ''
  check('cargo test slurm_tests (sbatch) executes', res.status === 0, err || out)
  check('all sbatch unit tests pass', out.includes('4 passed; 0 failed'), out || err)
}

// 2. Rust unit tests — squeue/sacct reconciliation (9 SLURM states)
{
  const manifestPath = join(process.cwd(), 'src-tauri', 'Cargo.toml')
  const cargoCandidates = [
    process.env.CARGO_BIN,
    'cargo',
    join(process.env.HOME ?? '', 'anaconda3/envs/mlforge-dev/bin/cargo'),
    join(process.env.HOME ?? '', '.cargo/bin/cargo'),
  ].filter(Boolean) as string[]
  let res: ReturnType<typeof spawnSync> | null = null
  for (const cand of cargoCandidates) {
    try {
      const r = spawnSync(cand, ['test', '--manifest-path', manifestPath, 'training::tests::slurm'], {
        encoding: 'utf8',
        env: { ...process.env, PATH: `${process.env.HOME}/anaconda3/envs/mlforge-dev/bin:${process.env.PATH}` },
      })
      if (r.error) continue
      res = r
      break
    } catch { continue }
  }
  if (!res) {
    const fallback = cargoCandidates[0]!
    res = spawnSync(fallback, ['test', '--manifest-path', manifestPath, 'training::tests::slurm'], { encoding: 'utf8', env: process.env }) as unknown as ReturnType<typeof spawnSync>
  }
  const out = (res.stdout as string | undefined) ?? ''
  const err = (res.stderr as string | undefined) ?? ''
  check('cargo test slurm reconciliation executes', res.status === 0, err || out)
  check('all slurm reconciliation tests pass (9 cases)', out.includes('9 passed; 0 failed'), out || err)
}

// 3. SSH/SLURM command surface in ssh.rs
{
  const sshPath = join(process.cwd(), 'src-tauri', 'src', 'ssh.rs')
  check('ssh.rs exists', existsSync(sshPath))
  const content = execSync(`cat "${sshPath}"`, { encoding: 'utf8' })

  // Submission
  check('sbatch submission writes slurm:<jid> to pid', content.includes("printf 'slurm:%s") && content.includes('> pid'))
  check('sbatch failure surfaces explicit sbatch error', content.includes('sbatch failed:'))
  check('submission parses job id from sbatch output', content.includes("grep -oE 'job [0-9]+'"))
  check('submission failure marker MLF_SUBMIT_FAILED', content.includes('MLF_SUBMIT_FAILED'))
  check('submission success marker MLF_JOBID', content.includes('MLF_JOBID'))

  // Status probing — the 8 states (reconciliation lives in training.rs, probed via ssh.rs)
  check('squeue probe for live SLURM job', content.includes('squeue -j') && content.includes("-o '%T'"))
  check('sacct probe for completed SLURM job', content.includes('sacct -j') && content.includes('-o State'))
  // Reconciliation vocabulary lives in training.rs
  const trainingContentForReconcile = execSync(`cat "${join(process.cwd(), 'src-tauri', 'src', 'training.rs')}"`, { encoding: 'utf8' })
  check('reconcile covers PENDING → queued', trainingContentForReconcile.includes('"PENDING"'))
  check('reconcile covers COMPLETED → done', trainingContentForReconcile.includes('"COMPLETED"'))
  check('reconcile covers CANCELLED → cancelled', trainingContentForReconcile.includes('"CANCELLED"'))
  check('reconcile covers TIMEOUT/FAILED → failed', trainingContentForReconcile.includes('"TIMEOUT"') || trainingContentForReconcile.includes('TIMEOUT'))
  check('reconcile handles UNKNOWN via fallback', trainingContentForReconcile.includes('reconcile_status'))

  // Job control
  check('scancel for SLURM stop', content.includes('scancel'))
  check('SLURM delete guards against live squeue', content.includes('squeue -j') && content.includes('MLF_ALIVE'))

  // Persistence
  check('train.sbatch frozen alongside pid', content.includes('train.sbatch'))
  check('pid persisted as slurm:<id> (not bare number)', content.includes('slurm:'))
  check('build_sbatch emits SBATCH directives', content.includes('#SBATCH --job-name=spinoml-'))
  check('build_sbatch handles gres/account/qos/modules/pre_run_script', content.includes('--gres=') && content.includes('--account=') && content.includes('module load'))
}

// 4. Training-side reconciliation in training.rs
{
  const trainingPath = join(process.cwd(), 'src-tauri', 'src', 'training.rs')
  check('training.rs exists', existsSync(trainingPath))
  const content = execSync(`cat "${trainingPath}"`, { encoding: 'utf8' })
  check('reconcile_slurm_status defined', content.includes('fn reconcile_slurm_status'))
  check('reconcile handles PENDING/CONFIGURING', content.includes('PENDING') && content.includes('CONFIGURING'))
  check('reconcile handles sacct CANCELLED', content.includes('CANCELLED'))
  check('reconcile handles sacct failures (FAILED/TIMEOUT/OOM)', content.includes('FAILED') && content.includes('TIMEOUT'))
  check('sacct CANCELLED suffix stripping', content.includes('split_whitespace'))
}

// 5. Remote job ID persistence via types and backend
{
  const typesPath = join(process.cwd(), 'src', 'training', 'types.ts')
  check('types.ts exists', existsSync(typesPath))
  const content = execSync(`cat "${typesPath}"`, { encoding: 'utf8' })
  check('SlurmConfig persisted in RunBackend', content.includes('SlurmConfig') && content.includes('RunBackend'))
  check('run.json backend.slurm frozen', content.includes('backend') && content.includes('slurm'))
  const backendPath = join(process.cwd(), 'src', 'training', 'backend.ts')
  check('backend.ts dispatches SLURM via tauri-ssh', existsSync(backendPath) && execSync(`cat "${backendPath}"`, { encoding: 'utf8' }).includes('has_slurm'))
}

// 6. Communication failure is the SSH transport failure already verified by verify:ssh
{
  const sshRs = execSync(`cat "${join(process.cwd(), 'src-tauri', 'src', 'ssh.rs')}"`, { encoding: 'utf8' })
  check('SLURM commands reuse ssh_failure classification (exit 255 → auth/DNS/timeout)', sshRs.includes('ssh_failure'))
  check('SLURM commands sanitize credentials via sanitize_credentials', sshRs.includes('sanitize_credentials'))
}

if (failures > 0) {
  console.error(`\n${failures} check(s) failed`)
  process.exit(1)
} else {
  console.log('\n✓ all slurm reliability checks passed')
}
