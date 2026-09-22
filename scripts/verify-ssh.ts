#!/usr/bin/env tsx
// Phase 34 — SSH reliability and explicit error classification.
// Tests the full suite of SSH transport and remote failure classifications:
// auth failure, host unavailable (DNS / network), timeout, connection loss,
// missing remote directory, permission denied, disk full, SFTP failure,
// and general remote command failure.

import { execSync, spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { join } from 'node:path'

let failures = 0
function check(name: string, condition: boolean, detail = '') {
  if (condition) console.log(`  ✓ ${name}`)
  else { failures++; console.log(`  ✗ ${name} ${detail}`) }
}

console.log('phase 34: ssh reliability')

// 1. Rust unit tests verify the explicit ssh_failure classification and sanitization
{
  const manifestPath = join(process.cwd(), 'src-tauri', 'Cargo.toml')
  const cargoCandidates = [
    process.env.CARGO_BIN,
    'cargo',
    join(process.env.HOME ?? '', 'anaconda3/envs/mlforge-dev/bin/cargo'),
    join(process.env.HOME ?? '', '.cargo/bin/cargo'),
  ].filter(Boolean) as string[]
  let res: ReturnType<typeof spawnSync> | null = null
  let cargoCmd = cargoCandidates[0]!
  for (const cand of cargoCandidates) {
    try {
      const r = spawnSync(cand, ['test', '--manifest-path', manifestPath, 'ssh_failure_tests'], {
        encoding: 'utf8',
        env: { ...process.env, PATH: `${process.env.HOME}/anaconda3/envs/mlforge-dev/bin:${process.env.PATH}` },
      })
      if (r.error) continue
      res = r
      cargoCmd = cand
      break
    } catch { continue }
  }
  if (!res) res = spawnSync(cargoCmd, ['test', '--manifest-path', manifestPath, 'ssh_failure_tests'], { encoding: 'utf8', env: process.env }) as unknown as ReturnType<typeof spawnSync>
  const out = (res.stdout as string | undefined) ?? ''
  const err = (res.stderr as string | undefined) ?? ''
  check('cargo test ssh_failure_tests executes', res.status === 0, err || out || String((res as unknown as { error?: Error }).error ?? ''))
  check('all ssh failure unit tests pass', out.includes('2 passed; 0 failed'), out || err)
}

// 2. SSH transport options and connection multiplexing structure
{
  const sshRsPath = join(process.cwd(), 'src-tauri', 'src', 'ssh.rs')
  check('ssh.rs exists', existsSync(sshRsPath))
  const content = execSync(`cat "${sshRsPath}"`, { encoding: 'utf8' })

  // Verify explicit timeouts and keepalives (Phase 34 requirement)
  check('SSH_OPTS configures ConnectTimeout=10', content.includes('ConnectTimeout=10'))
  check('SSH_OPTS configures ServerAliveInterval=20', content.includes('ServerAliveInterval=20'))
  check('SSH_OPTS configures ServerAliveCountMax=3', content.includes('ServerAliveCountMax=3'))
  check('SSH multiplexing configures ControlMaster=auto', content.includes('ControlMaster=auto'))
  check('SSH multiplexing configures ControlPersist', content.includes('ControlPersist'))

  // Verify explicit error mapping strings
  check('explicit auth / host-key failure message',
    content.includes('SSH authentication or host-key verification failed'))
  check('explicit DNS resolution failure message',
    content.includes('SSH host could not be resolved'))
  check('explicit connection timeout message',
    content.includes('SSH connection timed out'))
  check('explicit host unavailable / connection refused message',
    content.includes('SSH host is unavailable'))
  check('explicit connection loss / reset message',
    content.includes('SSH connection was lost'))
  check('explicit missing remote path message',
    content.includes('Remote path not found'))
  check('explicit remote permission denied message',
    content.includes('Remote permission denied'))
  check('explicit remote filesystem full message',
    content.includes('Remote filesystem is full'))
  check('explicit SFTP failure message',
    content.includes('Remote file transfer (SFTP) failed'))
  check('explicit remote command failure fallback message',
    content.includes('Remote SSH command failed'))
}

if (failures > 0) {
  console.error(`\n${failures} check(s) failed`)
  process.exit(1)
} else {
  console.log('\n✓ all ssh reliability checks passed')
}
