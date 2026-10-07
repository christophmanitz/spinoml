#!/usr/bin/env tsx
// Phase 35 — SSH and secret credential safety.
// Asserts that passwords, private keys, tokens, OAuth secrets, and SSH credentials
// are NEVER written to ordinary logs, experiment artifacts, or error traces.

import { spawnSync } from 'node:child_process'
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'

let failures = 0
function check(name: string, condition: boolean, detail = '') {
  if (condition) console.log(`  ✓ ${name}`)
  else { failures++; console.log(`  ✗ ${name} ${detail}`) }
}

console.log('phase 35: credential safety')

// 1. Signature patterns for sensitive credentials that must NEVER appear in logs or artifacts
const SENSITIVE_PATTERNS: { name: string; regex: RegExp }[] = [
  { name: 'Private Key header', regex: /-----BEGIN\s+(?:[A-Z0-9_-]+\s+)?PRIVATE\s+KEY/i },
  { name: 'PuTTY Private Key', regex: /PuTTY-User-Key-File/i },
  { name: 'Embedded password in URL', regex: /(?:https?|ssh|ftp|sftp):\/\/[^:\s'"]+:([^@\s'"]+)@[^/\s'"]+/i },
  { name: 'OpenAI / OpenCode secret key', regex: /\bsk-[a-zA-Z0-9_-]{24,}\b/ },
  { name: 'Anthropic API key', regex: /\bsk-ant-[a-zA-Z0-9_-]{24,}\b/ },
  { name: 'GitHub personal access token', regex: /\bgh[pousr]_[A-Za-z0-9_]{36,}\b/ },
  { name: 'GitLab personal access token', regex: /\bglpat-[A-Za-z0-9_-]{20,}\b/ },
  { name: 'HuggingFace user token', regex: /\bhf_[A-Za-z0-9]{34,}\b/ },
  { name: 'Slack token', regex: /\bxox[baprs]-[A-Za-z0-9_-]{10,}\b/ },
  { name: 'AWS secret access key pattern', regex: /\bAKIA[0-9A-Z]{16}\b/ },
]

function scanFileForSecrets(filePath: string): string[] {
  const content = readFileSync(filePath, 'utf8')
  const found: string[] = []
  for (const { name, regex } of SENSITIVE_PATTERNS) {
    if (regex.test(content)) {
      found.push(name)
    }
  }
  return found
}

function scanDirRecursive(dir: string, fileFilter?: (name: string) => boolean): string[] {
  const violations: string[] = []
  if (!existsSync(dir)) return violations

  function walk(current: string) {
    for (const entry of readdirSync(current)) {
      if (entry === 'node_modules' || entry === '.git' || entry === 'target' || entry === 'venv') {
        continue
      }
      const full = join(current, entry)
      try {
        const s = statSync(full)
        if (s.isDirectory()) {
          walk(full)
        } else if (s.isFile()) {
          if (!fileFilter || fileFilter(entry)) {
            const hits = scanFileForSecrets(full)
            if (hits.length > 0) {
              violations.push(`${full} (${hits.join(', ')})`)
            }
          }
        }
      } catch {
        // skip unreadable files
      }
    }
  }

  walk(dir)
  return violations
}

// Check 1: Experiment artifacts scan (experiments/ directory)
{
  const expDir = join(process.cwd(), 'experiments')
  const violations = scanDirRecursive(expDir)
  check('no secrets in experiment artifacts (experiments/ runs, metrics, configs)',
    violations.length === 0, violations.join('; '))
}

// Check 2: Pre-existing logs and docs scan
{
  const docsDir = join(process.cwd(), 'docs')
  const violations = scanDirRecursive(docsDir, (name) => name.endsWith('.log') || name.endsWith('.jsonl'))
  check('no secrets in documentation logs / jsonl', violations.length === 0, violations.join('; '))
}

// Check 3: Rust SSH credential sanitization integration
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
      const r = spawnSync(cand, ['test', '--manifest-path', manifestPath, 'sanitizes_credentials_in_diagnostics'], {
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
    res = spawnSync(fallback, ['test', '--manifest-path', manifestPath, 'sanitizes_credentials_in_diagnostics'], { encoding: 'utf8', env: process.env }) as unknown as ReturnType<typeof spawnSync>
  }
  const out = (res.stdout as string | undefined) ?? ''
  const err = (res.stderr as string | undefined) ?? ''
  check('Rust sanitize_credentials unit test executes and passes',
    res.status === 0 && out.includes('1 passed; 0 failed'),
    err || out || String((res as unknown as { error?: Error }).error ?? ''))
}

// Check 4: Frontend connection model does not store credentials
{
  const connStorePath = join(process.cwd(), 'src', 'connections', 'store.ts')
  const connContent = readFileSync(connStorePath, 'utf8')
  check('connection model delegates auth to ~/.ssh/config + agent (no password field)',
    !connContent.includes('password:') && !connContent.includes('privateKey:'))
  check('connection model notes explicit policy against storing secrets',
    connContent.includes('secrets are NOT stored'))
}

// Check 5: Training template environment recording does NOT dump os.environ
{
  const trainTemplatePath = join(process.cwd(), 'sidecar-torch', 'training_template.py')
  const templateContent = readFileSync(trainTemplatePath, 'utf8')
  // Find _env_info definition
  const envInfoStart = templateContent.indexOf('def _env_info(')
  check('_env_info function exists in training template', envInfoStart !== -1)
  const envInfoBlock = templateContent.slice(envInfoStart, envInfoStart + 1500)
  check('_env_info only captures whitelisted platform/torch info',
    !envInfoBlock.includes('os.environ') && !envInfoBlock.includes('environ.copy()'))
}

// Check 6: Runtime error output scrubber
{
  // Verify that error traces emitted in trainer failure handlers scrub or avoid printing raw env
  const trainPyPath = join(process.cwd(), 'sidecar-torch', 'training_template.py')
  const pyCode = readFileSync(trainPyPath, 'utf8')
  check('trainer does not log full environment on failure',
    !pyCode.includes('emit("run.failed", env=') && !pyCode.includes('traceback.format_exc(env='))
}

if (failures > 0) {
  console.error(`\n${failures} check(s) failed`)
  process.exit(1)
} else {
  console.log('\n✓ all credential safety checks passed')
}
