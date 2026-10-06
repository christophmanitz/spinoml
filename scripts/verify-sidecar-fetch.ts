#!/usr/bin/env tsx
// Phase 77/78 — every sidecar request must go through src/sidecars/auth.ts.
//
// Checks that no file under src/ (except the wrapper itself) calls bare
// `fetch(`. The wrapper `sidecarFetch` / the torch helper `torchFetch` have a
// capital F, so a case-sensitive `\bfetch\s*\(` deliberately ignores them.
// `currentTorchUrl` may still be imported for URL display, but a file that
// both imports it AND calls bare `fetch` is caught by the rule below anyway.
//
// Usage: tsx scripts/verify-sidecar-fetch.ts
// Exit 0 = clean, exit 1 = a bypass site exists (file:line listed).

import { readdirSync, readFileSync } from 'node:fs'
import { join, relative, sep } from 'node:path'

const ROOT = process.cwd()
const SRC = join(ROOT, 'src')
const WRAPPER = 'src/sidecars/auth.ts'

// Future legitimate non-sidecar fetches can be listed here by repo-relative path.
const ALLOW_LIST: string[] = []

const BARE_FETCH = /\bfetch\s*\(/

/** Replace comments with spaces (newlines preserved) so a `fetch(` mentioned in
 *  prose does not count, and line numbers stay exact. Simple by design. */
function stripComments(text: string): string {
  let out = ''
  let i = 0
  while (i < text.length) {
    const c = text[i]
    if (c === '/' && text[i + 1] === '/') {
      while (i < text.length && text[i] !== '\n') { out += ' '; i++ }
    } else if (c === '/' && text[i + 1] === '*') {
      out += '  '
      i += 2
      while (i < text.length && !(text[i] === '*' && text[i + 1] === '/')) {
        out += text[i] === '\n' ? '\n' : ' '
        i++
      }
      if (i < text.length) { out += '  '; i += 2 }
    } else {
      out += c
      i++
    }
  }
  return out
}

function listSourceFiles(dir: string): string[] {
  const out: string[] = []
  for (const ent of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, ent.name)
    if (ent.isDirectory()) out.push(...listSourceFiles(p))
    else if (ent.isFile() && (ent.name.endsWith('.ts') || ent.name.endsWith('.tsx'))) out.push(p)
  }
  return out.sort()
}

let failures = 0
const files = listSourceFiles(SRC)
console.log('phase 77/78: sidecar-fetch wrapper guard')
console.log(`  scanned ${files.length} src file(s)`)

for (const abs of files) {
  const rel = relative(ROOT, abs).split(sep).join('/')
  if (rel === WRAPPER) continue
  if (ALLOW_LIST.includes(rel)) continue
  const stripped = stripComments(readFileSync(abs, 'utf8'))
  const lines = stripped.split('\n')
  for (let i = 0; i < lines.length; i++) {
    if (BARE_FETCH.test(lines[i])) {
      failures++
      console.log(`  ✗ ${rel}:${i + 1} — bare fetch() bypasses sidecarFetch`)
    }
  }
}

if (failures > 0) {
  console.error(`\n${failures} sidecar-fetch bypass(es)`)
  process.exit(1)
}
console.log('\n✓ no src/ file bypasses the sidecarFetch wrapper')
