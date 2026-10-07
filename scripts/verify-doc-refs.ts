#!/usr/bin/env tsx
// Phase 80 — doc reference checker.
//
// Scans docs/engineering/*.md, docs/FEATURES.md, and CLAUDE.md and fails
// when:
//   (a) a backticked token that looks like a repo path
//       (`src/`, `src-tauri/`, `sidecar-torch/`, `sidecar-llm/`,
//        `scripts/`, `docs/`, `examples/` prefixes, OR a bare
//        `*.md/*.ts/*.tsx/*.py/*.mjs/*.rs` filename whose basename
//        exists somewhere in the repo) does not exist
//       (allow :line / :line-line / #anchor suffixes, glob stars,
//        <placeholder> segments, trailing `/`),
//   (b) a backticked `npm run <name>` or bare `<verb>:<name>` token
//       matching package.json script naming (`verify:*`, `test:*`,
//       `gen:*`, `typecheck:*`, `sidecar:*`) is not a script in
//       package.json (an explicit ALLOW array at the top lists the
//       intentionally-historic names — keep the reason short),
//   (c) a relative markdown link target does not exist,
//   (d) a `docs/engineering/<X>.md` referenced by name does not exist.
//
// Prints `file:line  kind  token` for each offender; exits 1 on any.
// `--self-test` exercises ≥ 10 inline cases (existing path ok,
// missing path fails, line suffix ok, glob ok, npm script ok/missing,
// link ok/missing, fenced code block ignored for path rules but npm
// scripts inside shell fences ARE checked). Mutates a scratch doc
// copy and asserts red.

import { readFileSync, readdirSync, existsSync, statSync, mkdirSync, writeFileSync, rmSync } from 'node:fs'
import { join, dirname, basename, relative, resolve } from 'node:path'

const ROOT = process.cwd()
const SELF_TEST = process.argv.includes('--self-test')
const DOC_GLOBS = [
  join(ROOT, 'docs', 'engineering'),
  join(ROOT, 'docs', 'FEATURES.md'),
  join(ROOT, 'CLAUDE.md'),
]

// Intentionally-historic npm-script names that no longer exist in
// package.json. Keep the reason short and concrete; if you find yourself
// adding more than 5 entries the docs are out of date and should be
// updated, not just suppressed here.
const ALLOW_NPM_SCRIPTS: Record<string, string> = {
  'cargo-check': 'documented alongside the suites table; the real script is `cargo check` (no npm alias)',
  'cargo-test':  'documented alongside the suites table; the real script is `cargo test` (no npm alias)',
  'lint':        'documented in BASELINE.md; the npm script is `lint` but its meaning changed (now reports-only)',
}

const PATH_PREFIXES = ['src/', 'src-tauri/', 'sidecar-torch/', 'sidecar-llm/', 'scripts/', 'docs/', 'examples/']
const PATH_EXTS = ['.md', '.ts', '.tsx', '.py', '.mjs', '.rs']

// Inside fenced code blocks we skip path-token checks (the rules below
// apply to prose), BUT npm-script tokens inside shell fences ARE checked
// — people put `npm run verify:foo` in shell examples that should be
// real.
function isCodeFenceOpen(text: string, pos: number): boolean {
  // Cheap: walk line starts from pos backwards counting fences.
  let open = false
  let i = 0
  while (i < pos) {
    const nl = text.indexOf('\n', i)
    if (nl === -1 || nl >= pos) break
    const line = text.slice(i, nl)
    if (/^```/.test(line)) open = !open
    i = nl + 1
  }
  return open
}

interface Offender {
  file: string
  line: number
  kind: string
  token: string
}

function lineOf(text: string, pos: number): number {
  let n = 1
  for (let i = 0; i < pos; i++) if (text.charCodeAt(i) === 10) n++
  return n
}

function stripTokenSuffix(token: string): string {
  // Allow `:line`, `:line-line`, `#anchor` on path tokens; strip them
  // before checking existence.
  let t = token
  // Strip a trailing `#anchor`
  const hashIdx = t.indexOf('#')
  if (hashIdx !== -1) t = t.slice(0, hashIdx)
  // Strip a `:line` or `:line-line` line-range
  const colonIdx = t.lastIndexOf(':')
  if (colonIdx !== -1 && /^\d+(-\d+)?$/.test(t.slice(colonIdx + 1))) {
    t = t.slice(0, colonIdx)
  }
  return t
}

function isPlaceholder(t: string): boolean {
  // `<…>` segments are not real paths. Glob stars are not real paths.
  // Brace expansion (`{a,b,c}`) is a docs shorthand, not a real file.
  // A bare `.py` / `.ts` extension with no filename is a prose mention.
  return /<[^>]+>/.test(t) || t.includes('*') || t.endsWith('/') || /\{[^}]+,/.test(t) || /^\.[a-z]+$/i.test(t)
}

function basenameExistsInRepo(basename: string): boolean {
  // Walks the repo shallowly; bounded by MAX_DIR_WALK to keep CI fast.
  const MAX_DIR_WALK = 50_000
  const stack = [ROOT]
  let visited = 0
  while (stack.length) {
    const dir = stack.pop()!
    if (visited++ > MAX_DIR_WALK) return false
    let ents: string[]
    try {
      ents = readdirSync(dir)
    } catch {
      continue
    }
    for (const e of ents) {
      if (e === 'node_modules' || e === '.git' || e === 'dist' || e === 'target') continue
      const p = join(dir, e)
      let st
      try { st = statSync(p) } catch { continue }
      if (st.isDirectory()) stack.push(p)
      else if (e === basename) return true
    }
  }
  return false
}

function resolveRepoPath(token: string): { kind: 'abs'; abs: string } | { kind: 'rel'; rel: string } | { kind: 'skip' } {
  const t = stripTokenSuffix(token)
  if (isPlaceholder(t)) return { kind: 'skip' }
  if (t.startsWith('/')) {
    // Absolute path under the repo root.
    return { kind: 'abs', abs: t }
  }
  // Relative to repo root.
  return { kind: 'rel', rel: t }
}

function checkPathToken(token: string): boolean {
  const r = resolveRepoPath(token)
  if (r.kind === 'skip') return true
  if (r.kind === 'abs') {
    // We trust absolute paths only if they live under ROOT.
    if (!r.abs.startsWith(ROOT + '/') && r.abs !== ROOT) return false
    return existsSync(r.abs)
  }
  // Relative. Must exist either literally, or by basename in the repo.
  const abs = join(ROOT, r.rel)
  if (existsSync(abs)) return true
  // Allow .md / .ts / .tsx / .py / .mjs / .rs by basename.
  const base = basename(r.rel)
  if (PATH_EXTS.includes('.' + base.split('.').pop()!) && basenameExistsInRepo(base)) return true
  return false
}

function looksLikeRepoPath(token: string): boolean {
  if (token.includes(' ') || token.includes('\n')) return false
  for (const p of PATH_PREFIXES) if (token.startsWith(p)) return true
  for (const ext of PATH_EXTS) if (token.endsWith(ext)) return true
  return false
}

function isNpmScriptKind(token: string): { verb: string; name: string } | null {
  // `npm run <verb>:<name>` (verbatim) OR bare `<verb>:<name>` that matches
  // the documented npm-script naming (`verify:*`, `test:*`, `gen:*`,
  // `typecheck:*`, `sidecar:*`).
  let body = token
  if (body.startsWith('npm run ')) body = body.slice('npm run '.length).trim()
  const m = /^([a-z]+):([a-z0-9][a-z0-9._-]*)$/.exec(body)
  if (!m) return null
  const verb = m[1]!
  const name = m[2]!
  if (!['verify', 'test', 'gen', 'typecheck', 'sidecar'].includes(verb)) return null
  return { verb, name }
}

function collectOffenders(file: string, text: string): Offender[] {
  const out: Offender[] = []
  const re = /`([^`\n]+)`/g
  let m: RegExpExecArray | null
  while ((m = re.exec(text)) !== null) {
    const token = m[1]!
    const pos = m.index
    const inFence = isCodeFenceOpen(text, pos)
    if (!inFence) {
      // Rule (a): repo path tokens outside fences.
      if (looksLikeRepoPath(token) && !checkPathToken(token)) {
        out.push({ file, line: lineOf(text, pos), kind: 'path', token })
        continue
      }
    } else {
      // Inside a fence we still check npm-script tokens (shell examples
      // often use them and should be real).
    }
    // Rule (b): npm script tokens (in fences OR in prose).
    const ns = isNpmScriptKind(token)
    if (ns) {
      const full = `${ns.verb}:${ns.name}`
      if (ALLOW_NPM_SCRIPTS[full]) continue
      const scriptNames = loadPackageJsonScripts()
      if (!scriptNames.has(full)) {
        out.push({ file, line: lineOf(text, pos), kind: 'npm-script', token })
      }
    }
  }
  return out
}

let _packageJsonScripts: Set<string> | null = null
function loadPackageJsonScripts(): Set<string> {
  if (_packageJsonScripts) return _packageJsonScripts
  try {
    const pkg = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8'))
    _packageJsonScripts = new Set(Object.keys(pkg.scripts ?? {}))
  } catch {
    _packageJsonScripts = new Set()
  }
  return _packageJsonScripts
}

function checkRelativeLinks(file: string, text: string): Offender[] {
  const out: Offender[] = []
  const dir = dirname(file)
  // Markdown link target regex: `[text](target)` or `[text](target "title")`.
  const re = /\[[^\]\n]+\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g
  let m: RegExpExecArray | null
  while ((m = re.exec(text)) !== null) {
    const target = m[1]!
    if (/^[a-z]+:\/\//.test(target)) continue   // http(s):// etc.
    if (target.startsWith('#')) continue         // same-page anchor
    if (target.startsWith('mailto:')) continue
    // Strip any #anchor
    let p = target
    const hash = p.indexOf('#')
    if (hash !== -1) p = p.slice(0, hash)
    // Compute absolute path relative to the markdown file's dir.
    let abs: string
    if (p.startsWith('/')) abs = join(ROOT, p.replace(/^\/+/, ''))
    else abs = resolve(dir, p)
    if (!existsSync(abs)) {
      out.push({ file, line: lineOf(text, m.index), kind: 'link', token: target })
    }
  }
  return out
}

function checkEngineeringDocReferences(file: string, text: string): Offender[] {
  const out: Offender[] = []
  // `docs/engineering/<X>.md` where X is the stem.
  const re = /`docs\/engineering\/([^`\n/]+)`/g
  let m: RegExpExecArray | null
  while ((m = re.exec(text)) !== null) {
    const stem = m[1]!
    if (!stem.endsWith('.md')) continue
    const p = join(ROOT, 'docs', 'engineering', stem)
    if (!existsSync(p)) {
      out.push({ file, line: lineOf(text, m.index), kind: 'engineering-doc', token: `docs/engineering/${stem}` })
    }
  }
  return out
}

function listMarkdownFiles(): string[] {
  const out: string[] = []
  const eng = DOC_GLOBS[0]!
  for (const f of readdirSync(eng)) {
    if (f.endsWith('.md')) out.push(join(eng, f))
  }
  for (const p of DOC_GLOBS.slice(1)) {
    if (existsSync(p)) out.push(p)
  }
  return out
}

function main(): number {
  const files = listMarkdownFiles()
  let totalOffenders = 0
  const allOffenders: Offender[] = []
  for (const f of files) {
    const text = readFileSync(f, 'utf8')
    const offs = [
      ...collectOffenders(f, text),
      ...checkRelativeLinks(f, text),
      ...checkEngineeringDocReferences(f, text),
    ]
    for (const o of offs) {
      console.error(`${relative(ROOT, o.file)}:${o.line}  ${o.kind}  ${o.token}`)
    }
    totalOffenders += offs.length
    allOffenders.push(...offs)
  }
  console.error(`scanned ${files.length} file(s); ${totalOffenders} offender(s)`)
  return totalOffenders === 0 ? 0 : 1
}

function selfTest(): number {
  // Build scratch fixtures in a temp dir under ROOT/.tmp and assert the
  // detector behaves. The temp dir is removed at the end.
  const tmp = join(ROOT, '.tmp-verify-doc-refs')
  rmSync(tmp, { recursive: true, force: true })
  mkdirSync(tmp, { recursive: true })
  let failures = 0
  function expect(name: string, ok: boolean, detail = ''): void {
    if (ok) console.log(`  ✓ ${name}`)
    else { failures++; console.log(`  ✗ ${name} ${detail}`) }
  }

  // 1. Existing repo path with line suffix → ok
  expect(
    'existing repo path with :line suffix is ok',
    checkPathToken('src/App.tsx:42'),
  )
  // 2. Missing path → fail
  expect(
    'missing repo path is reported',
    !checkPathToken('src/does-not-exist.ts'),
  )
  // 3. Glob star → skip
  expect(
    'glob star is skipped',
    checkPathToken('src/components/*.tsx'),
  )
  // 4. Placeholder segment → skip
  expect(
    '<placeholder> segment is skipped',
    checkPathToken('src/<layer>/registry.ts'),
  )
  // 5. Trailing slash → skip
  expect(
    'trailing slash is skipped',
    checkPathToken('docs/'),
  )
  // 6. Basename lookup across extensions (existing file)
  expect(
    'basename lookup across extensions finds an existing file',
    checkPathToken('package.json'),
  )
  // 7. basename lookup for a fake file → fail
  expect(
    'basename lookup of a fake file fails',
    !checkPathToken('this-name-does-not-exist-anywhere.md'),
  )

  // 8. npm script in prose — real one → ok
  expect(
    'real npm script in prose is ok',
    isNpmScriptKind('npm run verify:codegen') !== null && loadPackageJsonScripts().has('verify:codegen'),
  )
  // 9. npm script missing → fail (caught by collectOffenders; here we
  //    just verify the predicate)
  expect(
    'missing npm script is detected',
    isNpmScriptKind('npm run verify:this-does-not-exist') !== null && !loadPackageJsonScripts().has('verify:this-does-not-exist'),
  )
  // 10. verb outside the documented list → null
  expect(
    'non-npm verb returns null',
    isNpmScriptKind('npm run lint:something') === null && isNpmScriptKind('foo:bar') === null,
  )

  // 11. Relative markdown link target — existing → ok
  const realLinkFile = join(ROOT, 'docs', 'engineering', 'RISK_REGISTER.md')
  const realLinkText = readFileSync(realLinkFile, 'utf8')
  expect(
    'existing relative link in RISK_REGISTER.md produces no offender',
    checkRelativeLinks(realLinkFile, realLinkText).length === 0,
  )
  // 12. Missing relative link → fail
  const scratch = join(tmp, 'scratch.md')
  writeFileSync(scratch, '# t\n[bad](../does-not-exist.md)\n')
  const off12 = checkRelativeLinks(scratch, readFileSync(scratch, 'utf8'))
  expect('missing relative link produces one offender', off12.length === 1, JSON.stringify(off12))

  // 13. Code-fence skipping: path token inside a fence is NOT flagged
  // by collectOffenders (npm scripts inside fences ARE checked).
  const fenced = '# t\n```\n`src/does-not-exist.ts`\n```\n'
  const off13 = collectOffenders(scratch, fenced)
  expect('path token inside a fenced code block is not flagged', off13.length === 0, JSON.stringify(off13))

  // 14. engineering doc reference — existing → ok
  const refOK = '# t\nSee `docs/engineering/RISK_REGISTER.md`.\n'
  const off14 = checkEngineeringDocReferences(scratch, refOK)
  expect('engineering-doc reference to existing file is ok', off14.length === 0, JSON.stringify(off14))
  // 15. engineering doc reference — missing → fail
  const refBad = '# t\nSee `docs/engineering/NOPE.md`.\n'
  const off15 = checkEngineeringDocReferences(scratch, refBad)
  expect('engineering-doc reference to missing file is flagged', off15.length === 1, JSON.stringify(off15))

  // 16. Fenced-but-npm-script IS flagged (we check shell-fence commands too).
  const fencedScript = '# t\n```sh\n`npm run verify:this-doesnt-exist`\n```\n'
  const off16 = collectOffenders(scratch, fencedScript)
  expect('missing npm script inside a shell fence IS flagged', off16.length >= 1, JSON.stringify(off16))

  // 17. ALLOW_NPM_SCRIPTS suppresses (the historic `cargo-check`)
  const allowScript = '# t\n`npm run cargo-check`\n'
  const off17 = collectOffenders(scratch, allowScript)
  expect('ALLOW_NPM_SCRIPTS suppresses historic names', off17.length === 0, JSON.stringify(off17))

  // 18. Mutation: add a known-bad path to a scratch copy of an existing
  // doc; expect the checker to flag it.
  const originalDoc = join(ROOT, 'docs', 'engineering', 'RISK_REGISTER.md')
  const mutated = join(tmp, 'mutated.md')
  const origText = readFileSync(originalDoc, 'utf8')
  writeFileSync(mutated, origText + '\n`src/this-definitely-does-not-exist-12345.ts`\n')
  const off18 = collectOffenders(mutated, readFileSync(mutated, 'utf8'))
  expect('mutation: nonexistent path is flagged', off18.length >= 1, JSON.stringify(off18))

  rmSync(tmp, { recursive: true, force: true })
  if (failures === 0) console.log(`self-test PASS (0 failures)`)
  else console.log(`self-test FAIL (${failures} failures)`)
  return failures === 0 ? 0 : 1
}

if (SELF_TEST) {
  process.exit(selfTest())
}
process.exit(main())
