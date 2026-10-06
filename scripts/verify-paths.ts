#!/usr/bin/env tsx
// Phase 46 — symlink-aware containment for the LLM sidecar's local FS helpers.
//
// Proves sidecar-llm/path-scope.mjs against a real temp dir:
//   A. lexical rules (relative only, no .., no ~, no NUL, lengths, spaces,
//      unicode, emoji, backslash names, empty).
//   B. symlink semantics: in-root links, prefix-confusion, outside-target
//      reads/writes, dangling links, symlink loops, link chains, root-as-
//      symlink, non-existing tails, symlink target allow-list, isInside.
//   C. loadSymlinkTargets: env, file, both, skip rules, invalid JSON,
//      bad mode, nothing configured.
//   D. main.mjs wiring: import, helper use, no leftover path.join(ws.root …)
//      on a local FS call, safeRelpath kept (still used for remote/display),
//      loadSymlinkTargets() called per-call.

import {
  chmodSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs'
import { tmpdir } from 'node:os'
import { delimiter, join } from 'node:path'
import {
  resolveInWorkspace,
  loadSymlinkTargets,
  isInside,
} from '../sidecar-llm/path-scope.mjs'

let failures = 0
function check(name: string, cond: boolean, detail = '') {
  if (cond) console.log(`  ✓ ${name}`)
  else { failures++; console.log(`  ✗ ${name}${detail ? ` — ${detail}` : ''}`) }
}

function fail(name: string, detail: string) { check(name, false, detail) }

function symlink(target: string, link: string) {
  try { symlinkSync(target, link) }
  catch (e) { throw new Error(`symlink(${target} -> ${link}) failed: ${(e as Error).message}`, { cause: e }) }
}

console.log('phase 46: node sidecar symlink-aware path containment')

async function main() {
  const workdir = mkdtempSync(join(tmpdir(), 'spinoml-paths-'))
  const root = join(workdir, 'ws')
  mkdirSync(root, { recursive: true })
  const realRoot = realpathSync(root)

  // ── A. lexical ─────────────────────────────────────────────────────────────
  console.log('\n=== A. lexical ===')

  {
    const r = await resolveInWorkspace(root, 'a/b.txt')
    check('a/b.txt ok', r.ok === true && (r as { abs: string }).abs === join(realRoot, 'a', 'b.txt'))
  }
  {
    const r = await resolveInWorkspace(root, './a/b.txt')
    check('./a/b.txt ok', r.ok === true && (r as { abs: string }).abs === join(realRoot, 'a', 'b.txt'))
  }
  {
    const r = await resolveInWorkspace(root, 'my data/x y.csv')
    check('name with spaces ok', r.ok === true && (r as { abs: string }).abs === join(realRoot, 'my data', 'x y.csv'))
  }
  {
    const r = await resolveInWorkspace(root, 'Prüfung ✓/データ.csv')
    check('unicode + ✓ ok', r.ok === true && (r as { abs: string }).abs.includes('Prüfung ✓'))
  }
  {
    const r = await resolveInWorkspace(root, '😀/file.txt')
    check('emoji ok', r.ok === true)
  }
  {
    const r = await resolveInWorkspace(root, 'a/' + 'b'.repeat(255))
    check('255-byte segment ok', r.ok === true)
  }
  {
    const r = await resolveInWorkspace(root, 'a/' + 'b'.repeat(256))
    check('256-byte segment rejected', r.ok === false && (r as { code: string }).code === 'PATH_INVALID')
  }
  {
    const r = await resolveInWorkspace(root, 'a/' + 'b'.repeat(4096))
    check('4097-char path rejected', r.ok === false && (r as { code: string }).code === 'PATH_INVALID')
  }
  {
    const r = await resolveInWorkspace(root, '')
    check('empty rejected', r.ok === false && (r as { code: string }).code === 'PATH_INVALID')
  }
  {
    const r = await resolveInWorkspace(root, '..')
    check('.. rejected', r.ok === false && (r as { code: string }).code === 'PATH_INVALID')
  }
  {
    const r = await resolveInWorkspace(root, 'a/../../x')
    check('a/../../x rejected', r.ok === false && (r as { code: string }).code === 'PATH_INVALID')
  }
  {
    const r = await resolveInWorkspace(root, '/etc/passwd')
    check('/etc/passwd rejected', r.ok === false && (r as { code: string }).code === 'PATH_INVALID')
  }
  {
    const r = await resolveInWorkspace(root, '~/x')
    check('~/x rejected', r.ok === false && (r as { code: string }).code === 'PATH_INVALID')
  }
  {
    const r = await resolveInWorkspace(root, 'a\u0000b')
    check('NUL rejected', r.ok === false && (r as { code: string }).code === 'PATH_INVALID')
  }
  {
    const r = await resolveInWorkspace(root, 'a\\b.txt')
    check('backslash name ok', r.ok === true && (r as { abs: string }).abs === join(realRoot, 'a\\b.txt'))
  }

  // ── B. symlinks ────────────────────────────────────────────────────────────
  console.log('\n=== B. symlinks ===')

  const bdir = mkdtempSync(join(tmpdir(), 'spinoml-paths-b-'))
  const ws = join(bdir, 'ws'); mkdirSync(ws)
  const wsReal = realpathSync(ws)
  const wsEvil = join(bdir, 'ws-evil'); mkdirSync(wsEvil)
  const wsEvilReal = realpathSync(wsEvil)
  const outside = join(bdir, 'outside'); mkdirSync(outside)
  const outsideReal = realpathSync(outside)
  const outside2 = join(bdir, 'outside2'); mkdirSync(outside2)
  const outside2Real = realpathSync(outside2)
  mkdirSync(join(ws, 'sub'))
  writeFileSync(join(ws, 'sub', 'x.txt'), 'hello')
  symlink(join(ws, 'sub'), join(ws, 'linkdir'))

  // B1: in-root link to in-root dir
  {
    const r = await resolveInWorkspace(ws, 'linkdir/x.txt')
    check('in-root link -> in-root dir ok', r.ok === true && (r as { abs: string }).abs === join(wsReal, 'sub', 'x.txt'))
  }

  // B2: prefix confusion (ws-evil vs ws)
  symlink(wsEvil, join(ws, 'evil'))
  {
    const r = await resolveInWorkspace(ws, 'evil/x.txt')
    check('prefix-confusion symlink rejected', r.ok === false && (r as { code: string }).code === 'PATH_SYMLINK_OUTSIDE')
  }

  // B3: link to outside rejected for read
  symlink(outside, join(ws, 'out'))
  writeFileSync(join(outside, 'data.txt'), 'x')
  {
    const r = await resolveInWorkspace(ws, 'out/data.txt')
    check('outside link rejected for read', r.ok === false && (r as { code: string }).code === 'PATH_SYMLINK_OUTSIDE')
  }

  // B4: link to outside rejected for write of a new file
  {
    const r = await resolveInWorkspace(ws, 'out/new.txt', { forWrite: true })
    check('outside link rejected for write', r.ok === false && (r as { code: string }).code === 'PATH_SYMLINK_OUTSIDE')
  }

  // B5: allowed when outside is in symlinkTargets
  {
    const r = await resolveInWorkspace(ws, 'out/new.txt', { forWrite: true, symlinkTargets: [outsideReal] })
    check('outside link allowed when in targets', r.ok === true && (r as { abs: string }).abs === join(outsideReal, 'new.txt'))
  }

  // B6: still rejected for a different outside (outside2)
  symlink(outside2, join(ws, 'out2'))
  {
    const r = await resolveInWorkspace(ws, 'out2/new.txt', { forWrite: true, symlinkTargets: [outsideReal] })
    check('different outside still rejected', r.ok === false && (r as { code: string }).code === 'PATH_SYMLINK_OUTSIDE')
  }

  // B7: dangling link to an outside path rejected for write
  symlink(join(outside, 'missing.txt'), join(ws, 'dangle'))
  {
    const r = await resolveInWorkspace(ws, 'dangle', { forWrite: true })
    check('dangling outside link rejected for write',
      r.ok === false && (r as { code: string }).code === 'PATH_SYMLINK_OUTSIDE',
      r.ok ? '' : (r as { error: string }).error)
  }

  // B8: dangling link inside the root ok
  symlink(join(ws, 'inner-missing.txt'), join(ws, 'danglein'))
  {
    const r = await resolveInWorkspace(ws, 'danglein', { forWrite: true })
    check('dangling inside-root link ok', r.ok === true && (r as { abs: string }).abs === join(wsReal, 'inner-missing.txt'))
  }

  // B9: symlink loop -> PATH_INVALID, no throw, no hang
  symlink(join(ws, 'loopB'), join(ws, 'loopA'))
  symlink(join(ws, 'loopA'), join(ws, 'loopB'))
  {
    let r
    try { r = await resolveInWorkspace(ws, 'loopA/x.txt') } catch (e) { fail('symlink loop does not throw', (e as Error).message); return }
    check('symlink loop -> PATH_INVALID', r.ok === false && (r as { code: string }).code === 'PATH_INVALID', r.ok ? 'ok=true' : (r as { error: string }).error)
  }
  {
    const r = await resolveInWorkspace(ws, 'loopA')
    check('symlink loop on bare link -> PATH_INVALID', r.ok === false && (r as { code: string }).code === 'PATH_INVALID')
  }

  // B10: link chain A->B->outside rejected
  symlink(join(ws, 'c2'), join(ws, 'c1'))
  symlink(outside, join(ws, 'c2'))
  {
    const r = await resolveInWorkspace(ws, 'c1/data.txt')
    check('link chain A->B->outside rejected', r.ok === false && (r as { code: string }).code === 'PATH_SYMLINK_OUTSIDE')
  }

  // B11: root given as a symlink to the real root works
  const realForLink = join(bdir, 'real-target')
  mkdirSync(realForLink)
  writeFileSync(join(realForLink, 'a.txt'), 'a')
  const rootLink = join(bdir, 'rootLink')
  symlink(realForLink, rootLink)
  {
    const r = await resolveInWorkspace(rootLink, 'a.txt')
    check('root-as-symlink resolves inside',
      r.ok === true && (r as { abs: string }).abs === join(realpathSync(realForLink), 'a.txt'),
      r.ok ? '' : (r as { error: string }).error)
  }

  // B12: non-existing file under existing dir for write resolves inside
  {
    const r = await resolveInWorkspace(ws, 'sub/new.txt', { forWrite: true })
    check('non-existing file under existing dir resolves inside',
      r.ok === true && (r as { abs: string }).abs === join(wsReal, 'sub', 'new.txt'))
  }

  // B13: non-existing parent chain
  {
    const r = await resolveInWorkspace(ws, 'fresh/a/b/c.txt', { forWrite: true })
    check('non-existing parent chain resolves inside',
      r.ok === true && (r as { abs: string }).abs === join(wsReal, 'fresh', 'a', 'b', 'c.txt'))
  }

  // B14: nonexistent symlink target is skipped (no crash) and outside still rejected
  {
    const nonexistent = join(bdir, 'definitely-not-here')
    const r = await resolveInWorkspace(ws, 'out/new.txt', { forWrite: true, symlinkTargets: [nonexistent] })
    check('nonexistent symlink target skipped, still rejected',
      r.ok === false && (r as { code: string }).code === 'PATH_SYMLINK_OUTSIDE')
  }

  // isInside checks
  check('isInside child inside parent', isInside(join(realRoot, 'a', 'b'), realRoot) === true)
  check('isInside parent inside parent (self)', isInside(realRoot, realRoot) === true)
  check('isInside sibling prefix not inside', isInside(wsEvilReal, wsReal) === false)
  check('isInside with ..foo segment is inside', isInside(join(realRoot, '..foo'), realRoot) === true)

  // ── C. loadSymlinkTargets ──────────────────────────────────────────────────
  console.log('\n=== C. loadSymlinkTargets ===')
  const noHome = join(bdir, 'no-such-home')

  // C1: env only
  {
    const r = loadSymlinkTargets({ env: { SPINOML_SYMLINK_TARGETS: `${outsideReal}${delimiter}${outside2Real}` }, homedir: noHome })
    check('env only -> source env', r.source === 'env' && r.targets.includes(outsideReal) && r.targets.includes(outside2Real),
      `got source=${r.source} targets=${JSON.stringify(r.targets)}`)
  }

  // C2: env with relative + / + duplicate + empty entries: skipped + dedup + loadError
  {
    const raw = `${outsideReal}${delimiter}rel${delimiter}/${delimiter}${delimiter}${outsideReal}`
    const r = loadSymlinkTargets({ env: { SPINOML_SYMLINK_TARGETS: raw }, homedir: noHome })
    check('env dedupe + relative + / skipped',
      r.targets.length === 1 && r.targets[0] === outsideReal && r.loadError != null && /rel/.test(r.loadError) && /filesystem root/.test(r.loadError),
      `targets=${JSON.stringify(r.targets)} loadError=${r.loadError}`)
  }

  // C3: file only (valid 0600)
  {
    const xdg = mkdtempSync(join(tmpdir(), 'spinoml-xdg-'))
    const dir = join(xdg, 'spinoml'); mkdirSync(dir)
    const file = join(dir, 'scope.json')
    writeFileSync(file, JSON.stringify({ version: 1, symlink_targets: [outside2Real] }))
    chmodSync(file, 0o600)
    const r = loadSymlinkTargets({ env: { XDG_RUNTIME_DIR: xdg }, homedir: noHome })
    check('file only -> source file',
      r.source === 'file' && r.targets.length === 1 && r.targets[0] === outside2Real && r.loadError === null,
      `source=${r.source} targets=${JSON.stringify(r.targets)} loadError=${r.loadError}`)
  }

  // C4: both env and file
  {
    const xdg = mkdtempSync(join(tmpdir(), 'spinoml-xdg-'))
    const dir = join(xdg, 'spinoml'); mkdirSync(dir)
    const file = join(dir, 'scope.json')
    writeFileSync(file, JSON.stringify({ version: 1, symlink_targets: [outside2Real] }))
    chmodSync(file, 0o600)
    const r = loadSymlinkTargets({ env: { SPINOML_SYMLINK_TARGETS: outsideReal, XDG_RUNTIME_DIR: xdg }, homedir: noHome })
    check('both env + file -> source both', r.source === 'both' && r.targets.includes(outsideReal) && r.targets.includes(outside2Real) && r.loadError === null,
      `source=${r.source} targets=${JSON.stringify(r.targets)} loadError=${r.loadError}`)
  }

  // C5: invalid JSON -> none + loadError (no env)
  {
    const xdg = mkdtempSync(join(tmpdir(), 'spinoml-xdg-'))
    const dir = join(xdg, 'spinoml'); mkdirSync(dir)
    const file = join(dir, 'scope.json')
    writeFileSync(file, '{not json')
    chmodSync(file, 0o600)
    const r = loadSymlinkTargets({ env: { XDG_RUNTIME_DIR: xdg }, homedir: noHome })
    check('invalid JSON -> none + loadError', r.source === 'none' && r.targets.length === 0 && r.loadError != null && /JSON/i.test(r.loadError),
      `source=${r.source} loadError=${r.loadError}`)
  }

  // C6: mode 0666 ignored -> none + loadError
  {
    const xdg = mkdtempSync(join(tmpdir(), 'spinoml-xdg-'))
    const dir = join(xdg, 'spinoml'); mkdirSync(dir)
    const file = join(dir, 'scope.json')
    writeFileSync(file, JSON.stringify({ version: 1, symlink_targets: [outside2Real] }))
    chmodSync(file, 0o666)
    const r = loadSymlinkTargets({ env: { XDG_RUNTIME_DIR: xdg }, homedir: noHome })
    check('mode 0666 -> none + loadError', r.source === 'none' && r.targets.length === 0 && r.loadError != null && /writable/i.test(r.loadError),
      `source=${r.source} loadError=${r.loadError}`)
  }

  // C7: nothing configured
  {
    const r = loadSymlinkTargets({ env: {}, homedir: join(bdir, 'no-such-home-2') })
    check('nothing configured -> source none, no error', r.source === 'none' && r.targets.length === 0 && r.loadError === null,
      `source=${r.source} loadError=${r.loadError}`)
  }

  // C8: scope file with bad symlink_targets type
  {
    const xdg = mkdtempSync(join(tmpdir(), 'spinoml-xdg-'))
    const dir = join(xdg, 'spinoml'); mkdirSync(dir)
    const file = join(dir, 'scope.json')
    writeFileSync(file, JSON.stringify({ version: 1, symlink_targets: 'nope' }))
    chmodSync(file, 0o600)
    const r = loadSymlinkTargets({ env: { XDG_RUNTIME_DIR: xdg }, homedir: noHome })
    check('bad symlink_targets type -> none + loadError',
      r.source === 'none' && r.loadError != null && /array/i.test(r.loadError),
      `source=${r.source} loadError=${r.loadError}`)
  }

  // ── D. main.mjs wiring (static, on source text) ───────────────────────────
  console.log('\n=== D. main.mjs wiring ===')
  const mainPath = join(process.cwd(), 'sidecar-llm', 'main.mjs')
  const src = readFileSync(mainPath, 'utf8')

  check("imports from './path-scope.mjs'", src.includes("from './path-scope.mjs'"))
  check('uses resolveInWorkspace(', src.includes('resolveInWorkspace('))
  check('calls loadSymlinkTargets(', src.includes('loadSymlinkTargets('))

  // No `path.join(ws.root …)` left on a LOCAL fs call. Remote branches use
  // template strings (`${ws.root}/…`), which this check ignores; after the
  // conversion the only path.join(ws.root should be ZERO.
  {
    const lines = src.split('\n')
    const offenders: number[] = []
    lines.forEach((ln, i) => { if (ln.includes('path.join(ws.root')) offenders.push(i + 1) })
    check('no path.join(ws.root …) on local FS calls',
      offenders.length === 0,
      offenders.length ? `offenders at lines ${offenders.join(',')}` : '')
  }

  // safeRelpath must still exist (still used for remote paths + display).
  check('safeRelpath still defined', src.includes('function safeRelpath('))

  // The rejection message includes the code.
  check('error includes PATH_ code', src.includes('PATH_'))

  // ── cleanup ───────────────────────────────────────────────────────────────
  rmSync(workdir, { recursive: true, force: true })
  rmSync(bdir, { recursive: true, force: true })

  if (failures > 0) {
    console.error(`\n${failures} check(s) failed`)
    process.exit(1)
  } else {
    console.log('\n✓ all symlink-aware containment checks passed')
  }
}

main().catch((e) => { console.error(e); process.exit(2) })
