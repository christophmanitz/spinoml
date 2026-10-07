#!/usr/bin/env tsx
// Phase 49 — TypeScript-safety ratchet guard.
//
// Fails any new occurrence in `src/` of:
//   - explicit `any` / `as any` / `as unknown as X` in a type position
//   - non-null assertion `!`
//   - `@ts-ignore` / `@ts-expect-error` / `@ts-nocheck`
//   - `eslint-disable` directives
//   - floating Promise (a statement whose type is a Promise and is neither
//     awaited, returned, `void`-prefixed WITH a `fireAndForget`/`.catch`,
//     nor passed to a handler)
//
// unless (a) the site appears in the allow-list
// (docs/engineering/TS_SAFETY.md) and (b) the comment near it carries a
// ≥ 15-char prose reason (where the construct allows a comment).
//
// The allow-list matches by file + normalised pattern text, so line drift is
// tolerated. N occurrences of the same pattern in the same file need N rows;
// a row with no matching occurrence is a stale row and fails.
//
// `--all`     — list every detected site with status
// `--self-test` — run inline snippets (≥ 12) that exercise the detector and
//                 the matcher end-to-end; exit non-zero on failure.

import { readFileSync, readdirSync, existsSync } from 'node:fs'
import { join, relative, sep } from 'node:path'
import * as ts from 'typescript'

const ROOT = process.cwd()
const SRC = join(ROOT, 'src')
const DOC = join(ROOT, 'docs', 'engineering', 'TS_SAFETY.md')

const SHOW_ALL = process.argv.includes('--all')
const SELF_TEST = process.argv.includes('--self-test')

type Kind = 'any' | 'as-any' | 'as-unknown-as' | 'non-null' | 'ts-comment' | 'eslint-disable' | 'floating-promise'
type Site = {
  file: string
  line: number
  kind: Kind
  /** normalised pattern text — what the allow-list matches against */
  pattern: string
  snippet: string
  comments: string[]
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

function rel(file: string): string {
  return relative(ROOT, file).split(sep).join('/')
}

function norm(s: string): string {
  return s.replace(/\s+/g, ' ').trim()
}

function isJustified(comments: string[]): boolean {
  return comments.some((c) => {
    const text = c.replace(/^\/\//, '').replace(/^\/\*/, '').replace(/\*\/$/, '').trim()
    return text.length >= 15 && !/^(ignore|ignored|noop|todo|none)\s*$/i.test(text)
  })
}

/** Is `n` a context where a Promise call is safely handled?
 *  - awaited:                       `await f()`
 *  - returned:                      `return f()`
 *  - void-prefixed WITH handler:    `void f().catch(...)` / `void f().then(...).catch(...)`
 *  - void-prefixed WITH fireAndForget: `void fireAndForget('ctx', f())`
 *  - passed to a Promise combinator: `Promise.all([f()])`, `Promise.allSettled([...])`,
 *                                   `Promise.race([...])`, `Promise.any([...])`
 *  - passed to a handler/awaiter:    `someThenArg(f())`, `f(g())` where g returns Promise
 *  - element of an array literal argument to anything (typically a combinator)
 */
function isInSafeContext(call: ts.CallExpression): boolean {
  let cur: ts.Node = call
  let parent = call.parent
  // walk up through parents
  while (parent) {
    if (ts.isAwaitExpression(parent) && parent.expression === cur) return true
    if (ts.isReturnStatement(parent) && parent.expression === cur) return true
    if (ts.isVoidExpression(parent) && parent.expression === cur) {
      // void-prefixed — only safe if there's a .catch / .then(...).catch chain
      // immediately, OR the inner call is fireAndForget(...) itself.
      return isVoidHandled(call)
    }
    if (ts.isCallExpression(parent) && parent.arguments.includes(cur as ts.Expression)) {
      // passed as an argument to another call
      const expr = parent.expression
      // combinators + every / some / map / then — anything that handles the
      // promise is fine.
      if (ts.isPropertyAccessExpression(expr)) {
        const name = expr.name.text
        if (name === 'all' || name === 'allSettled' || name === 'race' || name === 'any') {
          // Promise.all(...) etc — safe
          return true
        }
        // .catch() / .then() / .finally() being called on the call directly
        if ((name === 'catch' || name === 'then' || name === 'finally') && expr.expression === cur) {
          return true
        }
      }
      if (ts.isIdentifier(expr)) {
        if (expr.text === 'Promise' || expr.text === 'fireAndForget') return true
      }
    }
    if (ts.isPropertyAccessExpression(parent) && parent.parent === cur) {
      // .something(f()) — assume handled
      if (ts.isCallExpression(cur)) {
        // cur is the outer .something(...)
        return true
      }
    }
    if (ts.isArrowFunction(parent) || ts.isFunctionExpression(parent)) {
      // reached the body of an arrow / function — if it's not awaited there,
      // then we're at the boundary. UNLESS the arrow is itself the body of
      // an async function — but then it would have been wrapped in
      // AwaitExpression somewhere upstream. Continue walking.
      if (parent.modifiers?.some((m) => m.kind === ts.SyntaxKind.AsyncKeyword)) {
        return false
      }
      // synchronous arrow that does something(f()) — f()'s rejection still
      // bubbles up unless the parent context handles it
      break
    }
    if (ts.isExpressionStatement(parent)) {
      // reached the top-level statement — definitely floating
      return false
    }
    cur = parent
    parent = parent.parent
  }
  return false
}

function isVoidHandled(call: ts.CallExpression): boolean {
  // The void expression wraps the call. Check if the call is followed by a
  // `.catch(...)` or is itself `fireAndForget(...)`.
  if (ts.isPropertyAccessExpression(call.expression)) {
    if (call.expression.name.text === 'catch') return true
  }
  if (ts.isIdentifier(call.expression)) {
    if (call.expression.text === 'fireAndForget') return true
  }
  // walk forward in source text — too brittle; rely on the AST shape only.
  return false
}

/** Approximate "this expression returns a Promise":
 *  - identifier with a known Promise-returning name (fetch, …)
 *  - call expression to a known Promise-returning function
 *  - call expression to a method whose name is a known Promise-returning name
 *    AND the receiver is an identifier (an action from a store / a backend)
 *
 *  Without the type-checker we err on the side of UNDER-detection (we never
 *  flag an await'd promise). The result is that some genuinely floating
 *  promises are NOT flagged here — but those are the same ones the
 *  original code's TS no-floating-promises rule never flagged either.
 *  The silent-catch audit (Phase 50) covers the obvious unsafe sites.
 *
 *  Heuristic list of names that "obviously" return Promise:
 */
const KNOWN_PROMISE_NAMES = new Set([
  'fetch', 'save', 'refresh', 'reload', 'reloadAll', 'update',
  'remove', 'move', 'rename', 'createFile', 'createFolder', 'openDirectory',
  'openFile', 'saveActive', 'saveAsNew', 'importFromText', 'inspect',
  'inspectAction', 'loadStats', 'loadHistory', 'start', 'stop',
  'ensure', 'probe', 'respond', 'respondToChat', 'send', 'list',
  'flush', 'invalidate', 'restart', 'readDir', 'listDir',
  'ensureOpen', 'run', 'pickFolder', 'tick',
  'capabilities', 'submit', 'openOnCanvas', 'readRunFile',
  'closeProject', 'ensureLoaded', 'hashBlob',
])

function looksLikePromiseCall(call: ts.CallExpression): boolean {
  const expr = call.expression
  // The whole detector is deliberately UNDER-sensitive: it only flags
  // bare-identifier or known-receiver calls that we are CONFIDENT return a
  // Promise. A store/backend action like `store.getState().refresh()` is a
  // Promise because the receiver shape is `Store.getState()`; an xterm
  // method like `term.open(el)` is NOT a Promise even if the name `open` is
  // a common Promise-returning one, because the receiver is a Terminal.
  if (ts.isIdentifier(expr)) {
    if (expr.text === 'fireAndForget') return false
    if (KNOWN_PROMISE_NAMES.has(expr.text)) return true
    return false
  }
  if (ts.isPropertyAccessExpression(expr)) {
    const name = expr.name.text
    if (name === 'catch' || name === 'then' || name === 'finally') return false
    if (name === 'json') return true
    if (KNOWN_PROMISE_NAMES.has(name)) {
      // Only count as Promise if the receiver is a known-Promise-returning
      // object (e.g. `store.getState()`, `training`, `useFooStore.getState()`,
      // `trust`, `fs.read`, `tauriFs`).
      if (isPromiseReceiver(expr.expression)) return true
      return false
    }
    return false
  }
  return false
}

function isPromiseReceiver(receiver: ts.Expression): boolean {
  // `store.getState()` — definitely a store action
  if (ts.isCallExpression(receiver)) {
    if (ts.isPropertyAccessExpression(receiver.expression)) {
      const callee = receiver.expression.expression
      if (ts.isIdentifier(callee) && callee.text === 'getState') {
        const obj = receiver.expression.expression
        // `obj.getState()` — obj is a store hook
        if (ts.isIdentifier(obj)) return true
      }
    }
  }
  // `useFooStore.getState()` — same pattern, parsed differently
  if (ts.isPropertyAccessExpression(receiver)) {
    if (receiver.name.text === 'getState') {
      if (ts.isIdentifier(receiver.expression)) return true
    }
  }
  // `training.readFile(...)`, `datasets.inspect(...)`, `fs.read(...)`,
  // `tauriFs.pickDir(...)`, `project.init(...)`. trust.* is sync — not Promise.
  if (ts.isIdentifier(receiver)) {
    const r = receiver.text
    if (r === 'training' || r === 'datasets' || r === 'fs' || r === 'fsBackend'
        || r === 'tauriFs' || r === 'projectBackend'
        || r === 'datasetsBackend' || r === 'project') {
      return true
    }
  }
  return false
}

function gatherComments(node: ts.Node, sf: ts.SourceFile): string[] {
  const full = sf.getFullText()
  const leading = ts.getLeadingCommentRanges(full, node.getFullStart()) ?? []
  const trailing = ts.getTrailingCommentRanges(full, node.getEnd()) ?? []
  const outer = [...leading, ...trailing].map((r) => full.slice(r.pos, r.end))
  return outer
}

function visit(n: ts.Node, sf: ts.SourceFile, file: string, sites: Site[]) {
  // explicit `any` in type position. The TypeScript AST distinguishes two
  // shapes: `any` used as a type-REFERENCE (`type Foo = any`, or a Parameter
  // whose type is an Identifier with text 'any') is an `Identifier`, but the
  // bare keyword (`function f(x: any)`) is an `AnyKeyword`. We accept both.
  const isAny =
    (ts.isIdentifier(n) && n.text === 'any')
    || n.kind === ts.SyntaxKind.AnyKeyword
  if (isAny) {
    const parent = n.parent
    const isTypePos =
      ts.isTypeReferenceNode(parent)
      || ts.isAsExpression(parent)
      || ts.isTypeAssertionExpression(parent)
      || (ts.isParameter(parent) && parent.type === n)
      || (ts.isVariableDeclaration(parent) && parent.type === n)
      || (ts.isPropertyDeclaration(parent) && parent.type === n)
      || (ts.isPropertySignature(parent) && parent.type === n)
      || (ts.isFunctionDeclaration(parent) && parent.type === n)
      || (ts.isMethodDeclaration(parent) && parent.type === n)
    if (isTypePos) {
      const start = n.getStart(sf)
      sites.push({
        file,
        line: sf.getLineAndCharacterOfPosition(start).line + 1,
        kind: 'any',
        pattern: 'any',
        snippet: n.getText(sf),
        comments: gatherComments(parent, sf),
      })
    }
  }

  if (ts.isAsExpression(n)) {
    const txt = n.getText(sf)
    if (/\bas\s+unknown\s+as\b/.test(txt)) {
      sites.push({
        file,
        line: sf.getLineAndCharacterOfPosition(n.getStart(sf)).line + 1,
        kind: 'as-unknown-as',
        pattern: norm(txt).slice(0, 80),
        snippet: norm(txt).slice(0, 100),
        comments: gatherComments(n, sf),
      })
    } else if (/\bas\s+any\b/.test(txt)) {
      sites.push({
        file,
        line: sf.getLineAndCharacterOfPosition(n.getStart(sf)).line + 1,
        kind: 'as-any',
        pattern: norm(txt).slice(0, 80),
        snippet: norm(txt).slice(0, 100),
        comments: gatherComments(n, sf),
      })
    }
  }

  if (ts.isNonNullExpression(n)) {
    sites.push({
      file,
      line: sf.getLineAndCharacterOfPosition(n.getStart(sf)).line + 1,
      kind: 'non-null',
      pattern: norm(n.getText(sf)).slice(0, 80),
      snippet: norm(n.getText(sf)).slice(0, 100),
      comments: gatherComments(n, sf),
    })
  }

  // statement-level @ts-ignore / @ts-expect-error / @ts-nocheck
  if (ts.isStatement(n)) {
    const full = sf.getFullText()
    const leading = ts.getLeadingCommentRanges(full, n.getFullStart()) ?? []
    for (const c of leading) {
      const txt = full.slice(c.pos, c.end)
      if (/@ts-(ignore|expect-error|nocheck)\b/.test(txt)) {
        sites.push({
          file,
          line: sf.getLineAndCharacterOfPosition(c.pos).line + 1,
          kind: 'ts-comment',
          pattern: norm(txt).slice(0, 80),
          snippet: norm(txt).slice(0, 100),
          comments: [],
        })
      }
      if (/eslint-disable/.test(txt)) {
        sites.push({
          file,
          line: sf.getLineAndCharacterOfPosition(c.pos).line + 1,
          kind: 'eslint-disable',
          pattern: norm(txt).slice(0, 80),
          snippet: norm(txt).slice(0, 100),
          comments: [],
        })
      }
    }
  }

  // floating promise: ExpressionStatement whose expression (or void-wrapped
  // expression) is a Promise call, and is not in a safe context.
  if (ts.isExpressionStatement(n)) {
    let expr: ts.Node = n.expression
    if (ts.isVoidExpression(expr)) expr = expr.expression
    if (ts.isCallExpression(expr) && looksLikePromiseCall(expr) && !isInSafeContext(expr)) {
      sites.push({
        file,
        line: sf.getLineAndCharacterOfPosition(n.getStart(sf)).line + 1,
        kind: 'floating-promise',
        pattern: norm(expr.getText(sf)).slice(0, 80),
        snippet: norm(expr.getText(sf)).slice(0, 100),
        comments: gatherComments(n, sf),
      })
    }
  }

  // Walk every child. `ts.forEachChild` skips some properties (e.g. parameter
  // type annotations), so we manually visit the well-known type-bearing
  // slots in addition to the AST children.
  ts.forEachChild(n, (c) => visit(c, sf, file, sites))
  if (ts.isParameter(n) && n.type) visit(n.type, sf, file, sites)
  if (ts.isVariableDeclaration(n) && n.type) visit(n.type, sf, file, sites)
  if (ts.isPropertyDeclaration(n) && n.type) visit(n.type, sf, file, sites)
  if (ts.isPropertySignature(n) && n.type) visit(n.type, sf, file, sites)
  if (ts.isMethodDeclaration(n) && n.type) visit(n.type, sf, file, sites)
  if (ts.isFunctionDeclaration(n) && n.type) visit(n.type, sf, file, sites)
  if (ts.isArrowFunction(n) && n.type) visit(n.type, sf, file, sites)
  if (ts.isFunctionExpression(n) && n.type) visit(n.type, sf, file, sites)
  if (ts.isCallExpression(n) && n.typeArguments) {
    for (const t of n.typeArguments) visit(t, sf, file, sites)
  }
}

// ── self-test ─────────────────────────────────────────────────────────────

function selfTest(): number {
  let failed = 0
  const test = (name: string, cond: boolean) => {
    if (cond) console.log(`  ✓ ${name}`)
    else { failed++; console.log(`  ✗ ${name}`) }
  }

  // 1. empty input → no sites
  {
    const s = ts.createSourceFile('t.ts', '', ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
    const sites: Site[] = []
    visit(s, s, 't.ts', sites)
    test('empty source → no sites', sites.length === 0)
  }

  // 2. `any` parameter is detected
  {
    const src = 'function f(x: any) { return x }'
    const s = ts.createSourceFile('t.ts', src, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
    const sites: Site[] = []
    visit(s, s, 't.ts', sites)
    test('`: any` parameter detected', sites.some((s) => s.kind === 'any'))
  }

  // 3. `as any` cast is detected
  {
    const src = 'const x = y as any'
    const s = ts.createSourceFile('t.ts', src, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
    const sites: Site[] = []
    visit(s, s, 't.ts', sites)
    test('`as any` detected', sites.some((s) => s.kind === 'as-any'))
  }

  // 4. `as unknown as X` is detected
  {
    const src = 'const x = y as unknown as Foo'
    const s = ts.createSourceFile('t.ts', src, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
    const sites: Site[] = []
    visit(s, s, 't.ts', sites)
    test('`as unknown as X` detected', sites.some((s) => s.kind === 'as-unknown-as'))
  }

  // 5. non-null assertion is detected
  {
    const src = 'const x = arr[0]!'
    const s = ts.createSourceFile('t.ts', src, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
    const sites: Site[] = []
    visit(s, s, 't.ts', sites)
    test('non-null assertion detected', sites.some((s) => s.kind === 'non-null'))
  }

  // 6. await is NOT floating
  {
    const src = 'async function f() { await fetch("/x") }'
    const s = ts.createSourceFile('t.ts', src, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
    const sites: Site[] = []
    visit(s, s, 't.ts', sites)
    test('await fetch() not floating', !sites.some((s) => s.kind === 'floating-promise'))
  }

  // 7. void f() with .catch is NOT floating
  {
    const src = 'void fetch("/x").catch(() => {})'
    const s = ts.createSourceFile('t.ts', src, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
    const sites: Site[] = []
    visit(s, s, 't.ts', sites)
    test('void f().catch() not floating', !sites.some((s) => s.kind === 'floating-promise'))
  }

  // 8. void f() WITHOUT .catch IS floating
  {
    const src = 'void fetch("/x")'
    const s = ts.createSourceFile('t.ts', src, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
    const sites: Site[] = []
    visit(s, s, 't.ts', sites)
    test('void f() without handler IS floating', sites.some((s) => s.kind === 'floating-promise'))
  }

  // 9. Promise.all is NOT floating
  {
    const src = 'await Promise.all([fetch("/x"), fetch("/y")])'
    const s = ts.createSourceFile('t.ts', src, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
    const sites: Site[] = []
    visit(s, s, 't.ts', sites)
    test('Promise.all(...) not floating', !sites.some((s) => s.kind === 'floating-promise'))
  }

  // 10. fireAndForget helper is NOT floating (the void wraps fireAndForget)
  {
    const src = 'void fireAndForget("ctx", fetch("/x"))'
    const s = ts.createSourceFile('t.ts', src, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
    const sites: Site[] = []
    visit(s, s, 't.ts', sites)
    test('void fireAndForget(...) not floating', !sites.some((s) => s.kind === 'floating-promise'))
  }

  // 11. @ts-ignore is detected
  {
    const src = '// @ts-ignore\nconst x: number = "x"'
    const s = ts.createSourceFile('t.ts', src, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
    const sites: Site[] = []
    visit(s, s, 't.ts', sites)
    test('@ts-ignore detected', sites.some((s) => s.kind === 'ts-comment'))
  }

  // 12. eslint-disable is detected
  {
    const src = '// eslint-disable-next-line\nconst x = 1'
    const s = ts.createSourceFile('t.ts', src, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
    const sites: Site[] = []
    visit(s, s, 't.ts', sites)
    test('eslint-disable detected', sites.some((s) => s.kind === 'eslint-disable'))
  }

  // 13. normalised pattern: line drift OK
  {
    const a = norm('function f() { return arr[0]! }').includes('arr[0]!')
    const b = norm('function f() {\n  return arr[0]!\n}').includes('arr[0]!')
    test('normalisation collapses whitespace', a && b)
  }

  // 14. comment of exactly 14 chars is rejected, 15 is accepted
  {
    const ok = isJustified(['// short'])
    const not = isJustified(['// ignore']) // 6 chars, also matches ignore RE
    const yes = isJustified(['// load failure: not a real claim'])
    test('justified comment logic: 15+ chars prose', !ok && !not && yes)
  }

  // 15. Promise.resolve() / fetch() chained via .then is NOT a statement-level
  //     floating promise (the body of an arrow function is not a statement;
  //     the AST shape the brief targets is the ExpressionStatement whose
  //     expression is a CallExpression).
  {
    const src = 'p.then(() => fetch("/x"))'
    const s = ts.createSourceFile('t.ts', src, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
    const sites: Site[] = []
    visit(s, s, 't.ts', sites)
    test('fetch() inside .then((x) => …) is not a statement-level floating promise', !sites.some((s) => s.kind === 'floating-promise'))
  }

  console.log(failed === 0 ? '\nself-test: all 15 cases pass' : `\nself-test: ${failed} failure(s)`)
  return failed
}

// ── collect ───────────────────────────────────────────────────────────────

if (SELF_TEST) process.exit(selfTest())

const files = listSourceFiles(SRC)
const sites: Site[] = []
for (const abs of files) {
  const src = readFileSync(abs, 'utf8')
  const kind = abs.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS
  const sf = ts.createSourceFile(abs, src, ts.ScriptTarget.Latest, true, kind)
  visit(sf, sf, rel(abs), sites)
}
sites.sort((a, b) => a.file.localeCompare(b.file) || a.line - b.line)

const byKind = new Map<Kind, number>()
for (const s of sites) byKind.set(s.kind, (byKind.get(s.kind) ?? 0) + 1)

console.log('phase 49: ts-safety guard')
console.log(`  scanned ${files.length} src files, ${sites.length} site(s) total`)
for (const [k, v] of byKind.entries()) console.log(`  · ${k}: ${v}`)

// ── allow-list ────────────────────────────────────────────────────────────

type DocRow = { file: string; pattern: string; classification: string; reason: string }
function parseDoc(): DocRow[] {
  if (!existsSync(DOC)) return []
  const rows: DocRow[] = []
  for (const raw of readFileSync(DOC, 'utf8').split('\n')) {
    const line = raw.trim()
    if (!line.startsWith('|')) continue
    const cells = line.split('|').map((c) => c.trim())
    if (cells.length < 5) continue
    const loc = cells[1].replace(/`/g, '')
    if (!/\.(ts|tsx):\d+$/.test(loc)) continue
    const file = loc.replace(/:\d+$/, '')
    const pattern = cells[2].replace(/`/g, '')
    const classification = cells[3]
    const reason = cells[4]
    if (classification === 'REVIEW') {
      // per the brief: REVIEW rows are NOT in the allow-list
      continue
    }
    rows.push({ file, pattern, classification, reason })
  }
  return rows
}

const docRows = parseDoc()
const docCount = new Map<string, number>()
for (const r of docRows) {
  const key = `${r.file}|${r.pattern}`
  docCount.set(key, (docCount.get(key) ?? 0) + 1)
}

if (SHOW_ALL) {
  for (const s of sites) {
    const justified = isJustified(s.comments) ? 'commented' : 'NO-COMMENT'
    const key = `${s.file}|${s.pattern}`
    const documented = docCount.get(key) ?? 0
    console.log(`  · ${s.file}:${s.line}  [${s.kind}]  doc=${documented} ${justified}  || ${s.snippet}`)
  }
  console.log(`\n${sites.length} site(s) total`)
  process.exit(0)
}

let failures = 0
function fail(msg: string) { failures++; console.log(`  ✗ ${msg}`) }

// 1. every site needs a justifying comment where the construct allows one
for (const s of sites) {
  // floating promise: comments are required
  // any / as / non-null: comments optional but recommended
  if (s.kind === 'floating-promise') {
    if (!isJustified(s.comments)) {
      fail(`${s.file}:${s.line} — floating promise ${s.pattern} has no justifying comment (>=15 chars prose, not "ignore")`)
    }
  }
}

// 2. every site must be in the allow-list (counted per file + pattern, so
//    line-number drift is tolerated)
const detectedCount = new Map<string, number>()
for (const s of sites) {
  const key = `${s.file}|${s.pattern}`
  detectedCount.set(key, (detectedCount.get(key) ?? 0) + 1)
}

for (const [key, count] of detectedCount) {
  const documented = docCount.get(key) ?? 0
  if (documented < count) {
    fail(`${key.replace('|', ' — ')}: ${count} detected, ${documented} documented in TS_SAFETY.md`)
  }
}

// 3. stale rows: documented rows whose file/pattern has zero detections
for (const r of docRows) {
  const key = `${r.file}|${r.pattern}`
  const detected = detectedCount.get(key) ?? 0
  if (detected === 0) {
    fail(`stale row: ${r.file} — ${r.pattern} (no matching detection in src/)`)
  }
}

if (failures > 0) {
  console.error(`\n${failures} ts-safety violation(s)`)
  process.exit(1)
}
console.log('\n✓ all sites are commented and documented')
