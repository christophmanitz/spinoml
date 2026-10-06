#!/usr/bin/env tsx
// Phase 48 — Rust panic / silent-swallow guard (mirror of
// scripts/verify-silent-catch.ts for src-tauri/src/*.rs).
//
// Without `syn` available this is a careful line scanner: every Rust file under
// src-tauri/src is masked (string / line / block / raw / char / byte literals
// and comments are replaced with spaces, preserving newlines and columns),
// every `#[cfg(test)]` item is tracked and excluded, and every remaining
// occurrence of the patterns below is reported. Limitations:
//   * Line-based: a pattern split across lines is missed (none exist today).
//   * Slice indexing `x[i]` / `s[a..b]`, `as` casts, integer arithmetic and
//     `Duration`/`Instant` subtraction are NOT scanned automatically (syn is
//     the proper tool). Those were reviewed by hand for this audit; see the
//     "Rust — manual scan" notes in docs/engineering/SILENT_EXCEPTIONS.md.
//   * Justification comments are gathered from the site line and the
//     contiguous comment/blank block above it (same rule as the Python guard).
//
// Every site must (a) carry a trailing or preceding `// <reason >= 15 chars of
// prose not "ignore"/"safe"/...>` comment AND (b) appear in the
// `## Rust allow-list` section of docs/engineering/SILENT_EXCEPTIONS.md
// (matched by file + normalised pattern text; line-number drift tolerated;
// N occurrences -> N rows; stale rows fail).
//
// Usage:
//   tsx scripts/verify-rust-panics.ts             # guard (exit 1 on violation)
//   tsx scripts/verify-rust-panics.ts --all        # list every detected site
//   tsx scripts/verify-rust-panics.ts --tsv        # machine-readable rows
//   tsx scripts/verify-rust-panics.ts --self-test

import { readFileSync, readdirSync, existsSync } from 'node:fs'
import { join, relative, sep } from 'node:path'

const ROOT = process.cwd()
const SRC = join(ROOT, 'src-tauri', 'src')
const DOC = join(ROOT, 'docs', 'engineering', 'SILENT_EXCEPTIONS.md')

const SHOW_ALL = process.argv.includes('--all')
const SHOW_TSV = process.argv.includes('--tsv')
const SELF_TEST = process.argv.includes('--self-test')

const IGNORE_RE = /^\s*(ignore|ignored|noop|todo|none|safe)\s*$/i

type Site = { file: string; line: number; pattern: string; snippet: string }

// Longest-first so `unwrap_or_else(` / `unwrap_or_default` are picked before
// the shorter `unwrap_or(` (their needles do not overlap, but order keeps the
// reported construct intuitive).
const PATTERNS: { name: string; needle: string }[] = [
  { name: 'unwrap_or_else(', needle: 'unwrap_or_else(' },
  { name: 'unwrap_or_default', needle: 'unwrap_or_default' },
  { name: 'unwrap_or(', needle: 'unwrap_or(' },
  { name: 'unwrap(', needle: 'unwrap(' },
  { name: 'expect(', needle: 'expect(' },
  { name: 'panic!', needle: 'panic!' },
  { name: 'unreachable!', needle: 'unreachable!' },
  { name: 'todo!', needle: 'todo!' },
  { name: 'unimplemented!', needle: 'unimplemented!' },
  { name: 'let _ =', needle: 'let _ =' },
  { name: '.ok()', needle: '.ok()' },
]

// ── masking ───────────────────────────────────────────────────────────────

type MaskResult = { masked: string; comments: string[][]; lineCommentStart: number[] }

function maskSource(src: string): MaskResult {
  let masked = ''
  const comments: string[][] = [[]]
  const lineCommentStart: number[] = [-1]
  let i = 0
  let line = 0
  let lineStart = 0
  const push = (ch: string, maskedChar?: string) => {
    masked += maskedChar !== undefined ? maskedChar : ch
    if (ch === '\n') {
      line++
      lineStart = masked.length
      if (!comments[line]) comments[line] = []
      if (lineCommentStart[line] === undefined) lineCommentStart[line] = -1
    }
  }

  while (i < src.length) {
    const c = src[i]
    const c2 = src[i + 1]

    if (c === '/' && c2 === '/') {
      let j = i
      while (j < src.length && src[j] !== '\n') j++
      comments[line].push(src.slice(i, j))
      if (lineCommentStart[line] === -1) lineCommentStart[line] = i - lineStart
      for (let k = i; k < j; k++) push(src[k], ' ')
      i = j
      continue
    }

    if (c === '/' && c2 === '*') {
      const startLine = line
      let text = '/*'
      let j = i + 2
      let d = 1
      push(' ', ' ')
      push(' ', ' ')
      while (j < src.length && d > 0) {
        if (src[j] === '/' && src[j + 1] === '*') {
          d++; text += '/*'; push(' ', ' '); push(' ', ' '); j += 2
        } else if (src[j] === '*' && src[j + 1] === '/') {
          d--; text += '*/'; push(' ', ' '); push(' ', ' '); j += 2
        } else {
          text += src[j]
          push(src[j], src[j] === '\n' ? '\n' : ' ')
          j++
        }
      }
      comments[startLine].push(text)
      i = j
      continue
    }

    // raw string: r"..." / r#"..."# / br#"..."#
    if (c === 'r' && (c2 === '"' || c2 === '#')) {
      let p = i + 1
      let hashes = 0
      while (src[p] === '#') { hashes++; p++ }
      if (src[p] === '"') {
        for (let k = i; k <= p; k++) push(src[k], ' ')
        p++
        while (p < src.length) {
          if (src[p] === '"') {
            let q = p + 1
            let ok = true
            for (let h = 0; h < hashes; h++) if (src[q + h] !== '#') { ok = false; break }
            if (ok) {
              q += hashes
              for (let k = p; k < q; k++) push(src[k], src[k] === '\n' ? '\n' : ' ')
              p = q
              break
            }
          }
          push(src[p], src[p] === '\n' ? '\n' : ' ')
          p++
        }
        i = p
        continue
      }
    }

    // normal / byte string
    if (c === '"') {
      push(c, ' ')
      i++
      while (i < src.length && src[i] !== '"') {
        if (src[i] === '\\' && src[i + 1] !== undefined) {
          push(src[i], ' ')
          push(src[i + 1], src[i + 1] === '\n' ? '\n' : ' ')
          i += 2
        } else {
          push(src[i], src[i] === '\n' ? '\n' : ' ')
          i++
        }
      }
      if (src[i] === '"') { push(src[i], ' '); i++ }
      continue
    }

    // char literal vs lifetime
    if (c === "'") {
      const n1 = src[i + 1]
      const n2 = src[i + 2]
      const isLifetime = n1 !== undefined && /[A-Za-z_]/.test(n1) && n2 !== "'"
      if (!isLifetime && n1 !== undefined) {
        push(c, ' ')
        i++
        while (i < src.length && src[i] !== "'") {
          if (src[i] === '\\' && src[i + 1] !== undefined) {
            push(src[i], ' ')
            push(src[i + 1], src[i + 1] === '\n' ? '\n' : ' ')
            i += 2
          } else {
            push(src[i], src[i] === '\n' ? '\n' : ' ')
            i++
          }
        }
        if (src[i] === "'") { push(src[i], ' '); i++ }
        continue
      }
    }

    push(c)
    i++
  }
  return { masked, comments, lineCommentStart }
}

// ── cfg(test) ranges ──────────────────────────────────────────────────────

function lineOf(text: string, offset: number): number {
  let line = 1
  for (let i = 0; i < offset && i < text.length; i++) if (text[i] === '\n') line++
  return line
}

function computeTestRanges(masked: string): Array<{ from: number; to: number }> {
  const ranges: Array<{ from: number; to: number }> = []
  const needle = '#[cfg(test)]'
  let from = 0
  while (true) {
    const idx = masked.indexOf(needle, from)
    if (idx === -1) break
    from = idx + needle.length
    const startLine = lineOf(masked, idx)
    let k = idx + needle.length
    // first `{` or `;` after the attribute
    let guard = 0
    while (k < masked.length && masked[k] !== '{' && masked[k] !== ';' && guard < 200000) { k++; guard++ }
    let endOffset = k
    if (masked[k] === '{') {
      let depth = 1
      k++
      while (k < masked.length && depth > 0) {
        if (masked[k] === '{') depth++
        else if (masked[k] === '}') depth--
        k++
      }
      endOffset = k
    }
    ranges.push({ from: startLine, to: lineOf(masked, endOffset) })
  }
  return ranges
}

// ── detection ─────────────────────────────────────────────────────────────

function findSites(src: string, rel: string, testRanges: Array<{ from: number; to: number }>): Site[] {
  const { masked, lineCommentStart } = maskSource(src)
  const lines = src.split('\n')
  const codeLines = masked.split('\n')
  const sites: Site[] = []
  for (let ln = 0; ln < lines.length; ln++) {
    const lineNo = ln + 1
    if (testRanges.some((r) => lineNo >= r.from && lineNo <= r.to)) continue
    const code = codeLines[ln] ?? ''
    if (!code.trim()) continue
    // The pattern key is the code BEFORE any same-line `//` comment, so a
    // justification comment appended to the line does not change the key.
    const cstart = lineCommentStart[ln] ?? -1
    const codeOnly = (cstart >= 0 ? lines[ln].slice(0, cstart) : lines[ln]).trim().replace(/\s+/g, ' ')
    for (const p of PATTERNS) {
      let from = 0
      for (;;) {
        const idx = code.indexOf(p.needle, from)
        if (idx === -1) break
        from = idx + 1
        const last = p.needle[p.needle.length - 1]
        const after = code[idx + p.needle.length]
        if (/[A-Za-z0-9_]/.test(last) && after && /[A-Za-z0-9_]/.test(after)) continue
        sites.push({ file: rel, line: lineNo, pattern: p.name, snippet: codeOnly })
      }
    }
  }
  return sites
}

// ── comments / justification ──────────────────────────────────────────────

function cleanComment(c: string): string {
  return c.replace(/^\/\//, '').replace(/^\/\*/, '').replace(/\*\/$/, '').trim()
}

function hasJustified(lineComments: string[][], lines: string[], siteLine: number): boolean {
  const ok = (arr: string[] | undefined) =>
    (arr ?? []).some((c) => { const t = cleanComment(c); return t.length >= 15 && !IGNORE_RE.test(t) })
  if (ok(lineComments[siteLine - 1])) return true
  for (let ln = siteLine - 2; ln >= 0; ln--) {
    const stripped = (lines[ln] ?? '').trim()
    if (stripped === '' || stripped.startsWith('//') || stripped.startsWith('/*') || stripped.startsWith('*')) {
      if (ok(lineComments[ln])) return true
      if (stripped === '') continue
      continue
    }
    break
  }
  return false
}

// ── allow-list ────────────────────────────────────────────────────────────

type DocRow = { file: string; pattern: string; classification: string }

function parseDoc(): DocRow[] {
  if (!existsSync(DOC)) return []
  const rows: DocRow[] = []
  let inRust = false
  for (const raw of readFileSync(DOC, 'utf8').split('\n')) {
    const stripped = raw.trim()
    if (stripped.startsWith('## ')) { inRust = stripped === '## Rust allow-list'; continue }
    if (!inRust) continue
    if (!stripped.startsWith('|')) continue
    // Split on the ` | ` cell delimiter, not on every `|` — Rust patterns
    // contain closure pipes (`map(|g| g.is_some())`). `\|` escapes are
    // unescaped first for markdown rendering.
    const unescaped = stripped.replace(/\\\|/g, '|')
    const cells = unescaped.replace(/^\|/, '').replace(/\|$/, '').split(' | ').map((c) => c.trim())
    if (cells.length < 4) continue
    const loc = cells[0].replace(/`/g, '')
    const m = loc.match(/^(.+\.rs):(\d+)$/)
    if (!m) continue
    const pattern = cells[1].replace(/`/g, '')
    rows.push({ file: m[1], pattern, classification: cells[2] })
  }
  return rows
}

// ── self-test ─────────────────────────────────────────────────────────────

function selfTest(): number {
  const cases: { name: string; src: string; expected: number }[] = [
    { name: 'unwrap call', src: 'fn f() {\n  let x = foo.unwrap();\n}\n', expected: 1 },
    { name: 'expect call', src: 'fn f() {\n  let x = foo.expect("x");\n}\n', expected: 1 },
    { name: 'panic macro', src: 'fn f() { panic!("x"); }\n', expected: 1 },
    { name: 'unreachable macro', src: 'fn f() { unreachable!(); }\n', expected: 1 },
    { name: 'todo macro', src: 'fn f() { todo!() }\n', expected: 1 },
    { name: 'unimplemented macro', src: 'fn f() { unimplemented!() }\n', expected: 1 },
    { name: 'let underscore', src: 'fn f() {\n  let _ = bad();\n}\n', expected: 1 },
    { name: 'ok discard', src: 'fn f() {\n  foo().ok();\n}\n', expected: 1 },
    { name: 'unwrap_or_default', src: 'fn f() {\n  let x = foo.unwrap_or_default();\n}\n', expected: 1 },
    { name: 'unwrap_or value', src: 'fn f() {\n  let x = foo.unwrap_or(0);\n}\n', expected: 1 },
    { name: 'unwrap_or_else', src: 'fn f() {\n  let x = foo.unwrap_or_else(|| 0);\n}\n', expected: 1 },
    { name: 'line comment ignored', src: '// foo.unwrap() is fine\nfn f() {}\n', expected: 0 },
    { name: 'string ignored', src: 'fn f() {\n  let s = "a.unwrap()";\n}\n', expected: 0 },
    { name: 'block comment ignored', src: '/* foo.unwrap() */\nfn f() {}\n', expected: 0 },
    { name: 'raw string ignored', src: 'fn f() {\n  let s = r#"unwrap() here"#;\n}\n', expected: 0 },
    { name: 'cfg(test) module ignored', src: '#[cfg(test)]\nmod tests {\n  fn t() {\n    let x = foo.unwrap();\n  }\n}\nfn g() {\n  let y = bar.expect("e");\n}\n', expected: 1 },
    { name: 'cfg(test) static ignored', src: '#[cfg(test)]\nstatic S: () = ();\nfn g() {\n  let y = bar.unwrap();\n}\n', expected: 1 },
    { name: 'identifier no false positive', src: 'fn unwrap_or_default_fn() {}\n', expected: 0 },
    { name: 'lifetime not char literal', src: "fn f<'a>(x: &'a str) { let _ = x; }\n", expected: 1 },
    { name: 'byte string ignored', src: 'fn f() {\n  let b = b"data.unwrap()";\n}\n', expected: 0 },
  ]
  let failed = 0
  for (const c of cases) {
    const { masked } = maskSource(c.src)
    const testRanges = computeTestRanges(masked)
    const sites = findSites(c.src, '<selftest>', testRanges)
    if (sites.length !== c.expected) {
      failed++
      console.log(`  X self-test '${c.name}': expected ${c.expected}, got ${sites.length} (${sites.map((s) => s.pattern).join(',')})`)
    }
  }
  if (failed) { console.log(`\n${failed} self-test case(s) failed`); return 1 }
  console.log(`OK self-test: ${cases.length} cases passed`)
  return 0
}

// ── main ──────────────────────────────────────────────────────────────────

function main(): number {
  if (SELF_TEST) return selfTest()

  const files = readdirSync(SRC, { withFileTypes: true })
    .filter((e) => e.isFile() && e.name.endsWith('.rs'))
    .map((e) => join(SRC, e.name))
    .sort()

  const allSites: Site[] = []
  const fileText = new Map<string, string>()
  for (const abs of files) {
    const src = readFileSync(abs, 'utf8')
    const rel = relative(ROOT, abs).split(sep).join('/')
    fileText.set(rel, src)
    const { masked } = maskSource(src)
    const testRanges = computeTestRanges(masked)
    allSites.push(...findSites(src, rel, testRanges))
  }
  allSites.sort((a, b) => a.file.localeCompare(b.file) || a.line - b.line)

  console.log('phase 48: rust panic/silent-swallow guard')
  console.log(`  scanned ${files.length} rust files, ${allSites.length} site(s)`)

  if (SHOW_ALL) {
    for (const s of allSites) console.log(`  · ${s.file}:${s.line}  [${s.pattern}]  || ${s.snippet}`)
    console.log(`\n${allSites.length} site(s) total`)
    return 0
  }
  if (SHOW_TSV) {
    for (const s of allSites) console.log(`${s.file}\t${s.line}\t${s.pattern}\t${s.snippet}`)
    return 0
  }

  let failures = 0
  const fail = (msg: string) => { failures++; console.log(`  X ${msg}`) }

  for (const s of allSites) {
    const src = fileText.get(s.file) ?? ''
    const { comments } = maskSource(src)
    const lines = src.split('\n')
    if (!hasJustified(comments, lines, s.line)) {
      fail(`${s.file}:${s.line} — ${s.pattern} has no justifying comment (>=15 chars prose, not "ignore"/"safe"): ${s.snippet}`)
    }
  }

  const docRows = parseDoc()
  const docCount = new Map<string, number>()
  for (const r of docRows) {
    if (r.classification === 'REVIEW') fail(`allow-list contains a REVIEW row: ${r.file} — ${r.pattern}`)
    const key = `${r.file}|${r.pattern}`
    docCount.set(key, (docCount.get(key) ?? 0) + 1)
  }
  const detCount = new Map<string, number>()
  for (const s of allSites) {
    const key = `${s.file}|${s.snippet}`
    detCount.set(key, (detCount.get(key) ?? 0) + 1)
  }
  for (const [key, count] of detCount) {
    const documented = docCount.get(key) ?? 0
    if (documented < count) fail(`${key.replace('|', ' — ')}: ${count} detected, ${documented} documented in SILENT_EXCEPTIONS.md`)
  }
  for (const [key, count] of docCount) {
    const detected = detCount.get(key) ?? 0
    if (detected < count) fail(`stale allow-list row: ${key.replace('|', ' — ')}: ${count} documented, ${detected} detected`)
  }

  for (const s of allSites) {
    const key = `${s.file}|${s.snippet}`
    const cls = docCount.get(key) ? 'EXPECTED' : '-'
    const src = fileText.get(s.file) ?? ''
    const { comments } = maskSource(src)
    const justified = hasJustified(comments, src.split('\n'), s.line) ? 'yes' : 'NO'
    console.log(`  ${s.file}:${s.line} | ${s.pattern} | ${cls} | comment=${justified} | ${s.snippet}`)
  }

  if (failures > 0) {
    console.error(`\n${failures} rust-panic violation(s)`)
    return 1
  }
  console.log('\nOK all sites are commented and documented')
  return 0
}

process.exit(main())
