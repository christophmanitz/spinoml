#!/usr/bin/env tsx
// Phase 50 — silent-exception guard.
//
// Stops new "swallowing" error handlers from appearing without a written
// justification. A handler is a swallow when it is a `catch` block whose body
// has no statement other than comments (or only `return` / `return <literal>`),
// or a `.catch(<arrow with empty/literal body>)`.
//
// Every such site must carry a comment of >= 15 characters of prose that is
// not merely "ignore"/"noop"/... AND must be listed in
// docs/engineering/SILENT_EXCEPTIONS.md (the allow-list). The document is
// matched by file + pattern text, so line-number drift is tolerated.
//
// Usage:
//   tsx scripts/verify-silent-catch.ts          # guard (exit 1 on violation)
//   tsx scripts/verify-silent-catch.ts --all     # list every detected site

import { readFileSync, readdirSync, existsSync } from 'node:fs'
import { join, relative, sep } from 'node:path'
import * as ts from 'typescript'

const ROOT = process.cwd()
const SRC = join(ROOT, 'src')
const DOC = join(ROOT, 'docs', 'engineering', 'SILENT_EXCEPTIONS.md')
const SHOW_ALL = process.argv.includes('--all')

type Site = {
  file: string
  line: number
  pattern: string
  comments: string[]
  snippet: string
}

const IGNORE_RE = /^\s*(ignore|ignored|noop|todo|none)\s*$/i

let failures = 0
function fail(msg: string) {
  failures++
  console.log(`  ✗ ${msg}`)
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

const isLiteral = (n: ts.Node): boolean => {
  if (ts.isParenthesizedExpression(n)) return isLiteral(n.expression)
  if (n.kind === ts.SyntaxKind.NullKeyword) return true
  if (ts.isNumericLiteral(n) || ts.isStringLiteral(n) || ts.isNoSubstitutionTemplateLiteral(n)) return true
  if (n.kind === ts.SyntaxKind.UndefinedKeyword || n.kind === ts.SyntaxKind.TrueKeyword || n.kind === ts.SyntaxKind.FalseKeyword) return true
  if (ts.isIdentifier(n) && n.text === 'undefined') return true
  if (ts.isArrayLiteralExpression(n) && n.elements.length === 0) return true
  if (ts.isObjectLiteralExpression(n) && n.properties.length === 0) return true
  return false
}

function commentsIn(content: string): string[] {
  const out: string[] = []
  const re = /\/\/[^\n]*|\/\*[\s\S]*?\*\//g
  let m: RegExpExecArray | null
  while ((m = re.exec(content))) out.push(m[0])
  return out
}

function cleanComment(c: string): string {
  return c.replace(/^\/\//, '').replace(/^\/\*/, '').replace(/\*\/$/, '').trim()
}

function hasJustifiedComment(comments: string[]): boolean {
  return comments.some((c) => {
    const text = cleanComment(c)
    return text.length >= 15 && !IGNORE_RE.test(text)
  })
}

function analyzeBlockBody(block: ts.Block): { swallow: boolean; pattern: string } {
  const stmts = block.statements
  if (stmts.length === 0) return { swallow: true, pattern: '{}' }
  if (stmts.length === 1 && ts.isReturnStatement(stmts[0])) {
    const r = stmts[0]
    if (!r.expression) return { swallow: true, pattern: '{ return }' }
    if (isLiteral(r.expression)) {
      return { swallow: true, pattern: `{ return ${r.expression.getText().replace(/\s+/g, ' ').trim()} }` }
    }
  }
  return { swallow: false, pattern: '' }
}

function gatherComments(node: ts.Node, sf: ts.SourceFile): string[] {
  const full = sf.getFullText()
  const leading = ts.getLeadingCommentRanges(full, node.getFullStart()) ?? []
  const trailing = ts.getTrailingCommentRanges(full, node.getEnd()) ?? []
  const outer = [...leading, ...trailing].map((r) => full.slice(r.pos, r.end))
  const inner = commentsIn(node.getText(sf))
  return [...outer, ...inner]
}

function visit(node: ts.Node, sf: ts.SourceFile, file: string, sites: Site[]) {
  if (ts.isCatchClause(node)) {
    const b = analyzeBlockBody(node.block)
    if (b.swallow) {
      sites.push({
        file,
        line: sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1,
        pattern: `catch ${b.pattern}`,
        comments: gatherComments(node, sf),
        snippet: node.getText(sf).replace(/\s+/g, ' ').slice(0, 100),
      })
    }
  }
  if (ts.isCallExpression(node)) {
    const expr = node.expression
    if (ts.isPropertyAccessExpression(expr) && expr.name.text === 'catch') {
      const arg = node.arguments[0]
      if (arg && (ts.isArrowFunction(arg) || ts.isFunctionExpression(arg))) {
        const body = arg.body
        let swallow = false
        let pattern = ''
        if (ts.isBlock(body)) {
          const b = analyzeBlockBody(body)
          swallow = b.swallow
          pattern = `.catch(() => ${b.pattern})`
        } else if (isLiteral(body)) {
          swallow = true
          pattern = `.catch(() => ${body.getText().replace(/\s+/g, ' ').trim()})`
        }
        if (swallow) {
          sites.push({
            file,
            line: sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1,
            pattern,
            comments: gatherComments(node, sf),
            snippet: node.getText(sf).replace(/\s+/g, ' ').slice(0, 100),
          })
        }
      }
    }
  }
  ts.forEachChild(node, (c) => visit(c, sf, file, sites))
}

// ── collect ──────────────────────────────────────────────────────────────

const sites: Site[] = []
for (const abs of listSourceFiles(SRC)) {
  const src = readFileSync(abs, 'utf8')
  const kind = abs.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS
  const sf = ts.createSourceFile(abs, src, ts.ScriptTarget.Latest, true, kind)
  visit(sf, sf, rel(abs), sites)
}
sites.sort((a, b) => a.file.localeCompare(b.file) || a.line - b.line)

// ── allow-list from the document ──────────────────────────────────────────

type DocRow = { file: string; pattern: string; classification: string }
function parseDoc(): DocRow[] {
  if (!existsSync(DOC)) return []
  const rows: DocRow[] = []
  for (const raw of readFileSync(DOC, 'utf8').split('\n')) {
    const line = raw.trim()
    if (!line.startsWith('|')) continue
    const cells = line.split('|').map((c) => c.trim())
    // cells[0] === '' (leading pipe)
    if (cells.length < 4) continue
    const loc = cells[1].replace(/`/g, '')
    if (!/\.(ts|tsx):\d+$/.test(loc)) continue
    const file = loc.replace(/:\d+$/, '')
    const pattern = cells[2].replace(/`/g, '')
    rows.push({ file, pattern, classification: cells[3] })
  }
  return rows
}

const docRows = parseDoc()
const docCount = new Map<string, number>()
for (const r of docRows) {
  const key = `${r.file}|${r.pattern}`
  docCount.set(key, (docCount.get(key) ?? 0) + 1)
}

// ── report ──────────────────────────────────────────────────────────────

console.log('phase 50: silent-exception guard')
console.log(`  scanned ${listSourceFiles(SRC).length} source files, ${sites.length} swallowing site(s)`)

if (SHOW_ALL) {
  for (const s of sites) {
    const justified = hasJustifiedComment(s.comments) ? 'commented' : 'NO-COMMENT'
    console.log(`  · ${s.file}:${s.line}  [${s.pattern}]  ${justified}  || ${s.snippet}`)
  }
  console.log(`\n${sites.length} swallowing site(s) total`)
  process.exit(0)
}

if (process.argv.includes('--tsv')) {
  // Machine-readable variant used to keep the allow-list in sync.
  for (const s of sites) {
    const justified = hasJustifiedComment(s.comments) ? 'ok' : 'NO'
    console.log(`${s.file}\t${s.line}\t${s.pattern}\t${justified}`)
  }
  process.exit(0)
}

// 1. every swallowing site needs a justifying comment
for (const s of sites) {
  if (!hasJustifiedComment(s.comments)) {
    fail(`${s.file}:${s.line} — ${s.pattern} has no justifying comment (>=15 chars prose, not "ignore")`)
  }
}

// 2. every swallowing site must be documented in the allow-list (counted per
//    file + pattern, so line-number drift is tolerated)
const detectedCount = new Map<string, number>()
for (const s of sites) {
  const key = `${s.file}|${s.pattern}`
  detectedCount.set(key, (detectedCount.get(key) ?? 0) + 1)
}
for (const [key, count] of detectedCount) {
  const documented = docCount.get(key) ?? 0
  if (documented < count) {
    fail(`${key.replace('|', ' — ')}: ${count} detected, ${documented} documented in SILENT_EXCEPTIONS.md`)
  }
}

if (failures > 0) {
  console.error(`\n${failures} silent-catch violation(s)`)
  process.exit(1)
}
console.log('\n✓ all swallowing sites are commented and documented')
