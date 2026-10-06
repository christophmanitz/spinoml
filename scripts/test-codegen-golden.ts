#!/usr/bin/env tsx
// Phase 6 (TODO §7) — codegen golden tests.
//
// Pin the EXACT text every generator emits so a silent change of generated code
// (an eps, an init order, a default) shows up as an explicit, reviewable diff.
// Fixtures + expected sources live under scripts/golden/ (committed). A normal
// run loads every *.graph.json, runs the REAL generator, and compares byte-exact.
//
//   npm run test:codegen-golden              verify (exit 1 on any mismatch)
//   npm run test:codegen-golden -- --update  rewrite expected files + fixtures
//   npm run test:codegen-golden -- --list    list cases
//
// Per case it additionally checks: (a) byte equality, (b) determinism (twice in
// this process AND once in a fresh child process), (c) the Python parses,
// (d) no volatile content (timestamp / absolute path / 0x address / uuid),
// (e) model cases run one forward pass and match the fixture's output shape
// (PyG cases are SKIPPED only when torch_geometric is not importable). A coverage
// guard then proves every registry entry is covered or explicitly excluded.
import { execFileSync, spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type { Edge } from '@xyflow/react'
import { generate } from '../src/codegen/generator'
import type { LayerNode } from '../src/canvas/GraphStore'
import { LAYERS } from '../src/layers/registry'
import { TRAINING_NODES } from '../src/training/graph/registry'
import { DATA_NODES } from '../src/data/graph/registry'
import { compileTrainingGraph } from '../src/codegen/trainingGenerator'
import { generateTrainingCode } from '../src/codegen/trainingCodegen'
import { compileDataGraph } from '../src/codegen/dataGenerator'
import { generateDataCode } from '../src/codegen/dataCodegen'
import { buildGoldenCases, serializeFixture, type GoldenInput, type GoldenKind } from './lib/golden-cases'

const ROOT = process.cwd()
const GOLDEN_DIR = join(ROOT, 'scripts', 'golden')
const PYTHON = process.env.PYTHON ?? 'python'
const TSX = join(ROOT, 'node_modules', '.bin', 'tsx')

type FixtureNode = {
  id: string
  layerType?: string
  trainingType?: string
  dataType?: string
  params: Record<string, unknown>
}
type Fixture = {
  kind: GoldenKind
  nodes: FixtureNode[]
  edges: { source: string; target: string }[]
  inputs?: GoldenInput[]
  expectedOutputShape?: number[] | Record<string, number[]>
  requires?: string[]
}
type LoadedCase = { name: string; fixture: Fixture; graphPath: string; expectedPath: string }

// ── generation ──────────────────────────────────────────────────────────────

function fixtureToGenerated(fx: Fixture, caseName: string): string {
  if (fx.kind === 'model') {
    const nodes: LayerNode[] = fx.nodes.map((n) => ({
      id: n.id,
      type: 'layer',
      position: { x: 0, y: 0 },
      data: { layerType: n.layerType ?? '', params: n.params },
    }))
    const edges: Edge[] = fx.edges.map((e, i) => ({ id: `e${i + 1}`, source: e.source, target: e.target }))
    const { code, issues } = generate(nodes, edges)
    if (issues.length) throw new Error(`${caseName}: generator reported issues: ${issues.join(' | ')}`)
    return code
  }
  if (fx.kind === 'training') {
    const snap = {
      nodes: fx.nodes.map((n) => ({ id: n.id, trainingType: n.trainingType ?? '', params: n.params })),
      edges: fx.edges,
    }
    const { ok, issues, plan } = compileTrainingGraph(snap)
    if (!ok || !plan) throw new Error(`${caseName}: training compile not ok: ${issues.join(' | ')}`)
    return generateTrainingCode(plan)
  }
  const snap = {
    nodes: fx.nodes.map((n) => ({ id: n.id, dataType: n.dataType ?? '', params: n.params })),
    edges: fx.edges,
  }
  const { ok, issues, plan } = compileDataGraph(snap)
  if (!ok || !plan) throw new Error(`${caseName}: data compile not ok: ${issues.join(' | ')}`)
  return generateDataCode(plan)
}

// ── loading ─────────────────────────────────────────────────────────────────

function loadCases(): LoadedCase[] {
  if (!existsSync(GOLDEN_DIR)) throw new Error(`missing ${GOLDEN_DIR} — run with --update`)
  const out: LoadedCase[] = []
  for (const file of readdirSync(GOLDEN_DIR).sort()) {
    if (!file.endsWith('.graph.json')) continue
    const name = file.slice(0, -'.graph.json'.length)
    const graphPath = join(GOLDEN_DIR, file)
    const expectedPath = join(GOLDEN_DIR, `${name}.expected.py`)
    const fixture = JSON.parse(readFileSync(graphPath, 'utf8')) as Fixture
    out.push({ name, fixture, graphPath, expectedPath })
  }
  return out
}

function emitAll(): void {
  const map: Record<string, string> = {}
  for (const c of loadCases()) map[c.name] = fixtureToGenerated(c.fixture, c.name)
  process.stdout.write(JSON.stringify(map))
}

// ── checks ──────────────────────────────────────────────────────────────────

const VOLATILE: { label: string; re: RegExp }[] = [
  { label: 'timestamp', re: /\d{4}-\d{2}-\d{2}/ },
  { label: 'absolute path', re: /(^|[\s'"=(,])\/(?:home|Users|tmp|var|opt|usr|root|scratch|private|mnt|data)\// },
  { label: 'memory address', re: /0x[0-9a-fA-F]+/ },
  { label: 'uuid', re: /[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}/ },
]

function volatileHits(text: string): string[] {
  const hits: string[] = []
  const lines = text.split('\n')
  for (let i = 0; i < lines.length; i++) {
    for (const v of VOLATILE) {
      if (v.re.test(lines[i])) hits.push(`${v.label} @ line ${i + 1}: ${lines[i].trim().slice(0, 120)}`)
    }
  }
  return hits
}

const FORWARD_HARNESS: string = [
  'import sys, json, importlib.util, inspect, torch',
  '',
  'def shape_of(x):',
  '    return [int(s) for s in x.shape]',
  '',
  'def build_arg(name, d):',
  '    if d.get("isGraph"):',
  '        from torch_geometric.data import Data',
  '        n, f = int(d["shape"][0]), int(d["shape"][1])',
  '        e = int(d.get("nEdges", 0))',
  '        parts = dict(x=torch.ones((n, f)), edge_index=torch.zeros((2, e), dtype=torch.long), batch=torch.zeros((n,), dtype=torch.long))',
  '        if int(d.get("edgeDim", 0)) > 0:',
  '            parts["edge_attr"] = torch.ones((e, int(d["edgeDim"])))',
  '        return Data(**parts)',
  '    if name == "batch":',
  '        # PyG graph membership: all-zeros = ONE graph (ones would mean two).',
  '        return torch.zeros(tuple(int(x) for x in d["shape"]), dtype=torch.long)',
  '    t = torch.ones(tuple(int(x) for x in d["shape"]))',
  '    if d.get("dtype") == "int64":',
  '        t = t.long()',
  '    return t',
  '',
  'def run_one(entry):',
  '    name = entry["name"]',
  '    meta = json.load(open(entry["meta"]))',
  '    spec = importlib.util.spec_from_file_location("golden_" + name.replace("-", "_"), entry["py"])',
  '    mod = importlib.util.module_from_spec(spec)',
  '    spec.loader.exec_module(mod)',
  '    model = mod.Model()',
  '    model.eval()',
  '    argnames = [p.name for p in inspect.signature(model.forward).parameters.values() if p.name != "self"]',
  '    by_name = {i["name"]: i for i in meta.get("inputs", [])}',
  '    args = []',
  '    for a in argnames:',
  '        if a not in by_name:',
  '            raise RuntimeError("no fixture input for forward arg " + a)',
  '        args.append(build_arg(a, by_name[a]))',
  '    with torch.no_grad():',
  '        out = model(*args)',
  '    exp = meta.get("expectedOutputShape")',
  '    if isinstance(exp, dict):',
  '        if not isinstance(out, dict):',
  '            raise RuntimeError("expected dict output, got " + type(out).__name__)',
  '        if set(out.keys()) != set(exp.keys()):',
  '            raise RuntimeError("dict keys %r != %r" % (sorted(out.keys()), sorted(exp.keys())))',
  '        for k in exp:',
  '            got = shape_of(out[k])',
  '            if got != list(exp[k]):',
  '                raise RuntimeError("shape[%s] %r != %r" % (k, got, list(exp[k])))',
  '    else:',
  '        got = shape_of(out)',
  '        if got != list(exp):',
  '            raise RuntimeError("shape %r != %r" % (got, list(exp)))',
  '',
  'def main():',
  '    torch_geometric_ok = True',
  '    try:',
  '        import torch_geometric  # noqa: F401',
  '    except Exception:',
  '        torch_geometric_ok = False',
  '    entries = json.load(open(sys.argv[1]))',
  '    for entry in entries:',
  '        name = entry["name"]',
  '        try:',
  '            if entry.get("requiresGeo") and not torch_geometric_ok:',
  '                print("CASE %s SKIP torch_geometric not importable" % name)',
  '                continue',
  '            run_one(entry)',
  '            print("CASE %s OK" % name)',
  '        except Exception as e:',
  '            print("CASE %s FAIL %s: %s" % (name, type(e).__name__, str(e).splitlines()[0][:300]))',
  '',
  'main()',
].join('\n')

function runForwardChecks(cases: LoadedCase[], gen: Map<string, string>, tmp: string): Map<string, { status: string; detail: string }> {
  const modelCases = cases.filter((c) => c.fixture.kind === 'model' && c.fixture.inputs && c.fixture.expectedOutputShape !== undefined)
  const entries: { name: string; py: string; meta: string; requiresGeo: boolean }[] = []
  for (const c of modelCases) {
    const pyPath = join(tmp, `${c.name}.py`)
    writeFileSync(pyPath, gen.get(c.name) ?? '', 'utf8')
    entries.push({
      name: c.name,
      py: pyPath,
      meta: c.graphPath,
      requiresGeo: (c.fixture.requires ?? []).includes('torch_geometric'),
    })
  }
  const harnessPath = join(tmp, 'forward_harness.py')
  writeFileSync(harnessPath, FORWARD_HARNESS, 'utf8')
  const entriesPath = join(tmp, 'forward_entries.json')
  writeFileSync(entriesPath, JSON.stringify(entries, null, 2), 'utf8')
  const res = new Map<string, { status: string; detail: string }>()
  let out: string
  try {
    out = execFileSync(PYTHON, [harnessPath, entriesPath], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024, timeout: 600_000 })
  } catch (e) {
    const err = e as { stdout?: string; stderr?: string }
    out = (err.stdout ?? '') + '\n' + (err.stderr ?? '')
  }
  for (const line of out.split('\n')) {
    const m = /^CASE (\S+) (OK|SKIP|FAIL)(?: (.*))?$/.exec(line)
    if (m) res.set(m[1], { status: m[2], detail: m[3] ?? '' })
  }
  for (const c of modelCases) if (!res.has(c.name)) res.set(c.name, { status: 'FAIL', detail: 'no harness result (python did not run?)' })
  return res
}

// ── coverage guard ──────────────────────────────────────────────────────────

const EXCLUDED: Record<GoldenKind, Record<string, string>> = {
  model: {},
  training: {},
  data: {},
}

function registryTypeKey(kind: GoldenKind): 'layerType' | 'trainingType' | 'dataType' {
  return kind === 'model' ? 'layerType' : kind === 'training' ? 'trainingType' : 'dataType'
}

function coverageGuard(cases: LoadedCase[]): { ok: boolean; lines: string[]; missing: string[] } {
  const used: Record<GoldenKind, Set<string>> = { model: new Set(), training: new Set(), data: new Set() }
  for (const c of cases) {
    const key = registryTypeKey(c.fixture.kind)
    for (const n of c.fixture.nodes) {
      const t = n[key]
      if (t) used[c.fixture.kind].add(t)
    }
  }
  const registries: Record<GoldenKind, Record<string, unknown>> = { model: LAYERS, training: TRAINING_NODES, data: DATA_NODES }
  const lines: string[] = []
  const missing: string[] = []
  for (const kind of ['model', 'training', 'data'] as GoldenKind[]) {
    const entries = Object.keys(registries[kind])
    const covered = entries.filter((t) => used[kind].has(t))
    const excluded = entries.filter((t) => !used[kind].has(t) && t in EXCLUDED[kind])
    const neither = entries.filter((t) => !used[kind].has(t) && !(t in EXCLUDED[kind]))
    for (const t of excluded) lines.push(`  ${kind}: EXCLUDED ${t} — ${EXCLUDED[kind][t]}`)
    for (const t of neither) missing.push(`${kind}.${t}`)
    lines.push(`  ${kind}: covered ${covered.length} / ${entries.length} (excluded ${excluded.length})`)
  }
  return { ok: missing.length === 0, lines, missing }
}

// ── update / list ───────────────────────────────────────────────────────────

function doUpdate(): void {
  mkdirSync(GOLDEN_DIR, { recursive: true })
  const cases = buildGoldenCases()
  let changed = 0
  for (const c of cases) {
    const name = c.name
    const graph = serializeFixture(c)
    const expected = fixtureToGenerated(JSON.parse(graph) as Fixture, name)
    const graphPath = join(GOLDEN_DIR, `${name}.graph.json`)
    const expectedPath = join(GOLDEN_DIR, `${name}.expected.py`)
    for (const [path, content] of [[graphPath, graph], [expectedPath, expected]] as [string, string][]) {
      const old = existsSync(path) ? readFileSync(path, 'utf8') : null
      if (old !== content) {
        writeFileSync(path, content, 'utf8')
        process.stdout.write(`  wrote ${path.slice(ROOT.length + 1)}\n`)
        changed++
      }
    }
  }
  process.stdout.write(`--update: ${changed} file(s) changed across ${cases.length} cases\n`)
}

// ── main ────────────────────────────────────────────────────────────────────

function listCases(cases: LoadedCase[]): void {
  for (const c of cases) process.stdout.write(`${c.name}\t${c.fixture.kind}\n`)
}

function main(): void {
  const args = process.argv.slice(2)
  if (args.includes('--emit-all')) {
    emitAll()
    return
  }
  if (args.includes('--list')) {
    listCases(loadCases())
    return
  }
  if (args.includes('--update')) doUpdate()

  const cases = loadCases()
  if (cases.length === 0) {
    process.stdout.write('✗ no fixtures found\n')
    process.exit(1)
  }

  let failed = 0

  // (b) fresh child process determinism.
  let childCodes: Record<string, string> = {}
  const childArgs = [join(ROOT, 'scripts', 'test-codegen-golden.ts'), '--emit-all']
  const child = existsSync(TSX)
    ? spawnSync(TSX, childArgs, { cwd: ROOT, encoding: 'utf8', maxBuffer: 128 * 1024 * 1024, timeout: 300_000 })
    : spawnSync('npx', ['tsx', ...childArgs], { cwd: ROOT, encoding: 'utf8', maxBuffer: 128 * 1024 * 1024, timeout: 300_000 })
  if (child.status === 0 && child.stdout) {
    try {
      childCodes = JSON.parse(child.stdout) as Record<string, string>
    } catch {
      childCodes = {}
    }
  } else {
    process.stdout.write(`  ✗ determinism: fresh child process failed (status ${child.status}): ${(child.stderr ?? '').slice(0, 200)}\n`)
    failed++
  }

  const tmp = mkdtempSync(join(tmpdir(), 'spinoml-golden-'))
  const gen = new Map<string, string>()
  const rows: string[] = []

  for (const c of cases) {
    const problems: string[] = []
    const code1 = fixtureToGenerated(c.fixture, c.name)
    gen.set(c.name, code1)
    const code2 = fixtureToGenerated(c.fixture, c.name)

    // (a) byte equality
    const expected = existsSync(c.expectedPath) ? readFileSync(c.expectedPath, 'utf8') : null
    if (expected === null) {
      problems.push('missing .expected.py')
    } else if (expected !== code1) {
      problems.push(`byte mismatch vs ${c.name}.expected.py (run --update to review)`)
      const a = expected.split('\n')
      const b = code1.split('\n')
      const diff: string[] = []
      const max = Math.max(a.length, b.length)
      for (let i = 0, shown = 0; i < max && shown < 60; i++) {
        if (a[i] !== b[i]) {
          diff.push(`@@ line ${i + 1}`)
          if (a[i] !== undefined) diff.push(`- ${a[i]}`)
          if (b[i] !== undefined) diff.push(`+ ${b[i]}`)
          shown++
        }
      }
      problems.push(diff.join('\n'))
    }

    // (b) determinism
    if (code1 !== code2) problems.push('non-deterministic within process')
    if (childCodes[c.name] !== undefined && childCodes[c.name] !== code1) problems.push('differs in a fresh child process')

    // (c) parses + (e) forward
    const pyPath = join(tmp, `${c.name}.py`)
    writeFileSync(pyPath, code1, 'utf8')
    try {
      execFileSync(PYTHON, ['-c', 'import ast,sys; ast.parse(open(sys.argv[1]).read())', pyPath], { stdio: 'pipe' })
    } catch (e) {
      const err = e as { stderr?: string }
      problems.push(`python ast.parse failed: ${(err.stderr ?? '').toString().split('\n').slice(0, 3).join(' ')}`)
    }

    // (d) volatile content
    const hits = volatileHits(code1)
    for (const h of hits) problems.push(`volatile content: ${h}`)

    rows.push(`${problems.length ? '✗' : '✓'} ${c.name} (${c.fixture.kind})`)
    if (problems.length) {
      failed++
      for (const p of problems) process.stdout.write(`  ✗ ${c.name}: ${p}\n`)
    }
  }

  // (e) forward pass, batched in one subprocess.
  const forward = runForwardChecks(cases, gen, tmp)
  for (const [name, r] of forward) {
    if (r.status === 'SKIP') {
      process.stdout.write(`  SKIPPED: ${name} — ${r.detail}\n`)
    } else if (r.status !== 'OK') {
      process.stdout.write(`  ✗ ${name}: forward check FAIL ${r.detail}\n`)
      failed++
    }
  }

  rmSync(tmp, { recursive: true, force: true })

  process.stdout.write('\nper-case:\n')
  for (const r of rows) process.stdout.write(`  ${r}\n`)

  const cov = coverageGuard(cases)
  process.stdout.write('\ncoverage:\n')
  for (const l of cov.lines) process.stdout.write(`${l}\n`)
  if (!cov.ok) {
    failed++
    process.stdout.write(`✗ coverage: add a golden fixture or exclude it: ${cov.missing.join(', ')}\n`)
  } else {
    process.stdout.write('  coverage guard: every registry entry is covered or explicitly excluded\n')
  }

  process.stdout.write(`\n${failed === 0 ? '✓' : '✗'} codegen golden: ${cases.length} case(s), ${failed} failure(s)\n`)
  process.exit(failed === 0 ? 0 : 1)
}

main()
