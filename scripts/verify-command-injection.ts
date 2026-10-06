#!/usr/bin/env tsx
// Phase 44 — command-injection / SSRF hardening of the LLM sidecar.
//
// Proves sidecar-llm/shell-safety.mjs and the three wired-in call sites in
// sidecar-llm/main.mjs against a REAL shell (sh -c printf) and a real temp dir:
//   A. splitArgs → quoteArgv round-trips through `sh` with no token interpreted
//      as an operator/substitution/glob and no SENTINEL file ever created.
//   B. splitArgs rejects hostile/oversized input.
//   C. checkDownloadUrl blocks SSRF targets (loopback, private, metadata,
//      obfuscated IPs) and allows real public dataset URLs.
//   D. checkSshTarget blocks option-like / shell-metachar targets.
//   E. static checks that main.mjs actually uses the helpers.

import { execFileSync } from 'node:child_process'
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import {
  splitArgs,
  quoteArgv,
  checkDownloadUrl,
  checkSshTarget,
  isBlockedAddress,
  safeFetch,
} from '../sidecar-llm/shell-safety.mjs'

interface FetchOptions {
  redirect?: 'manual' | 'follow' | 'error'
  signal?: AbortSignal
}

interface FetchCall {
  u: string
  o: FetchOptions
}

interface FakeResp {
  status: number
  ok: boolean
  headers: { get(name: string): string | null }
}

let failures = 0
function check(name: string, cond: boolean, detail = '') {
  if (cond) console.log(`  ✓ ${name}`)
  else { failures++; console.log(`  ✗ ${name}${detail ? ` — ${detail}` : ''}`) }
}

console.log('phase 44: node sidecar command-injection / SSRF')

// ── A. round trip through a real shell ──────────────────────────────────────
console.log('\n=== A. splitArgs/quoteArgv round-trip through sh ===')
const roundTripCases = [
  // benign
  '--epochs 10 --lr=0.1',
  '"a b" \'c d\'',
  'a\\ b',
  '--seed 0',
  '--name "hello world"',
  'a b c',
  "''",
  '""',
  'one',
  '--lr=0.1',
  '--config agent/config.yaml',
  // hostile
  '; touch SENTINEL',
  '$(touch SENTINEL)',
  '`touch SENTINEL`',
  'a && touch SENTINEL',
  'a | tee SENTINEL',
  '> SENTINEL',
  'a;b',
  '*',
  '~',
  '$HOME',
  '${IFS}',
  '-- --x',
  '--opt="x y"',
  'a < SENTINEL',
  'a >> SENTINEL',
  'SENTINEL;',
  '"touch SENTINEL"',
  '--x;touch SENTINEL',
  '$(id);touch SENTINEL',
]

for (const input of roundTripCases) {
  const parsed = splitArgs(input)
  if (!parsed.ok) { check(`round-trip ${JSON.stringify(input)}`, false, `splitArgs: ${parsed.error}`); continue }
  const argv = parsed.argv
  const quoted = quoteArgv(argv)
  const dir = mkdtempSync(join(tmpdir(), 'spinoml-cmdinj-'))
  let out = ''
  let execErr = ''
  try {
    out = execFileSync('sh', ['-c', 'printf "%s\\0" ' + quoted], { cwd: dir }).toString('utf8')
  } catch (e) { execErr = String(e) }
  const parts = out.split('\0')
  if (parts.length && parts[parts.length - 1] === '') parts.pop()
  const arrEq = parts.length === argv.length && parts.every((p, i) => p === argv[i])
  const sentinel = existsSync(join(dir, 'SENTINEL'))
  rmSync(dir, { recursive: true, force: true })
  const detail = execErr
    ? `exec failed: ${execErr}`
    : !arrEq
      ? `mismatch got ${JSON.stringify(parts)} want ${JSON.stringify(argv)}`
      : sentinel ? 'SENTINEL was created' : ''
  check(`round-trip ${JSON.stringify(input)}`, !execErr && arrEq && !sentinel, detail)
}

// ── B. splitArgs error cases ────────────────────────────────────────────────
console.log('\n=== B. splitArgs rejects bad input ===')
const badArgs: Array<[string, string]> = [
  ['unbalanced single quote', "a 'b"],
  ['unbalanced double quote', 'a "b'],
  ['newline', 'a\nb'],
  ['carriage return', 'a\rb'],
  ['NUL', 'a\u0000b'],
  ['more than 64 args', Array.from({ length: 65 }, (_, i) => `a${i}`).join(' ')],
  ['input longer than 4096', 'a'.repeat(4097)],
]
for (const [label, input] of badArgs) {
  const r = splitArgs(input)
  check(`reject ${label}`, r.ok === false, r.ok ? 'was accepted' : '')
}
{
  const r1 = splitArgs('')
  const r2 = splitArgs(undefined as unknown as string)
  check('empty string -> ok, argv []', r1.ok === true && r1.argv.length === 0)
  check('undefined -> ok, argv []', r2.ok === true && r2.argv.length === 0)
}

// ── C. checkDownloadUrl ─────────────────────────────────────────────────────
console.log('\n=== C. checkDownloadUrl SSRF policy ===')
const allowedUrls = [
  'https://rest.uniprot.org/uniprotkb/P12345.fasta',
  'https://files.rcsb.org/download/1ABC.pdb',
  'https://raw.githubusercontent.com/a/b/c/iris.csv',
  'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
  'https://93.184.216.34/x',
]
for (const u of allowedUrls) {
  const r = checkDownloadUrl(u)
  check(`allow ${u}`, r.ok === true, r.ok ? '' : r.error)
}
const blockedUrls = [
  'http://169.254.169.254/latest/meta-data/',
  'http://127.0.0.1:7421/infer',
  'http://localhost/x',
  'http://[::1]/x',
  'http://2130706433/',
  'http://0x7f.0.0.1/',
  'http://017700000001/',
  'http://127.1/',
  'http://[::ffff:127.0.0.1]/',
  'http://[::ffff:7f00:1]/',
  'http://10.0.0.5/',
  'http://192.168.1.1/',
  'http://172.16.0.1/',
  'http://100.64.0.1/',
  'http://metadata.internal/',
  'http://foo.local/',
  'https://user:pw@example.com/',
  'ftp://example.com/x',
  'file:///etc/passwd',
  'javascript:alert(1)',
  'https://example.com:22/',
  'https://example.com:7421/',
  '',
  'not a url',
]
for (const u of blockedUrls) {
  const r = checkDownloadUrl(u)
  check(`block ${JSON.stringify(u)}`, r.ok === false, r.ok ? `was allowed -> ${r.url}` : '')
}

// ── D. checkSshTarget ───────────────────────────────────────────────────────
console.log('\n=== D. checkSshTarget policy ===')
const goodTargets = ['leipzig-hpc', 'zw93onug@login01.sc.uni-leipzig.de']
for (const t of goodTargets) {
  const r = checkSshTarget(t)
  check(`accept ${JSON.stringify(t)}`, r.ok === true, r.ok ? '' : r.error)
}
const badTargets: Array<[string, string]> = [
  ['option-like -oProxyCommand=x', '-oProxyCommand=x'],
  ['option-like -J', '-J'],
  ['space', 'a b'],
  ['semicolon', 'a;b'],
  ['command substitution', '$(x)'],
  ['empty', ''],
  ['129 chars', 'a'.repeat(129)],
  ['newline', 'a\nb'],
]
for (const [label, t] of badTargets) {
  const r = checkSshTarget(t)
  check(`reject ${label}`, r.ok === false, r.ok ? 'was accepted' : '')
}

// ── E. static checks on main.mjs ────────────────────────────────────────────
console.log('\n=== E. main.mjs wiring ===')
const mainPath = join(process.cwd(), 'sidecar-llm', 'main.mjs')
const src = readFileSync(mainPath, 'utf8')
check("imports ./shell-safety.mjs", src.includes("from './shell-safety.mjs'"))

check('old `argStr = args ?` concatenation is gone', !src.includes('argStr = args ?'))

{
  const idx = src.indexOf("spawn('ssh'")
  const window = idx === -1 ? '' : src.slice(Math.max(0, idx - 700), idx + 200)
  check("spawn('ssh' is preceded by checkSshTarget", idx !== -1 && window.includes('checkSshTarget'))
  check("spawn('ssh' args contain '--' before target", idx !== -1 && window.includes("'--'") && window.includes('target'))
}

{
  const start = src.indexOf('downloadToDatasets')
  const end = src.indexOf('writeDatasetFile', start)
  const body = start === -1 ? '' : src.slice(start, end === -1 ? start + 4000 : end)
  check('downloadToDatasets calls checkDownloadUrl', body.includes('checkDownloadUrl'))
  check('downloadToDatasets adds --proto flags', body.includes('--proto'))
  check('downloadToDatasets uses safeFetch', body.includes('safeFetch'))
  check('downloadToDatasets no longer uses redirect:follow', !body.includes("redirect: 'follow'"))
}

// ── E2. runScript: a workspace-relative name can never be parsed as an option ─
{
  const src = readFileSync(join(process.cwd(), 'sidecar-llm', 'main.mjs'), 'utf8')
  const start = src.indexOf('async function runScript(')
  const end = src.indexOf('async function slurmStatus(', start)
  const body = start === -1 ? '' : src.slice(start, end === -1 ? start + 3000 : end)
  check("runScript prefixes the script with './' before quoting", body.includes("shellQuotePath(`./${safe}`)"))
  check('runScript sbatch call uses the ./-prefixed target', body.includes('sbatch ${target}'))
  check('runScript interpreter call uses the ./-prefixed target', body.includes('${interp}${target}'))
  check('runScript no longer quotes the bare relpath into a command', !body.includes('shellQuotePath(safe)'))
}

// ── F. isBlockedAddress table ────────────────────────────────────────────────
console.log('\n=== F. isBlockedAddress table ===')
const allowedIps = [
  '93.184.216.34', '8.8.8.8', '1.1.1.1', '140.82.121.4', '151.101.0.81',
  '2606:4700:4700::1111', '2001:4860:4860::8888', '2620:0:861:ed1a::1',
  '::ffff:8.8.8.8', '64:ff9b::808:808',
]
const blockedIps = [
  '127.0.0.1', '169.254.169.254', '10.1.2.3', '172.16.0.1', '192.168.1.1',
  '100.64.0.1', '224.0.0.1', '240.0.0.1', '198.18.0.1', '192.0.0.1',
  '0.0.0.0',
  '::1', '::ffff:7f00:1',
  '::ffff:127.0.0.1', '::ffff:10.0.0.1',
  '64:ff9b::7f00:1', '64:ff9b::a00:1', '64:ff9b:1::1', '64:ff9b::127.0.0.1',
  '2002:7f00:1::', '2002:a9fe:a9fe::1',
  '2001:0:4136:e378:8000:63bf:3fff:fdd2',
  '2001:db8::1',
  'fe80::1', 'fd00::1', 'ff02::1',
]
const invalidIps: Array<string | null | undefined | number> = [
  '',
  '127.1',
  '0x7f000001',
  '2130706433',
  '[::1]',
  '1.2.3',
  '1.2.3.4.5',
  '256.1.1.1',
  'abc',
  '1.1.1.1 ',
  '::1%eth0',
  '00.0.0.0',
  '01.2.3.4',
  '::g',
  'not an ip',
  null,
  undefined,
  42,
  {} as unknown as string,
]
for (const ip of allowedIps) {
  check(`isBlockedAddress allow ${ip}`, isBlockedAddress(ip) === false, `got ${isBlockedAddress(ip)}`)
}
for (const ip of blockedIps) {
  check(`isBlockedAddress block ${ip}`, isBlockedAddress(ip) === true, `got ${isBlockedAddress(ip)}`)
}
for (const ip of invalidIps) {
  check(`isBlockedAddress invalid/block ${JSON.stringify(ip)}`, isBlockedAddress(ip as unknown as string) === true, `got ${isBlockedAddress(ip as unknown as string)}`)
}

// ── G. safeFetch ─────────────────────────────────────────────────────────────
console.log('\n=== G. safeFetch SSRF-resolving fetch ===')
function makeResp({ status = 200, location = null }: { status?: number; location?: string | null } = {}): FakeResp {
  return {
    status,
    ok: status >= 200 && status < 300,
    headers: { get: (n: string) => (n && n.toLowerCase() === 'location' ? location : null) },
  }
}

// (a) hostname resolves to private -> reject; fetchImpl never called
{
  const calls: FetchCall[] = []
  const fetchImpl = async (u: string, o: FetchOptions) => { calls.push({ u, o }); return makeResp({ status: 200 }) }
  const lookup = async () => [{ address: '10.0.0.5', family: 4 }]
  let err: Error | null = null
  try { await safeFetch('http://example.com/', { lookup, fetchImpl }) }
  catch (e) { err = e as Error }
  check('safeFetch (a) hostname->private rejects', !!err && /blocked address/.test(err.message), err?.message ?? 'no error')
  check('safeFetch (a) fetchImpl never called', calls.length === 0, `called ${calls.length}`)
}

// (b) two resolved addresses, second private -> reject
{
  const calls: FetchCall[] = []
  const fetchImpl = async (u: string, o: FetchOptions) => { calls.push({ u, o }); return makeResp({ status: 200 }) }
  const lookup = async () => [
    { address: '93.184.216.34', family: 4 },
    { address: '10.0.0.5', family: 4 },
  ]
  let err: Error | null = null
  try { await safeFetch('http://example.com/', { lookup, fetchImpl }) }
  catch (e) { err = e as Error }
  check('safeFetch (b) second-private address rejects', !!err && /blocked address/.test(err.message), err?.message ?? 'no error')
  check('safeFetch (b) fetchImpl never called', calls.length === 0, `called ${calls.length}`)
}

// (c) public host 302 -> http://127.0.0.1/x -> reject on second hop
{
  const calls: FetchCall[] = []
  const fetchImpl = async (u: string, o: FetchOptions) => {
    calls.push({ u, o })
    return makeResp({ status: 302, location: 'http://127.0.0.1/x' })
  }
  const lookup = async () => [{ address: '93.184.216.34', family: 4 }]
  let err: Error | null = null
  try { await safeFetch('http://example.com/', { lookup, fetchImpl }) }
  catch (e) { err = e as Error }
  check('safeFetch (c) redirect to loopback rejects', !!err && /blocked address/.test(err.message), err?.message ?? 'no error')
  check('safeFetch (c) fetchImpl called once', calls.length === 1, `called ${calls.length}`)
}

// (d) redirect to hostname resolving to 192.168.1.9 -> reject
{
  const calls: FetchCall[] = []
  const fetchImpl = async (u: string, o: FetchOptions) => {
    calls.push({ u, o })
    return makeResp({ status: 302, location: 'http://internal.example/x' })
  }
  const lookup = async (host: string) => {
    if (host === 'example.com') return [{ address: '93.184.216.34', family: 4 }]
    if (host === 'internal.example') return [{ address: '192.168.1.9', family: 4 }]
    return []
  }
  let err: Error | null = null
  try { await safeFetch('http://example.com/', { lookup, fetchImpl }) }
  catch (e) { err = e as Error }
  check('safeFetch (d) redirect to private hostname rejects', !!err && /blocked address/.test(err.message), err?.message ?? 'no error')
  check('safeFetch (d) fetchImpl called once', calls.length === 1, `called ${calls.length}`)
}

// (e) 6 chained redirects -> throws "too many redirects"
{
  const calls: FetchCall[] = []
  const fetchImpl = async (u: string, o: FetchOptions) => {
    calls.push({ u, o })
    return makeResp({ status: 302, location: '/x' })
  }
  const lookup = async () => [{ address: '93.184.216.34', family: 4 }]
  let err: Error | null = null
  try { await safeFetch('http://example.com/', { lookup, fetchImpl }) }
  catch (e) { err = e as Error }
  check('safeFetch (e) 6 redirects throws too many', !!err && /too many redirects/.test(err.message), err?.message ?? 'no error')
  check('safeFetch (e) fetchImpl called 6 times', calls.length === 6, `called ${calls.length}`)
}

// (f) 302 to relative /other on same public host -> followed, final 200
{
  const calls: FetchCall[] = []
  let n = 0
  const fetchImpl = async (u: string, o: FetchOptions) => {
    calls.push({ u, o })
    n++
    if (n === 1) return makeResp({ status: 302, location: '/other' })
    return makeResp({ status: 200 })
  }
  const lookup = async () => [{ address: '93.184.216.34', family: 4 }]
  let resp: FakeResp | null = null
  let err: Error | null = null
  try { resp = await safeFetch('http://example.com/', { lookup, fetchImpl }) }
  catch (e) { err = e as Error }
  check('safeFetch (f) relative redirect followed + 200', !!resp && resp.status === 200, err?.message ?? `status=${resp?.status}`)
  check('safeFetch (f) fetchImpl called twice', calls.length === 2, `called ${calls.length}`)
  check('safeFetch (f) second URL resolved absolutely', calls[1]?.u === 'http://example.com/other', `second=${calls[1]?.u}`)
}

// (g) lookup failure -> reject
{
  const calls: FetchCall[] = []
  const fetchImpl = async (u: string, o: FetchOptions) => { calls.push({ u, o }); return makeResp({ status: 200 }) }
  const lookup = async () => { throw new Error('ENOTFOUND') }
  let err: Error | null = null
  try { await safeFetch('http://example.com/', { lookup, fetchImpl }) }
  catch (e) { err = e as Error }
  check('safeFetch (g) lookup failure rejects', !!err && /DNS lookup failed/.test(err.message), err?.message ?? 'no error')
  check('safeFetch (g) fetchImpl never called', calls.length === 0, `called ${calls.length}`)
}

// (h) fetchImpl is always called with redirect:'manual'
{
  const calls: FetchCall[] = []
  let n = 0
  const fetchImpl = async (u: string, o: FetchOptions) => {
    calls.push({ u, o })
    n++
    if (n === 1) return makeResp({ status: 302, location: '/x' })
    return makeResp({ status: 200 })
  }
  const lookup = async () => [{ address: '93.184.216.34', family: 4 }]
  try { await safeFetch('http://example.com/', { lookup, fetchImpl }) } catch { /* ignore */ }
  const allManual = calls.length > 0 && calls.every((c) => c.o && c.o.redirect === 'manual')
  check('safeFetch (h) fetchImpl always redirect:manual', allManual, `calls=${calls.length}`)
}

if (failures > 0) {
  console.error(`\n${failures} check(s) failed`)
  process.exit(1)
} else {
  console.log('\n✓ all command-injection / SSRF checks passed')
}
