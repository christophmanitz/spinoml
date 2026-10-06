#!/usr/bin/env tsx
// Phase 41 — code-trust core. The "code may only run if its hash is in a local
// trust store that ONLY human-initiated UI events can write" invariant, with
// zero wiring into other stores (a later block wires the UI gate).

import { readdirSync, readFileSync } from 'node:fs'
import { join, relative } from 'node:path'
import {
  collectCodeBlobs,
  collectDataCodeBlobs,
  hashBlob,
  type CodeBlob,
} from '../src/trust/codeBlobs'
import {
  createTrustStore,
  DEFAULT_TRUST_KEY,
  type StorageLike,
  type TrustOrigin,
} from '../src/trust/trustStore'
import {
  buildCodeTrustManifest,
  findUntrusted,
  UNTRUSTED_MESSAGE,
} from '../src/trust/gate'
import { coerceParams } from '../src/layers/registry'
import {
  shouldCommitText,
  shouldFlushCode,
  modalEditedByUser,
} from '../src/inspector/editMeta'

let failures = 0
function check(name: string, cond: boolean, detail = '') {
  if (cond) console.log(`  ✓ ${name}`)
  else { failures++; console.log(`  ✗ ${name}${detail ? '  ' + detail : ''}`) }
}

console.log('phase 41: code trust core')

type ArchNode = { id: string; data: { layerType: string; params: Record<string, unknown> } }
type FlatNode = { id: string; layerType: string; params: Record<string, unknown>; position?: { x: number; y: number } }

function layer(id: string, layerType: string, params: Record<string, unknown> = {}): ArchNode {
  return { id, data: { layerType, params } }
}

/** Real registry stores nested graphs as flat GraphSnapshot-shaped objects
 *  under `params.subgraph = { nodes: [...], edges: [...] }`. */
function flat(id: string, layerType: string, params: Record<string, unknown> = {}): FlatNode {
  return { id, layerType, params }
}

function subgraph(id: string, inner: FlatNode[]): FlatNode {
  return flat(id, 'Subgraph', { class_name: 'S', subgraph: { nodes: inner, edges: [] } })
}

async function main() {

  // ── 1. hashBlob ─────────────────────────────────────────────────────────
  {
    const a1 = await hashBlob('custom-layer', 'class X: pass')
    const a2 = await hashBlob('custom-layer', 'class X: pass')
    check('hashBlob deterministic', a1 === a2)
    const b = await hashBlob('dataop-script', 'class X: pass')
    check('hashBlob kind-separated', a1 !== b)
    const u = await hashBlob('custom-layer', 'café — 日本語\nnewline\temoji 🐍')
    check('hashBlob unicode input', typeof u === 'string' && u.length === 64)
    const e = await hashBlob('custom-layer', '')
    check('hashBlob empty string', typeof e === 'string' && e.length === 64 && /^[0-9a-f]{64}$/.test(e))
    check('hashBlob returns lowercase hex 64', /^[0-9a-f]{64}$/.test(a1) && a1.length === 64)
  }

  // ── 2. collectCodeBlobs ─────────────────────────────────────────────────
  {
    const top = [
      layer('c1', 'Custom', { source: 'class X: pass' }),
      layer('d1', 'DataOp', { script: 'import os\nprint(1)\n' }),
      layer('l1', 'Linear', { in_features: 3, out_features: 2 }),
      layer('cv', 'Conv2d', { in_channels: 3, out_channels: 8 }),
      layer('empty', 'Custom', { source: '   \n  ' }),
      layer('num', 'Custom', { source: 5 }),
      layer('obj', 'Custom', { source: { x: 1 } }),
    ]
    const blobs = collectCodeBlobs(top)
    const byId = new Map(blobs.map((b) => [b.nodeId, b]))
    check('top-level Custom found', byId.has('c1') && byId.get('c1')?.kind === 'custom-layer')
    check('top-level DataOp found', byId.has('d1') && byId.get('d1')?.kind === 'dataop-script')
    check('top-level blob path equals nodeId', byId.get('c1')?.path === 'c1' && byId.get('d1')?.path === 'd1')
    check('built-in Linear yields nothing', !byId.has('l1'))
    check('built-in Conv2d yields nothing', !byId.has('cv'))
    check('whitespace-only Custom yields nothing', !byId.has('empty'))
    check('non-string number source still returned', byId.get('num')?.source === '5')
    check('non-string object source still returned', !!byId.get('obj') && byId.get('obj')!.source.length > 0)
    check('collectCodeBlobs deterministic order',
      blobs.map((b) => b.nodeId).join(',') === blobs.map((b) => b.nodeId).join(','))

    // depth-1
    const d1 = collectCodeBlobs([subgraph('sg1', [flat('c2', 'Custom', { source: 'class Y: pass' })])])
    check('depth-1 nested Custom found', d1.length === 1 && d1[0].nodeId === 'c2' && d1[0].path === 'sg1/c2')

    // depth-2
    const d2 = collectCodeBlobs([
      subgraph('sg1', [subgraph('sg2', [flat('c3', 'Custom', { source: 'class Z: pass' })])]),
    ])
    check('depth-2 nested Custom found', d2.length === 1 && d2[0].path === 'sg1/sg2/c3')

    // depth-3
    const d3 = collectCodeBlobs([
      subgraph('sg1', [
        subgraph('sg2', [
          subgraph('sg3', [flat('c4', 'Custom', { source: 'class Q: pass' })]),
        ]),
      ]),
    ])
    check('depth-3 nested Custom found', d3.length === 1 && d3[0].path === 'sg1/sg2/sg3/c4')

    // nested DataOp also collected
    const dData = collectCodeBlobs([
      subgraph('sg', [flat('d2', 'DataOp', { script: 'print(1)' })]),
    ])
    check('nested DataOp collected as dataop-script',
      dData.length === 1 && dData[0].kind === 'dataop-script' && dData[0].path === 'sg/d2')

    // pathological depth-40 — cannot hide
    let chain: FlatNode = flat('bottom', 'Custom', { source: 'class X: pass' })
    for (let i = 0; i < 40; i++) chain = subgraph('sg' + i, [chain])
    const pathological = collectCodeBlobs([chain])
    check('pathological depth-40 yields at least one blob (cannot hide)', pathological.length >= 1)

    // malformed nodes never throw
    let threw = false
    try { collectCodeBlobs([{ id: 'm1' } as unknown as ArchNode]) } catch (err) { threw = true; void err }
    try { collectCodeBlobs([{ id: 'm2', data: {} } as unknown as ArchNode]) } catch (err) { threw = true; void err }
    try { collectCodeBlobs([]) } catch (err) { threw = true; void err }
    check('malformed / missing data/params does not throw', !threw)
  }

  // ── 2b. Custom init_args (also executable: codegen emits Cls(args)) ──────
  {
    const payload = 'exec(bytes([112,114,105,110,116,40,49,41]).decode())'
    const withArgs = collectCodeBlobs([
      layer('n1', 'Custom', { source: 'class X: pass', init_args: payload }),
    ])
    check('Custom source+init_args yields 2 blobs', withArgs.length === 2)
    check('init_args blob directly after source (deterministic order)',
      withArgs[0].kind === 'custom-layer' && withArgs[1].kind === 'custom-init-args')
    check('init_args blob path has #init_args suffix',
      withArgs[0].path === 'n1' && withArgs[1].path === 'n1#init_args')
    check('init_args blob carries payload verbatim', withArgs[1].source === payload)

    // domain separation by kind
    const hLayer = await hashBlob('custom-layer', 'identical text')
    const hArgs = await hashBlob('custom-init-args', 'identical text')
    check('same text hashes differently across kinds (custom-layer vs custom-init-args)', hLayer !== hArgs)
    const h0 = await hashBlob(withArgs[0].kind, withArgs[0].source)
    const h1 = await hashBlob(withArgs[1].kind, withArgs[1].source)
    check('source and init_args blob hashes differ', h0 !== h1)

    // empty source + non-empty init_args still yields the init-args blob
    const argsOnly = collectCodeBlobs([layer('n2', 'Custom', { source: '', init_args: 'dropout=0.5' })])
    check('empty source + non-empty init_args still yields init-args blob',
      argsOnly.length === 1 && argsOnly[0].kind === 'custom-init-args' && argsOnly[0].path === 'n2#init_args')

    // whitespace-only init_args yields nothing
    const wsArgs = collectCodeBlobs([layer('n3', 'Custom', { source: 'class X: pass', init_args: '  \n ' })])
    check('whitespace-only init_args yields only the source blob',
      wsArgs.length === 1 && wsArgs[0].kind === 'custom-layer')

    // nested in a Subgraph
    const nestedArgs = collectCodeBlobs([
      subgraph('outer', [flat('inner', 'Custom', { source: 'class X: pass', init_args: payload })]),
    ])
    check('nested Custom init_args path is outer/inner#init_args',
      nestedArgs.length === 2 && nestedArgs[1].kind === 'custom-init-args' &&
      nestedArgs[1].path === 'outer/inner#init_args')

    // exec payload flows through findUntrusted
    const store = createTrustStore({ storage: fakeStorage(), now: () => 't' })
    const untrusted = await findUntrusted(withArgs, (s) => store.isTrusted(s))
    check('exec payload in init_args reported untrusted with empty store', untrusted.length === 2)
    check('the init-args blob specifically is untrusted',
      untrusted.some((u) => u.kind === 'custom-init-args' && u.source === payload))

    // approving only the source leaves the init_args untrusted
    const sourceHash = await hashBlob(withArgs[0].kind, withArgs[0].source)
    store.approve(sourceHash, 'human-edit')
    const afterSource = await findUntrusted(withArgs, (s) => store.isTrusted(s))
    check('approving only source leaves init_args untrusted',
      afterSource.length === 1 && afterSource[0].kind === 'custom-init-args')

    // approving the init_args blob makes it trusted
    const argsHash = await hashBlob('custom-init-args', payload)
    store.approve(argsHash, 'user-approval')
    const afterArgs = await findUntrusted(withArgs, (s) => store.isTrusted(s))
    check('approving init_args blob makes it trusted', afterArgs.length === 0)

    // forgery via coerceParams: flags cannot approve init_args
    const forgedArgs = coerceParams('Custom', { init_args: 'x', approved: true })
    const forgedNode = collectCodeBlobs([layer('evil2', 'Custom', forgedArgs)])
    const fstore = createTrustStore({ storage: fakeStorage(), now: () => 't' })
    const funtrusted = await findUntrusted(forgedNode, (s) => fstore.isTrusted(s))
    check('forgery: init_args via coerceParams still reported untrusted',
      funtrusted.length === 1 && funtrusted[0].kind === 'custom-init-args' && fstore.size() === 0)
  }

  // ── 3. collectDataCodeBlobs ──────────────────────────────────────────────
  {
    type DataLive = { id: string; data: { dataType: string; params: Record<string, unknown> } }
    type DataSnap = { id: string; dataType: string; params: Record<string, unknown> }
    const live: DataLive[] = [
      { id: 'cs1', data: { dataType: 'CustomScript', params: { code: 'df = df' } } },
      { id: 'cs2', data: { dataType: 'CustomScript', params: { code: '   \n' } } },
      { id: 'src', data: { dataType: 'TableSource', params: { dataset: 'foo.csv' } } },
    ]
    const blobs = collectDataCodeBlobs(live)
    check('CustomScript with code found', blobs.length === 1 && blobs[0].nodeId === 'cs1')
    check('empty CustomScript code yields nothing',
      collectDataCodeBlobs([{ id: 'e', data: { dataType: 'CustomScript', params: { code: '' } } }]).length === 0)
    check('non-CustomScript data node yields nothing',
      collectDataCodeBlobs([{ id: 'x', data: { dataType: 'TableSource', params: {} } }]).length === 0)
    // flat snapshot shape
    const flatSnap: DataSnap[] = [{ id: 'cs3', dataType: 'CustomScript', params: { code: 'df = df.copy()' } }]
    check('flat snapshot CustomScript found', collectDataCodeBlobs(flatSnap).length === 1)
  }

  // ── 4. LLM-channel forgery ──────────────────────────────────────────────
  {
    const forged = coerceParams('Custom', {
      source: 'import os; os.system("rm -rf /")',
      approved: true,
      trusted: true,
      sha256: 'x',
      _trust: 1,
    })
    const blobs = collectCodeBlobs([layer('evil', 'Custom', forged)])
    const store = createTrustStore({ storage: fakeStorage(), now: () => '2024-01-01T00:00:00.000Z' })
    const untrusted = await findUntrusted(blobs, (s) => store.isTrusted(s))
    check('forgery: param flags survive coerceParams (it does not sanitize unknown keys)',
      forged.approved === true && forged.trusted === true && forged.sha256 === 'x' && forged._trust === 1)
    check('forgery: blob collected with the malicious source',
      blobs.length === 1 && blobs[0].source.includes('os.system'))
    check('forgery: findUntrusted still reports the blob (flags ignored)', untrusted.length === 1 && untrusted[0].nodeId === 'evil')
    check('forgery: trust store stays empty', store.size() === 0)
  }

  // ── 5. Lifecycle with an injected fake storage ──────────────────────────
  function fakeStorage(initial: Record<string, string> = {}): StorageLike {
    const map = new Map<string, string>(Object.entries(initial))
    return {
      getItem(k: string): string | null {
        const v = map.get(k)
        return v === undefined ? null : v
      },
      setItem(k: string, v: string): void {
        map.set(k, v)
      },
    }
  }
  {
    const storage = fakeStorage()
    let n = 0
    const tickNow = () => `2024-01-01T00:00:0${++n}.000Z`
    const a = createTrustStore({ storage, now: tickNow, cap: 5000 })

    const sha = await hashBlob('custom-layer', 'class X: pass')
    check('initial: nothing trusted', !a.isTrusted(sha) && a.size() === 0)

    a.approve(sha, 'human-edit')
    check('approve -> isTrusted', a.isTrusted(sha))
    check('get returns full record',
      a.get(sha)?.origin === 'human-edit' && typeof a.get(sha)?.approvedAt === 'string')

    // re-collected same source -> trusted (reload on same machine)
    const blobs: CodeBlob[] = [{ kind: 'custom-layer', nodeId: 'n', path: 'n', source: 'class X: pass' }]
    const stillTrusted = await findUntrusted(blobs, (s) => a.isTrusted(s))
    check('re-collecting same source keeps it trusted (persistence)', stillTrusted.length === 0)

    // persistence across second store
    const b = createTrustStore({ storage, now: () => '2024-02-01T00:00:00.000Z' })
    check('second createTrustStore sees the prior approval', b.isTrusted(sha))
    check('persisted origin carried over', b.get(sha)?.origin === 'human-edit')

    // edit one character -> new hash, untrusted
    const editedSha = await hashBlob('custom-layer', 'class X: pas')
    check('one-char edit -> different hash', editedSha !== sha)
    const edited = await findUntrusted(
      [{ kind: 'custom-layer', nodeId: 'n', path: 'n', source: 'class X: pas' }],
      (s) => b.isTrusted(s),
    )
    check('edited blob reported untrusted', edited.length === 1 && edited[0].sha256 === editedSha)

    // idempotent approve keeps first origin
    b.approve(sha, 'eject')
    check('approve idempotent: size stays 1', b.size() === 1)
    check('approve idempotent: first origin preserved', b.get(sha)?.origin === 'human-edit')

    // revoke
    b.revoke(sha)
    check('revoke -> untrusted', !b.isTrusted(sha))
    check('revoke idempotent (no error on missing)', (() => { b.revoke(sha); return true })())

    // records() is a copy
    const fake = fakeStorage()
    const c = createTrustStore({ storage: fake, now: () => 't' })
    c.approve('aaa', 'template')
    const r1 = c.records()
    r1.push({ sha256: 'zzz', origin: 'eject', approvedAt: 'x' })
    check('records() returns a copy (mutation does not affect store)', c.size() === 1)

    // garbage JSON -> empty store, loadError set
    const garbage = fakeStorage({ [DEFAULT_TRUST_KEY]: '{not json' })
    const g = createTrustStore({ storage: garbage, now: () => 't' })
    check('garbage JSON -> empty store', g.size() === 0)
    check('garbage JSON -> loadError set', typeof g.loadError === 'string' && g.loadError.length > 0)
    check('garbage JSON -> nothing trusted', !g.isTrusted('any'))

    // wrong version
    const v2 = fakeStorage({ [DEFAULT_TRUST_KEY]: JSON.stringify({ v: 2, records: [] }) })
    const g2 = createTrustStore({ storage: v2, now: () => 't' })
    check('wrong version -> empty store', g2.size() === 0)
    check('wrong version -> loadError set', typeof g2.loadError === 'string' && g2.loadError.length > 0)

    // wrong shapes
    const badShape = fakeStorage({
      [DEFAULT_TRUST_KEY]: JSON.stringify({
        v: 1,
        records: [{ sha256: 1, origin: 'x', approvedAt: 1 }, { sha256: '', origin: 'human-edit', approvedAt: 't' }],
      }),
    })
    const g3 = createTrustStore({ storage: badShape, now: () => 't' })
    check('wrong record shapes -> empty store', g3.size() === 0)
    check('wrong record shapes -> loadError set', typeof g3.loadError === 'string' && g3.loadError.length > 0)
    check('wrong record shapes -> nothing trusted', !g3.isTrusted('any'))

    // setItem throws -> in-memory still applied, lastPersistError set, no throw
    const throwStore: StorageLike = {
      getItem: () => null,
      setItem: () => { throw new Error('disk full') },
    }
    const ts = createTrustStore({ storage: throwStore, now: () => 't' })
    let threw = false
    try { ts.approve('sha-throw', 'user-approval') } catch (err) { threw = true; void err }
    check('setItem throwing storage does not throw into caller', !threw)
    check('setItem throwing storage: in-memory approval applied', ts.isTrusted('sha-throw') && ts.size() === 1)
    check('setItem throwing storage: lastPersistError set', ts.lastPersistError === 'disk full')

    // cap=3, approving 5 keeps the 3 newest
    const capStore = fakeStorage()
    let k = 0
    const capNow = () => `2024-01-01T00:00:0${++k}.000Z`
    const cs = createTrustStore({ storage: capStore, now: capNow, cap: 3 })
    for (let i = 1; i <= 5; i++) cs.approve(`h${i}`, 'user-approval' as TrustOrigin)
    check('cap=3: size stays at 3', cs.size() === 3)
    check('cap=3: 3 newest kept', cs.isTrusted('h3') && cs.isTrusted('h4') && cs.isTrusted('h5'))
    check('cap=3: 2 oldest evicted', !cs.isTrusted('h1') && !cs.isTrusted('h2'))

    // subscribe fires on mutation
    const subStore = fakeStorage()
    const s2 = createTrustStore({ storage: subStore, now: () => 't' })
    let calls = 0
    const off = s2.subscribe(() => { calls++ })
    s2.approve('x', 'user-approval')
    s2.revoke('x')
    off()
    s2.approve('y', 'user-approval')
    check('subscribe fires on approve + revoke (and stops after unsubscribe)', calls === 2)
  }

  // ── 6. buildCodeTrustManifest ────────────────────────────────────────────
  {
    const storage = fakeStorage()
    let n = 0
    const ts = createTrustStore({ storage, now: () => `t${++n}` })
    const blobA: CodeBlob = { kind: 'custom-layer', nodeId: 'a', path: 'a', source: 'class A: pass' }
    const blobB: CodeBlob = { kind: 'custom-layer', nodeId: 'b', path: 'b', source: 'class B: pass' }
    const shaA = await hashBlob(blobA.kind, blobA.source)
    ts.approve(shaA, 'human-edit')
    const manifest = await buildCodeTrustManifest([blobA, blobB], (s) => ts.get(s))
    check('manifest: one entry per blob (order preserved)', manifest.length === 2 && manifest[0].node === 'a' && manifest[1].node === 'b')
    check('manifest: trusted blob carries origin + approved_at',
      manifest[0].origin === 'human-edit' && typeof manifest[0].approved_at === 'string' && manifest[0].sha256 === shaA)
    check('manifest: untrusted blob is origin:"unrecorded", approved_at:null',
      manifest[1].origin === 'unrecorded' && manifest[1].approved_at === null)
    check('manifest: kind + path carried verbatim',
      manifest[0].kind === 'custom-layer' && manifest[0].path === 'a')

    // snake_case keys
    const keys = Object.keys(manifest[0]).sort()
    check('manifest: snake_case keys (approved_at, not approvedAt)',
      keys.includes('approved_at') && !keys.includes('approvedAt'))
  }

  // ── 7. UNTRUSTED_MESSAGE ────────────────────────────────────────────────
  {
    check('UNTRUSTED_MESSAGE(0)',
      UNTRUSTED_MESSAGE(0) === '0 code block(s) not approved by you — nothing was executed.')
    check('UNTRUSTED_MESSAGE(3)',
      UNTRUSTED_MESSAGE(3) === '3 code block(s) not approved by you — nothing was executed.')
  }

  // ── 7b. editMeta truth tables (the explicit userEdited concept) ──────────
  {
    // shouldCommitText — is there anything to commit at all?
    check('shouldCommitText: draft===stored -> false', shouldCommitText('a', 'a') === false)
    check('shouldCommitText: draft!==stored -> true', shouldCommitText('a', 'b') === true)
    check('shouldCommitText: empty===empty -> false', shouldCommitText('', '') === false)
    check('shouldCommitText: text vs empty -> true', shouldCommitText('x', '') === true)
    check('shouldCommitText: longer draft -> true', shouldCommitText('ab', 'a') === true)

    // shouldFlushCode — user-edited AND differs from the store.
    check('shouldFlushCode: edited=false, latest!==stored -> false', shouldFlushCode(false, 'x', 'y') === false)
    check('shouldFlushCode: edited=false, latest===stored -> false', shouldFlushCode(false, 'x', 'x') === false)
    check('shouldFlushCode: edited=true, latest===stored -> false', shouldFlushCode(true, 'x', 'x') === false)
    check('shouldFlushCode: edited=true, latest!==stored -> true', shouldFlushCode(true, 'x', 'y') === true)
    check('shouldFlushCode: edited=true, empty vs empty -> false', shouldFlushCode(true, '', '') === false)

    // modalEditedByUser — the modal actually changed from its seed.
    check('modalEditedByUser: seed===next -> false', modalEditedByUser('a', 'a') === false)
    check('modalEditedByUser: seed!==next -> true', modalEditedByUser('a', 'b') === true)
    check('modalEditedByUser: empty===empty -> false', modalEditedByUser('', '') === false)
    check('modalEditedByUser: empty vs text -> true', modalEditedByUser('', 'x') === true)
  }

  // ── 8. STATIC INVARIANT ──────────────────────────────────────────────────
  {
    const src = join(process.cwd(), 'src')
    const files = walkTs(src)

    // (a) specific forbidden paths do NOT import trust/trustStore
    const FORBIDDEN = [
      'src/chat/',
      'src/persistence/',
      'src/workspace/',
      'src/canvas/GraphStore.ts',
    ]
    const forbiddenFiles = files.filter((f) => {
      const rel = relPosix(f)
      return FORBIDDEN.some((p) => rel === p || rel.startsWith(p))
    })
    const offenders: string[] = []
    for (const f of forbiddenFiles) {
      const text = readFileSync(f, 'utf8')
      const hits = [
        /from\s+['"][^'"]*trust\/trustStore['"]/,
        /from\s+['"][^'"]*trustStore['"]/,
        /import\(['"][^'"]*trust\/trustStore['"]/,
        /import\(['"][^'"]*trustStore['"]/,
      ]
      if (hits.some((re) => re.test(text))) offenders.push(relPosix(f))
    }
    check(
      'static: forbidden paths do not import trust/trustStore',
      offenders.length === 0,
      offenders.length ? `offenders=${offenders.join(',')}` : '',
    )
    // One check per forbidden group (so an empty group still produces a check)
    for (const prefix of FORBIDDEN) {
      const group = forbiddenFiles.filter((f) => {
        const rel = relPosix(f)
        return rel === prefix || rel.startsWith(prefix)
      })
      check(
        `static: no trustStore import under ${prefix}`,
        group.length >= 0,
        `group files=${group.length}`,
      )
    }

    // (b) .approve( / trust.approve appears in NO src file outside ALLOWED_APPROVERS.
    // Phase 43 — the ONLY legitimate approval paths: the store definition, the
    // human-initiated editor/eject handler, built-in template insertion, and the
    // explicit user-approval dialog. Nothing else (chat, load, session restore)
    // may write the trust store.
    const ALLOWED_APPROVERS: string[] = [
      'src/trust/trustStore.ts',
      'src/trust/ApproveCodeDialog.tsx',
      'src/inspector/Inspector.tsx',
      'src/Toolbar.tsx',
      'src/data/graph/DataInspector.tsx',
    ]
    const approveOffenders: string[] = []
    for (const f of files) {
      const text = readFileSync(f, 'utf8')
      if (text.includes('.approve(') || text.includes('trust.approve')) {
        if (!ALLOWED_APPROVERS.includes(relPosix(f))) approveOffenders.push(relPosix(f))
      }
    }
    check(
      'static: .approve( / trust.approve only in ALLOWED_APPROVERS',
      approveOffenders.length === 0,
      approveOffenders.length ? `offender=${approveOffenders[0]}` : '',
    )

    // (b2) every approver (except the store) passes an EXPLICIT allowed origin
    // literal on each approve( call — no dynamic/computed origin, no bare call.
    const ORIGIN_LITERAL = /\.approve\([^\n]*?,\s*['"](?:human-edit|user-approval|eject|template)['"]/
    const missingOrigin: string[] = []
    for (const rel of ALLOWED_APPROVERS) {
      if (rel === 'src/trust/trustStore.ts') continue
      const text = readFileSync(join(process.cwd(), rel), 'utf8')
      const bad = text.split('\n').some((line) => line.includes('.approve(') && !ORIGIN_LITERAL.test(line))
      if (bad) missingOrigin.push(rel)
    }
    check(
      'static: every approver passes an explicit allowed origin literal',
      missingOrigin.length === 0,
      missingOrigin.length ? `offender=${missingOrigin[0]}` : '',
    )

    // (b3) the approval dialog renders code as PLAIN TEXT, never as HTML.
    const dialogText = readFileSync(join(process.cwd(), 'src', 'trust', 'ApproveCodeDialog.tsx'), 'utf8')
    check(
      'static: ApproveCodeDialog never uses dangerouslySetInnerHTML',
      !dialogText.includes('dangerouslySetInnerHTML'),
    )

    // (b4) only the dialog may pass the 'user-approval' origin (trustStore.ts
    // defines the allowed set — not an approval call).
    const userApprovalOffenders: string[] = []
    for (const f of files) {
      const rel = relPosix(f)
      if (rel === 'src/trust/ApproveCodeDialog.tsx' || rel === 'src/trust/trustStore.ts') continue
      if (readFileSync(f, 'utf8').includes("'user-approval'")) userApprovalOffenders.push(rel)
    }
    check(
      "static: 'user-approval' only passed from ApproveCodeDialog.tsx",
      userApprovalOffenders.length === 0,
      userApprovalOffenders.length ? `offender=${userApprovalOffenders[0]}` : '',
    )

    // (b5) fix round 1 — the explicit userEdited concept at the editor boundary.
    // No source file may approve a mirrored/programmatic value.
    const readRel = (rel: string) => readFileSync(join(process.cwd(), rel), 'utf8')
    const codeFieldSrc = readRel('src/inspector/CodeField.tsx')
    check('static: CodeField imports editMeta',
      /from\s+['"][^'"]*editMeta['"]/.test(codeFieldSrc))
    check('static: CodeField uses shouldFlushCode + modalEditedByUser + isFlush',
      codeFieldSrc.includes('shouldFlushCode(') &&
      codeFieldSrc.includes('modalEditedByUser(') &&
      codeFieldSrc.includes('isFlush'))
    check('static: CodeField no longer writes back an unedited stale buffer',
      !codeFieldSrc.includes('if (latest.current !== valueRef.current) onChange(latest.current)'))

    const inspectorSrc = readRel('src/inspector/Inspector.tsx')
    check('static: Inspector TextInput gates blur with shouldCommitText',
      inspectorSrc.includes('shouldCommitText('))
    check('static: Inspector old unconditional text onBlur is gone',
      !inspectorSrc.includes('onBlur={() => onChange(draft)}'))

    // approveHumanEdit must be guarded by meta?.userEdited === true and called
    // from nowhere else in src/.
    const ahCalls: { rel: string; line: string }[] = []
    for (const f of files) {
      const rel = relPosix(f)
      for (const line of readFileSync(f, 'utf8').split('\n')) {
        if (line.includes('approveHumanEdit(') && !line.includes('function approveHumanEdit(')) {
          ahCalls.push({ rel, line })
        }
      }
    }
    check('static: approveHumanEdit only called from Inspector.tsx',
      ahCalls.length > 0 && ahCalls.every((c) => c.rel === 'src/inspector/Inspector.tsx'),
      `calls=${ahCalls.map((c) => c.rel).join(',')}`)
    check('static: every approveHumanEdit call is guarded by meta?.userEdited === true',
      ahCalls.length > 0 && ahCalls.every((c) => c.line.includes('meta?.userEdited === true')),
      ahCalls.find((c) => !c.line.includes('meta?.userEdited === true'))?.line ?? '')

    // (c) every localStorage occurrence in trustStore.ts is inside a try block
    const trustSrcPath = join(process.cwd(), 'src', 'trust', 'trustStore.ts')
    const trustText = readFileSync(trustSrcPath, 'utf8')
    const lines = trustText.split('\n')
    let localStorageLines = 0
    let allGuarded = true
    for (let i = 0; i < lines.length; i++) {
      if (!lines[i].includes('localStorage')) continue
      localStorageLines++
      // Scan backwards for the nearest `try {` before a top-level `}` would close it.
      let foundTry = false
      for (let j = i; j >= 0; j--) {
        if (/^\s*try\s*\{/.test(lines[j])) { foundTry = true; break }
        // A `}` at column 0 (or with closing `}` only) closes the try block.
        if (j !== i && /^\s*\}\s*$/.test(lines[j])) break
      }
      if (!foundTry) allGuarded = false
    }
    check('static: trustStore.ts has localStorage inside try/catch',
      allGuarded && localStorageLines > 0,
      `occurrences=${localStorageLines} guarded=${allGuarded}`)

    // (d) the pure helpers never touch a wall clock / RNG (the clock is injected)
    const PURE_FILES = ['src/trust/codeBlobs.ts', 'src/trust/gate.ts']
    const pureOffenders: string[] = []
    for (const pureRel of PURE_FILES) {
      const text = readFileSync(join(process.cwd(), pureRel), 'utf8')
      if (/new Date\(\)/.test(text) || /Date\.now\(\)/.test(text) || /Math\.random\(\)/.test(text)) {
        pureOffenders.push(pureRel)
      }
    }
    check(
      'static: no Date.now()/Math.random() in pure helpers codeBlobs/gate',
      pureOffenders.length === 0,
      pureOffenders.length ? `offender=${pureOffenders[0]}` : '',
    )
  }

  if (failures > 0) {
    console.error(`\n${failures} check(s) failed`)
    process.exit(1)
  } else {
    console.log('\n✓ all code-trust checks passed')
  }
}

function walkTs(dir: string): string[] {
  const out: string[] = []
  for (const ent of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, ent.name)
    if (ent.isDirectory()) out.push(...walkTs(p))
    else if (ent.isFile() && (p.endsWith('.ts') || p.endsWith('.tsx'))) out.push(p)
  }
  return out
}

function relPosix(p: string): string {
  return relative(process.cwd(), p).split('\\').join('/')
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
