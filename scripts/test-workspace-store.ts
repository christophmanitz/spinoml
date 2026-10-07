#!/usr/bin/env tsx
// R018 — WORKSPACE store (src/workspace/store.ts). Two modes are exercised:
//   • browser  — the localStorage virtual FS (create/rename/remove/move,
//                activeFileId invariant 4, hostile persisted state, dirty
//                tracking, cross-file canvas binding).
//   • tauri    — by stubbing the Tauri bridge (`window.__TAURI_INTERNALS__.invoke`)
//                with an in-memory FS, so the .py-twin logic (invariant 7,
//                `pyTwinPath` sanitisation + `pyTwinError`) is covered too.
// The store is imported AFTER the DOM/localStorage/Tauri stubs are installed.
//
// Every case's comment names the bug it would catch. The mutation proofs
// (activeFileId reset, phantom invariant) are documented in the task.
//
// Run: npm run test:workspace-store   (needs a shell with node + tsx)

import type { Entry, Folder, File } from '../src/workspace/store'
import type { FsEntry } from '../src/workspace/tauri-fs'

let passed = 0
let failed = 0
function check(name: string, cond: boolean, detail = ''): void {
  if (cond) { passed++; process.stdout.write(`  ✓ ${name}\n`) }
  else { failed++; process.stdout.write(`  ✗ ${name}${detail ? '  ' + detail : ''}\n`) }
}

// ── global stubs (installed BEFORE the store is imported) ────────────────────

type InvokeFn = (cmd: string, args: Record<string, unknown>) => Promise<unknown>

class MemStorage {
  private map = new Map<string, string>()
  getItem(k: string): string | null { return this.map.has(k) ? this.map.get(k)! : null }
  setItem(k: string, v: string): void { this.map.set(k, String(v)) }
  removeItem(k: string): void { this.map.delete(k) }
  clear(): void { this.map.clear() }
  key(i: number): string | null { return [...this.map.keys()][i] ?? null }
  get length(): number { return this.map.size }
}

const storage = new MemStorage()

type WinStub = {
  addEventListener: (type: string, cb: unknown) => void
  removeEventListener: (type: string, cb: unknown) => void
  localStorage: Storage
  confirm: () => boolean
  alert: () => void
  __TAURI_INTERNALS__?: { invoke: InvokeFn }
}
const win: WinStub = {
  addEventListener: () => {},
  removeEventListener: () => {},
  localStorage: storage as unknown as Storage,
  confirm: () => true,
  alert: () => {},
}
;(globalThis as unknown as { window: WinStub }).window = win
;(globalThis as unknown as { localStorage: Storage }).localStorage = storage as unknown as Storage
;(globalThis as unknown as { alert: (m: string) => void }).alert = () => {}

// In-memory "disk" behind the fake Tauri invoke.
const diskFiles = new Map<string, string>()
const diskDirs = new Set<string>()
const writeLog: { relpath: string; content: string }[] = []
let failPyWrites = false

function listDisk(): FsEntry[] {
  const out: FsEntry[] = []
  for (const relpath of diskDirs) out.push({ name: relpath.split('/').pop() ?? relpath, relpath, is_dir: true })
  for (const relpath of diskFiles.keys()) out.push({ name: relpath.split('/').pop() ?? relpath, relpath, is_dir: false })
  return out
}
function moveKey(from: string, to: string): void {
  if (diskFiles.has(from)) { diskFiles.set(to, diskFiles.get(from)!); diskFiles.delete(from) }
  if (diskDirs.has(from)) {
    diskDirs.delete(from); diskDirs.add(to)
    for (const [k, v] of [...diskFiles]) if (k.startsWith(from + '/')) { diskFiles.set(to + k.slice(from.length), v); diskFiles.delete(k) }
    for (const d of [...diskDirs]) if (d.startsWith(from + '/')) { diskDirs.delete(d); diskDirs.add(to + d.slice(from.length)) }
  }
}
const invokeImpl: InvokeFn = async (cmd, args) => {
  const relpath = typeof args.relpath === 'string' ? args.relpath : ''
  switch (cmd) {
    case 'pick_workspace_dir': return '/fake/ws'
    case 'close_workspace_dir': return null
    case 'list_workspace': return listDisk()
    case 'write_workspace_file': {
      if (failPyWrites && relpath.endsWith('.py')) throw new Error('disk full')
      const content = typeof args.content === 'string' ? args.content : ''
      writeLog.push({ relpath, content })
      diskFiles.set(relpath, content)
      return null
    }
    case 'read_workspace_file': {
      const c = diskFiles.get(relpath)
      if (c === undefined) throw new Error(`no such file: ${relpath}`)
      return c
    }
    case 'mkdir_workspace': diskDirs.add(relpath); return null
    case 'delete_workspace_path': {
      diskFiles.delete(relpath)
      diskDirs.delete(relpath)
      for (const k of [...diskFiles.keys()]) if (k.startsWith(relpath + '/')) diskFiles.delete(k)
      for (const d of [...diskDirs]) if (d.startsWith(relpath + '/')) diskDirs.delete(d)
      return null
    }
    case 'rename_workspace_path': {
      moveKey(typeof args.fromRel === 'string' ? args.fromRel : '', typeof args.toRel === 'string' ? args.toRel : '')
      return null
    }
    default: return null
  }
}

// ── helpers (populated after the store is imported) ──────────────────────────

const STORAGE_KEY = 'spinoml.workspace.v1'
type WSMod = typeof import('../src/workspace/store')
type GraphMod = typeof import('../src/canvas/GraphStore')
type HistMod = typeof import('../src/history/store')
let useWorkspaceStore: WSMod['useWorkspaceStore']
let useGraphStore: GraphMod['useGraphStore']
let useHistoryStore: HistMod['useHistoryStore']

const ws = () => useWorkspaceStore.getState()
const entries = (): Record<string, Entry> => ws().entries
function entryName(id: string): string | undefined { return entries()[id]?.name }
function childIds(id: string): string[] {
  const e = entries()[id]
  return e?.kind === 'folder' ? (e as Folder).childIds : []
}
/** Invariant 4: activeFileId references an existing FILE or is null. */
function assertActiveInvariant(label: string): void {
  const s = ws()
  const e = s.activeFileId === null ? null : entries()[s.activeFileId]
  check(`invariant 4 ${label}: activeFileId is null or an existing file`,
    s.activeFileId === null || (!!e && e.kind === 'file'),
    `active=${String(s.activeFileId)}`)
}
function rootEntries(key: string, raw: string): void {
  storage.setItem(key, raw)
}

async function main(): Promise<void> {
  // imported only now, with the stubs in place
  const wsMod = await import('../src/workspace/store')
  useWorkspaceStore = wsMod.useWorkspaceStore
  const gMod = await import('../src/canvas/GraphStore')
  useGraphStore = gMod.useGraphStore
  const hMod = await import('../src/history/store')
  useHistoryStore = hMod.useHistoryStore

  // ═══════════════════════════════════════════════════════════════════════════
  console.log('browser mode: create / rename / remove / move')
  {
    useGraphStore.getState().resetGraph()
    const f1 = await ws().createFile('root', 'modelA.spinoml')
    check('createFile returns an id', typeof f1 === 'string' && f1.length > 0)
    check('created file is in entries', entries()[f1]?.kind === 'file')
    check('created file has an ISO savedAt', typeof (entries()[f1] as File).savedAt === 'string' && (entries()[f1] as File).savedAt.length > 0)
    check('root.childIds includes the file', childIds('root').includes(f1))
    check('createFile sets activeFileId', ws().activeFileId === f1)
    assertActiveInvariant('after createFile')

    const d1 = await ws().createFolder('root', 'sub')
    check('createFolder returns a folder id', entries()[d1]?.kind === 'folder')
    check('root.childIds includes the folder', childIds('root').includes(d1))
    const f2 = await ws().createFile(d1, 'nested.spinoml')
    check('nested file parent is the folder', (entries()[f2] as File).parentId === d1)
    check('folder.childIds includes the nested file', childIds(d1).includes(f2))

    // collision → suffixed, never overwrite
    const f3 = await ws().createFile('root', 'modelA.spinoml')
    check('collision gets a unique name (no overwrite)', entryName(f3) === 'modelA (2).spinoml', String(entryName(f3)))
    check('the original file is untouched', entryName(f1) === 'modelA.spinoml')

    // rename
    await ws().rename(f1, 'renamed.spinoml')
    check('rename updates the name', entryName(f1) === 'renamed.spinoml')
    await ws().rename(f1, '   ')
    check('rename to blank is a no-op (trim)', entryName(f1) === 'renamed.spinoml')
    await ws().rename(f1, 'renamed.spinoml')
    check('rename to the same name is a no-op', entryName(f1) === 'renamed.spinoml')
    await ws().rename(f3, 'renamed.spinoml')
    check('rename collision is suffixed', entryName(f3) === 'renamed (2).spinoml', String(entryName(f3)))

    // illegal-ish names: accepted verbatim (documented gap — no name validation)
    const weird = await ws().createFile('root', '..')
    check('name ".." is accepted verbatim (documented gap: no path-segment validation)', entryName(weird) === '..')
    const slashed = await ws().createFile('root', 'a/b.spinoml')
    check('name containing "/" is accepted verbatim (documented gap)', entryName(slashed) === 'a/b.spinoml')
    const unicode = await ws().createFile('root', '测试.spinoml')
    check('unicode name is preserved', entryName(unicode) === '测试.spinoml')
    const longName = await ws().createFile('root', 'x'.repeat(1000) + '.spinoml')
    check('very long name is accepted', entryName(longName) === 'x'.repeat(1000) + '.spinoml')
    const upper = await ws().createFile('root', 'ModelA.SPINOML')
    check('case-variant name is a distinct sibling (no case-folding collision)', entryName(upper) === 'ModelA.SPINOML')

    // remove a non-active sibling keeps activeFileId
    const activeBefore = ws().activeFileId
    await ws().remove(f3)
    check('removing a non-active file keeps activeFileId', ws().activeFileId === activeBefore)
    check('removed file is gone from entries', entries()[f3] === undefined)
    check('removed file is gone from root.childIds', !childIds('root').includes(f3))
    assertActiveInvariant('after removing a non-active file')
  }

  console.log('browser mode: invariant 4 under active-file deletion')
  {
    const f = await ws().createFile('root', 'temp.spinoml')
    check('new file is active', ws().activeFileId === f)
    await ws().remove(f)
    check('removing the ACTIVE file resets activeFileId to null', ws().activeFileId === null)
    assertActiveInvariant('after removing the active file')

    const dir = await ws().createFolder('root', 'victim')
    const child = await ws().createFile(dir, 'deep.spinoml')
    check('nested file is active', ws().activeFileId === child)
    await ws().remove(dir)
    check('removing an ANCESTOR folder removes the active child', entries()[child] === undefined)
    check('removing an ancestor folder resets activeFileId to null', ws().activeFileId === null)
    assertActiveInvariant('after removing an ancestor of the active file')

    const fA = await ws().createFile('root', 'active-rename.spinoml')
    await ws().rename(fA, 'active-renamed.spinoml')
    check('rename keeps activeFileId on the same id', ws().activeFileId === fA)
    assertActiveInvariant('after renaming the active file')

    const dest = await ws().createFolder('root', 'dest')
    await ws().move(fA, dest)
    check('move keeps activeFileId on the same id', ws().activeFileId === fA)
    check('moved file parent is updated', (entries()[fA] as File).parentId === dest)
    assertActiveInvariant('after moving the active file')

    // move cycle guard: a folder cannot be moved into its own descendant
    const p = await ws().createFolder('root', 'P')
    const c = await ws().createFolder(p, 'C')
    await ws().move(p, c)
    check('move cycle is prevented (folder stays put)', (entries()[p] as Folder).parentId === 'root')

    ws().closeActive()
    check('closeActive resets activeFileId to null', ws().activeFileId === null)
    assertActiveInvariant('after closeActive')
  }

  console.log('browser mode: dirty tracking (structural vs position-only)')
  {
    useGraphStore.getState().resetGraph()
    useHistoryStore.getState().clear()
    const f = await ws().createFile('root', 'dirty.spinoml')
    check('fresh active file is clean', ws().dirty === false)
    useGraphStore.getState().addLayer('ReLU', { x: 0, y: 0 })
    check('structural change marks dirty', ws().dirty === true)
    await ws().saveActive()
    check('save clears dirty', ws().dirty === false)
    check('save stored the content', (entries()[f] as File).content.includes('spinoml'))

    useGraphStore.getState().autoLayout()
    check('position-only (autoLayout) does NOT mark dirty', ws().dirty === false)
    useGraphStore.getState().onNodesChange([{ id: 'input', type: 'position', position: { x: 7, y: 9 } }])
    check('position-only (drag) does NOT mark dirty', ws().dirty === false)

    useGraphStore.getState().addLayer('GELU', { x: 0, y: 0 })
    check('structural change after moves marks dirty', ws().dirty === true)
    useHistoryStore.getState().undo()
    check('undo back to the saved structure is clean again', ws().dirty === false)
  }

  console.log('browser mode: hostile / corrupt persisted state never throws')
  {
    const cases: Array<[string, string]> = [
      ['null root', 'null'],
      ['array root', '[]'],
      ['number root', '5'],
      ['entries not an object', '{"entries":"nope","activeFileId":5,"expanded":"x"}'],
      ['entry pointing to a missing parent', JSON.stringify({
        entries: {
          root: { kind: 'folder', id: 'root', name: 'models', parentId: null, childIds: ['f'] },
          f: { kind: 'file', id: 'f', name: 'a.spinoml', parentId: 'ghost', content: '', savedAt: '' },
        }, activeFileId: null, expanded: [],
      })],
      ['parent cycle a<->b', JSON.stringify({
        entries: {
          root: { kind: 'folder', id: 'root', name: 'models', parentId: null, childIds: ['a'] },
          a: { kind: 'folder', id: 'a', name: 'A', parentId: 'b', childIds: ['b'] },
          b: { kind: 'folder', id: 'b', name: 'B', parentId: 'a', childIds: [] },
        }, activeFileId: null, expanded: [],
      })],
      ['duplicate ids under different keys', JSON.stringify({
        entries: {
          root: { kind: 'folder', id: 'root', name: 'models', parentId: null, childIds: ['x', 'y'] },
          x: { kind: 'file', id: 'x', name: 'x.spinoml', parentId: 'root', content: '', savedAt: '' },
          y: { kind: 'file', id: 'x', name: 'y.spinoml', parentId: 'root', content: '', savedAt: '' },
        }, activeFileId: null, expanded: [],
      })],
      ['entries missing root', JSON.stringify({ entries: { f: { kind: 'file', id: 'f', name: 'a', parentId: null, content: '', savedAt: '' } }, activeFileId: 'f', expanded: [] })],
    ]
    for (const [name, raw] of cases) {
      storage.clear()
      rootEntries(STORAGE_KEY, raw)
      let threw = false
      try { await ws().closeDirectory() } catch { threw = true }
      check(`hostile state "${name}" does not throw`, !threw)
      check(`hostile state "${name}" yields a root entry`, entries()['root']?.kind === 'folder')
      const s = ws()
      const okActive = s.activeFileId === null || (!!entries()[s.activeFileId] && entries()[s.activeFileId].kind === 'file')
      check(`hostile state "${name}" keeps activeFileId consistent`, okActive, `active=${String(s.activeFileId)}`)
      // The entry map must be a TREE under root: every id matches its key, each entry is listed by exactly
      // one folder that is also its parentId, and walking up the parent chain always reaches root.
      const ents = entries()
      const idsOk = Object.entries(ents).every(([k, e]) => e.id === k)
      const listedBy = new Map<string, string[]>()
      for (const e of Object.values(ents)) if (e.kind === 'folder') for (const c of e.childIds) listedBy.set(c, [...(listedBy.get(c) ?? []), e.id])
      const childrenOk = Object.values(ents).every((e) => e.id === 'root' ? e.parentId === null : (listedBy.get(e.id)?.length === 1 && listedBy.get(e.id)?.[0] === e.parentId))
      let terminates = true
      for (const e of Object.values(ents)) {
        let cur: Entry | undefined = e
        let hops = 0
        while (cur && cur.parentId !== null && hops < 100000) { cur = ents[cur.parentId]; hops++ }
        if (!cur || cur.id !== 'root') { terminates = false; break }
      }
      check(`hostile state "${name}": ids equal their keys`, idsOk)
      check(`hostile state "${name}": a consistent tree (one lister = parentId)`, childrenOk)
      check(`hostile state "${name}": every parent chain reaches root (no loop)`, terminates)
    }

    // activeFileId pointing to a missing entry — invariant 4 must repair it.
    storage.clear()
    rootEntries(STORAGE_KEY, JSON.stringify({
      entries: { root: { kind: 'folder', id: 'root', name: 'models', parentId: null, childIds: [] } },
      activeFileId: 'ghost', expanded: [],
    }))
    await ws().closeDirectory()
    check('activeFileId pointing to a missing entry is repaired to null', ws().activeFileId === null, `active=${String(ws().activeFileId)}`)
    assertActiveInvariant('after loading dangling activeFileId')

    // activeFileId pointing to a FOLDER must be repaired too.
    storage.clear()
    rootEntries(STORAGE_KEY, JSON.stringify({
      entries: {
        root: { kind: 'folder', id: 'root', name: 'models', parentId: null, childIds: ['sub'] },
        sub: { kind: 'folder', id: 'sub', name: 'sub', parentId: 'root', childIds: [] },
      }, activeFileId: 'sub', expanded: [],
    }))
    await ws().closeDirectory()
    check('activeFileId pointing to a folder is repaired to null', ws().activeFileId === null, `active=${String(ws().activeFileId)}`)

    // 10k entries load in bounded time without throwing
    {
      const big: Record<string, Entry> = { root: { kind: 'folder', id: 'root', name: 'models', parentId: null, childIds: [] } as Folder }
      for (let i = 0; i < 10000; i++) {
        const id = `f${i}`
        big[id] = { kind: 'file', id, name: `m${i}.spinoml`, parentId: 'root', content: '', savedAt: '' } as File
        ;(big.root as Folder).childIds.push(id)
      }
      storage.clear()
      rootEntries(STORAGE_KEY, JSON.stringify({ entries: big, activeFileId: null, expanded: [] }))
      const t0 = Date.now()
      let threw = false
      try { await ws().closeDirectory() } catch { threw = true }
      const ms = Date.now() - t0
      check('10k hostile entries load without throwing', !threw)
      check('10k hostile entries load in < 5s', ms < 5000, `${ms}ms`)
      check('10k hostile entries keep the root', entries()['root']?.kind === 'folder')
    }
    storage.clear()
    await ws().closeDirectory()
  }

  console.log('browser mode: no cross-file leak in the canvas binding')
  {
    useGraphStore.getState().resetGraph()
    useHistoryStore.getState().clear()
    const A = await ws().createFile('root', 'A.spinoml') // content = input-only
    useGraphStore.getState().addLayer('ReLU', { x: 0, y: 0 }) // edit file A (unsaved)
    const B = await ws().createFile('root', 'B.spinoml') // content captures the edit
    check('opening A after B loads only A (no ReLU leak)',
      await ws().openFile(A) === true && !useGraphStore.getState().nodes.some((n) => n.data.layerType === 'ReLU'))
    check('opening B loads B (ReLU present)',
      await ws().openFile(B) === true && useGraphStore.getState().nodes.some((n) => n.data.layerType === 'ReLU'))
    ws().closeActive()
  }

  // ═══════════════════════════════════════════════════════════════════════════
  console.log('tauri mode: .py twin path sanitisation (invariant 7)')
  {
    diskFiles.clear(); diskDirs.clear(); writeLog.length = 0; failPyWrites = false
    win.__TAURI_INTERNALS__ = { invoke: (cmd, args) => invokeImpl(cmd, args) }
    const opened = await ws().openDirectory()
    check('openDirectory enters tauri mode', opened === true && ws().mode === 'tauri')

    writeLog.length = 0
    await ws().createFile('root', 'my model.spinoml')
    const twin = writeLog.find((w) => w.relpath.endsWith('.py'))
    check('save writes a .py twin', !!twin)
    check('twin path sanitises spaces and uses the same stem', twin?.relpath === 'my_model.py', String(twin?.relpath))
    check('main .spinoml written unsanitised', writeLog.some((w) => w.relpath === 'my model.spinoml'))

    writeLog.length = 0
    await ws().createFile('root', '../evil.spinoml')
    check('twin sanitises ".." path traversal', writeLog.some((w) => /\/evil\.py$/.test(w.relpath) && !w.relpath.includes('..')), JSON.stringify(writeLog.map((w) => w.relpath)))

    writeLog.length = 0
    await ws().createFile('root', 'UPPER.SPINOML')
    check('twin strips the extension case-insensitively', writeLog.some((w) => w.relpath === 'UPPER.py'), JSON.stringify(writeLog.map((w) => w.relpath)))

    writeLog.length = 0
    await ws().createFile('root', 'noext')
    check('twin handles a missing .spinoml extension', writeLog.some((w) => w.relpath === 'noext.py'), JSON.stringify(writeLog.map((w) => w.relpath)))

    writeLog.length = 0
    await ws().createFile('root', '测试.spinoml')
    check('twin sanitises unicode', writeLog.some((w) => w.relpath === '_.py'), JSON.stringify(writeLog.map((w) => w.relpath)))
  }

  console.log('tauri mode: a failing twin write surfaces pyTwinError, main save truthful')
  {
    diskFiles.clear(); diskDirs.clear(); writeLog.length = 0
    const rel = await ws().createFile('root', 'good.spinoml')
    check('tauri createFile is active + clean', ws().activeFileId === rel && ws().dirty === false)
    check('successful create has no pyTwinError', ws().pyTwinError === null)
    // Regression: after createFile the in-memory content must be present, so the
    // dirty baseline is a real fingerprint (not null) — edits then register.
    check('createFile preserved the in-memory content', (entries()[rel] as File).content.length > 0)

    failPyWrites = true
    useGraphStore.getState().addLayer('ReLU', { x: 0, y: 0 })
    check('edit marked dirty', ws().dirty === true)
    await ws().saveActive()
    check('failing twin surfaces pyTwinError', typeof ws().pyTwinError === 'string' && ws().pyTwinError!.includes('Python-Zwilling'), String(ws().pyTwinError))
    check('main .spinoml still written (disk has it)', diskFiles.has(rel))
    check('main save state is truthful (dirty=false)', ws().dirty === false)
    failPyWrites = false

    // recover: a later successful save clears pyTwinError
    await ws().saveActive()
    check('successful save clears pyTwinError', ws().pyTwinError === null)

    // back to browser mode
    await ws().closeDirectory()
    delete win.__TAURI_INTERNALS__
    check('closeDirectory returns to browser mode', ws().mode === 'browser')
  }

  console.log(`\n${failed === 0 ? '✓ all workspace-store checks passed' : `✗ ${failed} check(s) failed`} (${passed} passed)`)
  process.exit(failed === 0 ? 0 : 1)
}

main().catch((e) => { console.error(e); process.exit(1) })
