// SpinoML LLM sidecar.
//
// Bridges the React frontend and the AI providers behind an AgentProvider
// abstraction. Hosts an HTTP server on 127.0.0.1:7422; the only meaningful
// endpoint is POST /chat which streams SSE events.
//
// Provider paths (payload.llm.kind):
//   subscription  → claude-agent-sdk + the local `claude` CLI (OAuth, Max)
//   anthropic     → direct Anthropic Messages API (API key)
//   openai-compat → OpenAI Chat Completions API (OpenAI, Gemini, Ollama, …)
//   opencode      → the first-class `opencode` CLI (default). The model is
//                   driven ONLY through a stdio MCP tool server
//                   (mcp-bridge.mjs) named "graph", so every operation it
//                   proposes flows through the SAME provider-neutral execTool
//                   pipeline (schema validation → GraphStore commit) as the
//                   Claude paths. Built-in opencode tools (read/bash/write/…)
//                   are disabled; opencode runs with `--format json` whose
//                   NDJSON event stream is mapped to the SSE event shapes.
//
// Each /chat request carries the full conversation history plus a snapshot
// of the user's current architecture. We rebuild a transient graph state
// here so the tool handlers can mutate and reason about it during the turn;
// each mutation is also pushed back to the frontend as an SSE action event
// so the GraphStore stays in lockstep.
//
// SSE event shapes:
//   {type: "status", value: "thinking"|"done"|"error", message?}
//   {type: "text",   value: "<assistant token chunk>"}
//   {type: "tool_use",    id, name, args}
//   {type: "tool_result", id, ok, result?, error?}
//   {type: "action", op, payload}     // mirror of the mutation, for the GraphStore
//   {type: "done"}

import { createServer } from 'node:http'
import { promises as fs } from 'node:fs'
import { readFileSync, readdirSync, rmSync } from 'node:fs'
import { spawn } from 'node:child_process'
import path from 'node:path'
import os from 'node:os'
import { z } from 'zod'
import { query, tool, createSdkMcpServer } from '@anthropic-ai/claude-agent-sdk'
import Anthropic from '@anthropic-ai/sdk'
import OpenAI from 'openai'
import { splitArgs, quoteArgv, checkDownloadUrl, checkSshTarget, safeFetch } from './shell-safety.mjs'
import { resolveInWorkspace, loadSymlinkTargets } from './path-scope.mjs'
import { validateNodeParams, wouldCreateCycle, redactSecrets, nodeKind } from './tool-validation.mjs'
import { validateModelName, validateJsonModelName } from './model-name.mjs'
import { randomUUID } from 'node:crypto'
import {
  loadConfig as loadAuthConfig,
  scrubEnv as scrubAuthEnv,
  decideFromHeaders,
  applyCorsHeaders,
  healthBody,
  limitedHealthBody,
  CODE_UNAUTHORIZED,
  CODE_BAD_ORIGIN,
  CODE_BAD_HOST,
  sessionSecretMatches,
  generateMcpSecret,
} from './auth.mjs'

// Authentication / Host / Origin gate (Phase 77/78; see
// docs/engineering/SIDECAR_AUTH.md). Read once at startup; SPINOML_SIDECAR_TOKEN
// is then scrubbed from process.env so subprocess children (opencode, Claude
// CLI, run_script helpers) cannot inherit it.
let AUTH_CONFIG
try {
  AUTH_CONFIG = loadAuthConfig(process.env)
} catch (e) {
  console.error(`[spinoml-llm] ${e.message}`)
  process.exit(2)
}
scrubAuthEnv(process.env)

// Default 7422 (the app and the frontend assume it). SPINOML_LLM_PORT lets the
// tests run a throw-away sidecar next to a running app; anything that is not a
// valid port is refused loudly instead of silently falling back.
const PORT = (() => {
  const raw = process.env.SPINOML_LLM_PORT
  if (raw === undefined || raw === '') return 7422
  const n = Number(raw)
  if (!Number.isInteger(n) || n < 1024 || n > 65535) {
    console.error(`[spinoml-llm] invalid SPINOML_LLM_PORT=${JSON.stringify(raw)} (expected an integer 1024-65535)`)
    process.exit(2)
  }
  return n
})()

// ── Lifecycle + child cleanup (Phase 13) ──────────────────────────────────
// Children (opencode per turn, run_script shell/slurm, ssh, the Claude CLI of the
// subscription SDK) are deliberately NOT detached: they stay in the sidecar's
// process group so a SIGKILL of the sidecar still reaps them. On SIGTERM/SIGINT
// (also what the Tauri parent's PR_SET_PDEATHSIG delivers when the app dies) we walk
// /proc for EVERY descendant of this process — not only the ones we spawned
// ourselves, an SDK-spawned CLI would otherwise leak — SIGTERM them, SIGKILL
// survivors after 1 s, and a 2.5 s watchdog hard-exits even if something is stuck.
// Track turn controllers so SIGTERM can abort in-flight /chat turns — the
// upstream provider stream is otherwise wedged on a write to a dead socket.
const trackedChildren = new Set()
// Disposable opencode session dirs of in-flight turns: removed synchronously on SIGTERM/SIGINT
// (a SIGKILL cannot be handled, so an empty tmp dir may still remain in that case).
const openSessionDirs = new Set()
const activeTurnControllers = new Set()
// Phase 51: thread-safe-enough (single event loop) request counter for /health.diag.
let requestCount = 0

function trackChild(child) {
  if (!child) return
  trackedChildren.add(child)
  const cleanup = () => trackedChildren.delete(child)
  child.once('close', cleanup)
  child.once('error', cleanup)
}

// Snapshot EVERY descendant (tracked children and anything else below this process,
// e.g. a CLI spawned by an SDK) BEFORE anything is aborted or killed: once a parent
// (say the `bash` of a run_script) dies, its children are reparented to init and can no
// longer be found by walking down from us, so the snapshot has to come first.
function snapshotDescendants() {
  const all = new Set()
  for (const child of trackedChildren) {
    if (typeof child.pid !== 'number') continue
    if (child.exitCode !== null || child.signalCode !== null) continue
    all.add(child.pid)
    collectDescendants(child.pid, all)
  }
  collectDescendants(process.pid, all)
  return all
}

function killPids(all) {
  for (const pid of all) {
    try {
      process.kill(pid, 'SIGTERM')
    } catch (e) {
      console.error(`[spinoml-llm] shutdown: kill ${pid} SIGTERM failed: ${e && e.message ? e.message : e}`)
    }
  }
  // 1 s grace then SIGKILL any survivors; the parent process itself exits
  // via the watchdog in shutdownAll(), so an unref'd timer is correct.
  const watchdog = setTimeout(() => {
    for (const pid of all) {
      try {
        process.kill(pid, 'SIGKILL')
      } catch (e) {
        console.error(`[spinoml-llm] shutdown: kill ${pid} SIGKILL failed: ${e && e.message ? e.message : e}`)
      }
    }
  }, 1000)
  watchdog.unref()
}

function collectDescendants(root, out) {
  // /proc/<pid>/task/*/children — Linux only. Reused from the torch
  // sidecar's design (sidecar-torch/main.py `_list_children`).
  let names
  try {
    names = readdirSync(`/proc/${root}/task`)
  } catch (e) {
    console.error(`[spinoml-llm] shutdown: readdir /proc/${root}/task failed: ${e && e.message ? e.message : e}`)
    return
  }
  for (const tid of names) {
    let text
    try {
      text = readFileSync(`/proc/${root}/task/${tid}/children`, 'utf8')
    } catch (e) {
      console.error(`[spinoml-llm] shutdown: readfile /proc/${root}/task/${tid}/children failed: ${e && e.message ? e.message : e}`)
      continue
    }
    for (const m of text.matchAll(/(\d+)/g)) {
      const c = Number(m[1])
      if (!Number.isNaN(c) && !out.has(c)) {
        out.add(c)
        collectDescendants(c, out)
      }
    }
  }
}

let shuttingDown = false
function shutdownAll(name) {
  if (shuttingDown) return
  shuttingDown = true
  const doomed = snapshotDescendants() // BEFORE the aborts below reparent grandchildren
  for (const ctrl of activeTurnControllers) {
    try {
      ctrl.abort()
    } catch (e) {
      console.error(`[spinoml-llm] shutdown: abort turn controller failed: ${e && e.message ? e.message : e}`)
    }
  }
  killPids(doomed)
  for (const dir of openSessionDirs) {
    try {
      rmSync(dir, { recursive: true, force: true })
    } catch (e) {
      console.error(`[spinoml-llm] shutdown: remove ${dir} failed: ${e && e.message ? e.message : e}`)
    }
  }
  try {
    server.close()
  } catch (e) {
    console.error(`[spinoml-llm] shutdown: server.close failed: ${e && e.message ? e.message : e}`)
  }
  const t = setTimeout(() => { process.exit(0) }, 2500)
  t.unref()
}

process.on('SIGTERM', () => shutdownAll('SIGTERM'))
process.on('SIGINT', () => shutdownAll('SIGINT'))

// Idle timeout for an upstream provider request: if no response headers (or no
// further stream chunk) arrive for this long, abort the request and surface an
// explicit `provider stalled` error. Validated like SPINOML_LLM_PORT — an
// invalid value is refused loudly at startup instead of silently defaulting.
const UPSTREAM_TIMEOUT_MS = (() => {
  const raw = process.env.SPINOML_LLM_UPSTREAM_TIMEOUT_MS
  if (raw === undefined || raw === '') return 120000
  const n = Number(raw)
  if (!Number.isInteger(n) || n <= 0) {
    console.error(`[spinoml-llm] invalid SPINOML_LLM_UPSTREAM_TIMEOUT_MS=${JSON.stringify(raw)} (expected a positive integer of milliseconds)`)
    process.exit(2)
  }
  return n
})()

// ────────────────────────────────────────────────────────────────────────────
// Graph state held only for the duration of one /chat turn.

function makeGraphContext(initial) {
  const nodes = new Map() // id → { id, layerType, params }
  const edges = new Map() // edge_key → { source, target }
  let counter = 0
  let inputShape = initial?.inputShape ?? [1, 3, 224, 224]

  for (const n of initial?.nodes ?? []) {
    nodes.set(n.id, { id: n.id, layerType: n.layerType, params: { ...(n.params ?? {}) } })
  }
  for (const e of initial?.edges ?? []) {
    edges.set(`${e.source}->${e.target}`, { source: e.source, target: e.target })
  }

  function nextId() {
    counter++
    while (nodes.has(`llm${counter}`)) counter++
    return `llm${counter}`
  }

  return {
    nodes,
    edges,
    nextId,
    setInputShape(s) { inputShape = s },
    snapshot() {
      return {
        input_shape: inputShape,
        nodes: [...nodes.values()],
        edges: [...edges.values()],
      }
    },
  }
}

// Transient TRAINING-graph state for one /chat turn — mirror of
// makeGraphContext but for the visual training graph (Phase 14).
function makeTrainingContext(initial) {
  const nodes = new Map() // id → { id, trainingType, params }
  const edges = new Map() // edge_key → { source, target }
  let counter = 0

  for (const n of initial?.nodes ?? []) {
    nodes.set(n.id, { id: n.id, trainingType: n.trainingType, params: { ...(n.params ?? {}) } })
  }
  for (const e of initial?.edges ?? []) {
    edges.set(`${e.source}->${e.target}`, { source: e.source, target: e.target })
  }

  function nextId() {
    counter++
    while (nodes.has(`tllm${counter}`)) counter++
    return `tllm${counter}`
  }

  return {
    nodes,
    edges,
    nextId,
    snapshot() {
      return { nodes: [...nodes.values()], edges: [...edges.values()] }
    },
  }
}

// Transient DATA-PROCESSING-graph state for one /chat turn — mirror of
// makeTrainingContext but for the visual data-prep pipeline canvas.
function makeDataContext(initial) {
  const nodes = new Map() // id → { id, dataType, params }
  const edges = new Map() // edge_key → { source, target }
  let counter = 0

  for (const n of initial?.nodes ?? []) {
    nodes.set(n.id, { id: n.id, dataType: n.dataType, params: { ...(n.params ?? {}) } })
  }
  for (const e of initial?.edges ?? []) {
    edges.set(`${e.source}->${e.target}`, { source: e.source, target: e.target })
  }

  function nextId() {
    counter++
    while (nodes.has(`dllm${counter}`)) counter++
    return `dllm${counter}`
  }

  return {
    nodes,
    edges,
    nextId,
    snapshot() {
      return { nodes: [...nodes.values()], edges: [...edges.values()] }
    },
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Per-turn action queue. Tool handlers push actions; the SSE writer drains.

function makeActionStream() {
  const queue = []
  let resolveNext = null
  let closed = false

  return {
    push(action) {
      if (closed) return
      queue.push(action)
      if (resolveNext) { const r = resolveNext; resolveNext = null; r() }
    },
    close() { closed = true; if (resolveNext) { const r = resolveNext; resolveNext = null; r() } },
    async *drain() {
      while (true) {
        while (queue.length) yield queue.shift()
        if (closed) return
        await new Promise((r) => { resolveNext = r })
      }
    },
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Workspace abstraction. The frontend tells us per turn whether file ops
// should hit the laptop's filesystem (local workspace) or an SSH target
// (remote-ssh workspace, root path is raw — may be tilde-prefixed). The
// helpers below dispatch on `ws.isRemote`. No new HTTP server: we just
// shell out to ssh from Node, which keeps auth + key handling identical
// to what Tauri does.

const SSH_OPTS = [
  '-o', 'BatchMode=yes',
  '-o', 'ConnectTimeout=10',
  '-o', 'ServerAliveInterval=20',
  '-o', 'ServerAliveCountMax=3',
]

function shellQuote(s) {
  return "'" + String(s).replace(/'/g, "'\\''") + "'"
}

/** Quote a path, expanding a leading `~/` to "$HOME" so the remote shell
 *  resolves it. Mirrors Rust's ssh.rs::shell_quote_path exactly. */
function shellQuotePath(s) {
  if (typeof s !== 'string' || s.length === 0) return shellQuote(s ?? '')
  if (s === '~') return '"$HOME"'
  if (s.startsWith('~/')) {
    return `"$HOME"${shellQuote('/' + s.slice(2))}`
  }
  return shellQuote(s)
}

function runSsh(target, remoteCmd, stdin) {
  const targetCheck = checkSshTarget(target)
  if (!targetCheck.ok) {
    return Promise.reject(new Error(`invalid ssh target: ${targetCheck.error}`))
  }
  return new Promise((resolve, reject) => {
    const child = spawn('ssh', [...SSH_OPTS, '--', target, remoteCmd], {
      stdio: ['pipe', 'pipe', 'pipe'],
    })
    let stdout = ''
    let stderr = ''
    child.stdout.on('data', (d) => { stdout += d.toString() })
    child.stderr.on('data', (d) => { stderr += d.toString() })
    child.on('error', reject)
    child.on('close', (code) => {
      if (code === 0) {
        resolve(stdout)
      } else {
        const trimmed = stderr.trim()
        const hint = code === 255 ? ' (255 = connection/auth; check ~/.ssh/config + agent)' : ''
        reject(new Error(`ssh exit ${code}: ${trimmed}${hint}`))
      }
    })
    if (stdin !== undefined && stdin !== null) {
      child.stdin.write(stdin)
    }
    child.stdin.end()
  })
}

function makeWorkspace(project) {
  if (!project || !project.root) return null
  return {
    root: String(project.root).replace(/\/$/, ''),
    sshTarget: project.ssh_target || null,
    isRemote: !!project.ssh_target,
  }
}

// ────────────────────────────────────────────────────────────────────────────
// MCP tool surface.

// Read a directory for a LISTING tool. A genuinely absent directory is an
// explicit empty list (a fresh project has no notes/ or runs/ yet); any other
// failure (permissions, IO, a scope rejection) propagates so the model is never
// told "nothing there" when the truth is "could not read it".
async function readdirOptional(abs, options) {
  try {
    return options ? await fs.readdir(abs, options) : await fs.readdir(abs)
  } catch (e) {
    if (e && e.code === 'ENOENT') return []
    throw e
  }
}

// Safe notes helpers — scoped to a single project root passed per-turn.
function safeNoteFilename(name) {
  const base = path.basename(name || '')
  if (!base || base.startsWith('.')) throw new Error('invalid note name')
  if (!/\.(md|txt)$/i.test(base)) throw new Error('note name must end in .md or .txt')
  return base
}

function safeDatasetFilename(name) {
  const base = path.basename(name || '')
  if (!base || base.startsWith('.') || base.includes('/') || base.includes('\\')) {
    throw new Error('invalid filename')
  }
  if (!/\.(csv|tsv|parquet|pq|json|jsonl|npy|npz|pt|pth|zip|tar|gz|tgz|pdb|sdf|smi|smiles|txt|manifest)$/i.test(base)) {
    throw new Error('unsupported extension; allowed: csv, tsv, parquet, pq, json, jsonl, npy, npz, pt, pth, zip, tar(.gz), pdb, sdf, smi, txt, manifest')
  }
  return base
}

async function notesList(ws) {
  if (ws.isRemote) {
    const dir = `${ws.root}/notes`
    const dirQ = shellQuotePath(dir)
    const out = await runSsh(
      ws.sshTarget,
      `mkdir -p ${dirQ} && find ${dirQ} -mindepth 1 -maxdepth 1 -type f ` +
      `\\( -name '*.md' -o -name '*.txt' \\) -printf '%f\\t%s\\n' 2>/dev/null`,
    )
    return out
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => {
        const [name, size] = line.split('\t')
        return { name, size: Number(size) || 0 }
      })
  }
  const dir = await resolveWsPath(ws, 'notes')
  const items = await readdirOptional(dir)
  const filtered = items.filter((f) => /\.(md|txt)$/i.test(f) && !f.startsWith('.'))
  const out = []
  for (const f of filtered.sort()) {
    try {
      const st = await fs.stat(path.join(dir, f))
      out.push({ name: f, size: st.size, mtime: new Date(st.mtimeMs).toISOString() })
    } catch (e) {
      // A stat failure must not claim a false size of 0; the notes tool renders
      // null as "(size unknown)" so the model can distinguish "0 bytes" from
      // "couldn't stat the file".
      out.push({ name: f, size: null, mtime: null, stat_error: e.message })
    }
  }
  return out
}

async function notesRead(ws, name) {
  const safe = safeNoteFilename(name)
  if (ws.isRemote) {
    const p = `${ws.root}/notes/${safe}`
    return await runSsh(ws.sshTarget, `cat ${shellQuotePath(p)}`)
  }
  return await fs.readFile(await resolveWsPath(ws, `notes/${safe}`), 'utf8')
}

async function notesAppend(ws, name, body) {
  const safe = safeNoteFilename(name)
  const text = body.endsWith('\n') ? body : body + '\n'
  if (ws.isRemote) {
    const dir = `${ws.root}/notes`
    const p = `${dir}/${safe}`
    await runSsh(
      ws.sshTarget,
      `mkdir -p ${shellQuotePath(dir)} && cat >> ${shellQuotePath(p)}`,
      text,
    )
  } else {
    const abs = await resolveWsPath(ws, `notes/${safe}`, { forWrite: true })
    await fs.mkdir(path.dirname(abs), { recursive: true })
    await fs.appendFile(abs, text, 'utf8')
  }
  return text.length
}

async function downloadToDatasets(ws, url, filename) {
  const checked = checkDownloadUrl(url)
  if (!checked.ok) {
    throw new Error(`url rejected: ${checked.error}`)
  }
  const safeUrl = checked.url
  const safe = safeDatasetFilename(filename)
  if (ws.isRemote) {
    const dir = `${ws.root}/datasets`
    const p = `${dir}/${safe}`
    // -fsSL → fail on HTTP errors, silent progress, follow redirects.
    // --max-time 300s keeps a hung download from hanging the chat turn.
    // --proto/--proto-redir/--max-redirs pin curl to http(s) and a small
    // redirect budget so an allowed URL can't bounce into another scheme.
    const out = await runSsh(
      ws.sshTarget,
      `mkdir -p ${shellQuotePath(dir)} && \
       curl -fsSL --max-time 300 --proto '=http,https' --proto-redir '=http,https' --max-redirs 5 ${shellQuote(safeUrl)} -o ${shellQuotePath(p)} && \
       wc -c < ${shellQuotePath(p)}`,
    )
    const bytes = parseInt(out.trim(), 10)
    if (!Number.isFinite(bytes) || bytes < 0) {
      // Never report an invented size of 0 for a file we just downloaded.
      throw new Error('download finished but its size could not be determined')
    }
    return { relpath: `datasets/${safe}`, bytes }
  }
  const dest = await resolveWsPath(ws, `datasets/${safe}`, { forWrite: true })
  const res = await safeFetch(safeUrl)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const buf = Buffer.from(await res.arrayBuffer())
  await fs.mkdir(path.dirname(dest), { recursive: true })
  await fs.writeFile(dest, buf)
  return { relpath: `datasets/${safe}`, bytes: buf.length }
}

// Write a text file (manifest JSON, a small CSV, a SMILES list, …) into the
// project's datasets/ folder. Text-only — binary payloads go through
// downloadToDatasets. Routes to the laptop FS or the SSH target like the rest.
async function writeDatasetFile(ws, filename, content) {
  const safe = safeDatasetFilename(filename)
  const text = String(content ?? '')
  if (ws.isRemote) {
    const dir = `${ws.root}/datasets`
    const p = `${dir}/${safe}`
    // `cat > file` truncates+writes; content arrives on stdin so no user
    // string is ever interpolated into the remote command.
    await runSsh(ws.sshTarget, `mkdir -p ${shellQuotePath(dir)} && cat > ${shellQuotePath(p)}`, text)
    return { relpath: `datasets/${safe}`, bytes: Buffer.byteLength(text) }
  }
  const abs = await resolveWsPath(ws, `datasets/${safe}`, { forWrite: true })
  await fs.mkdir(path.dirname(abs), { recursive: true })
  await fs.writeFile(abs, text, 'utf8')
  return { relpath: `datasets/${safe}`, bytes: Buffer.byteLength(text) }
}

// ── Generic read-only workspace helpers (local FS or SSH) — used by the run
//    tools. A missing dir/file is an explicit empty value; a read FAILURE
//    (permissions, IO, transport) propagates so a caller never reports an
//    empty/absent result as fact. ──
async function wsListDir(ws, reldir) {
  if (ws.isRemote) {
    // A missing dir is an explicit empty list; a permission/transport failure
    // exits non-zero and runSsh throws instead of reading as "no runs".
    const d = shellQuotePath(`${ws.root}/${reldir}`)
    const out = await runSsh(ws.sshTarget, `if [ ! -d ${d} ]; then exit 0; fi; ls -1 ${d}`)
    return out.split('\n').map((s) => s.trim()).filter(Boolean)
  }
  const abs = await resolveWsPath(ws, safeRelpath(reldir))
  return await readdirOptional(abs)
}

async function wsReadFile(ws, relpath) {
  if (ws.isRemote) {
    // An absent file yields '' (exit 0); a read failure (permissions, ssh)
    // exits non-zero and runSsh throws, so callers can tell missing from unreadable.
    const p = shellQuotePath(`${ws.root}/${relpath}`)
    return await runSsh(ws.sshTarget, `if [ -e ${p} ]; then cat ${p}; fi`)
  }
  const abs = await resolveWsPath(ws, safeRelpath(relpath))
  try {
    return await fs.readFile(abs, 'utf8')
  } catch (e) {
    if (e && e.code === 'ENOENT') return ''
    throw e
  }
}

// ── Generic agent FS + exec helpers (Phase: agent workspace tools) ──────────
// These back read_file / write_file / list_dir / run_script. Unlike the
// read-only helpers above, they surface errors (missing file → throw) so the
// agent gets honest feedback instead of an empty string.

// Reject absolute paths, tilde paths, and any `..` segment. The JS counterpart
// of Rust resolve() — keeps every path under the workspace root.
function safeRelpath(rel) {
  const s = String(rel ?? '').trim().replace(/^\.\//, '')
  if (!s) throw new Error('empty path')
  if (s.startsWith('/') || s.startsWith('~')) throw new Error('path must be relative to the workspace root')
  const segs = s.split('/').filter((seg) => seg.length && seg !== '.')
  if (segs.some((seg) => seg === '..')) throw new Error('path may not contain ".."')
  return segs.join('/')
}

// Symlink-aware containment for LOCAL workspace paths (remote/ssh paths stay
// lexical — they are shell-quoted relative paths on the host). safeRelpath is
// the lexical first pass; resolveInWorkspace then resolves the real path and
// requires it inside the realpath of the workspace root, OR inside a
// user-configured symlink target (env SPINOML_SYMLINK_TARGETS / scope.json).
// A rejection throws the same kind of explicit error the callers already
// surface; the code is included in the message. `abs` (the RESOLVED path) must
// be used for every later operation.
let lastScopeLoadError = null
async function resolveWsPath(ws, safe, { forWrite = false } = {}) {
  const scope = loadSymlinkTargets()
  if (scope.loadError && scope.loadError !== lastScopeLoadError) {
    lastScopeLoadError = scope.loadError
    console.error(`[path-scope] scope config ignored: ${scope.loadError}`)
  }
  const r = await resolveInWorkspace(ws.root, safe, { forWrite, symlinkTargets: scope.targets })
  if (!r.ok) throw new Error(`${r.code}: ${r.error}`)
  return r.abs
}

// write_file carries a UTF-8 string, so it can write any TEXT file. The path is
// confined to the workspace root by safeRelpath (no .. / absolute / ~) and the
// symlink-aware resolveWsPath (the real path must stay inside the root or an
// allow-listed target), which is the real safety boundary — within the root the
// agent may write any file in any directory (scripts, configs, models,
// manifests, notes). The only guard left is refusing known-binary extensions,
// which a text write would corrupt; binary payloads must go through
// download_to_datasets instead.
const BINARY_EXT_RE = /\.(pt|pth|ckpt|safetensors|h5|hdf5|pkl|pickle|joblib|onnx|msgpack|npy|npz|zip|tar|gz|tgz|bz2|xz|7z|db|sqlite|sqlite3|png|jpe?g|gif|bmp|webp|ico|tiff?|pdf|parquet|pq|feather|arrow|so|dylib|dll|exe|bin|wav|mp3|mp4|mov|woff2?|ttf|otf)$/i
// A few in-root dirs are never legitimate write targets and would turn an
// unconfirmed text write into code execution (a planted .git hook, or a
// .claude/.ssh file, runs OUTSIDE run_script's confirm gate). safeRelpath
// already confines writes to the workspace root; this denies the dangerous
// corners inside it while leaving all real data/script targets writable.
const WRITE_DENY_RE = /^(\.git|node_modules|\.ssh|\.claude)(\/|$)/
function guardWritePath(safe) {
  if (WRITE_DENY_RE.test(safe)) {
    throw new Error('write_file may not touch .git, node_modules, .ssh, or .claude')
  }
  if (BINARY_EXT_RE.test(safe)) {
    throw new Error('write_file is text-only and that extension is binary — use download_to_datasets for binary payloads')
  }
}

function capText(s, max = 8000) {
  const t = String(s ?? '')
  return t.length > max ? t.slice(0, max) + `\n…[truncated ${t.length - max} chars]` : t
}

async function wsWriteText(ws, relpath, text) {
  const safe = safeRelpath(relpath)
  if (ws.isRemote) {
    const p = `${ws.root}/${safe}`
    const dir = p.slice(0, p.lastIndexOf('/'))
    await runSsh(ws.sshTarget, `mkdir -p ${shellQuotePath(dir)} && cat > ${shellQuotePath(p)}`, text)
  } else {
    const abs = await resolveWsPath(ws, safe, { forWrite: true })
    await fs.mkdir(path.dirname(abs), { recursive: true })
    await fs.writeFile(abs, text, 'utf8')
  }
  return { relpath: safe, bytes: Buffer.byteLength(text) }
}

// Does a file/dir exist under the workspace root? Used to fail run_script fast
// with an actionable message instead of a cryptic `python: can't open file`
// when the target was never written (or written to a different path).
async function wsPathExists(ws, relpath) {
  const safe = safeRelpath(relpath)
  if (ws.isRemote) {
    const out = await runSsh(ws.sshTarget, `test -e ${shellQuotePath(`${ws.root}/${safe}`)} && echo MLF_YES || echo MLF_NO`)
    return out.includes('MLF_YES')
  }
  const abs = await resolveWsPath(ws, safe)
  try { await fs.access(abs); return true } catch { /* absent feeds an explicit "does not exist" error, never a write claim */ return false }
}

async function wsReadTextStrict(ws, relpath, cap = 100000) {
  const safe = safeRelpath(relpath)
  let text
  if (ws.isRemote) {
    // No `|| true` here — a missing file makes cat exit non-zero, which runSsh
    // turns into a thrown error the agent can see.
    text = await runSsh(ws.sshTarget, `cat ${shellQuotePath(`${ws.root}/${safe}`)}`)
  } else {
    text = await fs.readFile(await resolveWsPath(ws, safe), 'utf8')
  }
  return capText(text, cap)
}

async function wsListDirDetailed(ws, reldir) {
  const safe = reldir ? safeRelpath(reldir) : ''
  if (ws.isRemote) {
    const target = safe ? `${ws.root}/${safe}` : ws.root
    // Same contract as the local branch: absent dir → empty list, but a real
    // listing failure must not be reported as "empty or missing directory".
    const out = await runSsh(ws.sshTarget, `if [ ! -d ${shellQuotePath(target)} ]; then exit 0; fi; cd ${shellQuotePath(target)} && ls -1Ap`)
    return out.split('\n').map((s) => s.trim()).filter(Boolean).map((name) => ({
      name: name.replace(/\/$/, ''),
      is_dir: name.endsWith('/'),
    }))
  }
  const abs = await resolveWsPath(ws, safe || '.')
  const ents = await readdirOptional(abs, { withFileTypes: true })
  return ents.map((e) => ({ name: e.name, is_dir: e.isDirectory() }))
}

// Run a command, capturing stdout/stderr/exit-code WITHOUT throwing on a
// non-zero exit (scripts legitimately fail and the agent must see why).
// opts.onChunk(text, isErr) streams output live; opts.timeoutMs kills a run
// that overstays its welcome (the login-node guard) and sets timedOut.
function spawnCapture(file, args, opts = {}) {
  const { onChunk, timeoutMs, signal } = opts
  return new Promise((resolve) => {
    if (signal?.aborted) { resolve({ code: -1, stdout: '', stderr: 'aborted', timedOut: false, aborted: true }); return }
    // `signal` (the turn's AbortController) → Node sends SIGTERM to the child,
    // so hitting Stop in the chat actually kills the running script.
    // NOT detached: the child stays in the sidecar's process group, so a
    // SIGKILL to the sidecar (e.g. a test harness / the OS on an app crash)
    // still reaps it, and the SIGTERM handler reaps it via `trackedChildren`
    // + the /proc descendant walk (works whether or not the child detached).
    const child = spawn(file, args, { stdio: ['ignore', 'pipe', 'pipe'], signal })
    trackChild(child)
    let stdout = ''; let stderr = ''; let timedOut = false; let aborted = false
    if (signal) signal.addEventListener('abort', () => { aborted = true }, { once: true })
    const onData = (d, isErr) => {
      const s = d.toString()
      if (isErr) stderr += s; else stdout += s
      if (onChunk) { try { onChunk(s, isErr) } catch { /* never let a listener break the run */ } }
    }
    child.stdout.on('data', (d) => onData(d, false))
    child.stderr.on('data', (d) => onData(d, true))
    let timer = null
    if (timeoutMs) {
      timer = setTimeout(() => {
        timedOut = true
        try { child.kill('SIGTERM') } catch { /* process already exited — pure cleanup */ }
        setTimeout(() => { try { child.kill('SIGKILL') } catch { /* process already exited — pure cleanup */ } }, 3000)
      }, timeoutMs)
    }
    child.on('error', (e) => { if (timer) clearTimeout(timer); resolve({ code: -1, stdout, stderr: stderr + String(e), timedOut, aborted }) })
    child.on('close', (code) => { if (timer) clearTimeout(timer); resolve({ code, stdout, stderr, timedOut, aborted }) })
  })
}

function execIn(ws, cmd, opts) {
  return ws.isRemote
    ? spawnCapture('ssh', [...SSH_OPTS, ws.sshTarget, cmd], opts)
    : spawnCapture('bash', ['-lc', cmd], opts)
}

// A shell-mode run is for QUICK work. On a remote workspace it executes on the
// LOGIN NODE, so it's capped — heavy compute goes through SLURM. The cap matches
// the "~2 min" the tool + system prompt advertise (they were out of sync with a
// 60s value before); overshoot ⇒ not trivial ⇒ killed + told to use SLURM, so
// the user is never stuck waiting on the login node.
const SHELL_TIMEOUT_REMOTE_MS = 2 * 60 * 1000  // login node: ~2 min, then killed
const SHELL_TIMEOUT_LOCAL_MS = 10 * 60 * 1000  // local laptop: more headroom

// Execute a workspace script. 'shell' runs it now (python/.sh/raw) in the
// workspace root, streaming output via notify(); 'slurm' submits it via sbatch
// and parses the job id (returns immediately — the heavy work runs on a node).
async function runScript(ws, relpath, mode, args, notify, signal) {
  const safe = safeRelpath(relpath)
  const parsedArgs = splitArgs(args)
  if (!parsedArgs.ok) {
    return { mode, code: -1, stdout: '', stderr: `invalid args: ${parsedArgs.error}`, jobId: null, timedOut: false, aborted: false }
  }
  const argStr = parsedArgs.argv.length ? ` ${quoteArgv(parsedArgs.argv)}` : ''
  // `-u` → unbuffered Python so prints stream to the chat live (an ssh pipe is
  // not a TTY, so the default block buffering would otherwise hide all output
  // until the script exits, making a slow run look frozen).
  const interp = /\.py$/i.test(safe) ? 'python -u ' : /\.sh$/i.test(safe) ? 'bash ' : ''
  const cd = `cd ${shellQuotePath(ws.root)}`
  // './' prefix: a workspace-relative name that starts with '-' (e.g.
  // `--wrap=...`) must never be parsed by sbatch/python/bash as an OPTION.
  const target = shellQuotePath(`./${safe}`)
  if (mode === 'slurm') {
    if (notify) notify(`$ sbatch ${safe}${argStr}\n`)
    const r = await execIn(ws, `${cd} && sbatch ${target}${argStr}`, { timeoutMs: 60000, signal })
    const m = `${r.stdout}\n${r.stderr}`.match(/Submitted batch job (\d+)/)
    if (notify) notify(`${r.stdout}${r.stderr}`)
    return { mode, code: r.code, stdout: capText(r.stdout), stderr: capText(r.stderr), jobId: m ? m[1] : null, timedOut: r.timedOut, aborted: r.aborted }
  }
  const timeoutMs = ws.isRemote ? SHELL_TIMEOUT_REMOTE_MS : SHELL_TIMEOUT_LOCAL_MS
  if (notify) notify(`$ ${interp}${safe}${argStr}\n`)
  const onChunk = notify ? (s) => notify(s) : undefined
  const r = await execIn(ws, `${cd} && ${interp}${target}${argStr}`, { onChunk, timeoutMs, signal })
  return { mode, code: r.code, stdout: capText(r.stdout), stderr: capText(r.stderr), jobId: null, timedOut: r.timedOut, aborted: r.aborted }
}

async function slurmStatus(ws, jobId) {
  const id = String(jobId ?? '').replace(/[^0-9_]/g, '')
  if (!id) throw new Error('invalid job id')
  // squeue covers PENDING/RUNNING; sacct covers finished jobs. Try both.
  const r = await execIn(ws, `squeue -j ${id} -h -o %T 2>/dev/null; sacct -j ${id} --format=State -n -P 2>/dev/null | head -1`)
  // A timeout/abort or an ssh transport failure (exit 255) is NOT a job state —
  // surface it instead of inventing an "UNKNOWN" the model would read as fact.
  if (r.timedOut) throw new Error('slurm_status probe timed out')
  if (r.aborted) throw new Error('slurm_status probe was stopped')
  if (r.code === 255) {
    const tail = r.stderr.trim().split('\n').filter(Boolean).slice(-1)[0] || ''
    throw new Error(`ssh to the workspace host failed (255)${tail ? `: ${tail}` : ''}`)
  }
  const state = (r.stdout || '').split('\n').map((s) => s.trim()).filter(Boolean)[0] || 'UNKNOWN'
  return { jobId: id, state }
}

function isAffirmative(a) {
  if (a === true) return true
  if (a === false || a == null) return false
  return /^(y|yes|ok|confirm|approve|run|true|ausführen)$/i.test(String(a).trim())
}

// ── Bidirectional ask/answer channel ────────────────────────────────────────
// A tool handler running inside a live /chat turn can pause, ask the frontend a
// question (SSE `ask`), and await the answer posted to POST /respond. Backs
// both the ask_user tool and run_script's confirmation gate.
const pendingAsks = new Map() // askId → { resolve, reject }
const ASK_TIMEOUT_MS = 10 * 60 * 1000

// Build an askUser bound to one turn's emit + a registry for cleanup.
function makeAsker(pushEvent, requestId, registry) {
  return function askUser({ kind, prompt, options, payload }) {
    const askId = randomUUID()
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        if (pendingAsks.delete(askId)) { registry.delete(askId); reject(new Error('no answer (timed out)')) }
      }, ASK_TIMEOUT_MS)
      pendingAsks.set(askId, {
        resolve: (v) => { clearTimeout(timer); resolve(v) },
        reject: (e) => { clearTimeout(timer); reject(e) },
      })
      registry.add(askId)
      pushEvent({ type: 'ask', id: askId, kind, prompt, options: options ?? null, payload: payload ?? null })
    })
  }
}

function resolveAsk(askId, answer) {
  const entry = pendingAsks.get(askId)
  if (!entry) return false
  pendingAsks.delete(askId)
  entry.resolve(answer)
  return true
}

function safeRunId(id) {
  const base = path.basename(String(id || ''))
  if (!base || base.startsWith('.')) throw new Error('invalid run id')
  return base
}

// Read one JSON run artifact and report an EXPLICIT state. `missing` (no file)
// is a legitimate state — metrics.json is optional and externally launched runs
// may lack run.json — while `corrupt`/`unreadable` must never masquerade as an
// empty config the model would read as "this run has no settings".
async function readRunJson(ws, relpath) {
  let text
  try {
    text = await wsReadFile(ws, relpath)
  } catch (e) {
    return { value: {}, state: 'unreadable', error: e.message }
  }
  if (!text.trim()) return { value: {}, state: 'missing', error: null }
  try {
    return { value: JSON.parse(text), state: 'ok', error: null }
  } catch (e) {
    return { value: {}, state: 'corrupt', error: e.message }
  }
}

const RUNS_DIR = 'experiments/runs'
const RUN_LIST_CAP = 25

// Read ONLY the epoch.end / run.done lines of a run's events.jsonl — cheap even
// when the file carries thousands of per-batch lines (remote: grep on the host;
// local: filter after read). These lines carry the loss/epoch summary. A missing
// log is an explicit empty summary; an unreadable log returns an explicit error
// the caller surfaces — never a silent empty summary that looks like "no epochs".
async function readSummaryEvents(ws, id) {
  const p = `${RUNS_DIR}/${id}/events.jsonl`
  if (ws.isRemote) {
    const f = shellQuotePath(`${ws.root}/${p}`)
    // grep exits 1 for "no match" (a legitimately empty summary) and 2 for a
    // real read error: map only 1 → 0, let anything else make runSsh throw.
    const cmd = `if [ ! -e ${f} ]; then exit 0; fi; grep -hE '"(epoch.end|run.done)"' ${f} || { rc=$?; [ $rc -eq 1 ] && exit 0; exit $rc; }`
    try {
      return { text: await runSsh(ws.sshTarget, cmd), error: null }
    } catch (e) {
      return { text: '', error: e.message }
    }
  }
  let raw
  try {
    raw = await fs.readFile(await resolveWsPath(ws, p), 'utf8')
  } catch (e) {
    if (e && e.code === 'ENOENT') return { text: '', error: null }
    return { text: '', error: e.message }
  }
  return {
    text: raw.split('\n').filter((l) => l.includes('epoch.end') || l.includes('run.done')).join('\n'),
    error: null,
  }
}

// Derive (best_val_loss, completed_epochs) from the filtered events: min
// val_loss over epoch.end, run.done's best_val_loss, and the highest epoch seen.
function scanRunEvents(text) {
  let minVal = null; let doneBest = null; let maxEpoch = null
  for (const line of String(text).split('\n')) {
    const s = line.trim(); if (!s) continue
    let e; try { e = JSON.parse(s) } catch { continue }
    if (e.kind === 'epoch.end') {
      if (typeof e.val_loss === 'number') minVal = minVal == null ? e.val_loss : Math.min(minVal, e.val_loss)
      if (typeof e.epoch === 'number') maxEpoch = maxEpoch == null ? e.epoch : Math.max(maxEpoch, e.epoch)
    } else if (e.kind === 'run.done' && typeof e.best_val_loss === 'number') {
      doneBest = e.best_val_loss
    }
  }
  return { bestValLoss: doneBest ?? minVal, completedEpochs: maxEpoch == null ? null : maxEpoch + 1 }
}

// List recent training runs with a one-line summary each (status, best val
// loss, model). Reads run.json + metrics.json per run, falling back to
// events.jsonl so EXTERNALLY launched runs (sbatch / hand-written train.py that
// write events but no metrics.json) still report loss + epochs.
async function runsList(ws) {
  const ids = (await wsListDir(ws, RUNS_DIR)).filter((n) => !n.startsWith('.')).sort().reverse()
  const capped = ids.slice(0, RUN_LIST_CAP)
  const out = []
  for (const id of capped) {
    const cfgR = await readRunJson(ws, `${RUNS_DIR}/${id}/run.json`)
    const metR = await readRunJson(ws, `${RUNS_DIR}/${id}/metrics.json`)
    const evR = await readSummaryEvents(ws, id)
    const cfg = cfgR.value; const met = metR.value
    const warnings = []
    for (const [name, r] of [['run.json', cfgR], ['metrics.json', metR]]) {
      if (r.state === 'missing') warnings.push(`${name} missing`)
      else if (r.state !== 'ok') warnings.push(`${name} ${r.state}: ${r.error}`)
    }
    if (evR.error) warnings.push(`events.jsonl unreadable: ${evR.error}`)
    const ev = scanRunEvents(evR.text)
    out.push({
      run_id: id,
      label: cfg.run_label ?? null,
      status: met.status ?? cfg.status ?? 'unknown',
      model_path: cfg.model_path ?? null,
      dataset: cfg.dataset?.relpath ?? cfg.dataset?.path ?? null,
      best_val_loss: met.best_val_loss ?? cfg.best_val_loss ?? ev.bestValLoss ?? null,
      epochs: cfg.training?.epochs ?? cfg.epochs ?? ev.completedEpochs ?? null,
      warning: warnings.length ? warnings.join('; ') : null,
    })
  }
  return { runs: out, total: ids.length, shown: capped.length }
}

// Read ONE run in detail: its config (hyperparameters), the per-epoch metric
// history (epoch.end events) and the final outcome — enough to reason about
// over/underfitting, LR, capacity. Bulky events (per-step `batch`, prediction
// `sample.preds`) are dropped.
async function runRead(ws, runId) {
  const id = safeRunId(runId)
  const cfgR = await readRunJson(ws, `${RUNS_DIR}/${id}/run.json`)
  const metR = await readRunJson(ws, `${RUNS_DIR}/${id}/metrics.json`)
  const config = cfgR.value
  const metrics = metR.value
  const warnings = []
  for (const [name, r] of [['run.json', cfgR], ['metrics.json', metR]]) {
    if (r.state === 'missing') warnings.push(`${name} missing`)
    else if (r.state !== 'ok') warnings.push(`${name} ${r.state}: ${r.error}`)
  }

  const events = await wsReadFile(ws, `${RUNS_DIR}/${id}/events.jsonl`)
  const history = []
  let nParams = null; let done = null; let failed = null; let dataset = null; let earlystop = null
  for (const line of events.split('\n')) {
    const s = line.trim(); if (!s) continue
    let e; try { e = JSON.parse(s) } catch { continue }
    switch (e.kind) {
      case 'epoch.end':
        history.push({ epoch: e.epoch, train_loss: e.train_loss, val_loss: e.val_loss, val_acc: e.val_acc, metrics: e.metrics ?? null, lr: e.lr })
        break
      case 'model.built': nParams = e.n_params ?? null; break
      case 'dataset.loaded': { const { rows, ...rest } = e; dataset = rest; break }
      case 'run.done': done = { best_val_loss: e.best_val_loss, total_seconds: e.total_seconds }; break
      case 'run.failed': failed = { stage: e.stage, error: e.error }; break
      case 'run.earlystop': earlystop = { epoch: e.epoch, monitor: e.monitor, best: e.best }; break
      default: break
    }
  }
  return { run_id: id, status: metrics.status ?? config.status ?? 'unknown', config, warnings, n_params: nParams, dataset, history, earlystop, done, failed }
}

function buildToolSpecs(ctx, trainingCtx, dataCtx, actions, workspace, askUser, pushEvent, turnSignal, autoApproveShell = false) {
  // Local shadow of the SDK's tool() helper: collect provider-agnostic specs
  // instead of SDK tools. Same (name, description, zodShape, handler) signature,
  // so every tool definition below is reused verbatim across all providers.
  const tool = (name, description, schema, handler) => ({ name, description, schema, handler })
  const specs = [
    tool(
      'set_input_shape',
      'Change the model input tensor shape. Example shapes: [1, 3, 224, 224] for ImageNet RGB, [1, 16, 512] for a sequence of 16 tokens with 512 features.',
      { shape: z.array(z.number().int().positive()).min(2).max(6) },
      async ({ shape }) => {
        ctx.setInputShape(shape)
        actions.push({ op: 'set_input_shape', payload: { shape } })
        return content(`input shape set to [${shape.join(', ')}]`)
      },
    ),
    tool(
      'add_layer',
      'Add a new layer to the architecture. Use "after" to wire it after an existing node id. Supported layer_type values are: Conv1d, Conv2d, Conv3d, ConvTranspose2d, Linear, Flatten, Embedding, BatchNorm1d, BatchNorm2d, LayerNorm, GroupNorm, ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax, LogSoftmax, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, Dropout, Dropout2d, MultiheadAttention, TransformerEncoderLayer, TransformerEncoder, LSTM, GRU, RNN, GCNConv, GATConv, SAGEConv, GraphConv, GlobalMeanPool, GlobalMaxPool, GlobalAddPool, Reshape, View, Permute, Transpose, Concat, Add, Multiply, Stack, Output, Graph, Manifest, Sequence, ESPF, GraphTransformer, BuildGraph, DataOp. Pass params as a JSON object of layer-specific fields (e.g. {in_channels: 3, out_channels: 64} for Conv2d). DataOp is a DATA-stage node (not a model layer): it carries a Python `script` (params: {input_dataset, output_name, mode:"shell"|"slurm", cache, script}) that you run with run_script to download/tokenize/cache a dataset offline; it is a passthrough in the model forward. Drop one, write its `script` with the transform, then run_script it. Notes: Embedding needs an Input with dtype \'int64\' (token ids). LSTM/GRU/RNN take {input_size, hidden_size} and emit the sequence output (state is dropped). TransformerEncoder stacks num_layers encoder blocks. Reshape preserves the batch dim — its shape param is per-sample (use -1 to infer), e.g. {shape: [16, -1]}. Permute dims include the batch dim, e.g. {dims: [0, 2, 1]}. Graph = ONE node carrying a whole PyG Data (x + edge_index + batch + edge_attr) — the PRIMARY GNN input; feed it into a GNN/pool and the generator unpacks `x, edge_index, batch = g.x, …` automatically (put each branch of a dual-encoder in its own Subgraph). Manifest = a NON-emitting pairing descriptor: bind it to a .manifest and draw an edge from it to each typed input (Graph/Sequence/ESPF/Input) to declare which branch feeds which — it generates no model code, only feeds the paired data at smoke/train. Sequence = a token-id LongTensor input (e.g. a protein_seq manifest branch) for an Embedding/Transformer encoder. ESPF = a SMILES input tokenized into INTERPRETABLE SUBSTRUCTURE subword tokens (LongTensor) via the MolTrans ESPF BPE codebook (params {codebook:"drug"|"protein", shape, branch}); bind it to a manifest branch declared kind:"espf" and feed it into an Embedding (num_embeddings ≈ ESPF vocab, shown when you inspect the manifest) then a 1D-CNN/Transformer — like Sequence but chemically meaningful and each token ↔ a named substructure. GraphTransformer = PyG TransformerConv (multi-head graph attention). BuildGraph constructs edges from plain features (e.g. kNN) when you have no edge_index. GNN layers (GCNConv/GATConv/SAGEConv/GraphConv/GraphTransformer) operate on node features [N_nodes, in_channels] (no batch dim); GlobalMean/Max/AddPool are graph-level readouts. (Legacy/manual alternative to a Graph node: separate Input nodes named \'edge_index\' [int64, [2, E]] and \'batch\' [int64, [N_nodes]].)',
      {
        layer_type: z.string(),
        after: z.string().optional().describe('Optional source node id to connect from'),
        params: z.record(z.string(), z.unknown()).optional(),
      },
      async ({ layer_type, after, params }) => {
        // Validate BEFORE any mutation, including `after`. Defaults are NOT
        // filled here — the frontend fills them; we store only what was given.
        const v = validateNodeParams('layers', layer_type, params ?? {})
        if (!v.ok) return content(`error: ${v.error}`, true)
        if (after && !ctx.nodes.has(after)) return content(`error: source node "${after}" not found`, true)
        const id = ctx.nextId()
        ctx.nodes.set(id, { id, layerType: layer_type, params: v.params })
        actions.push({ op: 'add_layer', payload: { id, layer_type, params: v.params } })
        if (after) {
          ctx.edges.set(`${after}->${id}`, { source: after, target: id })
          actions.push({ op: 'connect', payload: { source: after, target: id } })
        }
        return content(`added ${layer_type} as ${id}${after ? ` after ${after}` : ''}`)
      },
    ),
    tool(
      'add_custom_node',
      'Add a Custom node — a free-form nn.Module you write yourself. This is the escape hatch for ops no '
      + 'built-in layer covers: bilinear/MLP heads, custom attention, relational message passing, fusing two '
      + 'branches, etc. `source` MUST define `class X(nn.Module)` with __init__ and forward; torch, nn and F '
      + '(torch.nn.functional) are pre-imported (put any other imports at the top of source). In forward the '
      + 'node receives ALL incoming edges as positional args in edge order — so wire each predecessor with a '
      + 'connect() (or use "after" for a single one). `init_args` is the constructor arg string '
      + '(e.g. "384, 256, dropout=0.1"); leave empty when __init__ takes no args (prefer nn.LazyLinear so '
      + 'shapes infer automatically).',
      {
        source: z.string().describe('Full Python: class X(nn.Module) with __init__ and forward'),
        init_args: z.string().optional().describe('Constructor args, e.g. "256, dropout=0.1"'),
        after: z.string().optional().describe('Optional source node id to connect from'),
      },
      async ({ source, init_args, after }) => {
        if (!/class\s+\w+\s*\(/.test(source)) {
          return content('error: source must define a `class X(nn.Module): ...`', true)
        }
        const v = validateNodeParams('layers', 'Custom', { source, init_args: init_args ?? '' })
        if (!v.ok) return content(`error: ${v.error}`, true)
        if (after && !ctx.nodes.has(after)) return content(`error: source node "${after}" not found`, true)
        const id = ctx.nextId()
        ctx.nodes.set(id, { id, layerType: 'Custom', params: v.params })
        actions.push({ op: 'add_layer', payload: { id, layer_type: 'Custom', params: v.params } })
        if (after) {
          ctx.edges.set(`${after}->${id}`, { source: after, target: id })
          actions.push({ op: 'connect', payload: { source: after, target: id } })
        }
        return content(`added Custom node ${id}${after ? ` after ${after}` : ''}`)
      },
    ),
    tool(
      'add_subgraph',
      'Add a Subgraph node — a node that is ITSELF a mini-graph, compiled to a nested nn.Module class. Use it '
      + 'to encapsulate a reusable block (an encoder, a residual block) or one branch of a dual-encoder. '
      + 'Provide the inner `nodes` (each {id, layer_type, params}) and `edges` ({source, target}); the SAME '
      + 'layer_type/params vocabulary as add_layer. Include at least one Input node (its `name` param becomes a '
      + 'forward arg) and one Output node. The parent wires its incoming edges to the inner Inputs (by matching '
      + 'name when possible, else positionally — so name inner Inputs to match the outer source vars). '
      + '`class_name` names the nested class. Use "after" to wire from a single predecessor.',
      {
        class_name: z.string(),
        nodes: z.array(z.object({
          id: z.string(),
          layer_type: z.string(),
          params: z.record(z.string(), z.unknown()).optional(),
        })).min(2),
        edges: z.array(z.object({ source: z.string(), target: z.string() })),
        after: z.string().optional(),
      },
      async ({ class_name, nodes, edges, after }) => {
        const hasInput = nodes.some((n) => n.layer_type === 'Input' || n.layer_type === 'Graph')
        const hasOutput = nodes.some((n) => n.layer_type === 'Output')
        if (!hasInput || !hasOutput) {
          return content('error: a subgraph needs at least one Input (or Graph) node and one Output node', true)
        }
        // Validate every inner node + every inner edge endpoint BEFORE mutating.
        const innerIds = new Set(nodes.map((n) => n.id))
        for (const n of nodes) {
          const v = validateNodeParams('layers', n.layer_type, n.params ?? {})
          if (!v.ok) return content(`error: inner node ${n.id}: ${v.error}`, true)
        }
        for (const e of edges) {
          if (!innerIds.has(e.source) || !innerIds.has(e.target)) {
            return content(`error: inner edge ${e.source}->${e.target} references an unknown node`, true)
          }
          if (wouldCreateCycle(edges, e.source, e.target)) {
            return content(`error: inner edge ${e.source}->${e.target} would create a cycle`, true)
          }
        }
        const v = validateNodeParams('layers', 'Subgraph', { class_name })
        if (!v.ok) return content(`error: ${v.error}`, true)
        if (after && !ctx.nodes.has(after)) return content(`error: source node "${after}" not found`, true)
        const id = ctx.nextId()
        const subgraph = {
          nodes: nodes.map((n, i) => ({
            id: n.id, layerType: n.layer_type, params: n.params ?? {},
            position: { x: 80 + (i % 3) * 180, y: 80 + Math.floor(i / 3) * 110 },
          })),
          edges: edges.map((e) => ({ source: e.source, target: e.target })),
        }
        const params = { class_name: v.params.class_name, subgraph }
        ctx.nodes.set(id, { id, layerType: 'Subgraph', params })
        actions.push({ op: 'add_layer', payload: { id, layer_type: 'Subgraph', params } })
        if (after) {
          ctx.edges.set(`${after}->${id}`, { source: after, target: id })
          actions.push({ op: 'connect', payload: { source: after, target: id } })
        }
        return content(`added Subgraph "${class_name}" (${nodes.length} inner nodes) as ${id}${after ? ` after ${after}` : ''}`)
      },
    ),
    tool(
      'connect',
      'Wire one node\'s output into another node\'s input.',
      { source: z.string(), target: z.string() },
      async ({ source, target }) => {
        if (!ctx.nodes.has(source)) return content(`error: source "${source}" not found`, true)
        if (!ctx.nodes.has(target)) return content(`error: target "${target}" not found`, true)
        if (source === target) return content(`error: cannot connect a node to itself ("${source}")`, true)
        const key = `${source}->${target}`
        // Idempotent duplicate: explicit non-error, but NO action (no change).
        if (ctx.edges.has(key)) return content(`edge ${source}->${target} already exists (no change)`)
        if (wouldCreateCycle(ctx.edges, source, target)) {
          return content(`error: edge ${source}->${target} would create a cycle`, true)
        }
        const targetKind = nodeKind('layers', ctx.nodes.get(target).layerType)
        const sourceKind = nodeKind('layers', ctx.nodes.get(source).layerType)
        if (targetKind === 'input' && sourceKind !== 'manifest') {
          return content(`error: cannot connect into Input node "${target}" (only a Manifest may feed an Input)`, true)
        }
        ctx.edges.set(key, { source, target })
        actions.push({ op: 'connect', payload: { source, target } })
        return content(`connected ${source} → ${target}`)
      },
    ),
    tool(
      'update_params',
      'Patch parameters on an existing node. Only the supplied keys change. For shape-typed params like LayerNorm.normalized_shape, pass a list of ints.',
      { id: z.string(), params: z.record(z.string(), z.unknown()) },
      async ({ id, params }) => {
        const n = ctx.nodes.get(id)
        if (!n) return content(`error: node "${id}" not found`, true)
        const v = validateNodeParams('layers', n.layerType, params, { partial: true })
        if (!v.ok) return content(`error: ${v.error}`, true)
        n.params = { ...n.params, ...v.params }
        actions.push({ op: 'update_params', payload: { id, params: v.params } })
        return content(`patched ${id}: ${JSON.stringify(v.params)}`)
      },
    ),
    tool(
      'delete_node',
      'Remove a node and its incident edges. The Input node cannot be deleted.',
      { id: z.string() },
      async ({ id }) => {
        if (id === 'input') return content(`error: cannot delete the Input node`, true)
        if (!ctx.nodes.has(id)) return content(`error: node "${id}" not found`, true)
        const removed = ctx.nodes.get(id)
        const wasInput = removed.layerType === 'Input'
        const wasOutput = removed.layerType === 'Output'
        ctx.nodes.delete(id)
        for (const [key, e] of ctx.edges) {
          if (e.source === id || e.target === id) ctx.edges.delete(key)
        }
        actions.push({ op: 'delete_node', payload: { id } })
        let msg = `deleted ${id}`
        if (wasInput && ![...ctx.nodes.values()].some((n) => n.layerType === 'Input')) {
          msg += '\nwarning: graph now has no Input'
        }
        if (wasOutput && ![...ctx.nodes.values()].some((n) => n.layerType === 'Output')) {
          msg += '\nwarning: graph now has no Output'
        }
        return content(msg)
      },
    ),
    // ─── Training-graph tools (Phase 14) ───────────────────────────────────
    tool(
      'add_training_node',
      'Add a node to the TRAINING graph (separate from the architecture graph above). '
      + 'node_type ∈ {DatasetSource, Split, DataLoader, ModelSource, Loss, Head, Optimizer, Scheduler, Metric, EarlyStopping, GradientClipping, MixedPrecision, TrainLoop}. '
      + 'Use "after" to also wire it from an existing training node id. Pass params as a JSON object of node-specific fields '
      + '(e.g. {kind:"AdamW", lr:0.001} for Optimizer, {dataset:"datasets/iris.csv", target:"species"} for DatasetSource, '
      + '{model:"models/iris-mlp.spinoml"} for ModelSource, {epochs:50} for TrainLoop). '
      + 'MULTITASK / JOINT training IS expressible here — DO NOT fall back to a hand-written script. '
      + 'Add one Head node per model output instead of a single Loss node; params: '
      + '{output:"<name matching an architecture Output node, e.g. binder/affinity>", target:"<dataset column>", '
      + 'loss:"BCEWithLogitsLoss"|"CrossEntropyLoss"|"MSELoss"|"L1Loss", weight:<float>}. The trainer optimizes the '
      + 'weighted sum of the heads and MASKS per head: rows whose target column is EMPTY for a head are skipped for that '
      + "head's loss + metrics (so an affinity-regression head trains on real binders only while decoy rows, with an empty "
      + 'value, simply don\'t contribute — exactly the masked joint loss). Each head reports per-task metrics + an '
      + 'eval.summary (confusion matrix for classification heads, scatter for regression heads). '
      + 'A runnable graph needs at least: DatasetSource, ModelSource(+model), Optimizer, TrainLoop, and EITHER a Loss(+target on DatasetSource) '
      + 'for single-task OR ≥1 Head for multitask — wire each into the TrainLoop. '
      + 'DIFFUSION (score/VE, epsilon-prediction): one Head with {output:"eps_hat", target_kind:"score", '
      + 'diff_branch:"<manifest branch with the CLEAN point set>", sigma_min:<float, default 0.05>, sigma_max:<float, default 6>, n_rep:<int>=4}; '
      + 'it then needs NO target column and MUST be the only Head — the trainer corrupts that branch itself (x = w + sigma*eps).',
      {
        node_type: z.string(),
        after: z.string().optional().describe('Optional source training-node id to connect from'),
        params: z.record(z.string(), z.unknown()).optional(),
      },
      async ({ node_type, after, params }) => {
        const v = validateNodeParams('training', node_type, params ?? {})
        if (!v.ok) return content(`error: ${v.error}`, true)
        if (after && !trainingCtx.nodes.has(after)) return content(`error: training source "${after}" not found`, true)
        const id = trainingCtx.nextId()
        trainingCtx.nodes.set(id, { id, trainingType: node_type, params: v.params })
        actions.push({ op: 'training:add_node', payload: { id, node_type, params: v.params } })
        if (after) {
          trainingCtx.edges.set(`${after}->${id}`, { source: after, target: id })
          actions.push({ op: 'training:connect', payload: { source: after, target: id } })
        }
        return content(`added training ${node_type} as ${id}${after ? ` after ${after}` : ''}`)
      },
    ),
    tool(
      'connect_training_nodes',
      'Wire one TRAINING node into another (typically a source/component into the TrainLoop).',
      { source: z.string(), target: z.string() },
      async ({ source, target }) => {
        if (!trainingCtx.nodes.has(source)) return content(`error: training source "${source}" not found`, true)
        if (!trainingCtx.nodes.has(target)) return content(`error: training target "${target}" not found`, true)
        if (source === target) return content(`error: cannot connect a training node to itself ("${source}")`, true)
        const key = `${source}->${target}`
        if (trainingCtx.edges.has(key)) return content(`edge ${source}->${target} already exists (no change)`)
        if (wouldCreateCycle(trainingCtx.edges, source, target)) {
          return content(`error: edge ${source}->${target} would create a cycle`, true)
        }
        trainingCtx.edges.set(key, { source, target })
        actions.push({ op: 'training:connect', payload: { source, target } })
        return content(`connected ${source} → ${target}`)
      },
    ),
    tool(
      'update_training_params',
      'Patch parameters on an existing TRAINING node. Only supplied keys change.',
      { id: z.string(), params: z.record(z.string(), z.unknown()) },
      async ({ id, params }) => {
        const n = trainingCtx.nodes.get(id)
        if (!n) return content(`error: training node "${id}" not found`, true)
        const v = validateNodeParams('training', n.trainingType, params, { partial: true })
        if (!v.ok) return content(`error: ${v.error}`, true)
        n.params = { ...n.params, ...v.params }
        actions.push({ op: 'training:update_params', payload: { id, params: v.params } })
        return content(`patched training ${id}: ${JSON.stringify(v.params)}`)
      },
    ),
    tool(
      'delete_training_node',
      'Remove a TRAINING node and its incident edges.',
      { id: z.string() },
      async ({ id }) => {
        if (!trainingCtx.nodes.has(id)) return content(`error: training node "${id}" not found`, true)
        trainingCtx.nodes.delete(id)
        for (const [key, e] of trainingCtx.edges) {
          if (e.source === id || e.target === id) trainingCtx.edges.delete(key)
        }
        actions.push({ op: 'training:delete_node', payload: { id } })
        return content(`deleted training ${id}`)
      },
    ),
    tool(
      'clear_training_graph',
      'Remove ALL training nodes and edges — use before building a fresh training graph from scratch.',
      {},
      async () => {
        trainingCtx.nodes.clear()
        trainingCtx.edges.clear()
        actions.push({ op: 'training:clear', payload: {} })
        return content('cleared the training graph')
      },
    ),
    // ─── Data-processing-graph tools (the third canvas) ───────────────────────
    tool(
      'add_data_node',
      'Add a node to the DATA-PROCESSING graph (separate from the architecture + training graphs). '
      + 'The data canvas is a DAG of data-prep steps that compiles to ONE reproducible Python pipeline '
      + 'script run via run_script. node_type ∈ {TableSource, DownloadColumn, RenameColumns, SelectColumns, '
      + 'FilterRows, Normalize, DropNA, ComputeColumn, SmilesToGraph, StructureToGraph, CustomScript, '
      + 'WriteDataset}. Use "after" to wire it from an existing data-node id. Pass params as a JSON object of '
      + 'node-specific fields, e.g. TableSource {dataset:"datasets/x.csv"}; DownloadColumn {id_column:"uniprot", '
      + 'url_template:"https://rest.uniprot.org/uniprotkb/{id}.fasta", out_dir:"datasets/seqs", '
      + 'filename_template:"{id}.fasta"} (use "https://files.rcsb.org/download/{id}.pdb" for PDB structures); '
      + 'Normalize {columns:"a,b", method:"zscore"}; RenameColumns {mapping:"old:new, a:b"}; '
      + 'SmilesToGraph {smiles_column:"smiles", out_name:"datasets/graphs/mol.pt", embed_3d:true}; '
      + 'WriteDataset {out_path:"datasets/processed.csv", format:"csv"}. The escape hatch is '
      + 'CustomScript {label:"...", code:"<free Python; `df` is the DataFrame from the previous node>"} — '
      + 'write the code yourself when no typed node fits. A typical chain: TableSource → (transforms / '
      + 'SmilesToGraph / DownloadColumn) → WriteDataset.',
      {
        node_type: z.string(),
        after: z.string().optional().describe('Optional source data-node id to connect from'),
        params: z.record(z.string(), z.unknown()).optional(),
      },
      async ({ node_type, after, params }) => {
        const v = validateNodeParams('data', node_type, params ?? {})
        if (!v.ok) return content(`error: ${v.error}`, true)
        if (after && !dataCtx.nodes.has(after)) return content(`error: data source "${after}" not found`, true)
        const id = dataCtx.nextId()
        dataCtx.nodes.set(id, { id, dataType: node_type, params: v.params })
        actions.push({ op: 'data:add_node', payload: { id, node_type, params: v.params } })
        if (after) {
          dataCtx.edges.set(`${after}->${id}`, { source: after, target: id })
          actions.push({ op: 'data:connect', payload: { source: after, target: id } })
        }
        return content(`added data ${node_type} as ${id}${after ? ` after ${after}` : ''}`)
      },
    ),
    tool(
      'connect_data_nodes',
      'Wire one DATA node into another (build the pipeline order: source → transforms → sink).',
      { source: z.string(), target: z.string() },
      async ({ source, target }) => {
        if (!dataCtx.nodes.has(source)) return content(`error: data source "${source}" not found`, true)
        if (!dataCtx.nodes.has(target)) return content(`error: data target "${target}" not found`, true)
        if (source === target) return content(`error: cannot connect a data node to itself ("${source}")`, true)
        const key = `${source}->${target}`
        if (dataCtx.edges.has(key)) return content(`edge ${source}->${target} already exists (no change)`)
        if (wouldCreateCycle(dataCtx.edges, source, target)) {
          return content(`error: edge ${source}->${target} would create a cycle`, true)
        }
        dataCtx.edges.set(key, { source, target })
        actions.push({ op: 'data:connect', payload: { source, target } })
        return content(`connected ${source} → ${target}`)
      },
    ),
    tool(
      'update_data_params',
      'Patch parameters on an existing DATA node. Only supplied keys change. For a CustomScript, pass '
      + '{code:"..."} to (re)write its Python.',
      { id: z.string(), params: z.record(z.string(), z.unknown()) },
      async ({ id, params }) => {
        const n = dataCtx.nodes.get(id)
        if (!n) return content(`error: data node "${id}" not found`, true)
        const v = validateNodeParams('data', n.dataType, params, { partial: true })
        if (!v.ok) return content(`error: ${v.error}`, true)
        n.params = { ...n.params, ...v.params }
        actions.push({ op: 'data:update_params', payload: { id, params: v.params } })
        return content(`patched data ${id}: ${JSON.stringify(v.params)}`)
      },
    ),
    tool(
      'delete_data_node',
      'Remove a DATA node and its incident edges.',
      { id: z.string() },
      async ({ id }) => {
        if (!dataCtx.nodes.has(id)) return content(`error: data node "${id}" not found`, true)
        dataCtx.nodes.delete(id)
        for (const [key, e] of dataCtx.edges) {
          if (e.source === id || e.target === id) dataCtx.edges.delete(key)
        }
        actions.push({ op: 'data:delete_node', payload: { id } })
        return content(`deleted data ${id}`)
      },
    ),
    tool(
      'clear_data_graph',
      'Remove ALL data-processing nodes and edges — use before building a fresh pipeline from scratch.',
      {},
      async () => {
        dataCtx.nodes.clear()
        dataCtx.edges.clear()
        actions.push({ op: 'data:clear', payload: {} })
        return content('cleared the data graph')
      },
    ),
    tool(
      'ask_user',
      'Ask the user a question through the GUI and WAIT for their answer. Use this — do NOT guess — '
      + 'when a decision is genuinely the user\'s to make: an ambiguous goal, a destructive choice, or '
      + 'missing information you cannot read yourself (try read_file / list_dir first). kind:"select" shows '
      + 'clickable option buttons (pass options[]); kind:"text" shows a free-text box. Returns the user\'s '
      + 'answer as a string. Ask ONE focused question rather than a wall of options.',
      {
        kind: z.enum(['select', 'text']),
        prompt: z.string(),
        options: z.array(z.string()).optional().describe('Required for kind:"select" — the clickable choices'),
      },
      async ({ kind, prompt, options }) => {
        if (!askUser) return content('error: interactive questions are unavailable in this session', true)
        try {
          const answer = await askUser({ kind, prompt, options })
          return content(`user answered: ${typeof answer === 'string' ? answer : JSON.stringify(answer)}`)
        } catch (e) {
          return content(`no answer: ${e.message}`, true)
        }
      },
    ),
  ]

  if (workspace) {
    specs.push(
      tool(
        'list_notes',
        'List markdown notes in the project\'s notes/ folder. Use this to discover what context already exists before making suggestions.',
        {},
        async () => {
          try {
            const items = await notesList(workspace)
            if (!items.length) return content('(no notes yet)')
            return content(items.map((n) => `${n.name} — ${n.size == null ? '(size unknown)' : `${n.size} bytes`}`).join('\n'))
          } catch (e) {
            return content(`error: ${e.message}`, true)
          }
        },
      ),
      tool(
        'read_note',
        'Read the full contents of a project note (markdown or text). Use to load decision logs, prior session notes, or the project README before suggesting changes.',
        { name: z.string() },
        async ({ name }) => {
          try {
            return content(await notesRead(workspace, name))
          } catch (e) {
            return content(`error: ${e.message}`, true)
          }
        },
      ),
      tool(
        'append_note',
        'Append text to a project note (creates the file if missing). Use to record decisions, things you tried, or hand-offs for the next session. Always include a timestamp header. Prefer a small set of notes — decisions.md, session-YYYY-MM-DD.md — over one note per turn.',
        { name: z.string(), content: z.string() },
        async ({ name, content: body }) => {
          try {
            const written = await notesAppend(workspace, name, body)
            return content(`appended ${written} chars to notes/${safeNoteFilename(name)}`)
          } catch (e) {
            return content(`error: ${e.message}`, true)
          }
        },
      ),
      tool(
        'record_step',
        'Append a structured, timestamped entry to the project REPRODUCIBILITY LAB NOTEBOOK '
        + '(notes/lab-notebook.md by default). This is how the user\'s methodology becomes reproducible '
        + 'for a paper, so use it CONTINUOUSLY — after every meaningful step: a dataset downloaded or '
        + 'preprocessed (record the source + the exact script + params + splits/seeds), an architecture '
        + 'change (what + why), a training config set, a run launched (record the run_id + hyperparameters), '
        + 'a result analyzed, or a decision made. The current model architecture + training-graph '
        + 'fingerprint and an ISO timestamp are embedded automatically — you supply the narrative. Keep '
        + 'each entry focused (one step per call), factual, and specific enough to reproduce.',
        {
          summary: z.string().describe('What was done in this step (one or two sentences)'),
          rationale: z.string().optional().describe('WHY — the scientific/engineering reason (a methods section needs this)'),
          category: z.enum(['data', 'architecture', 'training', 'result', 'decision', 'environment', 'other']).optional(),
          commands: z.string().optional().describe('Exact commands / script paths to reproduce this step (e.g. "run_script agent/prep.py --seed 0")'),
          results: z.string().optional().describe('Outcomes / metrics; include the run_id when it came from a training run'),
          refs: z.object({
            model: z.string().optional(),
            run_id: z.string().optional(),
            dataset: z.string().optional(),
          }).optional().describe('Traceability pointers: model file, run_id, dataset path'),
          notebook: z.string().optional().describe('Target file under notes/ (default "lab-notebook.md")'),
        },
        async ({ summary, rationale, category, commands, results, refs, notebook }) => {
          try {
            const ts = new Date().toISOString()
            const arch = [...ctx.nodes.values()]
            const archLine = arch.length
              ? `${arch.length} nodes: ${arch.map((n) => n.layerType).join(' → ')}`.slice(0, 500)
              : '(empty)'
            const tnodes = [...trainingCtx.nodes.values()]
            const trainLine = tnodes.length
              ? tnodes.map((n) => {
                  const p = n.params || {}
                  const kv = Object.keys(p).slice(0, 5).map((k) => `${k}=${JSON.stringify(p[k])}`).join(', ')
                  return kv ? `${n.trainingType}(${kv})` : n.trainingType
                }).join('; ').slice(0, 700)
              : '(none)'
            const lines = [`### ${ts} — ${(category || 'other').toUpperCase()}`, ``, `**Step:** ${summary}`]
            if (rationale) lines.push(``, `**Why:** ${rationale}`)
            if (commands) lines.push(``, `**Reproduce:** ${commands}`)
            if (results) lines.push(``, `**Results:** ${results}`)
            if (refs && (refs.model || refs.run_id || refs.dataset)) {
              const r = [
                refs.model && `model=\`${refs.model}\``,
                refs.run_id && `run_id=\`${refs.run_id}\``,
                refs.dataset && `dataset=\`${refs.dataset}\``,
              ].filter(Boolean).join(', ')
              lines.push(``, `**Refs:** ${r}`)
            }
            lines.push(``, `**Architecture:** ${archLine}`, `**Training graph:** ${trainLine}`, ``, `---`, ``)
            let file = notebook || 'lab-notebook.md'
            if (!/\.(md|txt)$/i.test(file)) file += '.md'
            await notesAppend(workspace, file, lines.join('\n'))
            return content(`recorded step to notes/${safeNoteFilename(file)}`)
          } catch (e) {
            return content(`record failed: ${e.message}`, true)
          }
        },
      ),
      tool(
        'download_to_datasets',
        'Download a URL into the project\'s datasets/ folder. Use this when the user asks for a standard dataset by name (iris, MNIST, california housing, fashion-mnist, etc.). Pick a known stable mirror — UCI archive raw, scikit-learn raw, sklearn-datasets GitHub, HuggingFace datasets resolve URLs, common tutorial GitHub repos — and a filename ending in .csv/.parquet/.npy/.json/.zip/etc. The dataset shows up live in SpinoML\'s Datasets tab the moment the download finishes. For binary archives (tar.gz, zip), let the user know they\'ll need to unpack — you can do this with a follow-up shell tool if one exists, otherwise tell them to extract via the Terminal tab.',
        {
          url: z.string().url(),
          filename: z.string().describe('Basename only (e.g. "iris.csv"), saved under datasets/'),
        },
        async ({ url, filename }) => {
          try {
            const r = await downloadToDatasets(workspace, url, filename)
            actions.push({ op: 'dataset-added', payload: { relpath: r.relpath, bytes: r.bytes } })
            const kb = (r.bytes / 1024).toFixed(1)
            return content(`saved ${kb} KB to ${r.relpath}`)
          } catch (e) {
            return content(`download failed: ${e.message}`, true)
          }
        },
      ),
      tool(
        'write_dataset_file',
        'Write a TEXT file into the project\'s datasets/ folder (creates or overwrites). '
        + 'Use this to author files the user can\'t download — most importantly a `.manifest` '
        + '(a JSON descriptor that glues a table to per-branch graph sources row-by-row for a '
        + 'dual-encoder, e.g. ligand SMILES + protein graph + a target column). Manifest schema:\n'
        + '{\n'
        + '  "table": "pairs.csv",            // CSV/TSV/parquet, relative to datasets/ (or an absolute path)\n'
        + '  "row": 0,                         // optional preview row\n'
        + '  "pairs": {                        // one entry per encoder branch\n'
        + '    "ligand":  {"column": "smiles",  "kind": "molecule"},        // build a graph from the SMILES cell (RDKit) → Graph node\n'
        + '    "lig_espf": {"column": "smiles", "kind": "espf", "codebook": "drug", "max_len": 50}, // ESPF substructure tokens → ESPF node\n'
        + '    "prot_seq": {"column": "sequence", "kind": "sequence", "vocab": "protein", "max_len": 1000}, // char tokens → Sequence node\n'
        + '    "protein": {"column": "uniprot", "dir": "/abs/path/to/graphs", "match": "exact", "ext": ".pt"}\n'
        + '  },\n'
        + '  "target": {"column": "label", "type": "classification"}        // or "regression"; omit if none\n'
        + '}\n'
        + 'Branch reference resolution: kind=="molecule" builds the graph inline from the SMILES cell (no dir); '
        + 'kind=="espf" tokenizes the SMILES (or protein) cell into INTERPRETABLE substructure subword ids via the '
        + 'MolTrans ESPF codebook (codebook:"drug" for SMILES / "protein" for sequences; optional max_len) — bind it to an '
        + 'ESPF node feeding an Embedding; kind=="sequence" tokenizes a string cell char/byte-level (vocab:"protein"/"smiles"/'
        + 'explicit-chars/omit; optional max_len) — bind it to a Sequence node; '
        + '`dir`+match=="exact" → <dir>/<cell><ext>; match=="contains" → first file in <dir> whose name CONTAINS '
        + 'the cell value; no `dir` → the cell IS a path. IMPORTANT: a `dir` must be ABSOLUTE (a leading ~/ is NOT '
        + 'expanded when joined to the manifest folder). target type "classification" reads the column as an int '
        + 'class index, "regression" as a float — so for a binary/integer label the column must already hold 0/1 ints. '
        + 'Other text files are fine too (a small CSV, a .smi list). Binary payloads must use download_to_datasets instead. '
        + 'The file shows up live in SpinoML\'s Datasets tab the moment it\'s written; a .manifest renders as a paired dataset.',
        {
          filename: z.string().describe('Basename only, e.g. "binder_decoy.manifest" or "pairs.csv" — saved under datasets/'),
          content: z.string().describe('Full file contents (for a manifest, the JSON text)'),
        },
        async ({ filename, content: body }) => {
          try {
            const r = await writeDatasetFile(workspace, filename, body)
            actions.push({ op: 'dataset-added', payload: { relpath: r.relpath, bytes: r.bytes } })
            return content(`wrote ${r.bytes} bytes to ${r.relpath}`)
          } catch (e) {
            return content(`write failed: ${e.message}`, true)
          }
        },
      ),
      tool(
        'list_runs',
        'List recent training runs in this project (experiments/runs/) with a one-line summary each: '
        + 'run_id, label, status, model_path, dataset, best_val_loss, epochs. Use this to find which runs '
        + 'exist before reading one in detail to optimize the model.',
        {},
        async () => {
          try {
            const r = await runsList(workspace)
            if (!r.runs.length) return content('(no training runs yet)')
            const lines = r.runs.map((x) =>
              `${x.run_id}  [${x.status}]  loss=${x.best_val_loss ?? '—'}  model=${x.model_path ?? '?'}  epochs=${x.epochs ?? '?'}`
              + (x.warning ? `  ⚠ ${x.warning}` : ''))
            const note = r.total > r.shown ? `\n(${r.shown} of ${r.total} shown — newest first)` : ''
            return content(lines.join('\n') + note)
          } catch (e) {
            return content(`error: ${e.message}`, true)
          }
        },
      ),
      tool(
        'read_run',
        'Read ONE training run in detail: its config (hyperparameters: optimizer/lr/weight_decay, loss, '
        + 'scheduler, epochs, batch_size, val_split), parameter count, dataset info, the per-epoch metric '
        + 'history (train_loss, val_loss, val_acc, lr) and the final outcome. Use this to diagnose '
        + 'over/underfitting (train↓ but val↑ → overfit; both high/flat → underfit or LR issue), then '
        + 'propose+apply concrete fixes via update_params / update_training_params / add_layer (e.g. add '
        + 'Dropout, lower lr, add a scheduler, raise/lower capacity). Returns JSON.',
        { run_id: z.string() },
        async ({ run_id }) => {
          try {
            const r = await runRead(workspace, run_id)
            return content(JSON.stringify(r, null, 2))
          } catch (e) {
            return content(`error: ${e.message}`, true)
          }
        },
      ),
      // ─── Agent workspace tools: read / write / list / run ────────────────
      tool(
        'read_file',
        'Read any text file in the workspace by its path RELATIVE to the workspace root '
        + '(e.g. "datasets/binder_decoy.manifest", "agent/tokenize.py", "experiments/runs/<id>/run.json"). '
        + 'ALWAYS read a file before overwriting it — never ask the user to paste contents you can read yourself.',
        { path: z.string().describe('Relative path from the workspace root') },
        async ({ path: rel }) => {
          try {
            const text = await wsReadTextStrict(workspace, rel)
            return content(text || '(empty file)')
          } catch (e) {
            return content(`read failed: ${e.message}`, true)
          }
        },
      ),
      tool(
        'write_file',
        'Create or overwrite a TEXT file ANYWHERE in the workspace (Python, shell/sbatch, YAML, JSON, '
        + 'CSV, configs, models, manifests, notes, …). Paths are relative to the workspace root and '
        + 'confined to it; parent dirs are created automatically. agent/ is your scratch dir for '
        + 'scripts you intend to run — write the script there, then call run_script on it. Always write '
        + 'code to a FILE, never paste long code into chat. Read a file with read_file before you '
        + 'overwrite it. For preprocessing that produces a dataset, have the script write its output '
        + 'under datasets/ (and extend a .manifest if pairing branches) so it shows up live in the '
        + 'Datasets tab. Binary payloads use download_to_datasets instead.',
        {
          path: z.string().describe('Relative path, e.g. "agent/tokenize_proteins.py"'),
          content: z.string().describe('Full file contents'),
        },
        async ({ path: rel, content: body }) => {
          try {
            const safe = safeRelpath(rel)
            guardWritePath(safe)
            const r = await wsWriteText(workspace, safe, String(body ?? ''))
            // datasets/ writes surface as a dataset in the UI (manifests render paired).
            if (r.relpath.startsWith('datasets/')) {
              actions.push({ op: 'dataset-added', payload: { relpath: r.relpath, bytes: r.bytes } })
            }
            return content(`wrote ${r.bytes} bytes to ${r.relpath}`)
          } catch (e) {
            return content(`write failed: ${e.message}`, true)
          }
        },
      ),
      tool(
        'list_dir',
        'List the contents of a workspace directory by its path relative to the root (empty = root). '
        + 'Returns each entry and whether it is a directory. Use this to discover what exists '
        + '(datasets/, agent/, experiments/runs/, a graphs dir) before reading or writing.',
        { path: z.string().optional().describe('Relative dir path; omit for the workspace root') },
        async ({ path: rel }) => {
          try {
            const items = await wsListDirDetailed(workspace, rel || '')
            if (!items.length) return content('(empty or missing directory)')
            return content(items.map((i) => `${i.is_dir ? '[dir]  ' : '       '}${i.name}`).join('\n'))
          } catch (e) {
            return content(`list failed: ${e.message}`, true)
          }
        },
      ),
      tool(
        'run_script',
        'Execute a script in the workspace — typically one you just wrote into agent/ with write_file. '
        + 'It runs WHERE THE WORKSPACE LIVES (locally for a local workspace, over ssh on the host for a '
        + 'remote one), so it operates on the real files. Live stdout/stderr streams to the chat, and '
        + 'the captured output + exit code come back so you can read errors and fix-then-rerun until it '
        + 'works. The user approves each run with a one-click GUI dialog — that is normal: call the tool '
        + 'and continue once approved (a declined run is a clear no). '
        + 'mode:"shell" runs it right now — use it freely for quick work (a download, a small script, a '
        + 'tokenizer over a few thousand rows). On a LOCAL workspace shell has a generous ~10-minute cap. '
        + 'On a REMOTE/HPC workspace shell runs on the LOGIN NODE and is capped at ~2 minutes, so keep '
        + 'login-node work light. mode:"slurm" submits an sbatch script (with #SBATCH directives) to the '
        + 'scheduler and returns a job id — use it for heavy compute on HPC (training, embedding a large '
        + 'dataset, GPU work, anything that needs minutes); poll with slurm_status, then read the job '
        + 'output with read_file. Two cheap HPC habits: for "does this file exist / how many" use '
        + 'list_dir + read_file (instant) instead of a script, and inside scripts prefer '
        + 'os.path.exists() on specific paths over globbing a huge networked cache dir (that can hang).',
        {
          path: z.string().describe('Relative path of the script, e.g. "agent/tokenize_proteins.py"'),
          mode: z.enum(['shell', 'slurm']).optional().describe('"shell" = run now (local ~10 min, remote login-node ~2 min); "slurm" = sbatch for heavy/long/GPU compute'),
          args: z.string().optional().describe('Extra CLI args appended after the script path'),
        },
        async ({ path: rel, mode, args }) => {
          const m = mode === 'slurm' ? 'slurm' : 'shell'
          let safe
          try { safe = safeRelpath(rel) } catch (e) { return content(`invalid path: ${e.message}`, true) }
          // Fail fast (and BEFORE bothering the user with a confirm dialog) when
          // the script isn't actually in the workspace — otherwise the run dies
          // with a cryptic `python: can't open file …`. run_script and write_file
          // share the same root, so a missing file means it was never written
          // (or written to a different path).
          try {
            if (!(await wsPathExists(workspace, safe))) {
              return content(
                `${safe} does not exist in the workspace${workspace.isRemote ? ' on the remote host' : ''}. `
                + 'Write it first with write_file (same workspace root), then run_script it. '
                + 'If you just wrote it, re-check the exact relative path.',
                true,
              )
            }
          } catch (e) {
            // A path-scope rejection (symlink escaping the workspace) is a hard
            // stop — never fall through to executing the script.
            if (e && /^PATH_/.test(e.message)) return content(`run failed: ${e.message}`, true)
            /* otherwise: if the existence probe itself fails, fall through and let the run surface the real error */
          }
          let preview = ''
          try { preview = await wsReadTextStrict(workspace, safe, 2000) } catch { /* show prompt without preview */ }
          const parsedArgs = splitArgs(args)
          if (!parsedArgs.ok) return content(`invalid args: ${parsedArgs.error}`, true)
          const argsQuoted = parsedArgs.argv.length ? ` ${quoteArgv(parsedArgs.argv)}` : ''
          // Every run is gated by a confirm dialog. shell runs are also capped
          // (60s on a remote login node), stream live, and are killable with Stop.
          const onLogin = m === 'shell' && workspace.isRemote
          // FEAT-3 Auto-Modus: auto-approve quick SHELL runs (no confirm dialog),
          // but SLURM jobs (queue/compute) ALWAYS confirm. Stay within the
          // curated MCP tools + workspace — never a generic permission bypass.
          // Stop still aborts the turn (turnSignal). Every auto-run is announced.
          const autoApproved = autoApproveShell && m === 'shell'
          if (autoApproved) {
            if (pushEvent) pushEvent({ type: 'log', value: `⚡ Auto-Modus: ${safe} automatisch freigegeben (shell)\n` })
          } else if (askUser) {
            let approved
            try {
              approved = await askUser({
                kind: 'confirm',
                prompt: (m === 'slurm'
                  ? `SLURM-Job abschicken: sbatch ${safe}${argsQuoted}?`
                  : `Skript ausführen: ${safe}${argsQuoted} (shell)?`)
                  + (onLogin ? '  ⚠ Login-Node, max. 2 min — für schwere Jobs SLURM.' : ''),
                payload: { path: safe, mode: m, args: args ?? '', preview, on_login_node: onLogin },
              })
            } catch (e) {
              return content(`run cancelled: ${e.message}`, true)
            }
            if (!isAffirmative(approved)) return content(`user declined to run ${safe}`, true)
          }
          // Stream live output to the chat so the user can watch the run.
          const notify = pushEvent ? (s) => pushEvent({ type: 'log', value: s }) : undefined
          try {
            const r = await runScript(workspace, safe, m, args, notify, turnSignal)
            if (r.aborted) return content(`stopped by user before ${safe} finished`, true)
            if (m === 'slurm') {
              if (r.jobId) return content(`submitted SLURM job ${r.jobId} (poll with slurm_status, then read its output with read_file)`)
              return content(`sbatch did not return a job id (exit ${r.code}).\nstdout:\n${r.stdout || '(empty)'}\nstderr:\n${r.stderr || '(empty)'}`, true)
            }
            if (r.timedOut) {
              return content(
                `KILLED: shell run exceeded the ${workspace.isRemote ? 'login-node 2-minute' : 'local 10-minute'} cap. `
                + `This is too heavy for mode:"shell". Rewrite it as an sbatch script (#SBATCH directives) and re-run with mode:"slurm".`,
                true,
              )
            }
            const body = `exit ${r.code}\n--- stdout ---\n${r.stdout || '(empty)'}\n--- stderr ---\n${r.stderr || '(empty)'}`
            return content(body, r.code !== 0)
          } catch (e) {
            return content(`run failed: ${e.message}`, true)
          }
        },
      ),
      tool(
        'slurm_status',
        'Check the state of a SLURM job submitted with run_script(mode:"slurm"). Returns the scheduler '
        + 'state (PENDING/RUNNING/COMPLETED/FAILED/…). Read the job\'s output files with read_file once it completes.',
        { job_id: z.string() },
        async ({ job_id }) => {
          try {
            const r = await slurmStatus(workspace, job_id)
            return content(`job ${r.jobId}: ${r.state}`)
          } catch (e) {
            return content(`status check failed: ${e.message}`, true)
          }
        },
      ),
    )
  }

  return specs
}

// Wrap provider-agnostic specs into an in-process MCP server for the
// claude-agent-sdk (subscription / OAuth) path.
function buildMcpServer(specs) {
  const tools = specs.map((s) => tool(s.name, s.description, s.schema, s.handler))
  return createSdkMcpServer({ name: 'spinoml-graph', version: '0.1.0', tools })
}

function content(text, isError = false) {
  return {
    content: [{ type: 'text', text }],
    ...(isError ? { isError: true } : {}),
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Provider-agnostic tool execution. The subscription path lets the agent SDK
// call handlers itself; the direct-API paths below run their own tool-use loop
// against the SAME specs + handlers, emitting the same SSE event shapes and
// pushing the same `actions` (so the GraphStore mirror is identical).

// Flatten a handler result (the content() shape) into { text, isError }.
function readToolResult(res) {
  const text = Array.isArray(res?.content)
    ? res.content.map((c) => c?.text ?? '').join('')
    : typeof res === 'string' ? res : ''
  return { text, isError: !!res?.isError }
}

// zod raw shape → JSON Schema for OpenAI/Anthropic tool definitions.
// `$schema` is dropped (OpenAI rejects unknown top-level keys in some modes).
function specToJsonSchema(spec) {
  const json = z.toJSONSchema(z.object(spec.schema ?? {}))
  delete json.$schema
  return json
}

// Central argument gate. Every provider path (anthropic / openai-compat /
// opencode MCP bridge / subscription) funnels through here, so a schema
// violation is rejected identically regardless of who proposed the call.
// `parseError` is set by the openai-compat loop when the provider streamed
// arguments that are not valid JSON — that must never become `{}`.
function validateToolArgs(spec, args, parseError) {
  if (parseError) return { ok: false, text: `error: arguments for ${spec.name} are not valid JSON: ${parseError}` }
  // zod's default object STRIPS unknown keys, so a hallucinated top-level argument
  // (e.g. `activation` on add_layer) would silently vanish while the model believes
  // it was applied. Reject it explicitly and name the valid arguments instead.
  if (args && typeof args === 'object' && !Array.isArray(args)) {
    const allowed = new Set(Object.keys(spec.schema ?? {}))
    const extra = Object.keys(args).filter((k) => !allowed.has(k))
    if (extra.length) {
      return {
        ok: false,
        text: `error: unknown argument${extra.length > 1 ? 's' : ''} ${extra.map((k) => JSON.stringify(k)).join(', ')} for ${spec.name} (valid: ${[...allowed].join(', ') || 'none'})`,
      }
    }
  }
  const parsed = z.object(spec.schema ?? {}).safeParse(args ?? {})
  if (!parsed.success) {
    const issues = parsed.error.issues
      .slice(0, 8)
      .map((i) => `${i.path.length ? i.path.join('.') : '(root)'}: ${i.message}`)
      .join('; ')
    return { ok: false, text: `error: invalid arguments for ${spec.name}: ${issues}` }
  }
  return { ok: true, data: parsed.data }
}

// Resolve a tool by name, validate its arguments, then run the handler. Shared
// by execTool (which mirrors the call to SSE) and the opencode MCP bridge route
// (which returns the result to opencode instead).
async function invokeTool(specsByName, name, args, parseError) {
  const spec = specsByName.get(name)
  if (!spec) return { text: `error: unknown tool "${name}"`, isError: true }
  const check = validateToolArgs(spec, args, parseError)
  if (!check.ok) return { text: check.text, isError: true }
  try { return readToolResult(await spec.handler(check.data)) }
  catch (e) { return { text: `error: ${e.message}`, isError: true } }
}

// Run one tool handler by name, emit the SSE tool_use/tool_result pair, and
// return { text, isError } for the provider loop to feed back to the model.
async function execTool(specsByName, emit, id, name, args, parseError) {
  emit({ type: 'tool_use', id, name, args: args ?? {} })
  const result = await invokeTool(specsByName, name, args, parseError)
  emit({ type: 'tool_result', id, ok: !result.isError, result: result.text, error: result.isError ? result.text : undefined })
  return result
}

const MAX_TOOL_TURNS = 100 // matches the subscription path's maxTurns; allows write→run→read→fix→rerun loops

// When a turn hits the step cap, ASK the user (via the GUI confirm card) whether to
// keep going instead of silently failing. Returns true to continue (reset the
// counter), false to stop. Without an asker (shouldn't happen) it stops.
async function confirmContinue(askUser, emit, steps) {
  if (!askUser) { emit({ type: 'status', value: 'error', message: 'hit max tool turns' }); return false }
  let answer
  try {
    answer = await askUser({
      kind: 'confirm',
      prompt: `Ich habe ${steps} Schritte gemacht und bin noch nicht fertig. Weitermachen?`,
      payload: { reason: 'max_turns', steps },
    })
  } catch {
    // An unanswerable continue-prompt stops the turn (the alternative is looping
    // forever); no tool result or stored artifact is affected.
    return false
  }
  if (!isAffirmative(answer)) { emit({ type: 'status', value: 'error', message: `gestoppt nach ${steps} Schritten` }); return false }
  emit({ type: 'status', value: 'thinking' })
  return true
}

// Direct Anthropic Messages API (API key, not the OAuth subscription).
async function runAnthropicApi(specs, systemPrompt, history, user, emit, opts, askUser) {
  const client = new Anthropic({ apiKey: opts.apiKey, ...(opts.baseUrl ? { baseURL: opts.baseUrl } : {}) })
  const specsByName = new Map(specs.map((s) => [s.name, s]))
  const tools = specs.map((s) => ({ name: s.name, description: s.description, input_schema: specToJsonSchema(s) }))
  const messages = [
    ...(history ?? []).filter((m) => m.role === 'user' || m.role === 'assistant').map((m) => ({ role: m.role, content: m.content })),
    { role: 'user', content: user },
  ]
  const mj = validateJsonModelName(opts.model)
  if (!mj.ok) throw new Error(`invalid model name: ${mj.error}`)
  const model = mj.name || 'claude-opus-4-8'

  let steps = 0
  while (true) {
    const stream = client.messages.stream({
      model, max_tokens: 16000, system: systemPrompt,
      thinking: { type: 'adaptive' }, tools, messages,
    })
    stream.on('text', (delta) => { if (delta) emit({ type: 'text', value: delta }) })
    const msg = await stream.finalMessage()
    messages.push({ role: 'assistant', content: msg.content })

    const toolUses = msg.content.filter((b) => b.type === 'tool_use')
    if (msg.stop_reason !== 'tool_use' || !toolUses.length) return

    const results = []
    for (const tu of toolUses) {
      const r = await execTool(specsByName, emit, tu.id, tu.name, tu.input)
      results.push({ type: 'tool_result', tool_use_id: tu.id, content: r.text, is_error: r.isError })
    }
    messages.push({ role: 'user', content: results })

    if (++steps % MAX_TOOL_TURNS === 0 && !(await confirmContinue(askUser, emit, steps))) return
  }
}

// OpenAI Chat Completions (covers OpenAI, Gemini's OpenAI-compatible endpoint,
// Ollama, OpenRouter, and any other OpenAI-compatible server via baseUrl).
async function runOpenAiCompat(specs, systemPrompt, history, user, emit, opts, askUser, abortSignal) {
  // maxRetries: 2 — the SDK default; we state it explicitly to guarantee a cap
  // on how long a flaky provider can retry before the turn fails.
  const client = new OpenAI({
    apiKey: opts.apiKey || 'no-key',
    ...(opts.baseUrl ? { baseURL: opts.baseUrl } : {}),
    maxRetries: 2,
  })
  const specsByName = new Map(specs.map((s) => [s.name, s]))
  const tools = specs.map((s) => ({
    type: 'function',
    function: { name: s.name, description: s.description, parameters: specToJsonSchema(s) },
  }))
  const messages = [
    { role: 'system', content: systemPrompt },
    ...(history ?? []).filter((m) => m.role === 'user' || m.role === 'assistant').map((m) => ({ role: m.role, content: m.content })),
    { role: 'user', content: user },
  ]
  const mj = validateJsonModelName(opts.model)
  if (!mj.ok) throw new Error(`invalid model name: ${mj.error}`)
  const model = mj.name || 'gpt-4o'
  const stallSeconds = Math.round(UPSTREAM_TIMEOUT_MS / 1000)

  let steps = 0
  while (true) {
    // Per-request idle guard: abort the upstream request when nothing arrives
    // for UPSTREAM_TIMEOUT_MS. A client abort (turnAbort) is merged in so the
    // upstream connection is torn down promptly too.
    const idleController = new AbortController()
    const signal = abortSignal ? AbortSignal.any([abortSignal, idleController.signal]) : idleController.signal
    let stalled = false
    const withIdle = async (promise) => {
      const timer = setTimeout(() => { stalled = true; idleController.abort() }, UPSTREAM_TIMEOUT_MS)
      try { return await promise } finally { clearTimeout(timer) }
    }

    let stream
    try {
      stream = await withIdle(client.chat.completions.create({ model, messages, tools, stream: true }, { signal }))
    } catch (e) {
      if (stalled) throw new Error(`provider stalled: no data for ${stallSeconds} s`)
      throw e
    }

    let text = ''
    const toolCalls = [] // accumulated by streamed index
    let chunkCount = 0
    let finishReason = null
    const iterator = stream[Symbol.asyncIterator]()
    for (;;) {
      let next
      try {
        next = await withIdle(iterator.next())
      } catch (e) {
        if (stalled) throw new Error(`provider stalled: no data for ${stallSeconds} s`)
        throw e
      }
      if (next.done) break
      const chunk = next.value
      chunkCount++
      const choice = chunk.choices?.[0]
      if (choice?.finish_reason) finishReason = choice.finish_reason
      const delta = choice?.delta
      if (!delta) continue
      if (delta.content) { text += delta.content; emit({ type: 'text', value: delta.content }) }
      for (const tc of delta.tool_calls ?? []) {
        const slot = (toolCalls[tc.index] ??= { id: '', name: '', arguments: '' })
        if (tc.id) slot.id = tc.id
        if (tc.function?.name) slot.name = tc.function.name
        if (tc.function?.arguments) slot.arguments += tc.function.arguments
      }
    }

    // An abort caused by the idle guard may surface as a clean end-of-stream
    // rather than a rejected `next()` — surface the stall explicitly.
    if (stalled) throw new Error(`provider stalled: no data for ${stallSeconds} s`)
    // Stream integrity: a non-SSE body yields no chunks, and a truncated stream
    // ends before any finish_reason. Both must be explicit errors, never a
    // silent `done`, and no partially-received tool call may run.
    if (chunkCount === 0) {
      throw new Error('provider returned no stream data (is the baseUrl an OpenAI-compatible endpoint?)')
    }
    if (!finishReason) {
      throw new Error('provider stream ended without a finish_reason (truncated?)')
    }

    const calls = toolCalls.filter(Boolean)
    messages.push({
      role: 'assistant',
      content: text || null,
      ...(calls.length ? { tool_calls: calls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.arguments || '{}' } })) } : {}),
    })
    if (!calls.length) return

    for (const c of calls) {
      let args = {}
      let parseError = null
      try {
        args = JSON.parse(c.arguments || '{}')
      } catch (e) {
        // Malformed arguments are NOT silently replaced by {}: the tool call is
        // rejected explicitly and the handler never runs.
        parseError = e instanceof Error ? e.message : String(e)
      }
      const r = await execTool(specsByName, emit, c.id, c.name, args, parseError)
      messages.push({ role: 'tool', tool_call_id: c.id, content: r.text })
    }

    if (++steps % MAX_TOOL_TURNS === 0 && !(await confirmContinue(askUser, emit, steps))) return
  }
}

// ────────────────────────────────────────────────────────────────────────────
// OpenCode CLI provider (first-class, the default).
//
// opencode is spawned per chat turn as `opencode run --format json -m <model>`
// — the CLI's machine-readable contract: NDJSON events on stdout. The model is
// exposed ONLY the MCP tool server `graph` (mcp-bridge.mjs, spawned from the
// inline config below); every built-in opencode tool (read/bash/write/…) is
// disabled so nothing can side-step the graph-validation gate. Tool calls come
// back as NDJSON tool_use events and are executed through the SAME
// handler/execTool pipeline as the other providers.

const OPENCODE_BIN = process.env.SPINOML_OPENCODE_BIN || 'opencode'
const OPENCODE_DEFAULT_MODEL = 'opencode/big-pickle'
// Optional overrides for tests: validated exactly like SPINOML_LLM_PORT /
// SPINOML_LLM_UPSTREAM_TIMEOUT_MS — a non-positive / non-integer value is
// refused loudly at startup (exit 2) instead of silently falling back.
const positiveIntEnv = (name, fallback) => {
  const raw = process.env[name]
  if (raw === undefined || raw === '') return fallback
  const n = Number(raw)
  if (!Number.isInteger(n) || n <= 0) {
    console.error(`[spinoml-llm] invalid ${name}=${JSON.stringify(raw)} (expected a positive integer of milliseconds)`)
    process.exit(2)
  }
  return n
}
const OPENCODE_TIMEOUT_MS = positiveIntEnv('SPINOML_OPENCODE_TIMEOUT_MS', 30 * 60 * 1000) // hard cap on one opencode run
// The FIRST step can take a while: one-time provider/account sync, model
// metadata fetch, backend retries (title agent). A chat turn is async — the UI
// shows "thinking" meanwhile — so give a generous window before giving up.
const OPENCODE_START_TIMEOUT_MS = positiveIntEnv('SPINOML_OPENCODE_START_TIMEOUT_MS', 90 * 1000)
const OPENCODE_MAX_TOOL_STEPS = 150 // opt-out guard; opencode has internal loops too
const OPENCODE_MODELS_TTL_MS = 60 * 1000 // opencode models is ~1s to run — cache it
const OPENCODE_MAX_JSON_CHUNK = 1_000_000 // guard on accumulated stdout

// Built-in opencode tools we never expose to the model. `mcp_*` also covers any
// MCP server the user has globally configured (opencode merges global config in).
const OPENCODE_DISABLED_TOOLS = [
  'read', 'write', 'edit', 'bash', 'glob', 'grep', 'list', 'webfetch', 'webfetch_search',
  'websearch', 'lsp', 'task', 'todo', 'todowrite', 'image', 'cheatsheet', 'notebook',
  'patch', 'kill', 'exec_command', 'mcp_*', 'mcp__*', 'agent_start', 'agent_start_reply', 'plan',
]

function opencodeDisabledTools() {
  return Object.fromEntries(OPENCODE_DISABLED_TOOLS.map((t) => [t, false]))
}

// Per-turn MCP session registry. The opencode bridge (mcp-bridge.mjs) proxies
// tools/list + tools/call requests here over HTTP; they run the SAME specs +
// handlers as every other provider, so actions/ask/log event routing is
// identical and no tool ever runs without hitting the graph-validation gate.
// Keyed by the /chat requestId; the session lives only for the turn.
//
// Each session is born with a per-turn random secret (32 bytes hex, generated
// here). The bridge authenticates by sending it as `X-SpinoML-Token` on every
// `/internal/mcp/<id>/…` request. The master token is NEVER accepted here —
// the opencode process must not see it (it could leak via `ps`, core dumps,
// environment dumps, etc.). The secret reaches the bridge through the
// `environment` of the opencode `mcp.graph` config entry, never argv.
const mcpSessions = new Map() // requestId → { specsByName, secret }

function registerMcpSession(requestId, specs) {
  const secret = generateMcpSecret()
  mcpSessions.set(requestId, {
    specsByName: new Map(specs.map((s) => [s.name, s])),
    secret,
  })
  return secret
}

function unregisterMcpSession(requestId) {
  mcpSessions.delete(requestId)
}

// HTTP handler for POST /internal/mcp/<requestId>/list | /call — the bridge's
// single point of contact (localhost-only, like the rest of the sidecar).
// Always authenticated with the per-turn session secret; an unauthenticated
// caller can neither reach a session nor learn whether one exists (same 401
// for a wrong secret and for an unknown requestId).
async function handleMcpRoute(req, res) {
  const m = req.url.match(/^\/internal\/mcp\/([^/]+)\/(list|call)$/)
  if (!m) {
    setCorsOrigin(res, RESPONSE_ORIGIN.get(res))
    return sendJson(res, 404, { error: 'not found' })
  }
  const [, requestId, action] = m
  const session = mcpSessions.get(requestId)
  const supplied = req.headers['x-spinoml-token']
  const suppliedStr = Array.isArray(supplied) ? supplied.join(', ') : (typeof supplied === 'string' ? supplied : '')
  const ok = !!session && sessionSecretMatches(suppliedStr, session.secret)
  if (!ok) {
    // Same body for "no session" and "wrong secret" — a probe must not be
    // able to enumerate live session ids.
    setCorsOrigin(res, RESPONSE_ORIGIN.get(res))
    return sendJson(res, 401, { error: CODE_UNAUTHORIZED, code: CODE_UNAUTHORIZED, reason: 'invalid' })
  }
  let body = ''
  for await (const chunk of req) {
    if (body.length > 1e6) return sendJson(res, 413, { error: 'body too large' })
    body += chunk
  }
  let payload = {}
  try { payload = body ? JSON.parse(body) : {} } catch { return sendJson(res, 400, { error: 'invalid json' }) }

  if (action === 'list') {
    const tools = [...session.specsByName.values()].map((s) => ({
      name: s.name, description: s.description, inputSchema: specToJsonSchema(s),
    }))
    return sendJson(res, 200, { tools })
  }

  const { name, args } = payload
  // Same validation gate as execTool: opencode's MCP bridge must not be able to
  // reach a handler with arguments that would fail the schema.
  const result = await invokeTool(session.specsByName, name, args)
  return sendJson(res, 200, { ok: !result.isError, result: result.text })
}

function stringifyPartOutput(output) {
  if (typeof output === 'string') return output
  if (output == null) return ''
  try { return JSON.stringify(output) } catch { return String(output) }
}

// Get the live provider list from the opencode CLI (`opencode models`).
let opencodeModelsCache = { at: 0, list: [] }
async function listOpenCodeModels() {
  const now = Date.now()
  if (now - opencodeModelsCache.at < OPENCODE_MODELS_TTL_MS) return opencodeModelsCache.list
  const r = await spawnCapture(OPENCODE_BIN, ['models'], { timeoutMs: 15_000 })
  // A failed/timed-out `opencode models` must not look like "no models": throw
  // so GET /opencode/models returns an explicit error instead of an empty list.
  if (r.timedOut) throw new Error(`\`${OPENCODE_BIN} models\` timed out`)
  if (r.aborted) throw new Error(`\`${OPENCODE_BIN} models\` was aborted`)
  if (r.code !== 0) {
    const tail = `${r.stdout}\n${r.stderr}`.split('\n').map((s) => s.trim()).filter(Boolean).slice(-3).join(' ')
    throw new Error(`\`${OPENCODE_BIN} models\` exited ${r.code}${tail ? `: ${tail}` : ''}`)
  }
  const lines = `${r.stdout}\n${r.stderr}`
    .split('\n').map((s) => s.trim()).filter((s) => /^[^\s/]+\/[^\s/]+$/.test(s))
  opencodeModelsCache = { at: Date.now(), list: [...new Set(lines)].sort() }
  return opencodeModelsCache.list
}

// Run one full opencode pass for a chat turn. Returns the assistant's final
// text (for the summary) once the event loop sees step_finish+stop. Tool
// results are fed back to opencode directly because it runs its OWN tool loop
// (unlike the other providers where this file orchestrates the loop); we only
// translate events and enforce budgets/timeouts/abort.
async function runOpenCode(specs, systemPrompt, history, user, emit, opts, requestId, turnAbort, mcpSecret) {
  // The model string is interpolated into argv (`spawn(... '--model', model ...)`);
  // a leading `-` or whitespace would be parsed by opencode as an OPTION, not the
  // model. Reject BEFORE any child process is spawned — `opts?.model` empty/absent
  // keeps the documented default.
  const m = validateModelName(opts?.model)
  if (!m.ok) {
    emit({ type: 'status', value: 'error', message: `invalid model name: ${m.error}` })
    return ''
  }
  const model = m.name || OPENCODE_DEFAULT_MODEL
  const prompt = `${systemPrompt}\n\n${formatHistoryAsPrompt(history ?? [], user)}`
  const bridgePath = new URL('./mcp-bridge.mjs', import.meta.url).pathname

  // Fresh, disposable config: inline config wins over global/session/project.
  // MCP tool server name "graph" → the model sees `graph_<tool>`.
  // The per-turn MCP secret is delivered via the `environment` field of the
  // MCP entry (opencode supports it for local MCP servers — see its docs /
  // `opencode mcp add --env KEY=VALUE`). The bridge reads it from
  // `SPINOML_MCP_SECRET` and sends it as `X-SpinoML-Token`. The master token
  // never reaches opencode.
  const opencodeConfig = {
    model,
    autoupdate: false,
    share: 'disabled',
    snapshot: false,
    tools: opencodeDisabledTools(),
    mcp: {
      graph: {
        type: 'local',
        command: [process.execPath, bridgePath, requestId],
        enabled: true,
        environment: { SPINOML_MCP_SECRET: mcpSecret },
      },
    },
  }

  const sessionDir = await fs.mkdtemp(path.join(os.tmpdir(), 'spinoml-opencode-'))
  openSessionDirs.add(sessionDir)
  const env = {
    ...process.env,
    OPENCODE_CONFIG_CONTENT: JSON.stringify(opencodeConfig),
    OPENCODE_CONFIG_DIR: sessionDir,
    // The MCP bridge defaults to 7422; when this sidecar runs on an overridden
    // port (SPINOML_LLM_PORT, used by tests / side-by-side dev) the bridge would
    // otherwise call the wrong port and every opencode tool call would fail.
    // Pin it to THIS sidecar's real address; it is a localhost-only URL.
    SPINOML_SIDECAR: `http://127.0.0.1:${PORT}`,
    NO_COLOR: '1',
  }

  let child
  try {
    // NOT detached: opencode stays in the sidecar's process group so a
    // SIGKILL to the sidecar reaps it too; the SIGTERM handler additionally
    // walks /proc for the whole opencode → bridge subtree and kills each PID.
    child = spawn(OPENCODE_BIN, ['run', '--format', 'json', '--model', model, '--pure', prompt], {
      cwd: sessionDir, stdio: ['ignore', 'pipe', 'pipe'], env, signal: turnAbort.signal,
    })
    trackChild(child)
  } catch (e) {
    emit({ type: 'status', value: 'error', message: `opencode start failed: ${e.message}` })
    return ''
  }

  let stdoutBuf = ''
  let stderrBuf = ''
  let haveEvent = false
  let forcedEnd = false
  let stepCount = 0
  let toolSteps = 0
  let toolSeq = 0
  let sawErrorEvent = false
  let finalText = ''

  const killHard = (soft = false) => {
    forcedEnd = true
    try { child.kill('SIGTERM') } catch { /* process already exited — best-effort kill */ }
    if (soft) {
      const t = setTimeout(() => { try { child.kill('SIGKILL') } catch { /* process already exited — best-effort kill */ } }, 3000)
      t.unref()
    }
  }

  const deadline = setTimeout(() => {
    killHard()
    emit({ type: 'status', value: 'error', message: `OpenCode nach ${OPENCODE_TIMEOUT_MS / 60000} min abgebrochen (Timeout)` })
  }, OPENCODE_TIMEOUT_MS)
  const startDeadline = setTimeout(() => {
    if (!haveEvent) {
      killHard()
      emit({ type: 'status', value: 'error', message: `OpenCode antwortet nicht (${OPENCODE_BIN} "${model}") — ist das CLI installiert und das Modell erreichbar?` })
    }
  }, OPENCODE_START_TIMEOUT_MS)

  child.stderr.on('data', (d) => {
    if (stderrBuf.length < 1_000_000) stderrBuf += d.toString()
  })

  function onEvent(ev) {
    if (!ev?.type) return
    switch (ev.type) {
      case 'text': {
        const t = ev.part?.text
        if (t) { finalText += t; emit({ type: 'text', value: t }) }
        break
      }
      case 'step_start': {
        haveEvent = true
        break
      }
      case 'tool_use': {
        haveEvent = true
        toolSteps++
        const part = ev.part ?? {}
        const raw = part.tool ?? ''
        const name = raw.startsWith('graph_') ? raw.slice('graph_'.length) : raw
        const callId = part.callID || `open_${++toolSeq}`
        const input = part.state?.input ?? {}
        emit({ type: 'tool_use', id: callId, name, args: input })
        // opencode may emit a tool_use part in a non-terminal state first;
        // only emit the RESULT once the state is terminal (output present).
        const status = part.state?.status
        if (status === 'completed' || status === 'error' || part.state?.output != null) {
          const output = stringifyPartOutput(part.state?.output)
          const isError = status === 'error'
          emit({ type: 'tool_result', id: callId, ok: !isError, result: output, error: isError ? output : undefined })
        }
        break
      }
      case 'step_finish': {
        haveEvent = true
        // OpenCode's own limit: lots of back-to-back tool passes signal a loop.
        if (ev.part?.reason === 'tool-calls') {
          if (toolSteps >= OPENCODE_MAX_TOOL_STEPS) {
            killHard()
            emit({ type: 'status', value: 'error', message: `OpenCode nach ${OPENCODE_MAX_TOOL_STEPS} Tool-Schritten abgebrochen` })
          }
          break
        }
        break
      }
      case 'error': {
        sawErrorEvent = true
        const msg = ev.error?.message || ev.error?.title || JSON.stringify(ev.error)
        emit({ type: 'status', value: 'error', message: `OpenCode: ${msg}` })
        break
      }
      default:
        break
    }
  }

  child.stdout.on('data', (d) => {
    if (stdoutBuf.length < OPENCODE_MAX_JSON_CHUNK) stdoutBuf += d.toString()
    let idx
    while ((idx = stdoutBuf.indexOf('\n')) !== -1) {
      const line = stdoutBuf.slice(0, idx).trim()
      stdoutBuf = stdoutBuf.slice(idx + 1)
      if (!line) continue
      let ev
      try { ev = JSON.parse(line) } catch { /* non-JSON noise → ignore */ continue }
      onEvent(ev)
    }
  })

  return await new Promise((resolve) => {
    // Deterministic cleanup: remove the disposable session dir BEFORE resolving,
    // so a completed/aborted turn never leaves a `spinoml-opencode-*` behind.
    // A fire-and-forget rm raced the process teardown (the lifecycle test
    // asserts no leftover temp dir, and a killed sidecar would keep them).
    const cleanupSessionDir = () => fs.rm(sessionDir, { recursive: true, force: true }).then(() => { openSessionDirs.delete(sessionDir) }).catch(() => { /* disposable opencode session dir cleanup */ })
    child.on('close', async (code) => {
      clearTimeout(deadline)
      clearTimeout(startDeadline)
      await cleanupSessionDir()
      if (turnAbort.signal.aborted) return resolve(finalText) // user aborted — no error noise
      if (forcedEnd) return resolve(finalText)
      if (code !== 0 && !sawErrorEvent) {
        const tail = stderrBuf.split('\n').map((s) => s.trim()).filter(Boolean).slice(-3).join(' ')
        emit({ type: 'status', value: 'error', message: `opencode exit ${code}${tail ? `: ${tail}` : ''}` })
      }
      resolve(finalText)
    })
    child.on('error', async (e) => {
      clearTimeout(deadline)
      clearTimeout(startDeadline)
      await cleanupSessionDir()
      resolve(finalText)
      emit({ type: 'status', value: 'error', message: `OpenCode-Start fehlgeschlagen: ${e.message}` })
    })
  })
}

// ────────────────────────────────────────────────────────────────────────────
// HTTP layer.
//
// Every code path that writes headers MUST honour the validated origin the
// auth gate produced for the request. We attach it to the `res` object via a
// WeakMap so no site can forget it and fall back to a wildcard. `applyCors`
// writes the (origin-conditional) CORS headers and `setCorsOrigin` plants the
// value BEFORE the first `res.writeHead`. 401/403/4xx responses also echo the
// allowed Origin (when there is one) so the webview can read them.

const RESPONSE_ORIGIN = new WeakMap()

function setCorsOrigin(res, origin) {
  RESPONSE_ORIGIN.set(res, origin)
}

function applyCors(res) {
  const origin = RESPONSE_ORIGIN.get(res)
  // Match the torch reference exactly: CORS headers ONLY for a validated
  // Origin — none at all when the request had no Origin or it was rejected.
  if (typeof origin === 'string' && origin.length > 0) {
    applyCorsHeaders(res, origin)
  }
}

function sendJson(res, status, obj) {
  applyCors(res)
  const payload = JSON.stringify(obj)
  // Explicit Content-Length: keeps the body a single framed response (no
  // chunked transfer-encoding), which the auth test's raw HTTP client and
  // any minimal client can read without a chunk decoder.
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(payload),
  })
  res.end(payload)
}

function rejectRequest(res, decision) {
  const code = decision.code
  const reason = decision.reason || undefined
  setCorsOrigin(res, decision.origin)
  sendJson(res, decision.status, reason === undefined ? { error: code, code } : { error: code, code, reason })
}

// FEAT-4 — the reproducibility-documentation instruction, by doc mode:
//   'verbose' (default) → document continuously + the mandatory environment capture
//   'compact'           → document only milestones, no per-step / no env-capture mandate
//   'off'               → never document proactively; only on an explicit user request
function docProtocolLines(docMode) {
  if (docMode === 'off') {
    return [
      '',
      '═══ DOCUMENTATION: OFF ═══',
      'The user turned OFF automatic documentation. Do NOT call record_step / append_note on your own.',
      'Only write to the lab notebook when the user EXPLICITLY asks you to document something.',
    ]
  }
  if (docMode === 'compact') {
    return [
      '',
      '═══ DOCUMENTATION: COMPACT ═══',
      'Document only the MILESTONES with record_step (appends a timestamped entry to notes/lab-notebook.md,',
      'auto-embedding the architecture + training-graph fingerprint): a dataset prepared, an architecture',
      'change + why, a run launched (with run_id + hyperparameters), a result analyzed. Keep entries short.',
      'Skip per-step logging and the full environment capture unless the user asks for paper-grade detail.',
    ]
  }
  return [
    '',
    '═══ REPRODUCIBILITY PROTOCOL (this user builds paper-grade models) ═══',
    'Their methodology MUST be fully reproducible by a third party. Document CONTINUOUSLY and',
    'comprehensively as you work — NOT sparingly. Your primary tool is record_step: it appends a',
    'timestamped, structured entry to notes/lab-notebook.md and auto-embeds the current architecture',
    '+ training-graph fingerprint, so you only supply the narrative. Call it after EVERY meaningful',
    'step, capturing enough that someone with only this workspace could reproduce the result:',
    ' - DATA: exact source (URL / path / version), the preprocessing script you wrote + its params,',
    '   filtering criteria, and the split sizes + random seeds. Have prep scripts PRINT these values.',
    ' - ARCHITECTURE: what changed and WHY — tie design choices to evidence (e.g. a prior run\'s curve).',
    ' - TRAINING: optimizer, lr, schedule, loss, epochs, batch size, seed, and the run_id once launched.',
    ' - RESULTS: metrics per run (reference the run_id — read_run gives the full history) + your reading.',
    ' - DECISIONS: every non-obvious choice and its reasoning, including dead ends you ruled out.',
    'ENVIRONMENT (mandatory for a methods section): once per session, and whenever it changes, write +',
    'run a small script that captures `python --version`, `pip freeze` (or `conda list`), `nvidia-smi`',
    'or `lscpu`, and the git commit if the project is a repo; save it to notes/environment.md (or',
    'experiments/) and record_step it. Versions + hardware make or break reproducibility.',
    'Use append_note / write_file for longer artifacts — a clean notes/METHODS.md paragraph when the',
    'user nears publishing, a data dictionary, etc. Use list_notes / read_note to see what already',
    'exists and EXTEND the running notebook; do not start a fresh file each turn. Keep entries focused.',
  ]
}

function buildSystemPrompt(snapshot, error, project, trainingSnapshot, dataSnapshot, docMode = 'verbose', autoMode = false) {
  const lines = [
    'You are an expert PyTorch architect embedded in SpinoML, a drag-and-drop GUI for building nn.Module architectures.',
    '',
    'You can SEE the user\'s current graph (below) and can MUTATE it via tools. Prefer tools over describing changes in prose — the user wants you to actually build, not just suggest.',
    '',
    'When the user describes a goal, take the smallest sequence of tool calls that reaches it. Common patterns:',
    ' - "make my CNN deeper" → repeated add_layer(Conv2d, after=...), add_layer(BatchNorm2d, ...), add_layer(ReLU, ...)',
    ' - "fix the failing layer" → inspect the error + shapes shown below, then update_params (or delete_node + add_layer when the layer choice itself is wrong, e.g. LayerNorm on a CNN body → swap to GroupNorm or BatchNorm2d).',
    ' - "give me a classification head" → add_layer(AdaptiveAvgPool2d) + add_layer(Flatten) + add_layer(Linear, params={in_features: <channels>, out_features: <num_classes>}).',
    '',
    'Layer params must match the input shape: Conv1d/2d/3d.in_channels = channel dim of input, BatchNorm1d/2d.num_features = channel dim, Linear.in_features = last dim, LayerNorm.normalized_shape = trailing dims, LSTM/GRU/RNN.input_size = last dim of a [N,L,C] sequence (batch_first), Embedding.embedding_dim becomes the new last dim. Inspect the shapes shown for each node before choosing parameters.',
    '',
    'After tool calls, briefly tell the user what you changed (one short sentence) — they can see the result on the canvas.',
    '',
    'FORMATTING: the chat panel renders GitHub-flavored Markdown — light **bold**, short `- ` bullet lists, `inline code` and fenced code blocks are fine and help readability. Keep replies concise; do NOT dump long code into chat — write it to a file with write_file instead.',
    '',
    'ASKING vs GUESSING: when a decision is genuinely the user\'s (an ambiguous goal, a destructive choice, missing info) FIRST try to find the answer yourself (read_file / list_dir / read_run). If you still need the user, call ask_user — it shows clickable options or a text box and WAITS for the answer in the same turn. NEVER stall by telling the user to paste a file or answer in prose: read it, or ask_user.',
  ]
  if (autoMode) {
    lines.push(
      '',
      '═══ AUTO MODE IS ON ═══',
      'The user enabled Auto mode: your shell run_script calls are auto-approved (no per-run click), so',
      'drive multi-step tasks through to completion — write a script, run it, read the output, fix and',
      're-run — without pausing for confirmation on each shell step. SLURM submissions STILL ask for a',
      'click (they cost queue/compute), so flag those clearly. Keep every run inside the workspace and use',
      'only your given tools. Still call ask_user for genuinely user-owned decisions (ambiguous goals,',
      'destructive or irreversible actions). The user can hit Stop at any time.',
    )
  }
  if (project) {
    lines.push(
      '',
      '═══ Project context ═══',
      `Name: ${project.name || '(unnamed)'}`,
    )
    if (project.description) lines.push(`Description: ${project.description}`)
    if (project.goal) lines.push(`Goal: ${project.goal}`)
    if (project.active_model) lines.push(`Active model: ${project.active_model}`)
    if (project.active_dataset) {
      lines.push(`Active dataset: ${project.active_dataset}`)
      if (project.active_dataset_inspect) {
        lines.push('Active dataset summary:', '```json', JSON.stringify(project.active_dataset_inspect, null, 2), '```')
      }
    }
    if (Array.isArray(project.recent_notes) && project.recent_notes.length) {
      lines.push('', 'Recent project notes (excerpts):')
      for (const n of project.recent_notes) {
        lines.push(`--- ${n.name} ---`, n.excerpt, '')
      }
    }
    lines.push(
      ...docProtocolLines(docMode),
      '',
      'You also have download_to_datasets(url, filename). When the user asks for a',
      'standard dataset by name (iris, MNIST, california housing, boston, wine, etc.)',
      'pick a stable raw mirror and download it. Iris CSV with header is at',
      'https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv',
      'or https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
      '(headerless). After download, mention the dataset appears in SpinoML\'s Datasets',
      'tab and suggest the next concrete step (e.g. "build an MLP with 4-input Input").',
      '',
      'You also have write_dataset_file(filename, content) for TEXT files you author',
      'yourself — above all a `.manifest` (JSON pairing a table to per-branch sources',
      'for a dual-encoder: ligand SMILES + protein graph + target column). When',
      'the user needs a manifest, gather the table path, the per-branch columns/dirs and',
      'the target column, then write it directly — do NOT tell the user you cannot access',
      'their files. A `dir` inside a manifest must be an ABSOLUTE path. Binary files still',
      'go through download_to_datasets. Each branch declares HOW its cell becomes a tensor',
      'and which typed input node consumes it: kind:"molecule" → a Graph node (RDKit graph);',
      'kind:"espf" → an ESPF node (interpretable SMILES substructure subword tokens, MolTrans',
      'codebook, codebook:"drug"/"protein") feeding an Embedding+1D-CNN/Transformer;',
      'kind:"sequence" → a Sequence node (char/byte token ids); a file branch (dir/ext/match)',
      '→ a Graph node loading a .pt. Draw one edge Manifest→input per branch. For a SMILES',
      'drug encoder prefer ESPF over raw char Sequence — the tokens are chemically meaningful',
      'and interpretable. Inspecting the manifest in the Datasets tab shows each branch\'s',
      'suggested node type + the ESPF/Sequence num_embeddings to size the Embedding.',
      '',
      'WORKSPACE FILES + SCRIPTS — you have FULL read/write/execute access to the user\'s workspace,',
      'confined to the workspace root. These are your ONLY system tools (there is no generic',
      'Bash/Read/Write/Edit) — and they are enough. Use them proactively: when a task needs a file',
      'written or a script run, just do it. Never tell the user to create a file, run a command, or',
      'paste contents you can read or write yourself.',
      ' - list_dir(path): see what exists (omit path for the root).',
      ' - read_file(path): read any text file. Read a file before you overwrite it.',
      ' - write_file(path, content): create/overwrite any text file ANYWHERE under the root —',
      '   scripts, configs, manifests, models, notes. Parent dirs are auto-created. Put scripts you',
      '   intend to run in agent/. Write code to a FILE; never paste long code into chat.',
      ' - run_script(path, mode, args?): execute a script. Output streams live; you get the exit code',
      '   and captured stdout/stderr back.',
      ' - slurm_status(job_id): poll a SLURM job submitted with run_script(mode:"slurm").',
      '',
      'STANDARD PROCEDURE for "run something / preprocess data / script a task":',
      ' 1. list_dir / read_file to understand the inputs (skip if you already know them).',
      ' 2. write_file a self-contained script into agent/ (e.g. agent/<task>.py). Make it print clear',
      '    progress and a final summary so the streamed output is useful.',
      ' 3. run_script it. Choose the mode by weight, not by fear:',
      '      - LOCAL workspace: mode:"shell" for almost everything (generous ~10-min cap).',
      '      - REMOTE/HPC workspace: mode:"shell" only for quick/light work (login node, ~2-min cap);',
      '        mode:"slurm" (an sbatch script with #SBATCH directives) for real compute — training,',
      '        embedding/tokenizing a large dataset, GPU work, anything that needs minutes. When unsure',
      '        on HPC, prefer slurm; then poll slurm_status and read the output file with read_file.',
      ' 4. Read the result. On a non-zero exit, fix the script and run it again — iterate to a green',
      '    run rather than handing the raw error back to the user.',
      'The user approves each run with a one-click GUI dialog; that is expected — call the tool and',
      'continue once approved. A declined run is a clear "no": adjust, or ask what they want instead.',
      'Data-prep specifically mirrors a DataOp node: have the script write its output under datasets/',
      '(and extend a .manifest if pairing branches) so it appears live in the Datasets tab — author',
      'the script, run it, the dataset appears.',
      'Two HPC habits: for "does X exist / how many" use list_dir + read_file (instant) instead of a',
      'script; inside scripts prefer os.path.exists() on specific paths over globbing a huge networked',
      'cache dir (that can hang for minutes).',
      '',
      'You can inspect TRAINING RUNS to optimize models: list_runs shows recent runs in',
      'experiments/runs/ (status, best_val_loss, model), and read_run(run_id) returns one',
      'run\'s hyperparameters + per-epoch history (train/val loss, lr) + outcome. Read a',
      'run, diagnose it (train↓ val↑ → overfit → add Dropout / weight_decay / less capacity;',
      'both high or flat → underfit or LR too low → more capacity / tune lr / add scheduler),',
      'then APPLY the fix with update_params / update_training_params / add_layer and say',
      'briefly what you changed and why. Prefer one focused change at a time.',
      '',
      'RUN EVENT PROTOCOL — when you write a CUSTOM train.py (e.g. a joint/multitask',
      'binder-classification + affinity-regression trainer) that should show up live in the',
      'Experiments UI, write it INTO a run dir experiments/runs/<run_id>/ and emit the same',
      'events SpinoML\'s built-in trainer does, one JSON object per line (flushed) to',
      'events.jsonl, plus a `status` file (queued|running|done|failed|cancelled). Schema:',
      ' - Also write a `pid` file in the run dir: under SLURM its content MUST be',
      '   "slurm:$SLURM_JOB_ID" (the viewer polls squeue for liveness); a direct/local run',
      '   writes the OS pid. A `running` status with no live process is shown as FAILED — this',
      '   is the #1 reason a healthy SLURM run looks failed, so always set the slurm: pid.',
      ' - run.start {pid}; dataset.loaded {n_rows, branches?, n_binders?, skipped?};',
      '   model.built {n_params}.',
      ' - epoch.end {epoch (0-based), train_loss, val_loss?, val_acc?, metrics?, lr}. metrics is',
      '   a flat {name: float} map. For MULTITASK namespace per-task metrics as "<output>/<metric>"',
      '   (e.g. {"binder/acc":0.93, "affinity/mae":0.21}) — the charts auto-discover those keys;',
      '   train_loss/val_loss stay the COMBINED (weighted) loss.',
      ' - checkpoint {epoch, path:"checkpoints/best.pt", val_loss, is_best:true} on each new best.',
      ' - eval.summary (re-emitted on each best, drives the "Auswertung" diagrams). SINGLE task:',
      '   {task:"classification"|"binary"|"regression", and EITHER confusion:{labels:[…],',
      '   matrix:[[…]]} (rows=truth, cols=pred) for classification/binary OR',
      '   scatter:{points:[[pred,truth],…], n_total, pred_label, truth_label} for regression}.',
      '   MULTITASK: {heads:[{output:"binder", task:"binary", confusion:{…}}, {output:"affinity",',
      '   task:"regression", scatter:{…}}]}. WITHOUT this event the Auswertung tab stays empty.',
      ' - sample.preds {epoch, rows:[{pred,truth,conf?,correct?}]} (single) or {epoch,',
      '   heads:[{output, task, rows:[…]}]} (multitask).',
      ' - run.done {total_seconds, best_val_loss} on success; run.failed {stage, error, traceback}.',
      'A joint run that omits eval.summary/sample.preds still charts loss+metrics but shows nothing',
      'under Auswertung — always emit them so the user sees the confusion matrix + affinity scatter.',
      '',
      'Beyond built-in layers you have two escape hatches: add_custom_node(source, init_args?)',
      'for a free-form nn.Module you write (heads, custom attention, branch fusion — torch/nn/F',
      'pre-imported, prefer nn.LazyLinear so shapes infer), and add_subgraph(class_name, nodes,',
      'edges) to encapsulate a reusable block or a dual-encoder branch as a nested module. Reach',
      'for these whenever no built-in layer expresses what the user needs.',
    )
    if (project.ssh_target) {
      lines.push(
        '',
        `This workspace lives on REMOTE host \`${project.ssh_target}\` at \`${project.root}\`. All your`,
        'workspace tools (read_file, write_file, list_dir, run_script, slurm_status, notes, datasets)',
        'route over ssh to THAT host and operate on the HPC filesystem directly, relative to the',
        'workspace root. run_script(mode:"shell") runs on the login node (~2-min cap); for heavy',
        'compute write an sbatch script (#SBATCH directives) and use run_script(mode:"slurm").',
      )
    }
    lines.push('═══════════════════════')
  }
  lines.push(
    '',
    'Current architecture snapshot:',
    '```json',
    JSON.stringify(snapshot, null, 2),
    '```',
  )
  lines.push(
    '',
    '═══ Training graph (separate from the architecture) ═══',
    'SpinoML also has a VISUAL TRAINING GRAPH — how a model is trained, built as nodes',
    'just like the architecture. Mutate it with the training tools: add_training_node,',
    'connect_training_nodes, update_training_params, delete_training_node, clear_training_graph.',
    'These are DIFFERENT from add_layer/connect (which only touch the architecture).',
    'Use the training tools when the user asks to set up / configure training, a training',
    'loop, optimizer, loss, schedule, callbacks, etc.',
    '',
    'Training node types and their key params:',
    ' - DatasetSource {dataset: "datasets/<file>", target: "<column>", features: [<columns>] (empty = all numeric)}. For a PAIRED GRAPH model (dual-encoder), set dataset to a ".manifest" file — it carries its own target + pairs the per-branch graphs, so target/features are ignored.',
    ' - Split {strategy: random|grouped, val_ratio: 0..0.9, seed, group_column} — strategy "grouped" needs group_column: every row sharing that value (e.g. one patient) stays in ONE partition, so nothing leaks train↔val.',
    ' - DataLoader {batch_size, shuffle, num_workers, drop_last}',
    ' - ModelSource {model: "models/<file>.spinoml"}',
    ' - Loss {kind: CrossEntropyLoss|BCEWithLogitsLoss|MSELoss|L1Loss, label_smoothing}',
    ' - Optimizer {kind: Adam|AdamW|SGD|RMSprop, lr, weight_decay, momentum}',
    ' - Scheduler {kind: none|StepLR|CosineAnnealingLR|ReduceLROnPlateau, step_size, gamma, patience}',
    ' - Metric {kind: accuracy|f1|precision|recall|mse|mae|r2|auroc|auprc|ef}  (add several for multiple metrics; auroc/auprc/ef are ranking metrics for a BINARY head)',
    ' - EarlyStopping {monitor: val_loss|val_acc|train_loss, patience, mode: min|max}',
    ' - GradientClipping {max_norm}',
    ' - MixedPrecision {dtype: fp16|bf16}',
    ' - TrainLoop {epochs, seed, log_every_n_steps, val_every_n_epochs, gradient_accumulation_steps}',
    '',
    'A runnable training graph needs at minimum: DatasetSource (a target column for',
    'tabular, OR a .manifest for paired graphs), ModelSource (with a .spinoml model),',
    'Loss, Optimizer, and TrainLoop — connect each source/component INTO the TrainLoop.',
    'Trainable dataset kinds: tabular (csv/tsv/parquet) and .manifest (paired graphs).',
    'Pick the loss to match the task: CrossEntropyLoss for classification, MSELoss for regression.',
    'When building from scratch, call clear_training_graph first. Use the model + dataset from',
    'the project context above when available.',
    '',
    'Current training-graph snapshot:',
    '```json',
    JSON.stringify(trainingSnapshot ?? { nodes: [], edges: [] }, null, 2),
    '```',
    '═══════════════════════',
  )
  lines.push(
    '',
    '═══ Data-processing graph (the third canvas) ═══',
    'SpinoML has a VISUAL DATA-PROCESSING CANVAS — a DAG of data-prep steps that compiles to ONE',
    'reproducible Python pipeline script (pandas + lazy rdkit/torch_geometric/biopython) which the user',
    'runs via run_script. Mutate it with the data tools: add_data_node, connect_data_nodes,',
    'update_data_params, delete_data_node, clear_data_graph (DISTINCT from the architecture + training tools).',
    'Use these whenever the user needs to PREPARE data: download files from a column of IDs (e.g. .pdb from',
    'RCSB or .fasta sequences from UniProt), build PyG graphs from SMILES (3D) or .pdb structures, or',
    'rename / normalize / filter / compute columns.',
    'Node types + key params:',
    ' - TableSource {dataset:"datasets/<file>"} — loads a csv/tsv/parquet as the DataFrame df.',
    ' - DownloadColumn {id_column, url_template (contains {id}), out_dir, filename_template, add_path_column}.',
    ' - RenameColumns {mapping:"old:new, a:b"} · SelectColumns {columns:"a,b"} · FilterRows {query:"label==1"}.',
    ' - Normalize {columns (empty=all numeric), method:zscore|minmax} · DropNA {subset} · ComputeColumn {name, expr}.',
    ' - SmilesToGraph {smiles_column, out_name(.pt), embed_3d, add_hydrogens} · StructureToGraph {path_column, out_name, contact_threshold}.',
    ' - CustomScript {label, code} — free Python; `df` is the running DataFrame (the escape hatch you can write).',
    ' - WriteDataset {out_path:"datasets/<file>", format:csv|parquet|pt} — writes the result so it shows in the Datasets tab + feeds training.',
    'A pipeline reads source → transforms/graph/fetch → WriteDataset; connect each step into the next.',
    'IMPORTANT: the generated script threads ONE shared DataFrame df, so each .spinodata must be a',
    'SINGLE LINEAR chain — exactly one source (TableSource, or a CustomScript that loads df) at the',
    'start, no branching, no second source chain. For separate data sources, build separate pipelines',
    'or merge them inside a CustomScript.',
    'To RUN it: the user clicks "Pipeline ausführen" on the canvas, OR you write the equivalent script to',
    'agent/ with write_file and run it with run_script (then record_step). When building from scratch,',
    'call clear_data_graph first.',
    '',
    'Current data-graph snapshot:',
    '```json',
    JSON.stringify(dataSnapshot ?? { nodes: [], edges: [] }, null, 2),
    '```',
    '═══════════════════════',
  )
  if (error) {
    lines.push('', `Current forward-pass error: ${error.message}`)
    if (error.failingNodeId) lines.push(`Failing node: ${error.failingNodeId} (${error.failingNodeLayerType ?? 'unknown'})`)
  }
  return lines.join('\n')
}

function formatHistoryAsPrompt(messages, latestUser) {
  const turns = []
  for (const m of messages ?? []) {
    if (m.role === 'user') turns.push(`User: ${m.content}`)
    else if (m.role === 'assistant') turns.push(`Assistant: ${m.content}`)
  }
  turns.push(`User: ${latestUser}`)
  turns.push('Assistant:')
  return turns.join('\n\n')
}

async function handleChat(req, res) {
  // Cap the request body. A normal /chat payload (system prompt + a long
  // history + a few graph/training/data nodes) is well under 256 KiB; 1 MiB
  // leaves headroom for unusually large system prompts while preventing an
  // unbounded stream from wedging the event loop or blowing memory.
  const MAX_BODY_BYTES = 1_048_576
  let body = ''
  let bodyBytes = 0
  for await (const chunk of req) {
    bodyBytes += chunk.length
    if (bodyBytes > MAX_BODY_BYTES) {
      return sendJson(res, 413, { error: `request body too large (>${MAX_BODY_BYTES} bytes)` })
    }
    body += chunk
  }
  let payload
  try { payload = JSON.parse(body || '{}') }
  catch (e) { return sendJson(res, 400, { error: `invalid json: ${e.message}` }) }
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return sendJson(res, 400, { error: 'body must be a JSON object' })
  }

  const { user, messages, graph, training_graph, data_graph, error, project, llm, autoMode, docMode } = payload
  if (typeof user !== 'string' || !user.trim()) {
    return sendJson(res, 400, { error: 'missing "user" string' })
  }
  // Type guards for the structural payload fields — the LLM is untrusted input
  // and a non-array `messages` would otherwise explode inside formatHistory.
  if (messages !== undefined && !Array.isArray(messages)) {
    return sendJson(res, 400, { error: '"messages" must be an array' })
  }
  if (graph !== undefined && (graph === null || typeof graph !== 'object' || Array.isArray(graph))) {
    return sendJson(res, 400, { error: '"graph" must be an object' })
  }
  // FEAT-3 Auto-Modus: shell run_script auto-approves (SLURM still confirms).
  // FEAT-4 Doku-Modus: 'verbose' | 'compact' | 'off' (default verbose).
  const autoApproveShell = autoMode === true
  const docModeVal = docMode === 'off' || docMode === 'compact' ? docMode : 'verbose'
  // Resolve the provider kind. A MISSING `llm`/`llm.kind` keeps the documented
  // default (`subscription` — the OAuth/CLI path) so older frontends stay
  // green. An UNKNOWN string kind must NOT silently fall back to subscription
  // (Phase 44/77 invariant: every failure becomes an explicit application
  // state — silently swapping providers would make a misconfigured frontend
  // appear to work). A wrong type is also an explicit error.
  const ALLOWED_KINDS = new Set(['opencode', 'subscription', 'anthropic', 'openai-compat'])
  let kind
  if (llm === undefined || llm === null || llm.kind === undefined || llm.kind === null || llm.kind === '') {
    kind = 'subscription'
  } else if (typeof llm.kind !== 'string') {
    return sendJson(res, 400, { error: `invalid llm.kind: expected string, got ${typeof llm.kind}` })
  } else if (!ALLOWED_KINDS.has(llm.kind)) {
    return sendJson(res, 400, { error: `unknown llm.kind: ${JSON.stringify(llm.kind)} (allowed: ${[...ALLOWED_KINDS].join(', ')})` })
  } else {
    kind = llm.kind
  }

  applyCors(res)
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  })

  // Guarded so a late write after the client disconnects can't crash the turn.
  // Every `status:error` message passes through secret redaction centrally, so
  // no provider path can leak the apiKey (or any token-shaped string) to the
  // SSE stream. Never log the payload/llm objects themselves.
  function emit(ev) {
    if (ev && ev.type === 'status' && ev.value === 'error' && typeof ev.message === 'string') {
      ev = { ...ev, message: redactSecrets(ev.message, [llm?.apiKey]) }
    }
    try { res.write(`data: ${JSON.stringify(ev)}\n\n`) } catch { /* client already disconnected — no reader left to mislead */ }
  }

  // WebKitGTK (the Tauri webview on Linux) buffers a streamed fetch() body and
  // doesn't surface bytes to the reader until enough accumulate or the stream
  // closes. During a quiet stretch — above all the run_script confirm `ask`,
  // after which the tool handler BLOCKS awaiting the answer — a small event sits
  // in that buffer and the confirm card never appears: a deadlock where the UI
  // waits for the event and the backend waits for the click. A periodic padded
  // SSE comment keeps bytes flowing so every event is delivered within ~1s, and
  // we flush a pad right after an `ask` so the card appears at once. Comment
  // lines (": …") are ignored by the EventSource spec and by our SSE parser.
  try { res.socket?.setNoDelay(true) } catch { /* not a TCP socket */ }
  const heartbeatPad = ':' + ' '.repeat(16384) + '\n\n'
  const heartbeat = setInterval(() => { try { res.write(heartbeatPad) } catch { /* client already disconnected — no reader left to mislead */ } }, 1000)

  // Per-turn ask/answer plumbing. The registry lets us reject any still-pending
  // question when the turn ends or the client disconnects (no hung handlers).
  const requestId = randomUUID()
  const askRegistry = new Set()
  const rejectPendingAsks = (reason) => {
    for (const askId of askRegistry) {
      const e = pendingAsks.get(askId)
      if (e) { pendingAsks.delete(askId); e.reject(new Error(reason)) }
    }
    askRegistry.clear()
  }
  // Aborted when the client disconnects (Stop/reset in the chat) → kills any
  // running run_script child so the user isn't stuck waiting on the login node.
  const turnAbort = new AbortController()
  // Register in the SIGTERM-aware set so a parent-initiated shutdown can
  // abort the in-flight provider stream too — without this the run_script
  // child would be reaped by killTrackedChildren while its provider is
  // still trying to write, which on some SDKs deadlocks the event loop.
  activeTurnControllers.add(turnAbort)
  // The client can go away via an explicit Stop/reset (the EventSource/fetch is
  // aborted). Node may report that on the request OR the response depending on
  // how the client tears the connection down, so watch both.
  const onClientGone = () => { turnAbort.abort(); rejectPendingAsks('client disconnected') }
  req.on('close', onClientGone)
  res.on('close', onClientGone)

  const actions = makeActionStream()
  // Single outbound-event lane. Events emitted from INSIDE a tool handler (ask,
  // log) go through this queue, not res.write directly — the same drained-loop
  // path that graph `action` events use and that reliably reaches the browser.
  // (Direct res.write from a nested SDK handler did not surface the confirm card.)
  const pushEvent = (ev) => actions.push({ event: ev })
  const askUser = makeAsker(pushEvent, requestId, askRegistry)

  const ctx = makeGraphContext({
    inputShape: graph?.input_shape ?? graph?.inputShape,
    nodes: graph?.nodes ?? [],
    edges: graph?.edges ?? [],
  })
  const trainingCtx = makeTrainingContext({
    nodes: training_graph?.nodes ?? [],
    edges: training_graph?.edges ?? [],
  })
  const dataCtx = makeDataContext({
    nodes: data_graph?.nodes ?? [],
    edges: data_graph?.edges ?? [],
  })

  // Drain the outbound queue to SSE as items are pushed, in parallel with the SDK.
  // A raw `{event}` is emitted verbatim; a `{op,payload}` is a graph action.
  const actionPump = (async () => {
    for await (const a of actions.drain()) {
      emit(a.event ? a.event : { type: 'action', op: a.op, payload: a.payload })
      // The confirm `ask` is immediately followed by a long quiet wait; push it
      // through the webview's read buffer at once so the card shows instantly
      // (the 1s heartbeat is only the backstop).
      if (a.event && a.event.type === 'ask') {
        try { res.write(heartbeatPad) } catch { /* client already disconnected — no reader left to mislead */ }
      }
    }
  })()

  const workspace = makeWorkspace(project)
  const specs = buildToolSpecs(ctx, trainingCtx, dataCtx, actions, workspace, askUser, pushEvent, turnAbort.signal, autoApproveShell)
  const systemPrompt = buildSystemPrompt(ctx.snapshot(), error, project, trainingCtx.snapshot(), dataCtx.snapshot(), docModeVal, autoApproveShell)

  emit({ type: 'status', value: 'thinking' })

  try {
    if (kind === 'anthropic') {
      if (!llm?.apiKey) throw new Error('Anthropic API key missing')
      await runAnthropicApi(specs, systemPrompt, messages, user, emit, llm, askUser)
    } else if (kind === 'openai-compat') {
      await runOpenAiCompat(specs, systemPrompt, messages, user, emit, llm, askUser, turnAbort.signal)
    } else if (kind === 'opencode') {
      // Register the per-turn MCP session BEFORE spawning opencode: the bridge
      // connects the moment the CLI starts. The generated secret is delivered
      // to the bridge through the opencode `environment` of the MCP config —
      // not argv, not a file (`ps` is world-readable). Unregistered in finally.
      const mcpSecret = registerMcpSession(requestId, specs)
      try {
        await runOpenCode(specs, systemPrompt, messages, user, emit, llm, requestId, turnAbort, mcpSecret)
      } finally {
        unregisterMcpSession(requestId)
      }
    } else {
      // Subscription / OAuth path via the claude-agent-sdk.
      const mcp = buildMcpServer(specs)
      const baseOptions = {
        // Our buildSystemPrompt IS the whole system prompt — same as the two
        // direct-API paths, so all three providers behave identically. We do
        // NOT use the claude_code preset: that prompt assumes built-in
        // Read/Write/Edit/Bash tools which we deliberately disable below, and
        // mixing it in gave the agent a split personality (it would try to
        // `npm run build` the repo). settingSources:[] keeps the dev repo's
        // CLAUDE.md / ~/.claude settings + hooks from leaking into this
        // embedded product assistant (OAuth credentials are loaded
        // separately and are unaffected).
        systemPrompt,
        settingSources: [],
        mcpServers: { graph: mcp },
        // CRITICAL: `tools: []` disables ALL built-in tools (Bash, Write,
        // Read, Edit, Glob, …). allowedTools alone does NOT do this — under
        // bypassPermissions every unlisted built-in tool is still auto-
        // approved, so the model would run scripts via built-in Bash on the
        // LAPTOP (not the workspace/HPC) and skip our GUI confirm gate. With
        // built-ins gone, the ONLY tools are our mcp__graph__* below, which
        // route to the workspace and gate run_script through askUser.
        tools: [],
        allowedTools: [
          'mcp__graph__set_input_shape',
          'mcp__graph__add_layer',
          'mcp__graph__add_custom_node',
          'mcp__graph__add_subgraph',
          'mcp__graph__connect',
          'mcp__graph__update_params',
          'mcp__graph__delete_node',
          'mcp__graph__add_training_node',
          'mcp__graph__connect_training_nodes',
          'mcp__graph__update_training_params',
          'mcp__graph__delete_training_node',
          'mcp__graph__clear_training_graph',
          'mcp__graph__add_data_node',
          'mcp__graph__connect_data_nodes',
          'mcp__graph__update_data_params',
          'mcp__graph__delete_data_node',
          'mcp__graph__clear_data_graph',
          'mcp__graph__ask_user',
          ...(workspace ? [
            'mcp__graph__list_notes',
            'mcp__graph__read_note',
            'mcp__graph__append_note',
            'mcp__graph__record_step',
            'mcp__graph__download_to_datasets',
            'mcp__graph__write_dataset_file',
            'mcp__graph__list_runs',
            'mcp__graph__read_run',
            'mcp__graph__read_file',
            'mcp__graph__write_file',
            'mcp__graph__list_dir',
            'mcp__graph__run_script',
            'mcp__graph__slurm_status',
          ] : []),
        ],
        permissionMode: 'bypassPermissions',
        maxTurns: MAX_TOOL_TURNS,
      }
      // On hitting maxTurns the SDK emits error_max_turns and the iterator ends.
      // We capture the session id, ASK the user whether to continue, and resume the
      // SAME session (full context preserved) for another batch — repeat until done
      // or the user stops.
      let sessionId = null
      let promptText = formatHistoryAsPrompt(messages, user)
      let batch = 0
      while (true) {
        let hitMax = false
        for await (const m of query({
          prompt: promptText,
          options: sessionId ? { ...baseOptions, resume: sessionId } : baseOptions,
        })) {
          if (m.type === 'result') {
            if (m.session_id) sessionId = m.session_id
            if (m.subtype === 'error_max_turns') hitMax = true
          } else if (m.type === 'system' && m.session_id) {
            sessionId = m.session_id
          }
          handleSdkMessage(m, emit)
        }
        if (!hitMax) break
        if (!(await confirmContinue(askUser, emit, MAX_TOOL_TURNS * (++batch)))) break
        promptText = 'Mach weiter mit der laufenden Aufgabe.'
      }
    }
    emit({ type: 'status', value: 'done' })
  } catch (e) {
    emit({ type: 'status', value: 'error', message: `${e.name}: ${e.message}` })
  } finally {
    activeTurnControllers.delete(turnAbort)
    clearInterval(heartbeat)
    rejectPendingAsks('chat turn ended')
    actions.close()
    await actionPump
    emit({ type: 'done' })
    res.end()
  }
}

function handleSdkMessage(m, emit) {
  // The SDK emits typed messages — we forward what the frontend cares about.
  if (m.type === 'assistant' && m.message?.content) {
    for (const block of m.message.content) {
      if (block.type === 'text' && typeof block.text === 'string' && block.text.length) {
        emit({ type: 'text', value: block.text })
      } else if (block.type === 'tool_use') {
        emit({ type: 'tool_use', id: block.id, name: block.name, args: block.input ?? {} })
      }
    }
  } else if (m.type === 'user' && m.message?.content) {
    for (const block of m.message.content) {
      if (block.type === 'tool_result') {
        const text = Array.isArray(block.content)
          ? block.content.map((c) => c.text ?? '').join('').trim()
          : typeof block.content === 'string' ? block.content : ''
        emit({
          type: 'tool_result',
          id: block.tool_use_id,
          ok: !block.is_error,
          result: text,
          error: block.is_error ? text : undefined,
        })
      }
    }
  }
  // error_max_turns is handled by the resume loop in the chat handler (it asks the
  // user whether to continue), not surfaced here as an error.
}

const HEALTH_FULL_BODY = { ok: true }

// Phase 51 — read-only diagnostic counters for the resource-leak / soak harnesses.
// Reads the module-level collections that already gate child/ask/turn lifecycle
// plus /proc/self. Every field is null when the platform cannot provide it
// (non-Linux) — it never throws. ONLY the full /health body carries `diag`;
// the limited unauthenticated body must not (process/thread/fd counts are info).
function diagSnapshot() {
  // When the process runs with --expose-gc (the soak/resource-leak harnesses do
  // this) collapse the V8 heap first, so rss_kb reports retained memory rather
  // than an arbitrary point on the sawtooth. No-op without the flag.
  if (typeof global.gc === 'function') global.gc()
  const readProcCount = (path) => {
    try {
      return readdirSync(path).length
    } catch (e) {
      console.error(`[spinoml-llm] diag: cannot list ${path}: ${e && e.message ? e.message : e}`)
      return null
    }
  }
  let rssKb = null
  try {
    const text = readFileSync('/proc/self/status', 'utf8')
    const m = /^VmRSS:\s+(\d+)/m.exec(text)
    if (m) rssKb = Number(m[1])
  } catch (e) {
    console.error(`[spinoml-llm] diag: cannot read /proc/self/status: ${e && e.message ? e.message : e}`)
  }
  return {
    pending_asks: pendingAsks.size,
    mcp_sessions: mcpSessions.size,
    active_turns: activeTurnControllers.size,
    tracked_children: trackedChildren.size,
    open_session_dirs: openSessionDirs.size,
    open_fds: readProcCount('/proc/self/fd'),
    rss_kb: rssKb,
    uptime_s: Math.round(process.uptime() * 1000) / 1000,
    requests_total: requestCount,
  }
}

const server = createServer(async (req, res) => {
  requestCount++
  // Per docs/engineering/SIDECAR_AUTH.md § 'Per-request enforcement order':
  // Host → Origin → OPTIONS → GET /health → token → otherwise. The auth
  // gate runs BEFORE any body is read so a malformed credential cannot
  // trigger work.
  const decision = decideFromHeaders(req.method, req.url ?? '/', req, AUTH_CONFIG)
  setCorsOrigin(res, decision.origin)
  if (decision.kind === 'reject') {
    return rejectRequest(res, decision)
  }
  if (decision.kind === 'ok_health_limited') {
    // Wrong / missing token on /health: limited body, no other keys.
    return sendJson(res, 200, limitedHealthBody())
  }
  // decision.kind === 'ok'

  if (req.method === 'OPTIONS') {
    applyCors(res)
    res.writeHead(204)
    res.end()
    return
  }
  if (req.method === 'GET' && req.url === '/health') {
    return sendJson(res, 200, healthBody(AUTH_CONFIG, { ...HEALTH_FULL_BODY, diag: diagSnapshot() }, decision.tokenOk))
  }
  if (req.method === 'POST' && req.url === '/respond') {
    let body = ''
    for await (const chunk of req) body += chunk
    let payload
    try { payload = JSON.parse(body || '{}') }
    catch (e) { return sendJson(res, 400, { error: `invalid json: ${e.message}` }) }
    const ok = resolveAsk(String(payload.askId ?? ''), payload.answer)
    return sendJson(res, ok ? 200 : 404, ok ? { ok: true } : { error: 'no such pending question' })
  }
  if (req.method === 'POST' && req.url?.startsWith('/internal/mcp/')) {
    try { return await handleMcpRoute(req, res) }
    catch (e) { return sendJson(res, 500, { error: `internal: ${e.name}: ${e.message}` }) }
  }
  if (req.method === 'GET' && req.url === '/opencode/models') {
    try {
      const models = await listOpenCodeModels()
      return sendJson(res, 200, { ok: true, provider: 'opencode', models })
    } catch (e) { return sendJson(res, 500, { error: `${e.name}: ${e.message}` }) }
  }
  if (req.method === 'POST' && req.url === '/chat') {
    try { await handleChat(req, res) }
    catch (e) {
      const msg = redactSecrets(`${e.name}: ${e.message}`, [])
      if (!res.headersSent) sendJson(res, 500, { error: msg })
      else { try { res.write(`data: ${JSON.stringify({ type: 'status', value: 'error', message: msg })}\n\n`); res.end() } catch { /* client already disconnected — already in the error path */ } }
    }
    return
  }
  sendJson(res, 404, { error: 'not found' })
})

server.on('error', (e) => {
  if (e && e.code === 'EADDRINUSE') {
    console.error(`[spinoml-llm] port ${PORT} is already in use — another sidecar running? (stop it or set SPINOML_LLM_PORT)`)
    process.exit(3)
  }
  console.error(`[spinoml-llm] server error: ${e && e.message ? e.message : String(e)}`)
  process.exit(1)
})

server.listen(PORT, '127.0.0.1', () => {
  console.log(`[spinoml-llm] listening on http://127.0.0.1:${PORT}`)
})
