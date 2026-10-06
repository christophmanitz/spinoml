// Pure, symlink-aware workspace path containment for the LLM sidecar.
//
// The lexical guard (main.mjs safeRelpath) only rejects absolute / `~` / `..`
// segments; the OS still follows symlinks, so a link inside the workspace can
// point outside it. This module resolves the real path and requires it to stay
// inside the realpath of the workspace root — OR inside an explicitly
// allow-listed "symlink target" root that the USER configured out-of-band
// (env SPINOML_SYMLINK_TARGETS plus an optional scope.json). The allow-list
// is never derived from the LLM/request.
//
// All I/O here is filesystem metadata only; nothing is written. The resolved
// `abs` is what callers must use for every later operation, so swapping an
// intermediate symlink after the check cannot redirect the operation.

import { promises as fs, statSync, readFileSync } from 'node:fs'
import path from 'node:path'
import os from 'node:os'

const MAX_PATH_LEN = 4096
const MAX_SEGMENT_BYTES = 255
const MAX_LINK_DEPTH = 64

// File results cached by (path, mtimeMs, size): the file is stat'd on every
// call, and only re-read/re-parsed when it actually changed.
const scopeFileCache = new Map()

// ── loadSymlinkTargets ──────────────────────────────────────────────────────
// Normalise one configured root entry. Absolute, non-`/` only; returns null
// for anything skipped and pushes the reason onto `reasons`.
function normalizeRootEntry(entry, label, reasons) {
  if (typeof entry !== 'string' || entry === '') {
    reasons.push(`${label} entry is not a non-empty string`)
    return null
  }
  if (!path.isAbsolute(entry)) {
    reasons.push(`${label} entry is not absolute: ${entry}`)
    return null
  }
  let abs
  try { abs = path.resolve(entry) } catch { abs = '' }
  if (!abs || abs === path.parse(abs).root) {
    reasons.push(`${label} entry is the filesystem root: ${entry}`)
    return null
  }
  return abs
}

function readScopeFile(file, now) {
  let stat
  try { stat = statSync(file) }
  catch { return { exists: false, targets: [], loadError: null } }

  if (!stat.isFile()) {
    return { exists: true, targets: [], loadError: `scope file is not a regular file: ${file}` }
  }
  if (typeof process.getuid === 'function' && stat.uid !== process.getuid()) {
    return { exists: true, targets: [], loadError: `scope file is not owned by the current uid: ${file}` }
  }
  if ((stat.mode & 0o022) !== 0) {
    return { exists: true, targets: [], loadError: `scope file is group/world writable: ${file}` }
  }

  const cached = scopeFileCache.get(file)
  if (cached && cached.mtimeMs === stat.mtimeMs && cached.size === stat.size) return cached

  let text
  try { text = readFileSync(file, 'utf8') }
  catch (e) { return cacheResult(file, stat, now, [], `scope file is unreadable: ${file}: ${e.message}`) }

  let parsed
  try { parsed = JSON.parse(text) }
  catch (e) { return cacheResult(file, stat, now, [], `scope file is not valid JSON: ${e.message}`) }

  const reasons = []
  let rawTargets = []
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    reasons.push('scope file root is not an object')
  } else {
    if (parsed.version != null && parsed.version !== 1) {
      reasons.push(`unsupported scope file version: ${parsed.version}`)
    }
    if (parsed.symlink_targets == null) rawTargets = []
    else if (!Array.isArray(parsed.symlink_targets)) {
      reasons.push('scope file symlink_targets is not an array')
    } else rawTargets = parsed.symlink_targets
  }

  const targets = []
  for (const entry of rawTargets) {
    const n = normalizeRootEntry(entry, 'scope file', reasons)
    if (n) targets.push(n)
  }
  const loadError = reasons.length ? reasons.join('; ') : null
  return cacheResult(file, stat, now, [...new Set(targets)], loadError)
}

function cacheResult(file, stat, now, targets, loadError) {
  const result = { exists: true, mtimeMs: stat.mtimeMs, size: stat.size, targets, loadError, at: now() }
  scopeFileCache.set(file, result)
  return result
}

export function loadSymlinkTargets({ env = process.env, homedir = os.homedir(), now = () => Date.now() } = {}) {
  const reasons = []

  const envTargets = []
  const rawEnv = env?.SPINOML_SYMLINK_TARGETS
  if (rawEnv != null && String(rawEnv) !== '') {
    for (const entry of String(rawEnv).split(path.delimiter)) {
      if (entry === '') continue
      const n = normalizeRootEntry(entry, 'env', reasons)
      if (n) envTargets.push(n)
    }
  }

  // Prefer the runtime-dir scope file when present, else the home cache.
  const candidates = []
  if (env?.XDG_RUNTIME_DIR) candidates.push(path.join(env.XDG_RUNTIME_DIR, 'spinoml', 'scope.json'))
  candidates.push(path.join(homedir, '.cache', 'spinoml', 'scope.json'))

  let fileResult = { exists: false, targets: [], loadError: null }
  for (const candidate of candidates) {
    const r = readScopeFile(candidate, now)
    if (r.exists) { fileResult = r; break }
  }

  const targets = [...new Set([...envTargets, ...fileResult.targets])]
  const envUsed = envTargets.length > 0
  const fileUsed = fileResult.exists && fileResult.targets.length > 0
  const source = envUsed && fileUsed ? 'both' : envUsed ? 'env' : fileUsed ? 'file' : 'none'

  const allReasons = [...reasons]
  if (fileResult.loadError) allReasons.push(fileResult.loadError)

  return {
    targets,
    source,
    loadError: allReasons.length ? allReasons.join('; ') : null,
  }
}

// ── containment ─────────────────────────────────────────────────────────────
// Exact containment: `/tmp/ws-evil` is NOT inside `/tmp/ws`. A path is inside
// itself. Uses path.relative so sibling prefix confusion cannot pass.
export function isInside(child, parent) {
  const rel = path.relative(parent, child)
  if (rel === '') return true
  return rel !== '..' && !rel.startsWith('..' + path.sep) && !path.isAbsolute(rel)
}

// ── lexical checks (mirror of main.mjs safeRelpath) ─────────────────────────
function parseRel(rel) {
  if (rel == null) return { ok: false, error: 'empty path' }
  if (typeof rel !== 'string') return { ok: false, error: 'path must be a string' }
  const s = rel.trim().replace(/^\.\//, '')
  if (!s) return { ok: false, error: 'empty path' }
  if (s.length > MAX_PATH_LEN) return { ok: false, error: `path too long (max ${MAX_PATH_LEN} characters)` }
  if (s.includes('\u0000')) return { ok: false, error: 'path may not contain NUL' }
  if (s.startsWith('/') || s.startsWith('~')) {
    return { ok: false, error: 'path must be relative to the workspace root' }
  }
  const segments = s.split('/').filter((seg) => seg.length && seg !== '.')
  if (segments.some((seg) => seg === '..')) return { ok: false, error: 'path may not contain ".."' }
  for (const seg of segments) {
    if (Buffer.byteLength(seg, 'utf8') > MAX_SEGMENT_BYTES) {
      return { ok: false, error: `path segment too long (max ${MAX_SEGMENT_BYTES} bytes): ${seg}` }
    }
  }
  return { ok: true, segments, display: rel }
}

// Resolve `p` following symlinks, tolerating a not-yet-existing tail (needed
// for writes): find the longest existing ancestor via lstat, realpath it, and
// re-append the remainder. A dangling symlink is resolved through its target
// (readlink relative to its directory) so the containment test sees the real
// destination rather than the link itself.
async function resolveExisting(p, depth = 0) {
  if (depth > MAX_LINK_DEPTH) {
    const e = new Error('too many levels of symbolic links')
    e.code = 'ELOOP'
    throw e
  }
  let cur = p
  const suffix = []
  while (true) {
    let st
    try { st = await fs.lstat(cur) }
    catch (e) {
      if (e.code !== 'ENOENT' && e.code !== 'ENOTDIR') throw e
      const parent = path.dirname(cur)
      if (parent === cur) throw e
      suffix.unshift(path.basename(cur))
      cur = parent
      continue
    }
    let real
    try { real = await fs.realpath(cur) }
    catch (e) {
      if (e.code === 'ENOENT' && st.isSymbolicLink()) {
        const link = await fs.readlink(cur)
        const targetAbs = path.resolve(path.dirname(cur), link)
        const resolvedTarget = await resolveExisting(targetAbs, depth + 1)
        return suffix.length ? path.join(resolvedTarget, ...suffix) : resolvedTarget
      }
      throw e
    }
    return suffix.length ? path.join(real, ...suffix) : real
  }
}

export async function resolveInWorkspace(root, rel, { forWrite = false, symlinkTargets } = {}) {
  const lexical = parseRel(rel)
  if (!lexical.ok) return { ok: false, code: 'PATH_INVALID', error: lexical.error }
  const { segments, display } = lexical

  let realRoot
  try { realRoot = await fs.realpath(root) }
  catch (e) {
    return { ok: false, code: 'PATH_INVALID', error: `workspace root is not accessible: ${e.message}` }
  }

  const candidate = segments.length ? path.join(realRoot, ...segments) : realRoot

  let resolved
  try { resolved = await resolveExisting(candidate) }
  catch (e) {
    if (e && e.code === 'ELOOP') {
      return { ok: false, code: 'PATH_INVALID', error: `symlink loop while resolving "${display}": ${e.message}` }
    }
    return { ok: false, code: 'PATH_INVALID', error: `could not resolve path "${display}": ${e.message}` }
  }

  if (isInside(resolved, realRoot)) return { ok: true, abs: resolved }

  const targets = Array.isArray(symlinkTargets) ? symlinkTargets : []
  for (const t of targets) {
    let realTarget
    try { realTarget = await fs.realpath(t) } catch { continue }
    if (realTarget === path.parse(realTarget).root) continue // never allow /
    if (isInside(resolved, realTarget)) return { ok: true, abs: resolved }
  }

  // The lexical path is always inside realRoot (segments carry no `..`); a
  // mismatch therefore means a symlink redirected it out.
  if (isInside(candidate, realRoot)) {
    return {
      ok: false,
      code: 'PATH_SYMLINK_OUTSIDE',
      error: `"${display}" resolves outside the workspace through a symlink (-> ${resolved}). `
        + 'If this is intended, add the target directory to SPINOML_SYMLINK_TARGETS.',
    }
  }
  return {
    ok: false,
    code: 'PATH_OUTSIDE',
    error: `"${display}" resolves outside the workspace (-> ${resolved}).`,
  }
}
