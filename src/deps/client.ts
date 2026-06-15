// HTTP client for the torch sidecar's dependency endpoints. The compatibility
// "smoke test" (/deps/check) resolves requested specs against the sidecar's
// Python env via `pip install --dry-run` WITHOUT installing — so you learn
// whether the desired versions are even compatible before committing.

import { currentTorchUrl } from '../sidecars/torchUrl'

export type DepReq = { spec: string; name: string; installed: string | null }

export type DepsCheckOk = {
  ok: true
  compatible: boolean
  python: string
  requested: DepReq[]
  would_install: string[]
  log?: string
  error?: string
}
export type DepsErr = { ok: false; error: string; offline?: true }
export type DepsCheckResult = DepsCheckOk | DepsErr

export type DepsInstallResult =
  | { ok: true; returncode: number; log: string }
  | { ok: false; error?: string; returncode?: number; log?: string; offline?: true }

async function post<T>(path: string, body: unknown): Promise<T | DepsErr> {
  try {
    const res = await fetch(`${currentTorchUrl()}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok) return { ok: false, error: `sidecar HTTP ${res.status}`, offline: true }
    return (await res.json()) as T
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e)
    return { ok: false, error: `sidecar unreachable: ${msg}`, offline: true }
  }
}

export function checkDeps(specs: string[]): Promise<DepsCheckResult> {
  return post<DepsCheckOk>('/deps/check', { specs })
}

export function installDeps(specs: string[]): Promise<DepsInstallResult> {
  return post<DepsInstallResult & object>('/deps/install', { specs }) as Promise<DepsInstallResult>
}

/** Parse a requirements.txt body into individual specs, dropping blanks and
 *  comments. Inline comments (` # …`) are stripped per PEP 508-ish convention. */
export function parseRequirements(text: string): string[] {
  return text
    .split('\n')
    .map((l) => l.replace(/\s+#.*$/, '').trim())
    .filter((l) => l && !l.startsWith('#'))
}
