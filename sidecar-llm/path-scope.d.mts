// Hand-written type declarations for sidecar-llm/path-scope.mjs.
// Mirrors the module's runtime exports exactly (no `any`): containment returns
// a discriminated union carrying the resolved absolute path, or one of the
// documented `PATH_*` failure codes.

export interface LoadSymlinkTargetsOptions {
  env?: Record<string, string | undefined>
  homedir?: string
  now?: () => number
}

export interface LoadSymlinkTargetsResult {
  targets: string[]
  source: 'env' | 'file' | 'both' | 'none'
  loadError: string | null
}

export function loadSymlinkTargets(
  opts?: LoadSymlinkTargetsOptions,
): LoadSymlinkTargetsResult

export function isInside(child: string, parent: string): boolean

export type ResolveInWorkspaceResult =
  | { ok: true; abs: string }
  | {
      ok: false
      code: 'PATH_INVALID' | 'PATH_SYMLINK_OUTSIDE' | 'PATH_OUTSIDE'
      error: string
    }

export interface ResolveInWorkspaceOptions {
  forWrite?: boolean
  symlinkTargets?: readonly string[]
}

export function resolveInWorkspace(
  root: string,
  rel: unknown,
  opts?: ResolveInWorkspaceOptions,
): Promise<ResolveInWorkspaceResult>
