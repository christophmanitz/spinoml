export type TrustOrigin = 'human-edit' | 'user-approval' | 'eject' | 'template'

export type TrustRecord = {
  sha256: string
  origin: TrustOrigin
  approvedAt: string
}

export type StorageLike = {
  getItem(k: string): string | null
  setItem(k: string, v: string): void
}

export const DEFAULT_TRUST_KEY = 'spinoml.codeTrust.v1'

const STORE_VERSION = 1
const DEFAULT_CAP = 5000
const ORIGINS: ReadonlyArray<TrustOrigin> = ['human-edit', 'user-approval', 'eject', 'template']

export type CodeTrustStore = {
  isTrusted(sha256: string): boolean
  get(sha256: string): TrustRecord | undefined
  approve(sha256: string, origin: TrustOrigin): void
  revoke(sha256: string): void
  clear(): void
  size(): number
  records(): TrustRecord[]
  subscribe(listener: () => void): () => void
  readonly lastPersistError: string | null
  readonly loadError: string | null
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

function memoryStorage(): StorageLike {
  const map = new Map<string, string>()
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

/** Resolve the default storage: try the browser storage API, fall back to an
 *  in-memory map. Every browser-storage access is inside a try block so a
 *  SecurityError (private mode, quota, SSR…) becomes the fallback, not a crash. */
function resolveStorage(): { storage: StorageLike; error: string | null } {
  try {
    const ls = (globalThis as { localStorage?: StorageLike }).localStorage
    if (ls && typeof ls.getItem === 'function' && typeof ls.setItem === 'function') {
      const probe = '__spinoml_code_trust_probe__'
      ls.setItem(probe, '1')
      ls.getItem(probe)
      return { storage: ls, error: null }
    }
    return { storage: memoryStorage(), error: null }
  } catch (err) {
    return { storage: memoryStorage(), error: errorMessage(err) }
  }
}

function isTrustOrigin(value: unknown): value is TrustOrigin {
  return typeof value === 'string' && (ORIGINS as ReadonlyArray<string>).includes(value)
}

/** Strict parse — any wrong shape (wrong version, non-array records, malformed
 *  entries) makes the whole persisted blob invalid. Caller treats that as
 *  fail-closed: empty store, `loadError` set, nothing trusted. */
function validatePersisted(parsed: unknown): TrustRecord[] | null {
  if (!isRecord(parsed)) return null
  if (parsed.v !== STORE_VERSION) return null
  const arr = parsed.records
  if (!Array.isArray(arr)) return null
  const out: TrustRecord[] = []
  for (const item of arr) {
    if (!isRecord(item)) return null
    const sha256 = item.sha256
    const origin = item.origin
    const approvedAt = item.approvedAt
    if (typeof sha256 !== 'string' || sha256 === '') return null
    if (!isTrustOrigin(origin)) return null
    if (typeof approvedAt !== 'string') return null
    out.push({ sha256, origin, approvedAt })
  }
  return out
}

export function createTrustStore(opts: {
  storage?: StorageLike
  now?: () => string
  cap?: number
  key?: string
} = {}): CodeTrustStore {
  const key = opts.key ?? DEFAULT_TRUST_KEY
  const cap = typeof opts.cap === 'number' && opts.cap > 0 ? Math.trunc(opts.cap) : DEFAULT_CAP
  const clock = opts.now ?? (() => new Date(0).toISOString())
  const resolved = opts.storage ? { storage: opts.storage, error: null } : resolveStorage()
  const storage = resolved.storage

  let stored: TrustRecord[] = []
  let loadError: string | null = resolved.error
  let lastPersistError: string | null = null
  const listeners = new Set<() => void>()

  function notify(): void {
    for (const listener of listeners) listener()
  }

  function enforceCap(): void {
    if (stored.length <= cap) return
    stored = stored.slice(stored.length - cap)
  }

  function persist(): void {
    try {
      storage.setItem(key, JSON.stringify({ v: STORE_VERSION, records: stored }))
      lastPersistError = null
    } catch (err) {
      lastPersistError = errorMessage(err)
    }
  }

  function commit(): void {
    enforceCap()
    persist()
    notify()
  }

  function load(): void {
    let raw: string | null
    try {
      raw = storage.getItem(key)
    } catch (err) {
      loadError = errorMessage(err)
      stored = []
      return
    }
    if (raw === null) return
    let parsed: unknown
    try {
      parsed = JSON.parse(raw)
    } catch (err) {
      loadError = `invalid JSON: ${errorMessage(err)}`
      stored = []
      return
    }
    const validated = validatePersisted(parsed)
    if (validated === null) {
      loadError = 'invalid trust store shape or version'
      stored = []
      return
    }
    stored = validated
    enforceCap()
  }

  function isTrusted(sha256: string): boolean {
    return stored.some((r) => r.sha256 === sha256)
  }

  function get(sha256: string): TrustRecord | undefined {
    return stored.find((r) => r.sha256 === sha256)
  }

  /** Idempotent: a second call with the same sha keeps the FIRST origin.
   *  This is the property that makes a re-imported identical file stay trusted
   *  while a one-byte edit produces a new hash that has to be approved again. */
  function approve(sha256: string, origin: TrustOrigin): void {
    if (stored.some((r) => r.sha256 === sha256)) return
    stored.push({ sha256, origin, approvedAt: clock() })
    commit()
  }

  function revoke(sha256: string): void {
    const next = stored.filter((r) => r.sha256 !== sha256)
    if (next.length === stored.length) return
    stored = next
    commit()
  }

  function clear(): void {
    stored = []
    commit()
  }

  function size(): number {
    return stored.length
  }

  function records(): TrustRecord[] {
    return stored.slice()
  }

  function subscribe(listener: () => void): () => void {
    listeners.add(listener)
    return () => {
      listeners.delete(listener)
    }
  }

  load()

  return {
    isTrusted,
    get,
    approve,
    revoke,
    clear,
    size,
    records,
    subscribe,
    get lastPersistError(): string | null {
      return lastPersistError
    },
    get loadError(): string | null {
      return loadError
    },
  }
}

/** App-wide singleton with a real clock. `now` is injected here (and only here)
 *  so the pure functions in codeBlobs / gate never touch the wall clock. */
export const trust = createTrustStore({ now: () => new Date().toISOString() })
