// Additive helpers for the Phase 51/52 resource-leak + soak harnesses.
//
// Real /proc sampling only — no mocked counters. Linux-only by nature; every
// reader returns null instead of throwing when /proc is unavailable, so a
// non-Linux host reports "null" rather than fabricating a number.

import { readFileSync, readdirSync } from 'node:fs'

export interface ProcStats {
  pid: number
  rssKb: number | null
  fds: number | null
  threads: number | null
  /** Direct + transitive child processes from /proc/<pid>/task/<tid>/children. */
  children: number | null
  /** utime+stime in seconds (USER_HZ assumed 100 — the Linux default). */
  cpuSeconds: number | null
}

const CLK_TCK = 100

export function readProcStats(pid: number): ProcStats {
  const stats: ProcStats = { pid, rssKb: null, fds: null, threads: null, children: null, cpuSeconds: null }
  try {
    const status = readFileSync(`/proc/${pid}/status`, 'utf8')
    const rss = /^VmRSS:\s+(\d+)/m.exec(status)
    if (rss) stats.rssKb = Number(rss[1])
    const thr = /^Threads:\s+(\d+)/m.exec(status)
    if (thr) stats.threads = Number(thr[1])
  } catch {
    // process gone — leave nulls
  }
  try {
    stats.fds = readdirSync(`/proc/${pid}/fd`).length
  } catch {
    // process gone — leave null
  }
  const kids = new Set<number>()
  try {
    for (const tid of readdirSync(`/proc/${pid}/task`)) {
      try {
        const text = readFileSync(`/proc/${pid}/task/${tid}/children`, 'utf8')
        for (const m of text.matchAll(/(\d+)/g)) kids.add(Number(m[1]))
      } catch {
        // thread vanished mid-walk
      }
    }
    stats.children = kids.size
  } catch {
    // process gone — leave null
  }
  try {
    const stat = readFileSync(`/proc/${pid}/stat`, 'utf8')
    const tail = stat.slice(stat.lastIndexOf(')') + 2).split(' ')
    const utime = Number(tail[11])
    const stime = Number(tail[12])
    if (Number.isFinite(utime) && Number.isFinite(stime)) stats.cpuSeconds = (utime + stime) / CLK_TCK
  } catch {
    // process gone — leave null
  }
  return stats
}

export function delta(after: number | null, before: number | null): number | null {
  if (after === null || before === null) return null
  return after - before
}

function mean(xs: number[]): number {
  return xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length
}

export function median(xs: number[]): number {
  if (xs.length === 0) return 0
  const s = [...xs].sort((a, b) => a - b)
  const mid = Math.floor(s.length / 2)
  return s.length % 2 === 0 ? (s[mid - 1] + s[mid]) / 2 : s[mid]
}

/**
 * Least-squares slope of y over x (x = seconds). Returns y-units per minute.
 * `n < 2` or a degenerate x spread returns 0.
 */
export function slopePerMin(xs: number[], ys: number[]): number {
  if (xs.length < 2 || xs.length !== ys.length) return 0
  const mx = mean(xs)
  const my = mean(ys)
  let num = 0
  let den = 0
  for (let i = 0; i < xs.length; i++) {
    num += (xs[i] - mx) * (ys[i] - my)
    den += (xs[i] - mx) ** 2
  }
  if (den === 0) return 0
  return (num / den) * 60
}

export interface TableRow {
  cells: string[]
}

export function printTable(headers: string[], rows: string[][]): void {
  const widths = headers.map((h, i) => Math.max(h.length, ...rows.map((r) => (r[i] ?? '').length)))
  const line = (cells: string[]): string => '  ' + cells.map((c, i) => c.padEnd(widths[i])).join('  ')
  console.log(line(headers))
  console.log('  ' + widths.map((w) => '-'.repeat(w)).join('  '))
  for (const row of rows) console.log(line(row))
}
