// Shared status pill for training runs.

const STYLES: Record<string, string> = {
  queued: 'bg-[#2a2f36] text-[#9aa1a8]',
  running: 'bg-[#13344f] text-[#6ab7ff]',
  done: 'bg-[#143d2a] text-[#5fd39a]',
  failed: 'bg-[#42191c] text-[#ff7a85]',
  cancelled: 'bg-[#3d3414] text-[#e6c34a]',
  unknown: 'bg-[#2a2f36] text-[#7a8088]',
}

export default function StatusPill({ status, alive }: { status: string; alive?: boolean }) {
  const cls = STYLES[status] ?? STYLES.unknown
  const pulse = status === 'running' && alive
  return (
    <span className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium ${cls}`}>
      {pulse && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />}
      {status}
    </span>
  )
}
