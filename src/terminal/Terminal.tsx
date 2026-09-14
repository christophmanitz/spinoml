import { useEffect, useRef, useState } from 'react'
import { Terminal as XTerm } from '@xterm/xterm'
import { FitAddon } from '@xterm/addon-fit'
import { WebLinksAddon } from '@xterm/addon-web-links'
import '@xterm/xterm/css/xterm.css'
import { invoke } from '@tauri-apps/api/core'
import { listen, type UnlistenFn } from '@tauri-apps/api/event'

import { useConnectionsStore, getCurrentConnection, sshTarget } from '../connections/store'
import { useWorkspaceStore } from '../workspace/store'
import { isTauri } from '../workspace/tauri-fs'

type SpawnArgs = {
  kind: 'local' | 'remote-ssh'
  local_cwd?: string
  alias?: string
  remote_root?: string
  cols?: number
  rows?: number
}

export default function Terminal() {
  const containerRef = useRef<HTMLDivElement | null>(null)
  // We re-mount when the active connection identity changes; subscribing
  // here keeps the effect dependency simple and avoids hidden coupling.
  const currentId = useConnectionsStore((s) => s.currentId)
  // Bumping restartNonce re-runs the effect → respawns the PTY. `exited` shows
  // the reconnect button. Without these a dropped connection left the terminal
  // dead with no way back short of switching connections.
  const [restartNonce, setRestartNonce] = useState(0)
  const [exited, setExited] = useState(false)
  const reconnect = () => setRestartNonce((n) => n + 1)

  useEffect(() => {
    if (!isTauri()) return
    const el = containerRef.current
    if (!el) return
    setExited(false)

    const conn = getCurrentConnection()
    const cwd = useWorkspaceStore.getState().workspaceRoot ?? undefined
    const args: SpawnArgs = conn.kind === 'remote-ssh'
      ? { kind: 'remote-ssh', alias: sshTarget(conn), remote_root: conn.root, local_cwd: cwd }
      : { kind: 'local', local_cwd: cwd }

    const term = new XTerm({
      fontFamily: 'ui-monospace, "JetBrains Mono", Menlo, Consolas, monospace',
      fontSize: 13,
      theme: {
        background: '#0a0c0f',
        foreground: '#e6e8eb',
        cursor: 'var(--accent)',
        selectionBackground: '#2a3038',
      },
      convertEol: false,
      cursorBlink: true,
      scrollback: 5000,
      allowProposedApi: true,
    })
    const fit = new FitAddon()
    term.loadAddon(fit)
    term.loadAddon(new WebLinksAddon())
    term.open(el)
    // First fit after open — element may still be 0×0 during initial layout;
    // ResizeObserver below will fire and fit again as the panel sizes itself.
    try { fit.fit() } catch { /* ignore zero-size */ }

    const initialBanner = conn.kind === 'remote-ssh'
      ? `\x1b[2mconnecting to ${sshTarget(conn)}:${conn.root} via ssh…\x1b[0m\r\n`
      : `\x1b[2mlocal shell · ${cwd ?? 'no workspace'}\x1b[0m\r\n`
    term.write(initialBanner)

    let sessionId: string | null = null
    let unlistenData: UnlistenFn | null = null
    let unlistenExit: UnlistenFn | null = null
    let disposed = false

    const spawn = async () => {
      try {
        const cols = Math.max(20, term.cols)
        const rows = Math.max(5, term.rows)
        const res = await invoke<{ id: string }>('pty_spawn', {
          args: { ...args, cols, rows },
        })
        if (disposed) {
          // Race: component unmounted before the spawn resolved. Kill child.
          invoke('pty_kill', { id: res.id }).catch(() => {})
          return
        }
        sessionId = res.id
        unlistenData = await listen<string>(`pty:${sessionId}:data`, (e) => {
          term.write(e.payload)
        })
        unlistenExit = await listen<string | null>(`pty:${sessionId}:exit`, (e) => {
          const tail = e.payload ? `\r\n${e.payload}` : ''
          term.writeln(`\r\n\x1b[33m[terminal exited]\x1b[0m${tail}`)
          term.writeln('\x1b[2mEnter drücken oder „Neu verbinden" klicken, um die Sitzung wiederherzustellen.\x1b[0m')
          sessionId = null
          setExited(true)
        })
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        term.writeln(`\x1b[31mPTY spawn failed:\x1b[0m ${msg}`)
      }
    }
    void spawn()

    const onData = term.onData((data) => {
      if (sessionId) {
        invoke('pty_write', { id: sessionId, data }).catch(() => {})
      } else if (data.includes('\r')) {
        // Session is dead — Enter respawns it (re-reads the current connection,
        // so it works again once ssh/network has recovered).
        reconnect()
      }
    })

    // Refit on container resize. Most layout changes (panel drag, window
    // resize, tab swap) flow through here. We also push the new dims down
    // to the OS PTY so child processes (vim, htop, …) re-flow.
    const ro = new ResizeObserver(() => {
      try { fit.fit() } catch { /* element may be hidden */ }
      if (sessionId) {
        const cols = Math.max(20, term.cols)
        const rows = Math.max(5, term.rows)
        invoke('pty_resize', { id: sessionId, cols, rows }).catch(() => {})
      }
    })
    ro.observe(el)

    // Re-fit when the panel becomes visible after being hidden (display:none
    // suppresses ResizeObserver in some browsers). IntersectionObserver fires
    // when the element re-enters layout.
    const io = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          try { fit.fit() } catch { /* ignore */ }
          if (sessionId) {
            invoke('pty_resize', { id: sessionId, cols: term.cols, rows: term.rows })
              .catch(() => {})
          }
          term.focus()
        }
      }
    })
    io.observe(el)

    return () => {
      disposed = true
      onData.dispose()
      ro.disconnect()
      io.disconnect()
      unlistenData?.()
      unlistenExit?.()
      if (sessionId) {
        invoke('pty_kill', { id: sessionId }).catch(() => {})
      }
      term.dispose()
    }
  }, [currentId, restartNonce])

  if (!isTauri()) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-[#6f767e]">
        Das Terminal braucht die Tauri-Desktop-App.
      </div>
    )
  }

  return (
    <div className="relative h-full w-full bg-[#0a0c0f]">
      <div ref={containerRef} className="h-full w-full" />
      {exited && (
        <button
          className="absolute right-2 top-2 rounded border border-[#3a4148] bg-[#13171b]/90 px-2 py-1 text-xs text-[#e6e8eb] shadow hover:bg-[#1a1f24]"
          onClick={reconnect}
          title="PTY-Sitzung neu starten"
        >↻ Neu verbinden</button>
      )}
    </div>
  )
}
