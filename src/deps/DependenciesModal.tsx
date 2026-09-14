import { useEffect, useState } from 'react'
import { fs } from '../connections/backend'
import { isTauri } from '../workspace/tauri-fs'
import { confirmDialog } from '../ui/confirm'
import {
  checkDeps, installDeps, parseRequirements,
  type DepsCheckResult, type DepsInstallResult,
} from './client'

const REQ_FILE = 'requirements.txt'

// Per-project Python dependencies. Edited as a requirements.txt the project
// owns (portable: `pip install -r requirements.txt` works on the HPC too).
// "Kompatibilität prüfen" is a smoke test — it resolves the specs against the
// sidecar's env via pip --dry-run WITHOUT installing.
export default function DependenciesModal({ onClose }: { onClose: () => void }) {
  const [text, setText] = useState('')
  const [loaded, setLoaded] = useState(false)
  const [loadNote, setLoadNote] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [savedAt, setSavedAt] = useState<number | null>(null)
  const [busy, setBusy] = useState<null | 'check' | 'install'>(null)
  const [check, setCheck] = useState<DepsCheckResult | null>(null)
  const [install, setInstall] = useState<DepsInstallResult | null>(null)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  useEffect(() => {
    let alive = true
    void (async () => {
      if (!isTauri()) { setLoadNote('Speichern braucht einen Workspace-Ordner (Tauri-App).'); setLoaded(true); return }
      try {
        const body = await fs.read(REQ_FILE)
        if (alive) setText(body)
      } catch {
        // No requirements.txt yet — start empty.
        if (alive) setLoadNote(`Kein ${REQ_FILE} im Projekt — wird beim Speichern angelegt.`)
      } finally {
        if (alive) setLoaded(true)
      }
    })()
    return () => { alive = false }
  }, [])

  const specs = parseRequirements(text)

  const runCheck = async () => {
    setBusy('check'); setInstall(null)
    setCheck(await checkDeps(specs))
    setBusy(null)
  }

  const save = async () => {
    if (!isTauri()) return
    setSaving(true)
    try {
      await fs.write(REQ_FILE, text.endsWith('\n') ? text : text + '\n')
      setSavedAt(Date.now())
    } catch (e) {
      setLoadNote(`Speichern fehlgeschlagen: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setSaving(false)
    }
  }

  const runInstall = async () => {
    if (!specs.length) return
    if (!(await confirmDialog(
      `${specs.length} Paket(e) jetzt in die Sidecar-Python-Umgebung installieren?\n\n${specs.join('\n')}`,
    ))) return
    setBusy('install'); setInstall(null)
    const res = await installDeps(specs)
    setInstall(res)
    setBusy(null)
    // Refresh installed versions / compatibility after a successful install.
    if (res.ok) setCheck(await checkDeps(specs))
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6" onClick={onClose}>
      <div
        className="flex h-[80vh] w-[min(820px,92vw)] flex-col overflow-hidden rounded border border-[#1f2429] bg-[#0e1216] shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-center justify-between border-b border-[#1f2429] px-3 py-2">
          <div className="flex items-baseline gap-2">
            <span className="text-sm text-[#e6e8eb]">Projekt-Dependencies</span>
            <span className="text-[10px] text-[#6f767e]">{REQ_FILE}</span>
          </div>
          <button
            className="rounded border border-[#1f2429] bg-[#13171b] px-2 py-1 text-xs hover:bg-[#1a1f24]"
            onClick={onClose}
            title="Esc"
          >✕</button>
        </div>

        <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto p-3 text-xs">
          <div className="text-[11px] text-[#9aa1a8]">
            Eine Zeile pro Paket, PEP-440-Specs erlaubt (z.&nbsp;B. <code className="text-[#cbd2d9]">torch_geometric==2.8.0</code>,
            {' '}<code className="text-[#cbd2d9]">rdkit&gt;=2024.3</code>). „Kompatibilität prüfen" löst die Versionen
            via <code className="text-[#cbd2d9]">pip --dry-run</code> auf, <strong>ohne</strong> zu installieren.
          </div>

          <textarea
            className="h-40 w-full resize-y rounded border border-[#1f2429] bg-[#0b0e11] p-2 font-mono text-[12px] text-[#e6e8eb] outline-none focus:border-[#3a4148]"
            placeholder={'# eine Zeile pro Paket\ntorch_geometric==2.8.0\nrdkit>=2024.3'}
            value={text}
            onChange={(e) => { setText(e.target.value); setSavedAt(null) }}
            disabled={!loaded}
            spellCheck={false}
          />

          <div className="flex flex-wrap items-center gap-2">
            <button
              className="rounded bg-[#1d4ed8]/80 px-2.5 py-1 text-[#e6e8eb] hover:bg-[#1d4ed8] disabled:opacity-50"
              onClick={runCheck}
              disabled={!!busy || specs.length === 0}
            >{busy === 'check' ? 'prüfe…' : 'Kompatibilität prüfen'}</button>
            <button
              className="rounded border border-[#1f2429] bg-[#13171b] px-2.5 py-1 hover:bg-[#1a1f24] disabled:opacity-50"
              onClick={save}
              disabled={!isTauri() || saving || !loaded}
              title={isTauri() ? `nach ${REQ_FILE} schreiben` : 'nur in der Tauri-App'}
            >{saving ? 'speichere…' : savedAt ? 'gespeichert ✓' : 'Speichern'}</button>
            <button
              className="rounded border border-amber-800/70 bg-amber-900/30 px-2.5 py-1 text-amber-100 hover:bg-amber-900/50 disabled:opacity-50"
              onClick={runInstall}
              disabled={!!busy || specs.length === 0}
              title="pip install in die Sidecar-Umgebung"
            >{busy === 'install' ? 'installiere…' : 'Installieren'}</button>
            <span className="ml-auto text-[10px] text-[#6f767e]">{specs.length} Paket(e)</span>
          </div>

          {loadNote && <div className="text-[10px] text-[#6f767e]">{loadNote}</div>}

          {check && <CheckResultView res={check} />}
          {install && <InstallResultView res={install} />}
        </div>
      </div>
    </div>
  )
}

function CheckResultView({ res }: { res: DepsCheckResult }) {
  if (!res.ok) {
    return (
      <div className="rounded border border-rose-900/60 bg-rose-950/40 px-2 py-1.5 text-[11px] text-rose-200">
        Sidecar nicht erreichbar: {res.error}. Läuft der Torch-Sidecar (Port 7421)?
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-2 rounded border border-[#1f2429] bg-[#0b0e11] p-2">
      <div className="flex items-center gap-2">
        <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
          res.compatible ? 'bg-emerald-900/50 text-emerald-200' : 'bg-rose-900/50 text-rose-200'
        }`}>
          {res.compatible ? '✓ kompatibel' : '✗ Konflikt'}
        </span>
        <span className="text-[10px] text-[#6f767e]">Python {res.python}</span>
        {res.compatible && res.log && <span className="text-[10px] text-[#6f767e]">· {res.log}</span>}
      </div>

      {res.requested.length > 0 && (
        <div className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-0.5 font-mono text-[10px]">
          <span className="text-[#6f767e]">Spec</span>
          <span className="text-[#6f767e]">installiert</span>
          {res.requested.map((r) => (
            <Row key={r.spec} left={r.spec} right={r.installed ?? '—'} dim={!r.installed} />
          ))}
        </div>
      )}

      {res.compatible && res.would_install.length > 0 && (
        <div className="text-[10px] text-[#9aa1a8]">
          <div className="mb-0.5 text-[#6f767e]">pip würde installieren/aktualisieren:</div>
          <div className="font-mono text-[#cbd2d9]">{res.would_install.join(', ')}</div>
        </div>
      )}

      {!res.compatible && res.error && (
        <pre className="max-h-40 overflow-auto whitespace-pre-wrap rounded bg-[#140b0d] p-2 font-mono text-[10px] text-rose-200/90">{res.error}</pre>
      )}
    </div>
  )
}

function Row({ left, right, dim }: { left: string; right: string; dim?: boolean }) {
  return (
    <>
      <span className="text-[#cbd2d9]">{left}</span>
      <span className={dim ? 'text-[#5b6168]' : 'text-emerald-300'}>{right}</span>
    </>
  )
}

function InstallResultView({ res }: { res: DepsInstallResult }) {
  return (
    <div className="flex flex-col gap-1">
      <div className={`text-[11px] ${res.ok ? 'text-emerald-300' : 'text-rose-300'}`}>
        {res.ok ? 'Installation erfolgreich ✓' : `Installation fehlgeschlagen${'returncode' in res && res.returncode != null ? ` (exit ${res.returncode})` : ''}`}
        {!res.ok && 'error' in res && res.error ? ` — ${res.error}` : ''}
      </div>
      {'log' in res && res.log && (
        <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded bg-[#0b0e11] p-2 font-mono text-[10px] text-[#9aa1a8]">{res.log}</pre>
      )}
    </div>
  )
}
