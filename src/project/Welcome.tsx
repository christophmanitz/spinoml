import { useState } from 'react'
import { useProjectStore, type ProjectStatus } from './store'
import { isTauri } from '../workspace/tauri-fs'
import { confirmDialog } from '../ui/confirm'
import {
  useConnectionsStore,
  type RemoteSshConnection,
} from '../connections/store'
import type { SshTestResult } from '../connections/tauri-ssh'
import { getRecentWorkspaces, removeRecentWorkspace } from '../workspace/recentWorkspaces'

type View = 'main' | 'remote-picker' | 'remote-form'

export default function Welcome() {
  const status = useProjectStore((s) => s.status)
  const pickFolder = useProjectStore((s) => s.pickFolder)
  const init = useProjectStore((s) => s.init)
  const migrate = useProjectStore((s) => s.migrate)
  const refresh = useProjectStore((s) => s.refresh)
  const closeProject = useProjectStore((s) => s.closeProject)
  const openConnection = useProjectStore((s) => s.openConnection)
  const openLocalPath = useProjectStore((s) => s.openLocalPath)

  const [recents, setRecents] = useState(() => getRecentWorkspaces())

  const saved = useConnectionsStore((s) => s.saved)
  const addRemote = useConnectionsStore((s) => s.addRemote)
  const updateRemote = useConnectionsStore((s) => s.updateRemote)
  const removeRemote = useConnectionsStore((s) => s.removeRemote)
  const testConnection = useConnectionsStore((s) => s.testConnection)

  const [view, setView] = useState<View>('main')
  const [showCreateFor, setShowCreateFor] = useState<null | 'new' | 'migrate'>(null)
  // When set, the remote form opens in edit mode for this connection.
  const [editing, setEditing] = useState<RemoteSshConnection | null>(null)

  if (!isTauri()) {
    return (
      <FullScreen>
        <Card title="Browser-Modus" subtitle="SpinoML läuft hier ohne echten Workspace.">
          <p className="text-sm leading-relaxed text-[#9aa1a8]">
            Projekte brauchen die Tauri-Desktop-App. Starte sie mit{' '}
            <code className="rounded bg-[#1f2429] px-1 py-0.5">npm run tauri dev</code>, oder
            arbeite hier weiter mit der lokalen Speicherung (kein <code>datasets/</code>,
            keine <code>notes/</code>).
          </p>
        </Card>
      </FullScreen>
    )
  }

  if (showCreateFor) {
    return (
      <CreateProjectForm
        mode={showCreateFor}
        onSubmit={async (name, description, goal) => {
          if (showCreateFor === 'new') await init(name, description, goal)
          else await migrate(name, description, goal)
          setShowCreateFor(null)
        }}
        onCancel={() => setShowCreateFor(null)}
        status={status}
      />
    )
  }

  if (status.kind === 'loading') {
    return <FullScreen><Card title="loading…" /></FullScreen>
  }

  if (status.kind === 'error') {
    return (
      <FullScreen>
        <Card title="Fehler beim Laden des Projekts">
          <p className="mb-3 text-sm text-rose-300">{status.error}</p>
          <Button onClick={() => void refresh()}>Erneut versuchen</Button>
          <span className="mx-2 text-[#5a6068]">·</span>
          <Button onClick={() => void closeProject()}>Anderen Ordner wählen</Button>
        </Card>
      </FullScreen>
    )
  }

  if (status.kind === 'remote-missing') {
    return (
      <FullScreen>
        <Card
          title="Leerer / fehlender Remote-Pfad"
          subtitle={`${status.alias}:${status.root}`}
        >
          <p className="mb-3 text-sm leading-relaxed text-[#9aa1a8]">
            Auf <code className="rounded bg-[#1f2429] px-1 py-0.5">{status.alias}</code>{' '}
            existiert unter <code className="rounded bg-[#1f2429] px-1 py-0.5">{status.root}</code>{' '}
            noch kein SpinoML-Projekt. Initialisiere eins — das legt
            <code className="mx-1 rounded bg-[#1f2429] px-1 py-0.5">spinoml.project.json</code>
            + die Standardordner (<code>models/</code>, <code>datasets/</code>,{' '}
            <code>notes/</code>, <code>experiments/</code>) auf dem Remote an.
          </p>
          <div className="flex gap-2">
            <Button primary onClick={() => setShowCreateFor('new')}>
              Projekt auf {status.alias} initialisieren
            </Button>
            <Button onClick={() => void closeProject()}>Andere Verbindung wählen</Button>
          </div>
        </Card>
      </FullScreen>
    )
  }

  if (status.kind === 'legacy') {
    const hasFiles = status.legacy_spinoml_count > 0
    return (
      <FullScreen>
        <Card
          title={hasFiles ? 'Ordner mit losen .spinoml-Dateien' : 'Leerer Ordner'}
          subtitle={status.root}
        >
          {hasFiles ? (
            <p className="mb-3 text-sm leading-relaxed text-[#9aa1a8]">
              Dieser Ordner enthält {status.legacy_spinoml_count}{' '}
              <code className="rounded bg-[#1f2429] px-1 py-0.5">.spinoml</code>-Datei
              {status.legacy_spinoml_count === 1 ? '' : 'en'}, aber noch keine
              Projekt-Struktur. Konvertiere ihn in ein SpinoML-Projekt — die Dateien
              landen in <code className="rounded bg-[#1f2429] px-1 py-0.5">models/</code>,
              und du bekommst <code>datasets/</code>, <code>notes/</code> und{' '}
              <code>experiments/</code> dazu.
            </p>
          ) : (
            <p className="mb-3 text-sm leading-relaxed text-[#9aa1a8]">
              Initialisiere hier ein neues SpinoML-Projekt — das legt
              <code className="mx-1 rounded bg-[#1f2429] px-1 py-0.5">spinoml.project.json</code>
              + die Standardordner an.
            </p>
          )}
          <div className="flex gap-2">
            <Button onClick={() => setShowCreateFor(hasFiles ? 'migrate' : 'new')} primary>
              {hasFiles ? 'In Projekt konvertieren' : 'Projekt initialisieren'}
            </Button>
            <Button onClick={() => void closeProject()}>Anderen Ordner wählen</Button>
          </div>
        </Card>
      </FullScreen>
    )
  }

  // status.kind === 'none' — pick a backend.

  if (view === 'remote-form') {
    return (
      <RemoteConnectionForm
        initial={editing}
        onTest={testConnection}
        onSave={(label, alias, root, user) => {
          const c = addRemote(label, alias, root, user)
          return c
        }}
        onUpdate={(id, patch) => updateRemote(id, patch)}
        onOpen={async (c) => {
          await openConnection(c.id)
          setView('main')
        }}
        onCancel={() => {
          setEditing(null)
          setView('remote-picker')
        }}
      />
    )
  }

  if (view === 'remote-picker') {
    return (
      <FullScreen>
        <Card
          title="Remote-Workspace (SSH)"
          subtitle="Arbeite auf einem anderen Server — z. B. dem HPC-Cluster."
        >
          {saved.length === 0 ? (
            <p className="mb-3 text-sm text-[#9aa1a8]">
              Noch keine Verbindungen gespeichert. Verbindungen nutzen deine
              vorhandene <code className="rounded bg-[#1f2429] px-1 py-0.5">~/.ssh/config</code>{' '}
              + ssh-agent — SpinoML speichert keine Passwörter oder Keys.
            </p>
          ) : (
            <div className="mb-3 space-y-1.5">
              {saved.map((c) => (
                <div
                  key={c.id}
                  className="flex items-center gap-2 rounded border border-[#2a3038] bg-[#1a1e22] px-2 py-1.5"
                >
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm text-[#e6e8eb]">{c.label}</div>
                    <div className="truncate text-[10px] text-[#6f767e]">
                      {c.user ? `${c.user}@${c.alias}` : c.alias}:<span className="text-[#9aa1a8]">{c.root}</span>
                    </div>
                  </div>
                  <Button
                    primary
                    onClick={async () => {
                      await openConnection(c.id)
                      setView('main')
                    }}
                  >
                    Öffnen
                  </Button>
                  <button
                    onClick={() => {
                      setEditing(c)
                      setView('remote-form')
                    }}
                    title="Verbindung bearbeiten"
                    className="rounded px-1.5 py-1 text-[#6f767e] hover:bg-[#2a3038] hover:text-[var(--accent)]"
                  >
                    ✎
                  </button>
                  <button
                    onClick={async () => {
                      if (await confirmDialog(`Verbindung „${c.label}" entfernen?`)) removeRemote(c.id)
                    }}
                    title="Verbindung löschen"
                    className="rounded px-1.5 py-1 text-[#6f767e] hover:bg-[#2a3038] hover:text-rose-300"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}
          <div className="flex gap-2">
            <Button primary onClick={() => { setEditing(null); setView('remote-form') }}>
              Neue Verbindung
            </Button>
            <Button onClick={() => setView('main')}>Zurück</Button>
          </div>
          <p className="mt-4 text-[10px] leading-relaxed text-[#5a6068]">
            <strong>Voraussetzung:</strong> Der SSH-Alias muss in deiner
            <code className="mx-1 rounded bg-[#1f2429] px-1 py-0.5">~/.ssh/config</code>
            stehen und ohne Passwort-Prompt erreichbar sein (Key + Agent, GSSAPI,
            ControlMaster, etc.). SpinoML ruft systemweites <code>ssh</code> auf.
          </p>
        </Card>
      </FullScreen>
    )
  }

  // view === 'main'
  return (
    <FullScreen>
      <Card
        title="SpinoML"
        subtitle="Drag-and-drop PyTorch-Architektur, mit einem KI-Assistenten an deiner Seite."
      >
        <p className="mb-4 text-sm leading-relaxed text-[#9aa1a8]">
          Ein <strong>SpinoML-Projekt</strong> ist ein Ordner mit deinen Modellen
          (<code>models/</code>), Datensätzen (<code>datasets/</code>),
          Notizen (<code>notes/</code>) und Experiment-Logs (<code>experiments/</code>).
          Der Assistent liest den Projektkontext bei jedem Chat, damit er fokussiert
          mitarbeiten kann.
        </p>
        <div className="flex flex-wrap gap-2">
          <Button onClick={() => void pickFolder()} primary>
            Lokaler Ordner…
          </Button>
          <Button onClick={() => setView('remote-picker')}>
            Remote-Workspace (SSH)
          </Button>
        </div>

        {recents.length > 0 && (
          <div className="mt-4">
            <div className="mb-1.5 text-[10px] uppercase tracking-wider text-[#6f767e]">Zuletzt geöffnet</div>
            <div className="space-y-1.5">
              {recents.map((r) => (
                <div
                  key={r.path}
                  className="flex items-center gap-2 rounded border border-[#2a3038] bg-[#1a1e22] px-2 py-1.5"
                >
                  <button
                    className="min-w-0 flex-1 text-left"
                    onClick={() => void openLocalPath(r.path)}
                    title={`Öffnen: ${r.path}`}
                  >
                    <div className="truncate text-sm text-[#e6e8eb]">{r.name}</div>
                    <div className="truncate text-[10px] text-[#6f767e]">{r.path}</div>
                  </button>
                  <button
                    onClick={() => { removeRecentWorkspace(r.path); setRecents(getRecentWorkspaces()) }}
                    title="Aus der Liste entfernen"
                    className="rounded px-1.5 py-1 text-[#6f767e] hover:bg-[#2a3038] hover:text-rose-300"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
        {saved.length > 0 && (
          <p className="mt-3 text-[11px] text-[#6f767e]">
            {saved.length} Remote-Verbindung{saved.length === 1 ? '' : 'en'} gespeichert —{' '}
            <button
              className="underline hover:text-[var(--accent)]"
              onClick={() => setView('remote-picker')}
            >
              auswählen
            </button>
          </p>
        )}
      </Card>
    </FullScreen>
  )
}

function RemoteConnectionForm({
  initial,
  onTest,
  onSave,
  onUpdate,
  onOpen,
  onCancel,
}: {
  initial?: RemoteSshConnection | null
  onTest: (target: string) => Promise<SshTestResult>
  onSave: (label: string, alias: string, root: string, user?: string) => RemoteSshConnection
  onUpdate: (id: string, patch: Partial<RemoteSshConnection>) => void
  onOpen: (c: RemoteSshConnection) => Promise<void>
  onCancel: () => void
}) {
  const editMode = !!initial
  const [label, setLabel] = useState(initial?.label ?? '')
  const [alias, setAlias] = useState(initial?.alias ?? '')
  const [user, setUser] = useState(initial?.user ?? '')
  const [root, setRoot] = useState(initial?.root ?? '~/spinoml')
  const [python, setPython] = useState(initial?.python ?? '')
  const [testing, setTesting] = useState(false)
  const [busy, setBusy] = useState(false)
  const [testResult, setTestResult] = useState<SshTestResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const aliasRe = /^[a-zA-Z0-9._-]+$/  // bare alias: no @, no :
  const userRe = /^[a-zA-Z0-9._-]+$/   // POSIX username chars
  const validAlias = aliasRe.test(alias)
  const validUser = user === '' || userRe.test(user)
  const validRoot = root.startsWith('/') || root.startsWith('~/') || root === '~'
  const composed = user.trim() ? `${user.trim()}@${alias.trim()}` : alias.trim()

  return (
    <FullScreen>
      <Card title={editMode ? 'SSH-Verbindung bearbeiten' : 'Neue SSH-Verbindung'}>
        <div className="space-y-3">
          <Field label="Label" hint="Wie soll das in der Liste auftauchen?">
            <input
              autoFocus
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="z.B. Leipzig HPC"
              className="w-full rounded border border-[#2a3038] bg-[#0e1216] px-2 py-1 text-sm text-[#e6e8eb] focus:border-[var(--accent)] focus:outline-none"
            />
          </Field>
          <Field
            label="SSH-Alias / Host"
            hint="Eintrag aus ~/.ssh/config (z.B. leipzig-hpc) ODER nackter Hostname (login01.sc.uni-leipzig.de)."
          >
            <input
              value={alias}
              onChange={(e) => {
                setAlias(e.target.value)
                setTestResult(null)
              }}
              placeholder="leipzig-hpc"
              className={`w-full rounded border bg-[#0e1216] px-2 py-1 text-sm text-[#e6e8eb] focus:outline-none ${
                alias && !validAlias ? 'border-rose-500' : 'border-[#2a3038] focus:border-[var(--accent)]'
              }`}
            />
          </Field>
          <Field
            label="User (optional)"
            hint="Leer lassen wenn ~/.ssh/config schon den User definiert. Sonst hier eintragen."
          >
            <input
              value={user}
              onChange={(e) => {
                setUser(e.target.value)
                setTestResult(null)
              }}
              placeholder="zw93onug"
              className={`w-full rounded border bg-[#0e1216] px-2 py-1 text-sm text-[#e6e8eb] focus:outline-none ${
                user && !validUser ? 'border-rose-500' : 'border-[#2a3038] focus:border-[var(--accent)]'
              }`}
            />
          </Field>
          <Field
            label="Remote-Pfad"
            hint="Absolut (/scratch/me/spinoml) oder Home-relativ (~/projects/spinoml)."
          >
            <input
              value={root}
              onChange={(e) => setRoot(e.target.value)}
              placeholder="~/spinoml"
              className={`w-full rounded border bg-[#0e1216] px-2 py-1 text-sm text-[#e6e8eb] focus:outline-none ${
                root && !validRoot ? 'border-rose-500' : 'border-[#2a3038] focus:border-[var(--accent)]'
              }`}
            />
          </Field>
          <Field
            label="Python (optional)"
            hint="Interpreter mit torch (+ pandas) für Trainings-Runs. Leer = <root>/.spinoml/venv/bin/python."
          >
            <input
              value={python}
              onChange={(e) => setPython(e.target.value)}
              placeholder="~/spinoml/.spinoml/venv/bin/python"
              className="w-full rounded border border-[#2a3038] bg-[#0e1216] px-2 py-1 text-sm text-[#e6e8eb] focus:border-[var(--accent)] focus:outline-none"
            />
          </Field>

          {validAlias && (
            <div className="rounded bg-[#1a1e22] px-2 py-1 text-[10px] text-[#6f767e]">
              SSH-Ziel: <code className="text-[#9aa1a8]">{composed}</code>
            </div>
          )}
          {testResult && (
            <div className="rounded border border-emerald-700/40 bg-emerald-900/15 px-2 py-1.5 text-xs text-emerald-300">
              <div className="font-medium">Verbindung OK</div>
              <div className="text-emerald-400/80">{testResult.uname}</div>
              <div className="text-emerald-400/60">HOME = {testResult.home}</div>
            </div>
          )}
          {error && (
            <div className="rounded bg-rose-900/20 px-2 py-1.5 text-xs text-rose-300 whitespace-pre-wrap">
              {error}
            </div>
          )}

          <div className="flex flex-wrap gap-2">
            <Button
              disabled={!validAlias || !validUser || testing}
              onClick={async () => {
                setTesting(true)
                setError(null)
                setTestResult(null)
                try {
                  const r = await onTest(composed)
                  setTestResult(r)
                } catch (e) {
                  setError(e instanceof Error ? e.message : String(e))
                } finally {
                  setTesting(false)
                }
              }}
            >
              {testing ? 'Teste…' : 'Verbindung testen'}
            </Button>
            {editMode ? (
              <Button
                primary
                disabled={!validAlias || !validUser || !validRoot || busy || !label.trim() || !initial}
                onClick={() => {
                  setError(null)
                  try {
                    if (!initial) return
                    onUpdate(initial.id, {
                      label: label.trim(),
                      alias: alias.trim(),
                      user: user.trim() || undefined,
                      root: root.trim(),
                      python: python.trim() || undefined,
                    })
                    onCancel()
                  } catch (e) {
                    setError(e instanceof Error ? e.message : String(e))
                  }
                }}
              >
                Speichern
              </Button>
            ) : (
              <Button
                primary
                disabled={!validAlias || !validUser || !validRoot || busy || !label.trim()}
                onClick={async () => {
                  setBusy(true)
                  setError(null)
                  try {
                    const c = onSave(label.trim(), alias.trim(), root.trim(), user.trim() || undefined)
                    if (python.trim()) onUpdate(c.id, { python: python.trim() })
                    await onOpen(c)
                  } catch (e) {
                    setError(e instanceof Error ? e.message : String(e))
                  } finally {
                    setBusy(false)
                  }
                }}
              >
                {busy ? 'Öffne…' : 'Anlegen & öffnen'}
              </Button>
            )}
            <Button onClick={onCancel} disabled={busy}>Abbrechen</Button>
          </div>
        </div>
      </Card>
    </FullScreen>
  )
}

function CreateProjectForm({
  mode, onSubmit, onCancel, status,
}: {
  mode: 'new' | 'migrate'
  onSubmit: (name: string, description: string, goal: string) => Promise<void>
  onCancel: () => void
  status: ProjectStatus
}) {
  const defaultName =
    (status.kind === 'legacy' || status.kind === 'loaded' || status.kind === 'remote-missing')
      ? status.root.split(/[\\/]/).filter(Boolean).pop() ?? ''
      : ''
  const [name, setName] = useState(defaultName)
  const [description, setDescription] = useState('')
  const [goal, setGoal] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const subtitle =
    status.kind === 'remote-missing' ? `${status.alias}:${status.root}` :
    status.kind === 'legacy' ? status.root :
    undefined

  return (
    <FullScreen>
      <Card
        title={mode === 'new' ? 'Neues Projekt' : 'Bestehende Dateien in Projekt konvertieren'}
        subtitle={subtitle}
      >
        <div className="space-y-3">
          <Field label="Name" hint="Kurz, prägnant — taucht in der Topbar auf.">
            <input
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full rounded border border-[#2a3038] bg-[#0e1216] px-2 py-1 text-sm text-[#e6e8eb] focus:border-[var(--accent)] focus:outline-none"
            />
          </Field>
          <Field label="Beschreibung" hint="Was ist das hier? 1–2 Sätze.">
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={2}
              className="w-full resize-none rounded border border-[#2a3038] bg-[#0e1216] px-2 py-1 text-sm text-[#e6e8eb] focus:border-[var(--accent)] focus:outline-none"
            />
          </Field>
          <Field label="Ziel" hint="Was willst du erreichen? Der Assistent liest das mit.">
            <textarea
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              rows={3}
              placeholder="z.B. CNN-Baseline für CIFAR-10, dann mit Dropout/BN vergleichen."
              className="w-full resize-none rounded border border-[#2a3038] bg-[#0e1216] px-2 py-1 text-sm text-[#e6e8eb] focus:border-[var(--accent)] focus:outline-none"
            />
          </Field>
          {error && <div className="rounded bg-rose-900/20 px-2 py-1.5 text-xs text-rose-300">{error}</div>}
          <div className="flex gap-2">
            <Button
              primary
              disabled={busy || !name.trim()}
              onClick={async () => {
                setBusy(true); setError(null)
                try { await onSubmit(name.trim(), description.trim(), goal.trim()) }
                catch (e) { setError(e instanceof Error ? e.message : String(e)) }
                finally { setBusy(false) }
              }}
            >
              {mode === 'new' ? 'Projekt anlegen' : 'Konvertieren'}
            </Button>
            <Button onClick={onCancel} disabled={busy}>Abbrechen</Button>
          </div>
        </div>
      </Card>
    </FullScreen>
  )
}

function FullScreen({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-full w-full items-center justify-center bg-[#0a0c0f] p-8">
      {children}
    </div>
  )
}

function Card({ title, subtitle, children }: { title: string; subtitle?: string; children?: React.ReactNode }) {
  return (
    <div className="w-full max-w-lg rounded-lg border border-[#1f2429] bg-[#0e1216] p-6 shadow-xl">
      <h1 className="text-xl font-semibold text-[#e6e8eb]">{title}</h1>
      {subtitle && <div className="mt-1 truncate text-xs text-[#6f767e]">{subtitle}</div>}
      <div className="mt-4">{children}</div>
    </div>
  )
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="mb-1 block text-xs uppercase tracking-wider text-[#6f767e]">{label}</label>
      {children}
      {hint && <div className="mt-0.5 text-[10px] text-[#5a6068]">{hint}</div>}
    </div>
  )
}

function Button({
  children, onClick, primary, disabled,
}: {
  children: React.ReactNode
  onClick?: () => void
  primary?: boolean
  disabled?: boolean
}) {
  const base = 'rounded px-3 py-1.5 text-sm transition-colors disabled:opacity-40 disabled:cursor-not-allowed'
  const cls = primary
    ? 'bg-[var(--accent)] text-[#0a0c0f] hover:bg-[#8cc7ff]'
    : 'border border-[#2a3038] bg-[#1a1e22] text-[#e6e8eb] hover:border-[var(--accent)]'
  return <button onClick={onClick} disabled={disabled} className={`${base} ${cls}`}>{children}</button>
}
