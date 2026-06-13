import { useState } from 'react'
import { useProjectStore, type ProjectStatus } from './store'
import { isTauri } from '../workspace/tauri-fs'

export default function Welcome() {
  const status = useProjectStore((s) => s.status)
  const pickFolder = useProjectStore((s) => s.pickFolder)
  const init = useProjectStore((s) => s.init)
  const migrate = useProjectStore((s) => s.migrate)
  const refresh = useProjectStore((s) => s.refresh)
  const closeProject = useProjectStore((s) => s.closeProject)

  const [showCreateFor, setShowCreateFor] = useState<null | 'new' | 'migrate'>(null)

  if (!isTauri()) {
    return (
      <FullScreen>
        <Card title="Browser-Modus" subtitle="MLForge läuft hier ohne echten Workspace.">
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
        </Card>
      </FullScreen>
    )
  }

  if (status.kind === 'legacy') {
    const hasFiles = status.legacy_mlforge_count > 0
    return (
      <FullScreen>
        <Card
          title={hasFiles ? 'Ordner mit losen .mlforge-Dateien' : 'Leerer Ordner'}
          subtitle={status.root}
        >
          {hasFiles ? (
            <p className="mb-3 text-sm leading-relaxed text-[#9aa1a8]">
              Dieser Ordner enthält {status.legacy_mlforge_count}{' '}
              <code className="rounded bg-[#1f2429] px-1 py-0.5">.mlforge</code>-Datei
              {status.legacy_mlforge_count === 1 ? '' : 'en'}, aber noch keine
              Projekt-Struktur. Konvertiere ihn in ein MLForge-Projekt — die Dateien
              landen in <code className="rounded bg-[#1f2429] px-1 py-0.5">models/</code>,
              und du bekommst <code>datasets/</code>, <code>notes/</code> und{' '}
              <code>experiments/</code> dazu.
            </p>
          ) : (
            <p className="mb-3 text-sm leading-relaxed text-[#9aa1a8]">
              Initialisiere hier ein neues MLForge-Projekt — das legt
              <code className="mx-1 rounded bg-[#1f2429] px-1 py-0.5">mlforge.project.json</code>
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

  // status.kind === 'none'
  return (
    <FullScreen>
      <Card
        title="MLForge"
        subtitle="Drag-and-drop PyTorch-Architektur, mit Claude an deiner Seite."
      >
        <p className="mb-4 text-sm leading-relaxed text-[#9aa1a8]">
          Ein <strong>MLForge-Projekt</strong> ist ein Ordner mit deinen Modellen
          (<code>models/</code>), Datensätzen (<code>datasets/</code>),
          Notizen (<code>notes/</code>) und Experiment-Logs (<code>experiments/</code>).
          Claude liest den Projektkontext bei jedem Chat, damit er fokussiert
          mitarbeiten kann.
        </p>
        <div className="flex gap-2">
          <Button onClick={() => void pickFolder()} primary>Projekt öffnen / Ordner wählen…</Button>
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
  const defaultName = (status.kind === 'legacy' || status.kind === 'loaded')
    ? status.root.split(/[\\/]/).filter(Boolean).pop() ?? ''
    : ''
  const [name, setName] = useState(defaultName)
  const [description, setDescription] = useState('')
  const [goal, setGoal] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  return (
    <FullScreen>
      <Card
        title={mode === 'new' ? 'Neues Projekt' : 'Bestehende Dateien in Projekt konvertieren'}
        subtitle={status.kind === 'legacy' ? status.root : undefined}
      >
        <div className="space-y-3">
          <Field label="Name" hint="Kurz, prägnant — taucht in der Topbar auf.">
            <input
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full rounded border border-[#2a3038] bg-[#0e1115] px-2 py-1 text-sm text-[#e6e8eb] focus:border-[#6ab7ff] focus:outline-none"
            />
          </Field>
          <Field label="Beschreibung" hint="Was ist das hier? 1–2 Sätze.">
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={2}
              className="w-full resize-none rounded border border-[#2a3038] bg-[#0e1115] px-2 py-1 text-sm text-[#e6e8eb] focus:border-[#6ab7ff] focus:outline-none"
            />
          </Field>
          <Field label="Ziel" hint="Was willst du erreichen? Claude liest das mit.">
            <textarea
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              rows={3}
              placeholder="z.B. CNN-Baseline für CIFAR-10, dann mit Dropout/BN vergleichen."
              className="w-full resize-none rounded border border-[#2a3038] bg-[#0e1115] px-2 py-1 text-sm text-[#e6e8eb] focus:border-[#6ab7ff] focus:outline-none"
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
    <div className="flex h-full w-full items-center justify-center bg-[#0b0d10] p-8">
      {children}
    </div>
  )
}

function Card({ title, subtitle, children }: { title: string; subtitle?: string; children?: React.ReactNode }) {
  return (
    <div className="w-full max-w-lg rounded-lg border border-[#1f2429] bg-[#0e1115] p-6 shadow-xl">
      <h1 className="text-xl font-semibold text-[#e6e8eb]">{title}</h1>
      {subtitle && <div className="mt-1 truncate text-xs text-[#7a8088]">{subtitle}</div>}
      <div className="mt-4">{children}</div>
    </div>
  )
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="mb-1 block text-xs uppercase tracking-wider text-[#7a8088]">{label}</label>
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
    ? 'bg-[#6ab7ff] text-[#0b0d10] hover:bg-[#8cc7ff]'
    : 'border border-[#2a3038] bg-[#1a1e22] text-[#e6e8eb] hover:border-[#6ab7ff]'
  return <button onClick={onClick} disabled={disabled} className={`${base} ${cls}`}>{children}</button>
}
