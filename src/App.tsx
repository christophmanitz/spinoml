import { useEffect, useState } from 'react'
import { Group, Panel, Separator, useDefaultLayout } from 'react-resizable-panels'
import Palette from './palette/Palette'
import LeftSidebar from './workspace/LeftSidebar'
import Canvas from './canvas/Canvas'
import Inspector from './inspector/Inspector'
import ChatPanel from './chat/ChatPanel'
import CodePreview from './codegen/CodePreview'
import Terminal from './terminal/Terminal'
import { useInferenceStore } from './inference/store'
import { sidecarHealthState } from './inference/client'
import { useChatStore } from './chat/store'
import { providerById, useProviderStore } from './chat/providerStore'
import { useVizStore } from './visualization/store'
import LayerExplain from './visualization/LayerExplain'
import { useManagedSidecars } from './sidecars/managed'
import { useApproveDialog } from './trust/useApproveDialog'
import ApproveCodeDialog from './trust/ApproveCodeDialog'
import { useProjectStore } from './project/store'
import Welcome from './project/Welcome'
import Toolbar from './Toolbar'
import { isTauri } from './workspace/tauri-fs'
import { useWorkspaceStore } from './workspace/store'
import { useDatasetsStore } from './datasets/store'
import DatasetDetail from './datasets/DatasetDetail'
import { useTrainingStore } from './training/store'
import NewRunModal from './training/NewRunModal'
import EvalRunModal from './training/EvalRunModal'
import RunDetailModal from './training/RunDetailModal'
import CompareModal from './training/CompareModal'
import { useViewModeStore } from './training/graph/viewMode'
import ModeToggle from './training/graph/ModeToggle'
import DiagnosticsBanner from './errors/DiagnosticsBanner'
import { fireAndForget } from './errors/report'
import TrainingPalette from './training/graph/TrainingPalette'
import TrainingCanvas from './training/graph/TrainingCanvas'
import TrainingInspector from './training/graph/TrainingInspector'
import TrainingCodePanel from './training/graph/TrainingCodePanel'
import DataPalette from './data/graph/DataPalette'
import DataCanvas from './data/graph/DataCanvas'
import DataInspector from './data/graph/DataInspector'
import DataCodePanel from './data/graph/DataCodePanel'
import { useConnectionsStore, getCurrentConnection, sshTarget } from './connections/store'
import { useRemoteSidecarStore } from './sidecars/remoteSidecar'

/** Poll the torch sidecar's /health for its auth mode. Used for the amber
 *  "ungesichert" chip in Tauri mode (browser-dev shows no chip). */
function useTorchAuthMode(): 'token' | 'unauthenticated-dev' | null {
  const [auth, setAuth] = useState<'token' | 'unauthenticated-dev' | null>(null)
  useEffect(() => {
    let alive = true
    const probe = async () => {
      const h = await sidecarHealthState()
      if (alive) setAuth(h.auth ?? null)
    }
    void fireAndForget('useTorchAuthMode.probe', probe())
    const t = setInterval(() => { void fireAndForget('useTorchAuthMode.poll', probe()) }, 5000)
    return () => { alive = false; clearInterval(t) }
  }, [])
  return auth
}

function UnsafeChip() {
  return (
    <span
      className="rounded bg-amber-900/40 px-2 py-0.5 text-amber-300"
      title="Sidecar läuft ohne Token (nicht von der App gestartet?)"
    >ungesichert</span>
  )
}

function InferenceBadge() {
  const status = useInferenceStore((s) => s.status)
  const nParams = useInferenceStore((s) => s.nParams)
  const error = useInferenceStore((s) => s.error)
  const untrusted = useInferenceStore((s) => s.untrusted)
  const managed = useManagedSidecars((s) => s.torch)
  const torchAuth = useTorchAuthMode()

  // Phase 43 — unapproved Custom/DataOp code blocks all execution. The badge is
  // the entry point to the approval dialog; it never approves by itself.
  if (status === 'untrusted') {
    return (
      <button
        onClick={() => useApproveDialog.getState().openFor(useInferenceStore.getState().untrusted)}
        className="rounded bg-amber-900/40 px-2 py-0.5 text-amber-300 hover:bg-amber-900/60"
        title={error ?? 'Code nicht freigegeben — klicken zum Prüfen'}
      >
        Code nicht freigegeben ({untrusted.length})
      </button>
    )
  }

  const prefix = managed ? 'shapes (auto)' : 'shapes'
  const label =
    status === 'idle' ? `${prefix}: idle`
    : status === 'inferring' ? `${prefix}: inferring…`
    : status === 'ok' ? `${prefix}: ok · ${nParams != null ? nParams.toLocaleString() + ' params' : ''}`
    : status === 'offline' ? `${prefix}: sidecar offline`
    : status === 'auth' ? `${prefix}: auth failed`
    : `${prefix}: ${error?.split(':')[0] ?? 'error'}`

  const color =
    status === 'ok' ? 'bg-emerald-900/40 text-emerald-300'
    : status === 'inferring' ? 'bg-[#1f2429] text-[#6f767e]'
    : status === 'offline' ? 'bg-[#1f2429] text-[#6f767e]'
    : status === 'error' || status === 'auth' ? 'bg-rose-900/40 text-rose-300'
    : 'bg-[#1f2429] text-[#6f767e]'

  return (
    <span className="flex items-center gap-1">
      <span className={`rounded px-2 py-0.5 ${color}`} title={error ?? ''}>{label}</span>
      {isTauri() && torchAuth === 'unauthenticated-dev' && <UnsafeChip />}
    </span>
  )
}

function RemoteSidecarBadge() {
  const status = useRemoteSidecarStore((s) => s.status)
  const ensure = useRemoteSidecarStore((s) => s.ensure)
  const conn = useConnectionsStore((s) => s.saved.find((c) => c.id === s.currentId))
  if (!conn) return null  // only meaningful for remote workspaces
  let label = ''
  let color = 'bg-[#1f2429] text-[#6f767e]'
  let title = ''
  switch (status.kind) {
    case 'idle':       label = 'hpc: idle'; break
    case 'preparing':  label = `hpc: ${status.phase}…`; title = status.message; color = 'bg-violet-900/30 text-violet-300'; break
    case 'starting':   label = 'hpc: starting…'; color = 'bg-violet-900/30 text-violet-300'; break
    case 'running':    label = `hpc: ok · :${status.local_port}`; color = 'bg-emerald-900/40 text-emerald-300'; title = `${status.alias}:${status.root}`; break
    case 'stopped':    label = 'hpc: stopped'; break
    case 'error':      label = `hpc: ${status.message.split(':')[0]}`; color = 'bg-rose-900/40 text-rose-300'; title = status.message; break
  }
  // Click to (re)connect the remote sidecar tunnel. refresh() reloads the
  // remote project and re-fires the ensure() bootstrap. Disabled mid-bootstrap.
  const busy = status.kind === 'preparing' || status.kind === 'starting'
  const hint = busy ? title : `${title ? title + ' — ' : ''}Klicken: neu verbinden`
  return (
    <button
        onClick={() => { if (!busy) void fireAndForget('RemoteSidecarBadge.reconnect', ensure(sshTarget(conn), conn.root, true)) }}
      disabled={busy}
      className={`rounded px-2 py-0.5 ${color} ${busy ? '' : 'cursor-pointer hover:brightness-125'}`}
      title={hint}
    >{label}{busy ? '' : ' ↻'}</button>
  )
}

function ExplainControls() {
  const viewMode = useViewModeStore((s) => s.mode)
  const explain = useVizStore((s) => s.explainMode)
  const running = useVizStore((s) => s.running)
  const playing = useVizStore((s) => s.playing)
  const hasData = useVizStore((s) => Object.keys(s.byNode).length > 0)
  const toggle = useVizStore((s) => s.toggleExplain)
  const run = useVizStore((s) => s.run)
  const playFlow = useVizStore((s) => s.playFlow)
  if (viewMode !== 'architecture') return null
  return (
    <div className="flex items-center gap-1">
      <button
        onClick={toggle}
        className={`rounded px-2 py-0.5 ${explain ? 'bg-violet-900/50 text-violet-200' : 'hover:bg-[#1f2429] hover:text-[#e6e8eb]'}`}
        title="Visualisiere, was durch die Layer fließt"
      >Explain</button>
      {explain && (
        <button
          onClick={() => run()}
          disabled={running}
          className="rounded bg-[#1d4ed8]/80 px-2 py-0.5 text-[#e6e8eb] hover:bg-[#1d4ed8] disabled:opacity-50"
          title="Ein Beispiel durch das Modell schicken"
        >{running ? '…' : '▶ Beispiel'}</button>
      )}
      {explain && hasData && (
        <button
          onClick={() => playFlow()}
          disabled={playing}
          className="rounded border border-[#1f2429] bg-[#13171b] px-2 py-0.5 text-[#9aa1a8] hover:bg-[#1a1f24] hover:text-[#e6e8eb] disabled:opacity-50"
          title="Fluss noch einmal abspielen"
        >{playing ? '… läuft' : '↻ Fluss'}</button>
      )}
    </div>
  )
}

function LLMBadge() {
  const online = useChatStore((s) => s.online)
  const status = useChatStore((s) => s.status)
  const managed = useManagedSidecars((s) => s.llm)
  const authFailed = useChatStore((s) => s.authFailed)
  const authMessage = useChatStore((s) => s.authMessage)
  const llmAuth = useChatStore((s) => s.llmAuth)
  const currentId = useProviderStore((s) => s.currentId)
  const provider = providerById(currentId)
  const prefix = managed ? 'LLM (auto)' : 'LLM'
  const state =
    online === null ? '…'
    : authFailed ? 'auth failed'
    : online === false ? 'offline'
    : status === 'streaming' ? 'thinking'
    : 'ready'
  const model = useProviderStore((s) => s.configs[currentId]?.model?.trim()) || provider.defaultModel
  const label = `${prefix} · ${provider.label}${model && provider.kind !== 'subscription' ? ` · ${model}` : ''}: ${state}`
  const color =
    authFailed ? 'bg-rose-900/40 text-rose-300'
    : online === false ? 'bg-[#1f2429] text-[#6f767e]'
    : status === 'streaming' ? 'bg-violet-900/40 text-violet-300'
    : online ? 'bg-emerald-900/40 text-emerald-300'
    : 'bg-[#1f2429] text-[#6f767e]'
  // Title: auth failure > unauthenticated-dev hint > nothing
  const badgeTitle = authFailed
    ? (authMessage ?? 'Sidecar-Authentifizierung fehlgeschlagen.')
    : !isTauri() && llmAuth === 'unauthenticated-dev'
      ? 'Sidecar läuft ohne Token (nicht von der App gestartet?)'
      : ''
  return (
    <span className="flex items-center gap-1">
      <span className={`rounded px-2 py-0.5 ${color}`} title={badgeTitle}>{label}</span>
      {isTauri() && llmAuth === 'unauthenticated-dev' && <UnsafeChip />}
    </span>
  )
}

function ProjectHeader() {
  const status = useProjectStore((s) => s.status)
  const closeProject = useProjectStore((s) => s.closeProject)
  const currentId = useConnectionsStore((s) => s.currentId)
  // Always subscribed (not behind the early return below): hook order must be
  // identical on every render or React would remount/crash the header whenever
  // status flips between loaded/not-loaded.
  const listError = useTrainingStore((s) => s.listError)
  if (status.kind !== 'loaded') {
    // Phase 38 — error while loading the project (e.g. ssh lost during
    // project.load) must NOT look like "not connected"; show the failure so
    // the user knows workspace/ssh is the problem, not just "none".
    if (status.kind === 'error') {
      const conn = getCurrentConnection()
      const isRemote = conn.kind === 'remote-ssh'
      return (
        <div className="flex items-center gap-2">
          <img src="/favicon.svg" alt="" className="h-5 w-5" />
          <span className="font-semibold tracking-tight">SpinoML</span>
          <span className="rounded bg-rose-900/40 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-rose-300" title={status.error}>
            {isRemote ? `ssh · ${conn.alias} — nicht verbunden` : 'nicht verbunden'} · {status.error.split(':')[0]}
          </span>
          <button onClick={() => void closeProject()} className="ml-1 text-[10px] text-[#5a6068] hover:text-[#9aa1a8]" title="close project">×</button>
        </div>
      )
    }
    return (
      <div className="flex items-center gap-2">
        <img src="/favicon.svg" alt="" className="h-5 w-5" />
        <span className="font-semibold tracking-tight">SpinoML</span>
      </div>
    )
  }
  const conn = getCurrentConnection()
  // Phase 38 — when the last training/ssh refresh failed (listError or
  // project error on remote), the header must not still claim "ssh · alias"
  // in violet as if connected. Use the training listError as a live
  // connectivity signal: a remote with a recent listError is "disconnected"
  // until the next successful poll.
  const disconnected = conn.kind === 'remote-ssh' && !!listError
  return (
    <div className="flex items-center gap-2">
      <img src="/favicon.svg" alt="" className="h-5 w-5" />
      <span className="font-semibold tracking-tight">SpinoML</span>
      <span className="text-[#6f767e]">·</span>
      <span className="text-[#e6e8eb]" title={status.meta.goal || status.meta.description}>
        {status.meta.name}
      </span>
      {conn.kind === 'remote-ssh' && (
        <span
          className={disconnected
            ? 'rounded bg-rose-900/40 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-rose-300'
            : 'rounded bg-violet-900/30 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-violet-300'}
          title={disconnected ? `${conn.alias}:${conn.root} — ${listError}` : `${conn.alias}:${conn.root}`}
          key={currentId}
        >
          ssh · {conn.alias}{disconnected ? ' — nicht verbunden' : ''}
        </span>
      )}
      <button
        onClick={() => void closeProject()}
        className="ml-1 text-[10px] text-[#5a6068] hover:text-[#9aa1a8]"
        title="close project"
      >
        ×
      </button>
    </div>
  )
}

type BottomTab = 'code' | 'terminal'

function BottomTabs() {
  const [tab, setTab] = useState<BottomTab>('code')
  // The Code tab follows the ACTIVE canvas: architecture → nn.Module, training →
  // training script, data → pipeline script. One unified, always-visible panel.
  const viewMode = useViewModeStore((s) => s.mode)
  const codeView = viewMode === 'training' ? <TrainingCodePanel />
    : viewMode === 'data' ? <DataCodePanel />
    : <CodePreview />
  return (
    <div className="flex h-full flex-col bg-[#0a0c0f]">
      <div className="flex shrink-0 border-b border-[#1f2429] bg-[#0e1216]">
        <TabBtn active={tab === 'code'} onClick={() => setTab('code')}>Code</TabBtn>
        <TabBtn active={tab === 'terminal'} onClick={() => setTab('terminal')}>Terminal</TabBtn>
      </div>
      <div className="relative min-h-0 flex-1">
        <div style={{ display: tab === 'code' ? 'block' : 'none' }} className="h-full">
          {codeView}
        </div>
        <div style={{ display: tab === 'terminal' ? 'block' : 'none' }} className="h-full">
          <Terminal />
        </div>
      </div>
    </div>
  )
}

function TabBtn({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  const base = 'px-3 py-1 text-xs transition-colors border-b-2'
  const cls = active
    ? 'border-[var(--accent)] text-[#e6e8eb]'
    : 'border-transparent text-[#6f767e] hover:text-[#e6e8eb]'
  return <button onClick={onClick} className={`${base} ${cls}`}>{children}</button>
}

const storage = typeof window !== 'undefined' ? window.localStorage : undefined
const HBAR =
  'w-px bg-[#1f2429] hover:w-[3px] hover:bg-[#3a4148] data-[separator-active]:w-[3px] data-[separator-active]:bg-[var(--accent)] transition-colors cursor-col-resize'
const VBAR =
  'h-px bg-[#1f2429] hover:h-[3px] hover:bg-[#3a4148] data-[separator-active]:h-[3px] data-[separator-active]:bg-[var(--accent)] transition-colors cursor-row-resize'

function useSaved(id: string) {
  const ctx = useDefaultLayout({ id, storage })
  return { defaultLayout: ctx.defaultLayout, onLayoutChanged: ctx.onLayoutChanged }
}

export default function App() {
  const cols = useSaved('spinoml.cols')
  const center = useSaved('spinoml.center')
  const right = useSaved('spinoml.right')
  const left = useSaved('spinoml.left')
  const status = useProjectStore((s) => s.status)
  const refresh = useProjectStore((s) => s.refresh)

  useEffect(() => {
    void fireAndForget('app.refresh', refresh())
  }, [refresh])

  // Workspace switched → drop caches keyed to the OLD root and reload the
  // datasets + experiments lists. Without this, those sections keep showing the
  // previous workspace's entries/runs (they only self-fetch when empty).
  const workspaceRoot = useWorkspaceStore((s) => s.workspaceRoot)
  useEffect(() => {
    if (!isTauri()) return
    useDatasetsStore.setState({ selectedRel: null, inspects: {}, stats: {}, smoke: {} })
    useTrainingStore.setState({ selectedRunId: null, runs: [] })
    if (!workspaceRoot) return
    void useDatasetsStore.getState().refresh()
    void useTrainingStore.getState().refresh()
  }, [workspaceRoot])

  const showWelcome = isTauri() && status.kind !== 'loaded'
  const selectedDataset = useDatasetsStore((s) => s.selectedRel)
  const selectedRun = useTrainingStore((s) => s.selectedRunId)
  const newRunOpen = useTrainingStore((s) => s.newRunOpen)
  const evalSourceId = useTrainingStore((s) => s.evalSourceId)
  const compareOpen = useTrainingStore((s) => s.compareOpen)
  const viewMode = useViewModeStore((s) => s.mode)
  const explainMode = useVizStore((s) => s.explainMode)

  return (
    <div className="flex h-screen w-screen flex-col bg-[#0a0c0f] text-[#e6e8eb]">
      <header className="flex h-10 shrink-0 items-center justify-between border-b border-[#1f2429] pl-4 pr-3">
        <div className="flex items-center gap-4">
          <ProjectHeader />
          <Toolbar />
          <ModeToggle />
        </div>
        <div className="flex items-center gap-2 text-xs text-[#6f767e]">
          <ExplainControls />
          <RemoteSidecarBadge />
          <InferenceBadge />
          <LLMBadge />
        </div>
      </header>

      {showWelcome ? (
        <div className="min-h-0 flex-1"><Welcome /></div>
      ) : (
        <Group
          orientation="horizontal"
          className="min-h-0 flex-1"
          defaultLayout={cols.defaultLayout}
          onLayoutChanged={cols.onLayoutChanged}
        >
          <Panel defaultSize="15%" minSize="140px">
            <Group
              orientation="vertical"
              defaultLayout={left.defaultLayout}
              onLayoutChanged={left.onLayoutChanged}
            >
              <Panel defaultSize="50%" minSize="100px">{viewMode === 'training' ? <TrainingPalette /> : viewMode === 'data' ? <DataPalette /> : <Palette />}</Panel>
              <Separator className={VBAR} />
              <Panel defaultSize="50%" minSize="120px"><LeftSidebar /></Panel>
            </Group>
          </Panel>
          <Separator className={HBAR} />

          <Panel defaultSize="58%" minSize="240px">
            <Group
              orientation="vertical"
              defaultLayout={center.defaultLayout}
              onLayoutChanged={center.onLayoutChanged}
            >
              <Panel defaultSize="70%" minSize="120px">{viewMode === 'training' ? <TrainingCanvas /> : viewMode === 'data' ? <DataCanvas /> : <Canvas />}</Panel>
              <Separator className={VBAR} />
              <Panel defaultSize="30%" minSize="80px"><BottomTabs /></Panel>
            </Group>
          </Panel>
          <Separator className={HBAR} />

          <Panel defaultSize="27%" minSize="240px">
            <Group
              orientation="vertical"
              defaultLayout={right.defaultLayout}
              onLayoutChanged={right.onLayoutChanged}
            >
              <Panel defaultSize="50%" minSize="100px">{viewMode === 'training' ? <TrainingInspector /> : viewMode === 'data' ? <DataInspector /> : (explainMode ? <LayerExplain /> : <Inspector />)}</Panel>
              <Separator className={VBAR} />
              <Panel defaultSize="50%" minSize="100px"><ChatPanel /></Panel>
            </Group>
          </Panel>
        </Group>
      )}

      {selectedDataset && <DatasetDetail relpath={selectedDataset} />}
      {newRunOpen && <NewRunModal />}
      {evalSourceId && <EvalRunModal />}
      {selectedRun && <RunDetailModal runId={selectedRun} />}
      {compareOpen && <CompareModal />}
      <ApproveCodeDialog />
      <DiagnosticsBanner />
    </div>
  )
}
