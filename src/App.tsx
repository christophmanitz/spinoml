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
import { useChatStore } from './chat/store'
import { useManagedSidecars } from './sidecars/managed'
import { useProjectStore } from './project/store'
import Welcome from './project/Welcome'
import Toolbar from './Toolbar'
import { isTauri } from './workspace/tauri-fs'
import { useDatasetsStore } from './datasets/store'
import DatasetDetail from './datasets/DatasetDetail'
import { useTrainingStore } from './training/store'
import NewRunModal from './training/NewRunModal'
import RunDetailModal from './training/RunDetailModal'
import CompareModal from './training/CompareModal'
import { useViewModeStore } from './training/graph/viewMode'
import ModeToggle from './training/graph/ModeToggle'
import TrainingPalette from './training/graph/TrainingPalette'
import TrainingCanvas from './training/graph/TrainingCanvas'
import TrainingInspector from './training/graph/TrainingInspector'
import { useConnectionsStore, getCurrentConnection } from './connections/store'
import { useRemoteSidecarStore } from './sidecars/remoteSidecar'

function InferenceBadge() {
  const status = useInferenceStore((s) => s.status)
  const nParams = useInferenceStore((s) => s.nParams)
  const error = useInferenceStore((s) => s.error)
  const managed = useManagedSidecars((s) => s.torch)

  const prefix = managed ? 'shapes (auto)' : 'shapes'
  const label =
    status === 'idle' ? `${prefix}: idle`
    : status === 'inferring' ? `${prefix}: inferring…`
    : status === 'ok' ? `${prefix}: ok · ${nParams != null ? nParams.toLocaleString() + ' params' : ''}`
    : status === 'offline' ? `${prefix}: sidecar offline`
    : `${prefix}: ${error?.split(':')[0] ?? 'error'}`

  const color =
    status === 'ok' ? 'bg-emerald-900/40 text-emerald-300'
    : status === 'inferring' ? 'bg-[#1f2429] text-[#7a8088]'
    : status === 'offline' ? 'bg-[#1f2429] text-[#7a8088]'
    : status === 'error' ? 'bg-rose-900/40 text-rose-300'
    : 'bg-[#1f2429] text-[#7a8088]'

  return <span className={`rounded px-2 py-0.5 ${color}`} title={error ?? ''}>{label}</span>
}

function RemoteSidecarBadge() {
  const status = useRemoteSidecarStore((s) => s.status)
  const conn = useConnectionsStore((s) => s.saved.find((c) => c.id === s.currentId))
  if (!conn) return null  // only meaningful for remote workspaces
  let label = ''
  let color = 'bg-[#1f2429] text-[#7a8088]'
  let title = ''
  switch (status.kind) {
    case 'idle':       label = 'hpc: idle'; break
    case 'preparing':  label = `hpc: ${status.phase}…`; title = status.message; color = 'bg-violet-900/30 text-violet-300'; break
    case 'starting':   label = 'hpc: starting…'; color = 'bg-violet-900/30 text-violet-300'; break
    case 'running':    label = `hpc: ok · :${status.local_port}`; color = 'bg-emerald-900/40 text-emerald-300'; title = `${status.alias}:${status.root}`; break
    case 'stopped':    label = 'hpc: stopped'; break
    case 'error':      label = `hpc: ${status.message.split(':')[0]}`; color = 'bg-rose-900/40 text-rose-300'; title = status.message; break
  }
  return <span className={`rounded px-2 py-0.5 ${color}`} title={title}>{label}</span>
}

function LLMBadge() {
  const online = useChatStore((s) => s.online)
  const status = useChatStore((s) => s.status)
  const managed = useManagedSidecars((s) => s.llm)
  const prefix = managed ? 'LLM (auto)' : 'LLM'
  const label =
    online === null ? `${prefix}: …`
    : online === false ? `${prefix}: offline`
    : status === 'streaming' ? `${prefix}: thinking`
    : `${prefix}: ready`
  const color =
    online === false ? 'bg-[#1f2429] text-[#7a8088]'
    : status === 'streaming' ? 'bg-violet-900/40 text-violet-300'
    : online ? 'bg-emerald-900/40 text-emerald-300'
    : 'bg-[#1f2429] text-[#7a8088]'
  return <span className={`rounded px-2 py-0.5 ${color}`}>{label}</span>
}

function ProjectHeader() {
  const status = useProjectStore((s) => s.status)
  const closeProject = useProjectStore((s) => s.closeProject)
  const currentId = useConnectionsStore((s) => s.currentId)
  if (status.kind !== 'loaded') {
    return <span className="font-semibold tracking-tight">MLForge</span>
  }
  const conn = getCurrentConnection()
  return (
    <div className="flex items-baseline gap-2">
      <span className="font-semibold tracking-tight">MLForge</span>
      <span className="text-[#7a8088]">·</span>
      <span className="text-[#e6e8eb]" title={status.meta.goal || status.meta.description}>
        {status.meta.name}
      </span>
      {conn.kind === 'remote-ssh' && (
        <span
          className="rounded bg-violet-900/30 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-violet-300"
          title={`${conn.alias}:${conn.root}`}
          key={currentId}
        >
          ssh · {conn.alias}
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
  return (
    <div className="flex h-full flex-col bg-[#0b0d10]">
      <div className="flex shrink-0 border-b border-[#1f2429] bg-[#0e1115]">
        <TabBtn active={tab === 'code'} onClick={() => setTab('code')}>Code</TabBtn>
        <TabBtn active={tab === 'terminal'} onClick={() => setTab('terminal')}>Terminal</TabBtn>
      </div>
      <div className="relative min-h-0 flex-1">
        <div style={{ display: tab === 'code' ? 'block' : 'none' }} className="h-full">
          <CodePreview />
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
    ? 'border-[#6ab7ff] text-[#e6e8eb]'
    : 'border-transparent text-[#7a8088] hover:text-[#e6e8eb]'
  return <button onClick={onClick} className={`${base} ${cls}`}>{children}</button>
}

const storage = typeof window !== 'undefined' ? window.localStorage : undefined
const HBAR =
  'w-px bg-[#1f2429] hover:w-[3px] hover:bg-[#3a4148] data-[separator-active]:w-[3px] data-[separator-active]:bg-[#6ab7ff] transition-colors cursor-col-resize'
const VBAR =
  'h-px bg-[#1f2429] hover:h-[3px] hover:bg-[#3a4148] data-[separator-active]:h-[3px] data-[separator-active]:bg-[#6ab7ff] transition-colors cursor-row-resize'

function useSaved(id: string) {
  const ctx = useDefaultLayout({ id, storage })
  return { defaultLayout: ctx.defaultLayout, onLayoutChanged: ctx.onLayoutChanged }
}

export default function App() {
  const cols = useSaved('mlforge.cols')
  const center = useSaved('mlforge.center')
  const right = useSaved('mlforge.right')
  const left = useSaved('mlforge.left')
  const status = useProjectStore((s) => s.status)
  const refresh = useProjectStore((s) => s.refresh)

  useEffect(() => {
    void refresh()
  }, [refresh])

  const showWelcome = isTauri() && status.kind !== 'loaded'
  const selectedDataset = useDatasetsStore((s) => s.selectedRel)
  const selectedRun = useTrainingStore((s) => s.selectedRunId)
  const newRunOpen = useTrainingStore((s) => s.newRunOpen)
  const compareOpen = useTrainingStore((s) => s.compareOpen)
  const viewMode = useViewModeStore((s) => s.mode)

  return (
    <div className="flex h-screen w-screen flex-col bg-[#0b0d10] text-[#e6e8eb]">
      <header className="flex h-10 shrink-0 items-center justify-between border-b border-[#1f2429] pl-4 pr-3">
        <div className="flex items-center gap-4">
          <ProjectHeader />
          <Toolbar />
          <ModeToggle />
        </div>
        <div className="flex items-center gap-2 text-xs text-[#7a8088]">
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
              <Panel defaultSize="50%" minSize="100px">{viewMode === 'training' ? <TrainingPalette /> : <Palette />}</Panel>
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
              <Panel defaultSize="70%" minSize="120px">{viewMode === 'training' ? <TrainingCanvas /> : <Canvas />}</Panel>
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
              <Panel defaultSize="50%" minSize="100px">{viewMode === 'training' ? <TrainingInspector /> : <Inspector />}</Panel>
              <Separator className={VBAR} />
              <Panel defaultSize="50%" minSize="100px"><ChatPanel /></Panel>
            </Group>
          </Panel>
        </Group>
      )}

      {selectedDataset && <DatasetDetail relpath={selectedDataset} />}
      {newRunOpen && <NewRunModal />}
      {selectedRun && <RunDetailModal runId={selectedRun} />}
      {compareOpen && <CompareModal />}
    </div>
  )
}
