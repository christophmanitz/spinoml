import { Group, Panel, Separator, useDefaultLayout } from 'react-resizable-panels'
import Palette from './palette/Palette'
import Canvas from './canvas/Canvas'
import Inspector from './inspector/Inspector'
import ChatPanel from './chat/ChatPanel'
import CodePreview from './codegen/CodePreview'
import { useInferenceStore } from './inference/store'
import { useChatStore } from './chat/store'
import Toolbar from './Toolbar'

function InferenceBadge() {
  const status = useInferenceStore((s) => s.status)
  const nParams = useInferenceStore((s) => s.nParams)
  const error = useInferenceStore((s) => s.error)

  const label =
    status === 'idle' ? 'shapes: idle'
    : status === 'inferring' ? 'shapes: inferring…'
    : status === 'ok' ? `shapes: ok · ${nParams != null ? nParams.toLocaleString() + ' params' : ''}`
    : status === 'offline' ? 'shapes: sidecar offline'
    : `shapes: ${error?.split(':')[0] ?? 'error'}`

  const color =
    status === 'ok' ? 'bg-emerald-900/40 text-emerald-300'
    : status === 'inferring' ? 'bg-[#1f2429] text-[#7a8088]'
    : status === 'offline' ? 'bg-[#1f2429] text-[#7a8088]'
    : status === 'error' ? 'bg-rose-900/40 text-rose-300'
    : 'bg-[#1f2429] text-[#7a8088]'

  return <span className={`rounded px-2 py-0.5 ${color}`} title={error ?? ''}>{label}</span>
}

function LLMBadge() {
  const online = useChatStore((s) => s.online)
  const status = useChatStore((s) => s.status)
  const label =
    online === null ? 'LLM: …'
    : online === false ? 'LLM: offline'
    : status === 'streaming' ? 'LLM: thinking'
    : 'LLM: ready'
  const color =
    online === false ? 'bg-[#1f2429] text-[#7a8088]'
    : status === 'streaming' ? 'bg-violet-900/40 text-violet-300'
    : online ? 'bg-emerald-900/40 text-emerald-300'
    : 'bg-[#1f2429] text-[#7a8088]'
  return <span className={`rounded px-2 py-0.5 ${color}`}>{label}</span>
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

  return (
    <div className="flex h-screen w-screen flex-col bg-[#0b0d10] text-[#e6e8eb]">
      <header className="flex h-10 shrink-0 items-center justify-between border-b border-[#1f2429] pl-4 pr-3">
        <div className="flex items-center gap-4">
          <span className="font-semibold tracking-tight">MLForge</span>
          <Toolbar />
        </div>
        <div className="flex items-center gap-2 text-xs text-[#7a8088]">
          <InferenceBadge />
          <LLMBadge />
        </div>
      </header>

      <Group
        orientation="horizontal"
        className="min-h-0 flex-1"
        defaultLayout={cols.defaultLayout}
        onLayoutChanged={cols.onLayoutChanged}
      >
        <Panel defaultSize="15%" minSize="120px"><Palette /></Panel>
        <Separator className={HBAR} />

        <Panel defaultSize="58%" minSize="240px">
          <Group
            orientation="vertical"
            defaultLayout={center.defaultLayout}
            onLayoutChanged={center.onLayoutChanged}
          >
            <Panel defaultSize="70%" minSize="120px"><Canvas /></Panel>
            <Separator className={VBAR} />
            <Panel defaultSize="30%" minSize="80px"><CodePreview /></Panel>
          </Group>
        </Panel>
        <Separator className={HBAR} />

        <Panel defaultSize="27%" minSize="240px">
          <Group
            orientation="vertical"
            defaultLayout={right.defaultLayout}
            onLayoutChanged={right.onLayoutChanged}
          >
            <Panel defaultSize="50%" minSize="100px"><Inspector /></Panel>
            <Separator className={VBAR} />
            <Panel defaultSize="50%" minSize="100px"><ChatPanel /></Panel>
          </Group>
        </Panel>
      </Group>
    </div>
  )
}
