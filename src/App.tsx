import Palette from './palette/Palette'
import Canvas from './canvas/Canvas'
import Inspector from './inspector/Inspector'
import ChatPanel from './chat/ChatPanel'
import CodePreview from './codegen/CodePreview'
import { useInferenceStore } from './inference/store'
import { useChatStore } from './chat/store'

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

export default function App() {
  return (
    <div className="flex h-screen w-screen flex-col bg-[#0b0d10] text-[#e6e8eb]">
      <header className="flex h-10 shrink-0 items-center justify-between border-b border-[#1f2429] px-4">
        <div className="flex items-center gap-3">
          <span className="font-semibold tracking-tight">MLForge</span>
          <span className="text-xs text-[#7a8088]">PyTorch architecture builder · phase 3</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-[#7a8088]">
          <InferenceBadge />
          <LLMBadge />
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <Palette />
        <div className="flex min-w-0 flex-1 flex-col">
          <Canvas />
          <CodePreview />
        </div>
        <div className="flex w-[380px] shrink-0 flex-col border-l border-[#1f2429]">
          <Inspector />
          <ChatPanel />
        </div>
      </div>
    </div>
  )
}
