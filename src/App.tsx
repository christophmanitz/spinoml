import Palette from './palette/Palette'
import Canvas from './canvas/Canvas'
import Inspector from './inspector/Inspector'
import ChatPanel from './chat/ChatPanel'
import CodePreview from './codegen/CodePreview'

export default function App() {
  return (
    <div className="flex h-screen w-screen flex-col bg-[#0b0d10] text-[#e6e8eb]">
      <header className="flex h-10 shrink-0 items-center justify-between border-b border-[#1f2429] px-4">
        <div className="flex items-center gap-3">
          <span className="font-semibold tracking-tight">MLForge</span>
          <span className="text-xs text-[#7a8088]">PyTorch architecture builder · phase 0</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-[#7a8088]">
          <span className="rounded bg-[#1f2429] px-2 py-0.5">LLM: disconnected</span>
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
