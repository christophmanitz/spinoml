export default function ChatPanel() {
  return (
    <div className="flex h-1/2 min-h-0 flex-col p-3 text-sm">
      <div className="mb-2 text-xs uppercase tracking-wide text-[#7a8088]">Chat</div>
      <div className="flex-1 overflow-y-auto rounded border border-[#1f2429] bg-[#0e1216] p-2 text-xs text-[#7a8088]">
        LLM not connected yet. Phase 4 wires Claude Agent SDK sidecar.
      </div>
      <div className="mt-2 flex gap-1">
        <input
          className="flex-1 rounded border border-[#1f2429] bg-[#13171b] px-2 py-1 text-xs outline-none focus:border-[#3a4148]"
          placeholder="Message…"
          disabled
        />
        <button
          className="rounded border border-[#1f2429] bg-[#13171b] px-3 py-1 text-xs text-[#7a8088]"
          disabled
        >
          Send
        </button>
      </div>
    </div>
  )
}
