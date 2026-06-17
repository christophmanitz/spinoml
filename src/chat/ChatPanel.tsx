import { useEffect, useRef, useState } from 'react'
import { useChatStore, type ChatMessage, type ToolCall } from './store'
import { providerById, useProviderStore } from './providerStore'
import ProviderSettings from './ProviderSettings'

export default function ChatPanel() {
  const messages = useChatStore((s) => s.messages)
  const status = useChatStore((s) => s.status)
  const online = useChatStore((s) => s.online)
  const send = useChatStore((s) => s.send)
  const reset = useChatStore((s) => s.reset)
  const currentId = useProviderStore((s) => s.currentId)
  const provider = providerById(currentId)

  const [draft, setDraft] = useState('')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [messages])

  const onSend = async () => {
    const text = draft
    setDraft('')
    await send(text)
  }

  const disabled = status === 'streaming' || online === false
  const statusHint =
    online === false ? 'sidecar offline — run npm run sidecar:llm'
    : status === 'streaming' ? 'LLM is working…'
    : online === null ? 'checking sidecar…'
    : 'ready'

  return (
    <div className="flex h-full min-h-0 flex-col p-3 text-sm">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-xs uppercase tracking-wide text-[#7a8088]">Chat</div>
        <div className="flex items-center gap-2 text-[10px] text-[#7a8088]">
          <button
            className="rounded px-1 text-[#9aa1a8] hover:bg-[#1f2429] hover:text-[#e6e8eb]"
            onClick={() => setSettingsOpen(true)}
            title="LLM-Quelle wählen"
          >{provider.label} ⚙</button>
          <span>{statusHint}</span>
          {messages.length > 0 && (
            <button className="hover:text-[#e6e8eb]" onClick={reset}>reset</button>
          )}
        </div>
      </div>
      {settingsOpen && <ProviderSettings onClose={() => setSettingsOpen(false)} />}

      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto rounded border border-[#1f2429] bg-[#0e1216] p-2 text-xs"
      >
        {messages.length === 0 && (
          <div className="text-[#7a8088]">
            Ask Claude to build, debug, or extend your architecture. Try “add a small CNN
            classifier for 10 classes” or “fix the failing LayerNorm”.
          </div>
        )}
        {messages.map((m) => <MessageView key={m.id} message={m} />)}
      </div>

      <div className="mt-2 flex gap-1">
        <textarea
          className="min-h-[34px] max-h-[120px] flex-1 resize-y rounded border border-[#1f2429] bg-[#13171b] px-2 py-1 text-xs outline-none focus:border-[#3a4148]"
          placeholder={online === false ? 'sidecar offline' : 'Message…'}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              if (!disabled) onSend()
            }
          }}
          rows={1}
          disabled={online === false}
        />
        <button
          className="rounded border border-[#1f2429] bg-[#13171b] px-3 py-1 text-xs text-[#e6e8eb] hover:bg-[#1a1f24] disabled:text-[#5b6168]"
          onClick={onSend}
          disabled={disabled || !draft.trim()}
        >
          {status === 'streaming' ? '…' : 'Send'}
        </button>
      </div>
    </div>
  )
}

function MessageView({ message }: { message: ChatMessage }) {
  if (message.role === 'user') {
    return (
      <div className="mb-2">
        <div className="mb-0.5 text-[10px] uppercase tracking-wide text-[#7a8088]">you</div>
        <div className="whitespace-pre-wrap rounded bg-[#13171b] px-2 py-1 text-[#e6e8eb]">
          {message.content}
        </div>
      </div>
    )
  }
  return (
    <div className="mb-2">
      <div className="mb-0.5 flex items-center gap-1 text-[10px] uppercase tracking-wide text-[#7a8088]">
        <span>claude</span>
        {message.status === 'streaming' && <span className="text-[#5b6168]">·streaming</span>}
        {message.status === 'error' && <span className="text-rose-400">·error</span>}
      </div>
      {message.toolCalls.map((t) => <ToolCallView key={t.id} call={t} />)}
      {message.content && (
        <div className="whitespace-pre-wrap rounded bg-[#13171b] px-2 py-1 text-[#e6e8eb]">
          {message.content}
        </div>
      )}
      {message.status === 'error' && message.error && (
        <div className="mt-1 rounded bg-rose-950/40 px-2 py-1 font-mono text-[10px] text-rose-300">
          {message.error}
        </div>
      )}
    </div>
  )
}

function ToolCallView({ call }: { call: ToolCall }) {
  const friendly = call.name.replace(/^mcp__graph__/, '')
  const color =
    call.status === 'ok' ? 'border-emerald-900/60 bg-emerald-950/30 text-emerald-200'
    : call.status === 'error' ? 'border-rose-900/60 bg-rose-950/30 text-rose-200'
    : 'border-[#1f2429] bg-[#13171b] text-[#9aa1a8]'
  return (
    <div className={`mb-1 rounded border px-2 py-1 font-mono text-[10px] leading-snug ${color}`}>
      <div>
        <span className="text-[#7a8088]">→ </span>
        <span>{friendly}</span>
        <span className="text-[#7a8088]">({summarizeArgs(call.args)})</span>
      </div>
      {call.status === 'error' && call.error && (
        <div className="mt-0.5 text-rose-300">{call.error}</div>
      )}
    </div>
  )
}

function summarizeArgs(args: Record<string, unknown>): string {
  const parts: string[] = []
  for (const [k, v] of Object.entries(args)) {
    if (Array.isArray(v)) parts.push(`${k}=[${v.join(',')}]`)
    else if (typeof v === 'object' && v !== null) parts.push(`${k}=${JSON.stringify(v)}`)
    else parts.push(`${k}=${String(v)}`)
  }
  return parts.join(', ')
}
