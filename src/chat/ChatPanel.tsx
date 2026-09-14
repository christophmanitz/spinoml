import { useEffect, useRef, useState } from 'react'
import { useChatStore, type ChatMessage, type ToolCall } from './store'
import { providerById, useProviderStore } from './providerStore'
import { useChatUi } from './uiStore'
import ProviderSettings from './ProviderSettings'
import QuestionCard from './QuestionCard'
import Markdown from '../ui/Markdown'

export default function ChatPanel() {
  const messages = useChatStore((s) => s.messages)
  const status = useChatStore((s) => s.status)
  const online = useChatStore((s) => s.online)
  const send = useChatStore((s) => s.send)
  const reset = useChatStore((s) => s.reset)
  const compact = useChatStore((s) => s.compact)
  const stop = useChatStore((s) => s.stop)
  const pendingAsk = useChatStore((s) => s.pendingAsk)
  const answerAsk = useChatStore((s) => s.answerAsk)
  const currentId = useProviderStore((s) => s.currentId)
  const provider = providerById(currentId)
  const basePx = useChatUi((s) => s.basePx)
  const bumpFontScale = useChatUi((s) => s.bumpFontScale)
  const autoMode = useChatUi((s) => s.autoMode)
  const setAutoMode = useChatUi((s) => s.setAutoMode)
  const docMode = useChatUi((s) => s.docMode)
  const setDocMode = useChatUi((s) => s.setDocMode)

  const [draft, setDraft] = useState('')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    // pendingAsk in the deps so the confirm/question card scrolls into view the
    // moment it appears — otherwise it renders below the fold and the turn looks
    // hung when it's actually waiting for the user to answer.
  }, [messages, pendingAsk])

  const onSend = async () => {
    const text = draft
    setDraft('')
    // If a question/confirm is open, the input box answers it (works even if the
    // QuestionCard isn't visible). confirm → yes/no parsed from the text.
    if (pendingAsk) {
      if (pendingAsk.kind === 'confirm') answerAsk(/^\s*(j|y|ok|ausf|run|go|1|true)/i.test(text))
      else answerAsk(text.trim())
      return
    }
    await send(text)
  }

  const disabled = status === 'streaming' || online === false
  const statusHint =
    pendingAsk ? '⏳ wartet auf deine Antwort'
    : online === false ? 'Sidecar offline — npm run sidecar:llm'
    : status === 'streaming' ? 'LLM arbeitet…'
    : online === null ? 'prüfe Sidecar…'
    : 'bereit'
  // Status as a single Ampel dot (hover shows the detail via title).
  const dotColor =
    online === false ? 'bg-rose-500'
    : pendingAsk ? 'bg-sky-400'
    : status === 'streaming' ? 'bg-amber-400'
    : online === null ? 'bg-zinc-500'
    : 'bg-emerald-500'

  return (
    <div className="flex h-full min-h-0 flex-col p-3 text-sm">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-xs uppercase tracking-wide text-[#6f767e]">
          <span
            className={`inline-block h-2 w-2 rounded-full ${dotColor} ${status === 'streaming' ? 'animate-pulse' : ''}`}
            title={statusHint}
          />
          Chat
        </div>
        <div className="flex items-center gap-2 text-[10px] text-[#6f767e]">
          <div className="flex items-center rounded border border-[#1f2429]">
            <button
              className="px-1.5 leading-none text-[#9aa1a8] hover:bg-[#1f2429] hover:text-[#e6e8eb]"
              onClick={() => bumpFontScale(-0.1)}
              title="Schrift kleiner"
            >A−</button>
            <button
              className="px-1.5 leading-none text-[#9aa1a8] hover:bg-[#1f2429] hover:text-[#e6e8eb]"
              onClick={() => bumpFontScale(0.1)}
              title="Schrift größer"
            >A+</button>
          </div>
          <button
            className="rounded px-1 text-[#9aa1a8] hover:bg-[#1f2429] hover:text-[#e6e8eb]"
            onClick={() => setSettingsOpen(true)}
            title="LLM-Quelle wählen"
          >{provider.label} ⚙</button>
          {messages.length > 6 && (
            <button className="hover:text-[#e6e8eb]" onClick={compact} title="Älteren Verlauf ausblenden (Kontext kürzen, Zustand bleibt)">kürzen</button>
          )}
          {messages.length > 0 && (
            <button className="hover:text-[#e6e8eb]" onClick={reset}>reset</button>
          )}
        </div>
      </div>

      {/* FEAT-3 / FEAT-4 — Auto-Modus + Doku-Modus toggles. */}
      <div className="mb-2 flex items-center gap-2 text-[10px]">
        <button
          onClick={() => setAutoMode(!autoMode)}
          title={autoMode
            ? 'Auto-Modus AN: Shell-Skripte werden automatisch freigegeben (SLURM fragt weiterhin). Stop jederzeit möglich.'
            : 'Auto-Modus AUS: jede run_script-Ausführung wird per Klick bestätigt.'}
          className={`flex items-center gap-1 rounded border px-1.5 py-0.5 transition-colors ${
            autoMode
              ? 'border-amber-600/60 bg-amber-950/40 text-amber-300'
              : 'border-[#1f2429] text-[#6f767e] hover:text-[#9aa1a8]'}`}
        >
          <span className={`inline-block h-1.5 w-1.5 rounded-full ${autoMode ? 'bg-amber-400' : 'bg-[#3a4148]'}`} />
          Auto-Modus {autoMode ? 'AN' : 'AUS'}
        </button>
        <div className="flex items-center gap-1 text-[#6f767e]">
          <span title="Wie ausführlich der Chatbot in notes/lab-notebook.md dokumentiert.">Doku:</span>
          {(['off', 'compact', 'verbose'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setDocMode(m)}
              className={`rounded px-1.5 py-0.5 transition-colors ${
                docMode === m
                  ? 'bg-[var(--accent-sel)] text-[var(--accent)]'
                  : 'text-[#6f767e] hover:bg-[#1a1e22] hover:text-[#9aa1a8]'}`}
              title={m === 'off' ? 'Keine automatische Doku — nur auf explizite Bitte.'
                : m === 'compact' ? 'Nur Meilensteine dokumentieren.'
                : 'Ausführliche, paper-taugliche Doku (Standard).'}
            >{m === 'off' ? 'Aus' : m === 'compact' ? 'Kompakt' : 'Ausführlich'}</button>
          ))}
        </div>
      </div>

      {settingsOpen && <ProviderSettings onClose={() => setSettingsOpen(false)} />}

      <div
        ref={scrollRef}
        className="flex-1 space-y-3 overflow-y-auto rounded border border-[#1f2429] bg-[#0e1216] p-3"
        style={{ fontSize: `${basePx}px` }}
      >
        {messages.length === 0 && (
          <div className="text-[0.92em] leading-relaxed text-[#6f767e]">
            Ask the assistant to build, debug, or extend your architecture. Try “add a small CNN
            classifier for 10 classes” or “fix the failing LayerNorm”.
          </div>
        )}
        {messages.map((m) => <MessageView key={m.id} message={m} />)}
        {pendingAsk && <QuestionCard ask={pendingAsk} />}
      </div>

      <div className="mt-2 flex gap-1" style={{ fontSize: `${basePx}px` }}>
        <textarea
          className="min-h-[38px] max-h-[200px] flex-1 resize-y rounded border border-[#1f2429] bg-[#13171b] px-2 py-1.5 text-[1em] outline-none focus:border-[#3a4148]"
          placeholder={
            online === false ? 'sidecar offline'
            : pendingAsk?.kind === 'confirm' ? 'ja / nein + Enter, um zu antworten…'
            : pendingAsk ? 'Antwort + Enter…'
            : 'Message…'
          }
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              // Allow answering an open question even while the turn streams.
              if (!disabled || pendingAsk) onSend()
            }
          }}
          rows={1}
          disabled={online === false}
        />
        {status === 'streaming' ? (
          <button
            className="rounded border border-rose-900/60 bg-rose-950/40 px-3 py-1 text-[1em] text-rose-200 hover:bg-rose-900/50"
            onClick={stop}
            title="Laufenden Zug + Skript abbrechen"
          >⏹ Stop</button>
        ) : (
          <button
            className="rounded border border-[#1f2429] bg-[#13171b] px-3 py-1 text-[1em] text-[#e6e8eb] hover:bg-[#1a1f24] disabled:text-[#5b6168]"
            onClick={onSend}
            disabled={disabled || !draft.trim()}
          >Send</button>
        )}
      </div>
    </div>
  )
}

function MessageView({ message }: { message: ChatMessage }) {
  const currentId = useProviderStore((s) => s.currentId)
  const providerLabel = providerById(currentId).label
  if (message.role === 'user') {
    return (
      <div>
        <div className="mb-1 text-[0.72em] font-semibold uppercase tracking-wide text-[#6f767e]">you</div>
        <div className="whitespace-pre-wrap rounded-md border border-[#1f2429] bg-[#13171b] px-3 py-2 leading-relaxed text-[#e6e8eb]">
          {message.content}
        </div>
      </div>
    )
  }
  return (
    <div>
      <div className="mb-1 flex items-center gap-1.5 text-[0.72em] font-semibold uppercase tracking-wide text-[#6f767e]">
        <span className="text-[#9aa1a8]">{providerLabel}</span>
        {message.status === 'streaming' && <span className="text-[#5b6168]">· streaming</span>}
        {message.status === 'error' && <span className="text-rose-400">· error</span>}
      </div>
      {message.toolCalls.map((t) => <ToolCallView key={t.id} call={t} />)}
      {message.log && <ScriptLog log={message.log} live={message.status === 'streaming'} />}
      {message.content && (
        <div className="rounded-md border border-[#1f2429] bg-[#11161b] px-3 py-2 text-[#d4d7db]">
          <Markdown>{message.content}</Markdown>
        </div>
      )}
      {message.status === 'streaming' && !message.content && message.toolCalls.length === 0 && (
        <div className="px-1 text-[0.92em] text-[#5b6168]">…</div>
      )}
      {message.status === 'error' && message.error && (
        <div className="mt-1 rounded bg-rose-950/40 px-2 py-1 font-mono text-[0.8em] text-rose-300">
          {message.error}
        </div>
      )}
    </div>
  )
}

// Markdown renderer tuned for the dark chat surface. Element overrides give
// readable paragraphs, lists, headings and code without the typography plugin.

// Live terminal-style view of a running script's stdout/stderr. Auto-scrolls to
// the bottom as new output streams in so the user watches the run unfold.
function ScriptLog({ log, live }: { log: string; live: boolean }) {
  const ref = useRef<HTMLPreElement>(null)
  const [secs, setSecs] = useState(0)
  useEffect(() => { if (ref.current) ref.current.scrollTop = ref.current.scrollHeight }, [log])
  useEffect(() => {
    if (!live) return
    const t = setInterval(() => setSecs((s) => s + 1), 1000)
    return () => clearInterval(t)
  }, [live])
  return (
    <div className="mb-1 overflow-hidden rounded-md border border-[#1f2429] bg-[#0b0e11]">
      <div className="flex items-center gap-1.5 border-b border-[#1f2429] px-2 py-1 text-[0.72em] uppercase tracking-wide text-[#6f767e]">
        <span>script output</span>
        {live && <span className="text-emerald-400">● running {secs}s</span>}
        {live && <span className="text-[#5b6168]">· „Stop" zum Abbrechen</span>}
      </div>
      <pre ref={ref} className="max-h-48 overflow-auto px-2 py-1.5 font-mono text-[0.78em] leading-snug text-[#b8c0c8] whitespace-pre-wrap">
        {log}
      </pre>
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
    <div className={`mb-1 rounded border px-2 py-1 font-mono text-[0.72em] leading-snug ${color}`}>
      <div>
        <span className="text-[#6f767e]">→ </span>
        <span>{friendly}</span>
        <span className="text-[#6f767e]">({summarizeArgs(call.args)})</span>
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
