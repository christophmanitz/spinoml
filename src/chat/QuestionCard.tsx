import { useState } from 'react'
import { useChatStore, type PendingAsk } from './store'

// Inline card shown when the LLM (or a gated run_script) asks the user something
// mid-turn and is WAITING for an answer. Replaces the old broken flow where a
// question was just text the user couldn't actually answer.
//   confirm → Ausführen / Abbrechen (boolean)
//   select  → one button per option (string)
//   text    → a text box + submit (string)
export default function QuestionCard({ ask }: { ask: PendingAsk }) {
  const answerAsk = useChatStore((s) => s.answerAsk)
  const [text, setText] = useState('')

  const preview = typeof ask.payload?.preview === 'string' ? (ask.payload.preview as string) : ''

  return (
    <div className="rounded-md border-2 border-[var(--accent)]/70 bg-[#11161d] px-3 py-2.5 shadow-[0_0_0_3px_rgba(36,200,219,0.12)]">
      <div className="mb-1 text-[0.72em] font-semibold uppercase tracking-wide text-[var(--accent)]">
        {ask.kind === 'confirm' ? 'Bestätigung' : 'Frage'}
      </div>
      <div className="whitespace-pre-wrap leading-relaxed text-[#e6e8eb]">{ask.prompt}</div>

      {preview && (
        <pre className="mt-2 max-h-40 overflow-auto rounded border border-[#1f2429] bg-[#0b0e11] p-2 font-mono text-[0.8em] leading-snug text-[#cdd3da]">
          {preview}
        </pre>
      )}

      {ask.kind === 'confirm' && (
        <div className="mt-2.5 flex gap-2">
          <button
            className="rounded border border-emerald-700/60 bg-emerald-900/40 px-3 py-1 text-emerald-100 hover:bg-emerald-800/50"
            onClick={() => answerAsk(true)}
          >Ausführen</button>
          <button
            className="rounded border border-[#1f2429] bg-[#13171b] px-3 py-1 text-[#9aa1a8] hover:bg-[#1a1f24]"
            onClick={() => answerAsk(false)}
          >Abbrechen</button>
        </div>
      )}

      {ask.kind === 'select' && (
        <div className="mt-2.5 flex flex-wrap gap-2">
          {(ask.options ?? []).map((opt) => (
            <button
              key={opt}
              className="rounded border border-[#3a4148] bg-[#13171b] px-3 py-1 text-[#e6e8eb] hover:bg-[#1a1f24]"
              onClick={() => answerAsk(opt)}
            >{opt}</button>
          ))}
        </div>
      )}

      {ask.kind === 'text' && (
        <div className="mt-2.5 flex gap-1">
          <input
            autoFocus
            className="flex-1 rounded border border-[#1f2429] bg-[#13171b] px-2 py-1 text-[1em] outline-none focus:border-[#3a4148]"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && text.trim()) { e.preventDefault(); answerAsk(text.trim()) }
            }}
            placeholder="Antwort…"
          />
          <button
            className="rounded border border-[#1f2429] bg-[#13171b] px-3 py-1 text-[1em] text-[#e6e8eb] hover:bg-[#1a1f24] disabled:text-[#5b6168]"
            onClick={() => answerAsk(text.trim())}
            disabled={!text.trim()}
          >Senden</button>
        </div>
      )}
    </div>
  )
}
