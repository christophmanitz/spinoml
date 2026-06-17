import { useEffect } from 'react'
import { PROVIDERS, providerById, useProviderStore } from './providerStore'

const INPUT =
  'w-full rounded border border-[#1f2429] bg-[#0b0e11] px-2 py-1 text-[12px] text-[#e6e8eb] outline-none focus:border-[#6ab7ff]'

export default function ProviderSettings({ onClose }: { onClose: () => void }) {
  const currentId = useProviderStore((s) => s.currentId)
  const configs = useProviderStore((s) => s.configs)
  const setCurrent = useProviderStore((s) => s.setCurrent)
  const setConfig = useProviderStore((s) => s.setConfig)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const provider = providerById(currentId)
  const cfg = configs[provider.id] ?? {}

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6" onClick={onClose}>
      <div
        className="flex max-h-[80vh] w-[min(480px,92vw)] flex-col overflow-hidden rounded border border-[#1f2429] bg-[#0e1216] shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-center justify-between border-b border-[#1f2429] px-3 py-2">
          <span className="text-sm text-[#e6e8eb]">LLM-Quelle</span>
          <button
            className="rounded border border-[#1f2429] bg-[#13171b] px-2 py-1 text-xs hover:bg-[#1a1f24]"
            onClick={onClose}
            title="Esc"
          >✕</button>
        </div>

        <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3 text-xs">
          <label className="flex flex-col gap-1">
            <span className="text-[11px] uppercase tracking-wide text-[#7a8088]">Provider</span>
            <select
              className={INPUT}
              value={currentId}
              onChange={(e) => setCurrent(e.target.value)}
            >
              {PROVIDERS.map((p) => (
                <option key={p.id} value={p.id}>{p.label}</option>
              ))}
            </select>
          </label>

          {provider.hint && <div className="text-[11px] text-[#9aa1a8]">{provider.hint}</div>}

          {provider.kind !== 'subscription' && (
            <>
              {provider.needsKey && (
                <label className="flex flex-col gap-1">
                  <span className="text-[11px] uppercase tracking-wide text-[#7a8088]">API-Key</span>
                  <input
                    type="password"
                    className={INPUT}
                    placeholder="sk-…"
                    autoComplete="off"
                    value={cfg.apiKey ?? ''}
                    onChange={(e) => setConfig(provider.id, { apiKey: e.target.value })}
                  />
                </label>
              )}

              <label className="flex flex-col gap-1">
                <span className="text-[11px] uppercase tracking-wide text-[#7a8088]">Modell</span>
                <input
                  className={INPUT}
                  list={`models-${provider.id}`}
                  placeholder={provider.defaultModel}
                  value={cfg.model ?? ''}
                  onChange={(e) => setConfig(provider.id, { model: e.target.value })}
                />
                <datalist id={`models-${provider.id}`}>
                  {(provider.models ?? []).map((m) => <option key={m} value={m} />)}
                </datalist>
              </label>

              <label className="flex flex-col gap-1">
                <span className="text-[11px] uppercase tracking-wide text-[#7a8088]">Base-URL (optional)</span>
                <input
                  className={INPUT}
                  placeholder={provider.baseUrl ?? 'Standard'}
                  value={cfg.baseUrl ?? ''}
                  onChange={(e) => setConfig(provider.id, { baseUrl: e.target.value })}
                />
              </label>
            </>
          )}

          <div className="text-[10px] text-[#5b6168]">
            Einstellungen werden lokal gespeichert (localStorage). Keys verlassen das Gerät nicht.
          </div>
        </div>
      </div>
    </div>
  )
}
