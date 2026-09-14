import { create } from 'zustand'

// LLM source selection for the chatbot. Transport kinds map onto the
// sidecar's code paths (sidecar-llm/main.mjs):
//   opencode       → the `opencode` CLI (DEFAULT). The model is driven ONLY
//                    through the sidecar's MCP tool bridge — every operation
//                    it proposes flows through the same validation gate as
//                    the Claude paths. Model = `opencode/<model>`.
//   subscription  → claude-agent-sdk via the local `claude` CLI (OAuth, no key)
//   anthropic     → direct Anthropic Messages API (API key)
//   openai-compat → OpenAI Chat Completions API (OpenAI, Gemini, Ollama, …)
//
// Config (incl. API keys) is persisted to localStorage — same approach as the
// connections store. Keys live in the desktop app's webview storage only.

export type LlmKind = 'opencode' | 'subscription' | 'anthropic' | 'openai-compat'

export type ProviderSpec = {
  id: string
  label: string
  kind: LlmKind
  /** Model used when the user hasn't picked one. */
  defaultModel?: string
  /** Suggested models (datalist); the field stays free-text. */
  models?: string[]
  /** Built-in base URL for openai-compat providers; editable in the UI. */
  baseUrl?: string
  /** Whether an API key is required (hidden + skipped when false). */
  needsKey: boolean
  /** Short hint shown in the settings modal. */
  hint?: string
}

export const PROVIDERS: ProviderSpec[] = [
  {
    id: 'opencode',
    label: 'OpenCode',
    kind: 'opencode',
    needsKey: false,
    defaultModel: 'opencode/big-pickle',
    models: ['opencode/big-pickle', 'opencode/mimo-v2.5-free', 'opencode/ling-3.0-flash-fin-free'],
    hint: 'Standard-Provider. Lokale opencode-CLI; Modellliste wird vom CLi geladen.',
  },
  {
    id: 'claude-subscription',
    label: 'Claude (Subscription)',
    kind: 'subscription',
    needsKey: false,
    hint: 'Nutzt die lokale claude-CLI / Max-Subscription. Kein API-Key nötig.',
  },
  {
    id: 'anthropic-api',
    label: 'Anthropic API',
    kind: 'anthropic',
    needsKey: true,
    defaultModel: 'claude-opus-4-8',
    models: ['claude-opus-4-8', 'claude-sonnet-4-6', 'claude-haiku-4-5', 'claude-fable-5'],
    hint: 'Direkter API-Zugriff mit eigenem Anthropic-Key.',
  },
  {
    id: 'openai',
    label: 'OpenAI',
    kind: 'openai-compat',
    needsKey: true,
    defaultModel: 'gpt-4o',
    models: ['gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'o4-mini'],
    hint: 'OpenAI API mit eigenem Key.',
  },
  {
    id: 'gemini',
    label: 'Google Gemini',
    kind: 'openai-compat',
    needsKey: true,
    baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai/',
    defaultModel: 'gemini-2.0-flash',
    models: ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash'],
    hint: 'Gemini über den OpenAI-kompatiblen Endpoint.',
  },
  {
    id: 'ollama',
    label: 'Ollama (lokal)',
    kind: 'openai-compat',
    needsKey: false,
    baseUrl: 'http://localhost:11434/v1',
    defaultModel: 'llama3.1',
    models: ['llama3.1', 'qwen2.5', 'mistral', 'gemma2'],
    hint: 'Lokaler Ollama-Server. Modellname frei wählbar; kein Key nötig.',
  },
]

export function providerById(id: string): ProviderSpec {
  return PROVIDERS.find((p) => p.id === id) ?? PROVIDERS[0]
}

export type ProviderConfig = { apiKey?: string; model?: string; baseUrl?: string }

const STORAGE_KEY = 'spinoml.chat.provider.v1'

type Persisted = {
  currentId: string
  configs: Record<string, ProviderConfig>
}

function loadPersisted(): Persisted {
  if (typeof window === 'undefined') return { currentId: PROVIDERS[0].id, configs: {} }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return { currentId: PROVIDERS[0].id, configs: {} }
    const parsed = JSON.parse(raw)
    const currentId =
      typeof parsed?.currentId === 'string' && PROVIDERS.some((p) => p.id === parsed.currentId)
        ? parsed.currentId
        : PROVIDERS[0].id
    const configs =
      parsed?.configs && typeof parsed.configs === 'object' ? (parsed.configs as Record<string, ProviderConfig>) : {}
    return { currentId, configs }
  } catch {
    return { currentId: PROVIDERS[0].id, configs: {} }
  }
}

function persist(state: Persisted): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
  } catch {
    /* quota / private-mode — best effort */
  }
}

type State = {
  currentId: string
  configs: Record<string, ProviderConfig>
  setCurrent: (id: string) => void
  setConfig: (id: string, patch: Partial<ProviderConfig>) => void
}

const initial = loadPersisted()

export const useProviderStore = create<State>((set, get) => ({
  currentId: initial.currentId,
  configs: initial.configs,

  setCurrent: (id) => {
    set({ currentId: id })
    persist({ currentId: id, configs: get().configs })
  },

  setConfig: (id, patch) => {
    const configs = { ...get().configs, [id]: { ...get().configs[id], ...patch } }
    set({ configs })
    persist({ currentId: get().currentId, configs })
  },
}))

export type LlmRequest = {
  kind: LlmKind
  model?: string
  apiKey?: string
  baseUrl?: string
}

/** Read the active provider as a sidecar `llm` payload (non-React contexts). */
export function getCurrentLlmRequest(): LlmRequest {
  const { currentId, configs } = useProviderStore.getState()
  const p = providerById(currentId)
  if (p.kind === 'subscription') return { kind: 'subscription' }
  const cfg = configs[p.id] ?? {}
  return {
    kind: p.kind,
    model: cfg.model?.trim() || p.defaultModel,
    apiKey: cfg.apiKey?.trim() || undefined,
    baseUrl: cfg.baseUrl?.trim() || p.baseUrl || undefined,
  }
}
