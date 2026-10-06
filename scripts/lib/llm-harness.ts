// Harness for driving the REAL LLM sidecar against the fake OpenAI server.
//
// Spawns `node sidecar-llm/main.mjs` on a throw-away free port, waits for
// /health, then POSTs /chat exactly like the frontend (kind:"openai-compat",
// baseUrl pointing at the fake server) and parses the SSE response into typed
// event lists. It also supports answering `ask` events via POST /respond and
// aborting an in-flight /chat request.

import { spawn, type ChildProcessByStdio } from 'node:child_process'
import { createServer } from 'node:net'
import type { Readable } from 'node:stream'
import { setTimeout as delay } from 'node:timers/promises'
import { FakeOpenAI, type FakeStep } from './fake-openai.ts'

export interface GraphNode {
  id: string
  layerType: string
  params: Record<string, unknown>
}

export interface GraphSnapshot {
  input_shape: number[]
  nodes: GraphNode[]
  edges: { source: string; target: string }[]
}

export interface LlmConfig {
  kind: string
  apiKey: string
  baseUrl: string
  model: string
}

export interface ChatOptions {
  script: FakeStep[]
  graph?: GraphSnapshot
  llm?: Partial<LlmConfig>
  project?: { root: string; ssh_target?: string } | null
  signal?: AbortSignal
  onAsk?: (ask: AskEvent) => string | boolean | Promise<string | boolean>
  timeoutMs?: number
  autoMode?: boolean
  loop?: boolean
  user?: string
  // Phase 78: forward as X-SpinoML-Token on the /chat POST (and /respond),
  // so tests can drive a sidecar started in mode `token`. No token by default
  // — the existing unauthenticated-dev tests stay green.
  token?: string
  // Phase 78: forward as Origin (exercises the Host/Origin gate).
  origin?: string
}

export interface AskEvent {
  id: string
  kind: string
  prompt: string
  payload: Record<string, unknown>
}

export interface ChatResult {
  events: Record<string, unknown>[]
  actions: { op: string; payload: Record<string, unknown> }[]
  toolResults: { id: string; ok: boolean; result: string; error?: string }[]
  toolUses: { id: string; name: string; args: Record<string, unknown> }[]
  statuses: { value: string; message?: string }[]
  asks: AskEvent[]
  texts: string[]
  ended: boolean
  aborted: boolean
  fetchError?: string
  elapsedMs: number
}

export const BASE_GRAPH: GraphSnapshot = {
  input_shape: [1, 4],
  nodes: [
    { id: 'in', layerType: 'Input', params: { name: 'x', shape: [1, 4], dtype: 'float32' } },
    { id: 'fc', layerType: 'Linear', params: { in_features: 4, out_features: 2 } },
  ],
  edges: [{ source: 'in', target: 'fc' }],
}

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = createServer()
    srv.once('error', reject)
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address()
      if (addr && typeof addr === 'object') {
        const p = addr.port
        srv.close(() => resolve(p))
      } else {
        srv.close(() => reject(new Error('no free port')))
      }
    })
  })
}

function str(v: unknown): string {
  return typeof v === 'string' ? v : ''
}

function asRecord(v: unknown): Record<string, unknown> {
  return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : {}
}

export class LlmHarness {
  readonly port: number
  readonly fake: FakeOpenAI
  private readonly child: ChildProcessByStdio<null, Readable, Readable>
  stdout = ''
  stderr = ''
  private readonly base: string

  private constructor(port: number, fake: FakeOpenAI, child: ChildProcessByStdio<null, Readable, Readable>) {
    this.port = port
    this.fake = fake
    this.child = child
    this.base = `http://127.0.0.1:${port}`
    child.stdout.on('data', (d: Buffer) => {
      this.stdout += d.toString()
    })
    child.stderr.on('data', (d: Buffer) => {
      this.stderr += d.toString()
    })
  }

  static async start(opts: { env?: Record<string, string> } = {}): Promise<LlmHarness> {
    const fake = await FakeOpenAI.start()
    const port = await freePort()
    const child = spawn('node', ['sidecar-llm/main.mjs'], {
      cwd: process.cwd(),
      env: { ...process.env, SPINOML_LLM_PORT: String(port), ...opts.env },
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    const h = new LlmHarness(port, fake, child)
    const healthy = await h.waitHealthy(15000)
    if (!healthy) {
      const stderr = h.stderr
      await h.stop()
      throw new Error(`sidecar-llm did not become healthy on port ${port}.\nstderr:\n${stderr}`)
    }
    return h
  }

  // Phase 78: connect to a sidecar the caller already started (e.g. via
  // scripts/lib/auth-probe's startSidecar) and pair it with a freshly-spun
  // fake OpenAI server. The harness only owns the fake server's lifecycle
  // from this point on; stop() tears it down. No new sidecar is spawned.
  static async connect(opts: { baseUrl: string }): Promise<LlmHarness> {
    const fake = await FakeOpenAI.start()
    const h = Object.create(LlmHarness.prototype) as LlmHarness
    ;(h as unknown as { fake: FakeOpenAI }).fake = fake
    ;(h as unknown as { port: number }).port = 0
    ;(h as unknown as { base: string }).base = opts.baseUrl
    ;(h as unknown as { stdout: string }).stdout = ''
    ;(h as unknown as { stderr: string }).stderr = ''
    return h
  }

  get baseUrl(): string {
    return this.base
  }

  async waitHealthy(timeoutMs: number): Promise<boolean> {
    const deadline = Date.now() + timeoutMs
    while (Date.now() < deadline) {
      if (this.child.exitCode !== null) return false
      if (await this.health()) return true
      await delay(150)
    }
    return false
  }

  async health(): Promise<boolean> {
    try {
      const r = await fetch(`${this.base}/health`)
      return r.ok
    } catch {
      return false
    }
  }

  async healthBody(): Promise<string> {
    try {
      const r = await fetch(`${this.base}/health`)
      return await r.text()
    } catch {
      return ''
    }
  }

  async opencodeModelsBody(): Promise<string> {
    try {
      const r = await fetch(`${this.base}/opencode/models`)
      return await r.text()
    } catch {
      return ''
    }
  }

  async respond(askId: string, answer: string | boolean, token?: string): Promise<boolean> {
    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      if (typeof token === 'string' && token.length > 0) headers['X-SpinoML-Token'] = token
      const r = await fetch(`${this.base}/respond`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ askId, answer }),
      })
      return r.ok
    } catch {
      return false
    }
  }

  async chat(opts: ChatOptions): Promise<ChatResult> {
    this.fake.setScript(opts.script, { loop: opts.loop ?? false })
    const apiKey = opts.llm?.apiKey ?? `sk-TEST-${Math.random().toString(36).slice(2)}`
    const body = {
      user: opts.user ?? 'test turn',
      messages: [],
      graph: opts.graph ?? BASE_GRAPH,
      training_graph: { nodes: [], edges: [] },
      data_graph: { nodes: [], edges: [] },
      project: opts.project ?? null,
      autoMode: opts.autoMode ?? false,
      llm: {
        kind: 'openai-compat',
        apiKey,
        baseUrl: this.fake.baseUrl,
        model: 'fake',
        ...opts.llm,
      },
    }

    const start = Date.now()
    const signals: AbortSignal[] = [AbortSignal.timeout(opts.timeoutMs ?? 20000)]
    if (opts.signal) signals.push(opts.signal)
    const signal = AbortSignal.any(signals)

    const result: ChatResult = {
      events: [],
      actions: [],
      toolResults: [],
      toolUses: [],
      statuses: [],
      asks: [],
      texts: [],
      ended: false,
      aborted: false,
      elapsedMs: 0,
    }
    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      if (typeof opts.token === 'string' && opts.token.length > 0) {
        headers['X-SpinoML-Token'] = opts.token
      }
      if (typeof opts.origin === 'string' && opts.origin.length > 0) {
        headers.Origin = opts.origin
      }
      const res = await fetch(`${this.base}/chat`, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal,
      })
      if (!res.body) throw new Error(`no response body (HTTP ${res.status})`)
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''
      for (;;) {
        const { value, done } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        let idx: number
        while ((idx = buf.indexOf('\n\n')) !== -1) {
          const frame = buf.slice(0, idx)
          buf = buf.slice(idx + 2)
          const line = frame.split('\n').find((l) => l.startsWith('data:'))
          if (!line) continue
          const json = line.slice(5).trim()
          if (!json) continue
          let ev: Record<string, unknown>
          try {
            ev = asRecord(JSON.parse(json))
          } catch {
            continue
          }
          result.events.push(ev)
          this.applyEvent(ev, result, opts)
        }
      }
    } catch (e) {
      if (e instanceof Error && e.name === 'AbortError') result.aborted = true
      else if (signal.aborted) result.aborted = true
      else result.fetchError = e instanceof Error ? `${e.name}: ${e.message}` : String(e)
    }
    result.elapsedMs = Date.now() - start
    return result
  }

  private applyEvent(ev: Record<string, unknown>, result: ChatResult, opts: ChatOptions): void {
    const type = str(ev.type)
    if (type === 'action') {
      result.actions.push({ op: str(ev.op), payload: asRecord(ev.payload) })
    } else if (type === 'tool_use') {
      result.toolUses.push({ id: str(ev.id), name: str(ev.name), args: asRecord(ev.args) })
    } else if (type === 'tool_result') {
      result.toolResults.push({
        id: str(ev.id),
        ok: ev.ok === true,
        result: str(ev.result),
        error: typeof ev.error === 'string' ? ev.error : undefined,
      })
    } else if (type === 'status') {
      result.statuses.push({
        value: str(ev.value),
        message: typeof ev.message === 'string' ? ev.message : undefined,
      })
    } else if (type === 'text') {
      result.texts.push(str(ev.value))
    } else if (type === 'ask') {
      const ask: AskEvent = { id: str(ev.id), kind: str(ev.kind), prompt: str(ev.prompt), payload: asRecord(ev.payload) }
      result.asks.push(ask)
      if (opts.onAsk) {
        void Promise.resolve(opts.onAsk(ask)).then((answer) => {
          if (answer !== undefined) void this.respond(ask.id, answer)
        })
      }
    } else if (type === 'done') {
      result.ended = true
    }
  }

  async stop(): Promise<void> {
    // connect() doesn't spawn a sidecar — only the fake server has lifecycle.
    if (this.child && this.child.exitCode === null) {
      this.child.kill('SIGTERM')
      const exited = await Promise.race([
        new Promise<boolean>((resolve) => this.child.once('exit', () => resolve(true))),
        delay(3000).then(() => false as const),
      ])
      if (!exited && this.child.exitCode === null) this.child.kill('SIGKILL')
    }
    await this.fake.close()
  }
}
