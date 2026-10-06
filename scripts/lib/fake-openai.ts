// Fake OpenAI-compatible server for the LLM-safety harness.
//
// It speaks just enough of `POST /v1/chat/completions` (streaming SSE) for the
// official `openai` npm SDK, driven by a per-test SCRIPT so a test can make the
// "model" do exactly one thing per request: emit text, emit one or more tool
// calls, return an HTTP error, send a non-SSE body, truncate the stream, delay,
// or hang forever.
//
// It records every request (headers incl. Authorization + the full message
// list) so a test can assert what the sidecar actually sent back to the model,
// and it exposes the contents of `role:"tool"` messages in order — the tool
// results the real sidecar produced.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import { setTimeout as delay } from 'node:timers/promises'

export interface ToolCallSpec {
  name: string
  /** Raw JSON argument string exactly as the provider would stream it. */
  arguments: string
}

export interface FakeStep {
  /** Assistant text (content deltas). */
  text?: string
  /** Tool calls to stream. Multiple entries = one assistant turn, executed in order. */
  toolCalls?: ToolCallSpec[]
  /** HTTP error status. Combined with `body` for the literal error body. */
  status?: number
  /** Literal response body for `status` or `raw`. */
  body?: string
  /** Literal body sent as text/event-stream (may be intentionally malformed). */
  raw?: string
  /** Write the SSE stream up to this many bytes, then destroy the socket. */
  truncateAfterBytes?: number
  /** Wait this long before responding (simulates a slow provider). */
  delayMs?: number
  /** Never respond; the connection stays open. */
  hang?: boolean
  /** Send a single SSE chunk, then keep the socket open without ever finishing. */
  stall?: boolean
  /** Keep answering this same step for every subsequent request. */
  repeat?: boolean
}

export interface RecordedRequest {
  headers: Record<string, string | string[] | undefined>
  body: unknown
  raw: string
}

interface ChatMessage {
  role?: unknown
  content?: unknown
}

export class FakeOpenAI {
  private readonly server: Server
  readonly port: number
  readonly requests: RecordedRequest[] = []
  private steps: FakeStep[] = []
  private index = 0
  private loop = false
  /** Number of requests that were told to hang. */
  hangStarts = 0
  /** Number of hanging requests whose client socket has since closed. */
  hangCloses = 0

  private constructor(server: Server, port: number) {
    this.server = server
    this.port = port
  }

  static async start(): Promise<FakeOpenAI> {
    const server = createServer()
    const port = await new Promise<number>((resolve, reject) => {
      server.once('error', reject)
      server.listen(0, '127.0.0.1', () => {
        const addr = server.address()
        if (addr && typeof addr === 'object') resolve(addr.port)
        else reject(new Error('no port'))
      })
    })
    const fake = new FakeOpenAI(server, port)
    server.on('request', (req, res) => {
      void fake.handle(req, res)
    })
    return fake
  }

  /** Replace the script; clears recorded requests and resets the cursor. */
  setScript(steps: FakeStep[], opts: { loop?: boolean } = {}): void {
    this.steps = steps
    this.index = 0
    this.loop = opts.loop ?? false
    this.requests.length = 0
    this.hangStarts = 0
    this.hangCloses = 0
  }

  get baseUrl(): string {
    return `http://127.0.0.1:${this.port}/v1`
  }

  authHeaders(): string[] {
    return this.requests
      .map((r) => (typeof r.headers.authorization === 'string' ? r.headers.authorization : ''))
      .filter((h) => h.length > 0)
  }

  /** Contents of every `role:"tool"` message the sidecar sent back, in order. */
  toolResultsSeen(): string[] {
    const out: string[] = []
    for (const req of this.requests) {
      const body = req.body as { messages?: unknown } | null
      const messages = Array.isArray(body?.messages) ? body.messages : []
      for (const m of messages as ChatMessage[]) {
        if (m.role === 'tool') out.push(typeof m.content === 'string' ? m.content : JSON.stringify(m.content ?? ''))
      }
    }
    return out
  }

  async waitForHungClose(timeoutMs: number): Promise<boolean> {
    const deadline = Date.now() + timeoutMs
    while (Date.now() < deadline) {
      if (this.hangStarts > 0 && this.hangCloses > 0) return true
      await delay(50)
    }
    return this.hangCloses > 0
  }

  async close(): Promise<void> {
    if (typeof this.server.closeAllConnections === 'function') this.server.closeAllConnections()
    await new Promise<void>((resolve) => this.server.close(() => resolve()))
  }

  private nextStep(): FakeStep {
    if (this.steps.length === 0) return { text: 'done' }
    if (this.index >= this.steps.length) {
      if (!this.loop) return { text: 'done' }
      this.index = 0
    }
    const step = this.steps[this.index]
    if (!step.repeat) this.index++
    return step
  }

  private async handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const url = req.url ?? ''
    if (req.method !== 'POST' || !url.endsWith('/chat/completions')) {
      res.writeHead(404, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ error: { message: 'not found' } }))
      return
    }

    let raw = ''
    for await (const chunk of req) raw += chunk.toString()
    let body: unknown
    try {
      body = JSON.parse(raw)
    } catch {
      body = null
    }
    this.requests.push({ headers: req.headers, body, raw })

    const step = this.nextStep()

    if (step.hang) {
      this.hangStarts++
      // The request body has already been consumed, so `req`'s own 'close' may
      // never fire on a client abort; watch the response and the raw socket too
      // (counted once).
      let closed = false
      const markClose = (): void => {
        if (!closed) {
          closed = true
          this.hangCloses++
        }
      }
      res.on('close', markClose)
      req.on('close', markClose)
      req.socket?.on('close', markClose)
      return
    }

    if (step.delayMs && step.delayMs > 0) await delay(step.delayMs)

    if (step.status !== undefined) {
      const payload = step.body ?? JSON.stringify({ error: { message: 'fake provider error', type: 'invalid_request_error' } })
      res.writeHead(step.status, { 'Content-Type': 'application/json' })
      res.end(payload)
      return
    }

    if (step.raw !== undefined) {
      res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' })
      res.end(step.raw)
      return
    }

    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    })

    if (step.stall) {
      // One chunk (the assistant role delta), then hold the connection open with
      // no finish_reason — the sidecar's idle timeout must fire.
      const frames = this.buildFrames({})
      if (frames.length > 0) res.write(frames[0])
      return
    }

    const frames = this.buildFrames(step)
    if (step.truncateAfterBytes !== undefined) {
      let written = 0
      for (const frame of frames) {
        const remaining = step.truncateAfterBytes - written
        if (remaining <= 0) break
        if (frame.length > remaining) {
          res.write(frame.slice(0, remaining))
          break
        }
        res.write(frame)
        written += frame.length
      }
      res.socket?.destroy()
      return
    }
    for (const frame of frames) res.write(frame)
    res.end()
  }

  private buildFrames(step: FakeStep): string[] {
    const base = { id: 'chatcmpl-fake', object: 'chat.completion.chunk', created: 1700000000, model: 'fake' }
    const frame = (delta: Record<string, unknown>, finish: string | null): string =>
      `data: ${JSON.stringify({ ...base, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`

    const frames: string[] = [frame({ role: 'assistant' }, null)]
    if (step.text) {
      for (const part of step.text.match(/[\s\S]{1,16}/g) ?? [step.text]) frames.push(frame({ content: part }, null))
    }
    if (step.toolCalls && step.toolCalls.length > 0) {
      step.toolCalls.forEach((tc, i) => {
        frames.push(frame({ tool_calls: [{ index: i, id: `call_${i + 1}`, type: 'function', function: { name: tc.name, arguments: '' } }] }, null))
      })
      step.toolCalls.forEach((tc, i) => {
        const half = Math.ceil(tc.arguments.length / 2)
        const first = tc.arguments.slice(0, half)
        const second = tc.arguments.slice(half)
        if (first.length > 0) frames.push(frame({ tool_calls: [{ index: i, function: { arguments: first } }] }, null))
        if (second.length > 0) frames.push(frame({ tool_calls: [{ index: i, function: { arguments: second } }] }, null))
      })
      frames.push(frame({}, 'tool_calls'))
    } else {
      frames.push(frame({}, 'stop'))
    }
    return frames
  }
}
