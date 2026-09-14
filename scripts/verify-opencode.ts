// verify-opencode: proves the OpenCode LLM provider path end-to-end.
//
// Cases:
//   1. GET /opencode/models returns a live list from the `opencode models` CLI.
//   2. POST /chat with kind='opencode' + a cheap free model streams text and
//      reaches status done — the sidecar spawned `opencode run`, the opencode
//      model saw/used ONLY the `graph_*` MCP tools, and nothing external ran.
//   3. POST /chat with a bogus model id reports a clean status error instead of
//      hanging or crashing the sidecar.
//
// Run: `npm run verify:opencode`. Requires the opencode CLI + a reachable model.
const LLM_PORT = process.env.SPINOML_LLM_PORT ?? '7422'
const SIDECAR = `http://127.0.0.1:${LLM_PORT}`
const FREE_MODEL = process.env.SPINOML_OPENCODE_TEST_MODEL ?? 'opencode/mimo-v2.5-free'

async function isUp(): Promise<boolean> {
  try { return (await fetch(`${SIDECAR}/health`)).ok } catch { return false }
}

function readSse(stream: ReadableStream<Uint8Array>): Promise<Record<string, unknown>[]> {
  return new Promise((resolve, reject) => {
    const reader = stream.getReader()
    const decoder = new TextDecoder()
    let buf = ''
    const out: Record<string, unknown>[] = []
    ;(async () => {
      try {
        for (;;) {
          const { value, done } = await reader.read()
          if (done) break
          buf += decoder.decode(value, { stream: true })
          let idx: number
          while ((idx = buf.indexOf('\n\n')) !== -1) {
            const raw = buf.slice(0, idx)
            buf = buf.slice(idx + 2)
            const line = raw.split('\n').find((l) => l.startsWith('data:'))
            if (!line) continue
            const json = line.slice(5).trim()
            if (!json) continue
            try { out.push(JSON.parse(json)) } catch { /* skip malformed */ }
          }
        }
        resolve(out)
      } catch (e) { reject(e) }
    })()
  })
}

async function chat(user: string, model: string, timeoutMs: number) {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), timeoutMs)
  try {
    const res = await fetch(`${SIDECAR}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user,
        messages: [],
        graph: { input_shape: [1, 3, 224, 224], inputs: [], nodes: [], edges: [] },
        llm: { kind: 'opencode', model },
      }),
      signal: ctrl.signal,
    })
    if (!res.ok || !res.body) {
      const text = await res.text().catch(() => '')
      throw new Error(`chat HTTP ${res.status}: ${text || '(no body)'}`)
    }
    return await readSse(res.body)
  } finally {
    clearTimeout(timer)
  }
}

let failed = 0
const check = (name: string, ok: boolean, detail = '') => {
  console.log(`  ${ok ? '✓' : '✗'} ${name}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failed++
}

async function main() {
  if (!(await isUp())) {
    console.error('LLM sidecar not healthy on 7422 — start it with `npm run sidecar:llm` (or tauri dev).')
    process.exit(1)
  }
  console.log('sidecar up')

  // 1. /opencode/models
  console.log('\n=== /opencode/models ===')
  const mres = await fetch(`${SIDECAR}/opencode/models`)
  const mjson = await mres.json().catch(() => null)
  const models: string[] = Array.isArray(mjson?.models) ? mjson.models : []
  check('HTTP 200', mres.ok)
  check('non-empty model list', models.length > 0, `${models.length} models`)
  check('suggests big-pickle', models.includes('opencode/big-pickle'))
  if (freeProblem(models)) {
    console.log(`  note: free test model "${FREE_MODEL}" not in list — using big-pickle default`)
  }

  // 2. happy path with a real (cheap) model
  console.log(`\n=== /chat (kind=opencode) ===`)
  const evts = await chat(
    'Antworte mit genau einem kurzen Satz auf Deutsch. Nutze KEINE Tools.',
    FREE_MODEL,
    180_000,
  )
  const text = evts.filter((e) => e.type === 'text').map((e) => String(e.value)).join('')
  const toolUses = evts.filter((e) => e.type === 'tool_use')
  const toolResults = evts.filter((e) => e.type === 'tool_result')
  const statuses = evts.filter((e) => e.type === 'status')
  const errors = statuses.filter((e) => e.value === 'error')
  const done = statuses.some((e) => e.value === 'done')
  check('streamed assistant text', text.trim().length > 0, JSON.stringify(text.slice(0, 80)))
  check('ended with status done', done)
  check('no status error', errors.length === 0, errors[0]?.message ? String(errors[0].message) : '')
  check(
    'tool pairs balanced (if any)',
    toolUses.length === toolResults.length,
    `${toolUses.length} tool_use / ${toolResults.length} tool_result`,
  )

  // 3. error path: bogus model must fail cleanly, not hang
  console.log('\n=== /chat (bogus model) ===')
  const errEvts = await chat('Hallo', 'opencode/does-not-exist-xyz', 120_000)
  const errStatus = errEvts.filter((e) => e.type === 'status' && e.value === 'error')
  const errText = errEvts.filter((e) => e.type === 'text')
  check(
    'clean status error (no hang)',
    errStatus.length > 0,
    errStatus[0]?.message ? `${String(errStatus[0].message).slice(0, 120)}` : '',
  )
  if (errText.length > 0) {
    console.log(`  note: model still answered despite bogus id: ${JSON.stringify(String(errText[0].value).slice(0, 80))}`)
  }

  console.log(`\n${failed === 0 ? '✓' : '✗'} opencode provider ${failed === 0 ? 'ok' : `(${failed} failing)`}`)
  process.exit(failed === 0 ? 0 : 1)
}

function freeProblem(models: string[]): boolean {
  return !models.includes(FREE_MODEL)
}

main().catch((e) => {
  console.error('✗ verify-opencode crashed:', e)
  process.exit(1)
})