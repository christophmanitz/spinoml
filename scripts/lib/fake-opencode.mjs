// Fake `opencode` CLI for scripts/test-opencode-lifecycle.ts.
//
// The real LLM sidecar spawns `OPENCODE_BIN` (default `opencode`) with
// `OPENCODE_CONFIG_CONTENT` carrying an inline `mcp.graph` entry whose
// `command` is [node, mcp-bridge.mjs, requestId] and whose `environment` holds
// the per-turn MCP session secret. This script plays that CLI: it reads the
// config, spawns the REAL bridge with `{...process.env, ...mcp.graph.environment}`
// (exactly what real opencode does), speaks MCP over the bridge's stdio
// (initialize / tools/list / tools/call), and prints opencode-format NDJSON
// event lines on stdout. Behaviour is selected by FAKE_OPENCODE_SCENARIO (JSON).
//
// No network, no API key, no real opencode. It exits on SIGTERM and kills its
// bridge child so the lifecycle tests can assert no process is left behind.

import { spawn } from 'node:child_process'
import { writeFileSync } from 'node:fs'

function parseObject(raw) {
  if (typeof raw !== 'string' || raw === '') return {}
  try {
    const v = JSON.parse(raw)
    return v && typeof v === 'object' && !Array.isArray(v) ? v : {}
  } catch {
    return {}
  }
}

const scenario = parseObject(process.env.FAKE_OPENCODE_SCENARIO)
const name =
  typeof scenario.scenario === 'string'
    ? scenario.scenario
    : typeof scenario.name === 'string'
      ? scenario.name
      : 'silent'

const config = parseObject(process.env.OPENCODE_CONFIG_CONTENT)
const graphEntry =
  config.mcp && typeof config.mcp === 'object' && config.mcp.graph && typeof config.mcp.graph === 'object'
    ? config.mcp.graph
    : {}
const command = Array.isArray(graphEntry.command) ? graphEntry.command.map(String) : []
const environment =
  graphEntry.environment && typeof graphEntry.environment === 'object' ? graphEntry.environment : {}
const requestId = typeof command[2] === 'string' ? command[2] : ''
const mcpSecret = typeof environment.SPINOML_MCP_SECRET === 'string' ? environment.SPINOML_MCP_SECRET : ''

let bridgeChild = null

function mergedBridgeEnv() {
  return { ...process.env, ...environment }
}

function dump(extra) {
  if (typeof scenario.dump !== 'string' || scenario.dump === '') return
  const bridgeEnv = mergedBridgeEnv()
  const payload = {
    scenario: name,
    requestId,
    selfPid: process.pid,
    childPid: bridgeChild && bridgeChild.pid ? bridgeChild.pid : null,
    fakeEnvHasMasterToken: Object.prototype.hasOwnProperty.call(process.env, 'SPINOML_SIDECAR_TOKEN'),
    fakeEnvKeys: Object.keys(process.env).sort(),
    fakeArgv: process.argv.slice(),
    bridgeArgv: command.slice(),
    fakeSecretInArgv: mcpSecret !== '' && process.argv.slice().some((a) => a === mcpSecret),
    bridgeSecretInArgv: mcpSecret !== '' && command.some((a) => a === mcpSecret),
    bridgeEnvHasSecret: typeof environment.SPINOML_MCP_SECRET === 'string' && environment.SPINOML_MCP_SECRET !== '',
    bridgeEnvHasMasterToken: Object.prototype.hasOwnProperty.call(bridgeEnv, 'SPINOML_SIDECAR_TOKEN'),
    ...(extra || {}),
  }
  // The dump only records KEY NAMES and a boolean for the master token — never
  // the token value itself. Best-effort: a missing dir must not crash the fake.
  try {
    writeFileSync(scenario.dump, JSON.stringify(payload))
  } catch {
    // best-effort marker file
  }
}

function writePidFile() {
  if (typeof scenario.pidFile !== 'string' || scenario.pidFile === '') return
  const payload = { pid: process.pid, childPid: bridgeChild && bridgeChild.pid ? bridgeChild.pid : null }
  try {
    writeFileSync(scenario.pidFile, JSON.stringify(payload))
  } catch {
    // best-effort marker file
  }
}

let shuttingDown = false
function shutdown(code) {
  if (shuttingDown) return
  shuttingDown = true
  const child = bridgeChild
  if (!child || child.exitCode !== null) {
    process.exit(code)
    return
  }
  try {
    child.stdin.end()
  } catch {
    // already closed
  }
  try {
    child.kill('SIGTERM')
  } catch {
    // already gone
  }
  const hard = setTimeout(() => {
    try {
      if (child.exitCode === null) child.kill('SIGKILL')
    } catch {
      // already gone
    }
    process.exit(code)
  }, 750)
  hard.unref()
  child.once('exit', () => process.exit(code))
  setTimeout(() => process.exit(code), 2000).unref()
}

process.on('SIGTERM', () => shutdown(0))
process.on('SIGINT', () => shutdown(0))
process.on('SIGHUP', () => shutdown(0))

function spawnBridge(mutateEnv) {
  if (command.length === 0) return null
  const env = mergedBridgeEnv()
  if (typeof mutateEnv === 'function') mutateEnv(env)
  const child = spawn(command[0], command.slice(1), { env, stdio: ['pipe', 'pipe', 'pipe'] })
  child.stderr.on('data', () => {
    // drain — a full pipe would otherwise wedge the bridge
  })
  child.on('error', () => {
    // surfaced as an MCP timeout / missing tool result
  })
  bridgeChild = child
  return child
}

function mcpRequest(child, id, method, params, timeoutMs) {
  return new Promise((resolve) => {
    if (!child || !child.stdin || child.stdin.destroyed) {
      resolve(null)
      return
    }
    let buf = ''
    let settled = false
    const finish = (v) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      child.stdout.removeListener('data', onData)
      resolve(v)
    }
    const onData = (d) => {
      buf += d.toString()
      let idx
      while ((idx = buf.indexOf('\n')) !== -1) {
        const line = buf.slice(0, idx).trim()
        buf = buf.slice(idx + 1)
        if (!line) continue
        let msg
        try {
          msg = JSON.parse(line)
        } catch {
          continue
        }
        if (msg && msg.id === id) {
          finish(msg)
          return
        }
      }
    }
    const timer = setTimeout(() => finish(null), typeof timeoutMs === 'number' ? timeoutMs : 10000)
    child.stdout.on('data', onData)
    try {
      child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n')
    } catch {
      finish(null)
    }
  })
}

function mcpNotify(child, method, params) {
  if (!child || !child.stdin || child.stdin.destroyed) return
  try {
    child.stdin.write(JSON.stringify({ jsonrpc: '2.0', method, params }) + '\n')
  } catch {
    // closed
  }
}

async function runToolFlow(child, tool, args) {
  await mcpRequest(child, 1, 'initialize', {
    protocolVersion: '2025-03-26',
    capabilities: {},
    clientInfo: { name: 'fake-opencode', version: '0.0.0' },
  })
  mcpNotify(child, 'notifications/initialized', {})
  await mcpRequest(child, 2, 'tools/list', {})
  return await mcpRequest(child, 3, 'tools/call', { name: tool, arguments: args })
}

function resultOutput(res) {
  const content = res && res.result && res.result.content
  if (Array.isArray(content)) {
    return content.map((c) => (c && typeof c.text === 'string' ? c.text : '')).join('')
  }
  if (res && res.error && typeof res.error.message === 'string') return res.error.message
  return ''
}

function resultIsError(res) {
  return !!(res && (res.result && res.result.isError === true || res.error != null))
}

function emit(ev) {
  process.stdout.write(JSON.stringify(ev) + '\n')
}

async function main() {
  if (name === 'dump-env') {
    dump()
    return shutdown(0)
  }

  if (name === 'exit-nonzero') {
    process.stderr.write('fake-opencode: exit-nonzero stderr tail line\n')
    return shutdown(3)
  }

  if (name === 'error-event') {
    emit({ type: 'error', error: { message: 'fake-error-message' } })
    return shutdown(1)
  }

  if (name === 'garbage') {
    process.stdout.write('this is not json\n{ also broken\n')
    emit({ type: 'text', part: { text: 'garbage-ok' } })
    return shutdown(0)
  }

  if (name === 'silent') {
    spawnBridge()
    dump()
    writePidFile()
    // Keep the process (and the bridge child) alive until SIGTERM/SIGINT.
    setInterval(() => {}, 1000)
    return
  }

  let child = null
  if (name === 'bridge-no-secret') {
    child = spawnBridge((env) => {
      delete env.SPINOML_MCP_SECRET
    })
  } else if (name === 'bridge-wrong-secret') {
    child = spawnBridge((env) => {
      env.SPINOML_MCP_SECRET = 'x'.repeat(64)
    })
  } else {
    child = spawnBridge()
  }
  dump()
  writePidFile()

  const tool = typeof scenario.tool === 'string' ? scenario.tool : 'graph_add_layer'
  const args =
    scenario.args && typeof scenario.args === 'object'
      ? scenario.args
      : { layer_type: 'ReLU', after: 'fc' }

  const res = await runToolFlow(child, tool, args)
  const output = resultOutput(res)
  const isError = resultIsError(res)

  emit({ type: 'step_start', part: {} })
  emit({
    type: 'tool_use',
    part: {
      tool,
      callID: 'call_1',
      state: { status: isError ? 'error' : 'completed', input: args, output },
    },
  })
  emit({ type: 'step_finish', part: { reason: 'stop' } })
  emit({ type: 'text', part: { text: output || 'done' } })
  return shutdown(0)
}

main().catch((e) => {
  process.stderr.write(`fake-opencode: ${e && e.message ? e.message : String(e)}\n`)
  shutdown(1)
})
