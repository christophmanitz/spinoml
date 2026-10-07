#!/usr/bin/env tsx
// SpinoML — CUDA hardware suite wrapper (Node → Python).
//
// What npm run test:hardware-cuda invokes. Resolves Python (honouring PYTHON
// env var like other suites), spawns scripts/hardware-cuda.py with pass-through
// stdio, exits with the python process's exit code. Pure node builtins.

import { spawn, type SpawnOptions } from 'node:child_process'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const SCRIPT = join(HERE, 'hardware-cuda.py')

/**
 * Resolve the Python invocation. Order:
 *   1. process.env.PYTHON (explicit override; e.g. PYTHON=python3.11 or /abs/path)
 *   2. `python` — the active conda env's python (the standard pattern for verify-*)
 *   3. `conda run --no-capture-output -n mlforge-dev python` — last-resort fallback
 *      mirroring scripts/verify-reference.ts.
 *
 * Returns { cmd, args } so the caller can wire stdio 1:1 to the parent's.
 */
function resolvePython(): { cmd: string; args: string[] } {
  if (process.env.PYTHON && process.env.PYTHON.trim()) {
    return { cmd: process.env.PYTHON, args: [] }
  }
  return { cmd: 'python', args: [] }
}

async function main(): Promise<number> {
  const extraArgs = process.argv.slice(2)
  const { cmd, args } = resolvePython()
  const fullArgs = [...args, SCRIPT, ...extraArgs]

  const opts: SpawnOptions = {
    stdio: 'inherit',
    env: process.env,
  }

  return await new Promise<number>((resolve) => {
    const child = spawn(cmd, fullArgs, opts)
    child.on('error', (err: NodeJS.ErrnoException) => {
      // `code === 'ENOENT'` means the binary itself wasn't found. On CPU nodes
      // `python` is on PATH via the conda env; on GPU nodes the user/CI must
      // arrange that (or set PYTHON). Don't swallow — surface the error verbatim.
      console.error(`hardware-cuda: failed to spawn ${cmd}: ${err.message}`)
      if (err.code === 'ENOENT' && !process.env.PYTHON) {
        console.error('  hint: set PYTHON=/path/to/python or run inside the mlforge-dev conda env')
      }
      resolve(127)
    })
    child.on('close', (code) => {
      resolve(code ?? 1)
    })
  })
}

const exitCode = await main()
process.exit(exitCode)
