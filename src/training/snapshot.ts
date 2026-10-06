// Phase 20 — immutable launch-time snapshot. startRun/startEvalRun compute
// content-hashes of the TWO artifacts the experiment actually executes (the
// model.spinoml graph + the generated model.py) plus the preprocessing
// (DataOp script) steps baked into the graph, and freeze them into run.json
// under `snapshot`. train.py then re-hashes the RUN-DIR copies before any
// training and refuses to start if they drifted — so the running experiment
// is provably the launch-time bytes, never mutable UI state.

import type { GraphSnapshot } from '../canvas/GraphStore'
import { parseFile } from '../persistence/file'
import type { PreprocessingStep, RunSnapshot } from './types'
import { collectCodeBlobs } from '../trust/codeBlobs'
import { buildCodeTrustManifest } from '../trust/gate'
import { trust } from '../trust/trustStore'

/** Plain SHA-256 of the UTF-8 bytes — must match train.py's hashlib.sha256
 *  over the RUN-DIR file bytes. */
export async function sha256Hex(text: string): Promise<string> {
  const data = new TextEncoder().encode(text)
  const buf = await globalThis.crypto.subtle.digest('SHA-256', data)
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('')
}

function extractPreprocessing(graph: GraphSnapshot): PreprocessingStep[] {
  const steps: PreprocessingStep[] = []
  for (const node of graph.nodes) {
    if (node.layerType !== 'DataOp') continue
    const p = node.params ?? {}
    const script = typeof p.script === 'string' ? p.script : ''
    if (!script) continue
    steps.push({
      node: node.id,
      script,
      input_dataset: String(p.input_dataset ?? ''),
      output_name: String(p.output_name ?? ''),
      mode: p.mode === 'slurm' ? 'slurm' : 'shell',
      cache: p.cache !== false,
    })
  }
  return steps
}

/** The frozen snapshot for a run being launched. `modelSpinoml`/`modelPy` are
 *  the EXACT strings also handed to training.start (→ written verbatim into the
 *  run dir), so their hashes prove executor bytes == launch bytes. */
export async function buildRunSnapshot(
  modelSpinoml: string,
  modelPy: string,
): Promise<RunSnapshot> {
  let graph: GraphSnapshot | null = null
  try { graph = parseFile(modelSpinoml) } catch { graph = null }
  const [graph_sha256, model_py_sha256] = await Promise.all([
    sha256Hex(modelSpinoml),
    sha256Hex(modelPy),
  ])
  const blobs = graph ? collectCodeBlobs(graph.nodes) : []
  const code_trust = await buildCodeTrustManifest(blobs, (sha) => trust.get(sha))
  return {
    version: 1,
    graph_sha256,
    model_py_sha256,
    preprocessing: graph ? extractPreprocessing(graph) : [],
    code_trust,
  }
}