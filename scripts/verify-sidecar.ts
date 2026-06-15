import { spawn } from 'node:child_process'
import { setTimeout as wait } from 'node:timers/promises'
import type { Edge } from '@xyflow/react'
import { generate } from '../src/codegen/generator'
import { defaultParamsFor } from '../src/layers/registry'
import type { LayerNode } from '../src/canvas/GraphStore'

function mkNode(id: string, layerType: string, overrides: Record<string, unknown> = {}): LayerNode {
  return {
    id,
    type: 'layer',
    position: { x: 0, y: 0 },
    data: { layerType, params: { ...defaultParamsFor(layerType), ...overrides } },
  }
}

type Case = {
  name: string
  nodes: LayerNode[]
  edges: Edge[]
  expectOk: boolean
  expectShapeFor?: { id: string; shape: number[] }
  expectErrorContains?: string
}

const cases: Case[] = [
  {
    name: 'sequential CNN',
    nodes: [
      mkNode('input', 'Input', { shape: [1, 3, 32, 32] }),
      mkNode('n1', 'Conv2d', { in_channels: 3, out_channels: 16 }),
      mkNode('n2', 'BatchNorm2d', { num_features: 16 }),
      mkNode('n3', 'ReLU'),
      mkNode('n4', 'AdaptiveAvgPool2d', { output_size: [1, 1] }),
      mkNode('n5', 'Flatten'),
      mkNode('n6', 'Linear', { in_features: 16, out_features: 10 }),
    ],
    edges: [
      { id: 'e1', source: 'input', target: 'n1' },
      { id: 'e2', source: 'n1', target: 'n2' },
      { id: 'e3', source: 'n2', target: 'n3' },
      { id: 'e4', source: 'n3', target: 'n4' },
      { id: 'e5', source: 'n4', target: 'n5' },
      { id: 'e6', source: 'n5', target: 'n6' },
    ],
    expectOk: true,
    expectShapeFor: { id: 'n6', shape: [1, 10] },
  },
  {
    name: 'channel mismatch (expect forward error)',
    nodes: [
      mkNode('input', 'Input', { shape: [1, 3, 32, 32] }),
      mkNode('n1', 'Conv2d', { in_channels: 3, out_channels: 16 }),
      mkNode('n2', 'Conv2d', { in_channels: 99, out_channels: 32 }),
    ],
    edges: [
      { id: 'e1', source: 'input', target: 'n1' },
      { id: 'e2', source: 'n1', target: 'n2' },
    ],
    expectOk: false,
    expectErrorContains: 'channels',
  },
  {
    name: 'lstm sequence (tuple output unpacked)',
    nodes: [
      mkNode('input', 'Input', { shape: [1, 16, 32] }),
      mkNode('n1', 'LSTM', { input_size: 32, hidden_size: 64, num_layers: 1, batch_first: true }),
      mkNode('n2', 'Linear', { in_features: 64, out_features: 5 }),
    ],
    edges: [
      { id: 'e1', source: 'input', target: 'n1' },
      { id: 'e2', source: 'n1', target: 'n2' },
    ],
    expectOk: true,
    expectShapeFor: { id: 'n2', shape: [1, 16, 5] },
  },
  {
    name: 'embedding from int64 token ids',
    nodes: [
      mkNode('input', 'Input', { shape: [1, 16], dtype: 'int64' }),
      mkNode('n1', 'Embedding', { num_embeddings: 1000, embedding_dim: 64 }),
    ],
    edges: [
      { id: 'e1', source: 'input', target: 'n1' },
    ],
    expectOk: true,
    expectShapeFor: { id: 'n1', shape: [1, 16, 64] },
  },
]

const SIDECAR = 'http://127.0.0.1:7421'

async function isUp(): Promise<boolean> {
  try {
    const r = await fetch(`${SIDECAR}/health`)
    return r.ok
  } catch {
    return false
  }
}

async function infer(code: string, inputShapes: number[][], inputDtypes: string[]) {
  const r = await fetch(`${SIDECAR}/infer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code, input_shapes: inputShapes, input_dtypes: inputDtypes }),
  })
  return r.json() as Promise<{
    ok: boolean
    shapes: Record<string, number[]>
    n_params?: number
    error?: string
    stage?: string
  }>
}

async function main() {
  let owned = false
  let child: ReturnType<typeof spawn> | null = null
  if (!(await isUp())) {
    console.log('starting sidecar…')
    child = spawn('python', ['sidecar-torch/main.py'], {
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: false,
    })
    owned = true
    const deadline = 8000
    const t0 = Number(process.hrtime.bigint() / 1_000_000n)
    while (true) {
      await wait(200)
      if (await isUp()) break
      const now = Number(process.hrtime.bigint() / 1_000_000n)
      if (now - t0 > deadline) {
        console.error('sidecar did not become healthy within 8s')
        child.kill()
        process.exit(1)
      }
    }
  }

  let failed = 0
  for (const c of cases) {
    const { code, attrMap, inputs, issues } = generate(c.nodes, c.edges)
    if (issues.length) {
      console.log(`  ✗ ${c.name}: codegen issues: ${JSON.stringify(issues)}`)
      failed++
      continue
    }
    const res = await infer(code, inputs.map((i) => i.shape), inputs.map((i) => i.dtype))
    console.log(`\n=== ${c.name} ===`)
    console.log(`  ok=${res.ok}  stage=${res.stage ?? '-'}  n_params=${res.n_params ?? '-'}`)
    if (Object.keys(res.shapes).length) console.log('  shapes:', res.shapes)
    if (res.error) console.log('  error:', res.error.split('\n')[0])

    if (c.expectOk && !res.ok) {
      console.log(`  ✗ expected ok=true`)
      failed++
      continue
    }
    if (!c.expectOk && res.ok) {
      console.log(`  ✗ expected ok=false`)
      failed++
      continue
    }
    if (c.expectShapeFor) {
      const attr = attrMap[c.expectShapeFor.id]
      const got = attr ? res.shapes[attr] : undefined
      const want = c.expectShapeFor.shape
      if (!got || got.join(',') !== want.join(',')) {
        console.log(`  ✗ expected ${c.expectShapeFor.id} (${attr}) shape ${want}, got ${got}`)
        failed++
        continue
      }
      console.log(`  ✓ ${c.expectShapeFor.id} → [${got.join(', ')}]`)
    }
    if (c.expectErrorContains && !(res.error ?? '').includes(c.expectErrorContains)) {
      console.log(`  ✗ expected error containing "${c.expectErrorContains}"`)
      failed++
      continue
    }
    if (c.expectErrorContains) console.log(`  ✓ error mentioned "${c.expectErrorContains}"`)
  }

  console.log(`\n${failed === 0 ? '✓' : '✗'} ${cases.length - failed}/${cases.length} passed`)

  if (owned && child) child.kill()
  process.exit(failed === 0 ? 0 : 1)
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
