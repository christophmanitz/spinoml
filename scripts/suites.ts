#!/usr/bin/env tsx
// Phase 82 — Test suite registry.
//
// Single source of truth for `npm run ci` / `npm run suites`. Every npm script
// in package.json that matches `verify:*` or `test:*` must appear here AND
// every `npmScript` here must exist in package.json — the `--check` flag in
// run-all.ts enforces both directions and runs as the first step of CI.
//
// Categories follow the test pyramid (TODO §67):
//   unit         pure logic, no real trainer, no network
//   contract     a single boundary under test
//   integration  multiple components touching
//   e2e          real trainer, real checkpoint, real metrics
//   scientific   reference / property / fuzz vs independent oracle
//   remote       live ssh/slurm (none exist yet — see `remote-live` BLOCKED)
//   hardware     CUDA (none exists — see `hardware-cuda` BLOCKED)
//   infrastructure  build / lint / cargo / opencode
//
// `needs` is the set of capabilities the suite requires to run meaningfully.
// Anything in `needs` that is missing → BLOCKED with the exact reason. See
// run-all.ts `detectCapability()` for the probe logic.
//
// `timeoutSec` ≈ 5x measured duration (min 30s); the runner kills the whole
// process group on TIMEOUT and reports it as a failure.

export type Category =
  | 'unit'
  | 'contract'
  | 'integration'
  | 'e2e'
  | 'scientific'
  | 'remote'
  | 'hardware'
  | 'infrastructure'

export type Need =
  | 'torch-env'
  | 'cargo'
  | 'cuda'
  | 'ssh-host'
  | 'slurm'
  | 'network'
  | 'llm-key'
  | 'private-data'

export interface Suite {
  readonly name: string
  readonly npmScript?: string
  readonly command?: readonly string[]
  readonly category: Category
  readonly timeoutSec: number
  readonly needs: readonly Need[]
  readonly why: string
}

export const SUITES: readonly Suite[] = [
  // ── unit ────────────────────────────────────────────────────────────
  {
    name: 'test:graphstore',
    npmScript: 'test:graphstore',
    category: 'unit',
    timeoutSec: 60,
    needs: [],
    why: 'GraphStore invariants + mutation guards',
  },
  {
    name: 'test:persistence',
    npmScript: 'test:persistence',
    category: 'unit',
    timeoutSec: 60,
    needs: [],
    why: 'Persistence round-trip + malformed-file fail-safety',
  },
  {
    name: 'test:determinism',
    npmScript: 'test:determinism',
    category: 'unit',
    timeoutSec: 60,
    needs: ['torch-env'],
    why: 'Codegen repeatability + shuffled-array determinism',
  },
  {
    name: 'test:verifier',
    npmScript: 'test:verifier',
    category: 'unit',
    timeoutSec: 60,
    needs: [],
    why: 'Fail-closed model gate (decision matrix + e2e)',
  },
  {
    name: 'test:races',
    npmScript: 'test:races',
    category: 'unit',
    timeoutSec: 60,
    needs: [],
    why: 'Async inference staleness guard (fetch mock)',
  },
  {
    name: 'test:deps-policy',
    npmScript: 'test:deps-policy',
    category: 'unit',
    timeoutSec: 60,
    needs: ['torch-env'],
    why: 'Pip spec policy (reject flags, paths, extras)',
  },
  {
    name: 'test:run-script',
    npmScript: 'test:run-script',
    category: 'unit',
    timeoutSec: 60,
    needs: ['torch-env'],
    why: 'torch /run_script relpath policy (subprocess mock)',
  },
  {
    name: 'test:llm-validation-parity',
    npmScript: 'test:llm-validation-parity',
    category: 'contract',
    timeoutSec: 120,
    needs: [],
    why: 'Sidecar tool validation vs the frontend registry (8k cases): never accepts a value the frontend would change or reject',
  },
  {
    name: 'test:llm-safety',
    npmScript: 'test:llm-safety',
    category: 'contract',
    timeoutSec: 180,
    needs: [],
    why: 'LLM sidecar vs a fake OpenAI provider: hostile tool calls, provider failures, secrets (Node only)',
  },
  {
    name: 'verify:command-injection',
    npmScript: 'verify:command-injection',
    category: 'unit',
    timeoutSec: 60,
    needs: [],
    why: 'splitArgs+quoteArgv round-trip vs real sh + SSRF + ssh target policy',
  },
  {
    name: 'verify:paths',
    npmScript: 'verify:paths',
    category: 'unit',
    timeoutSec: 60,
    needs: [],
    why: 'Node symlink-aware path containment',
  },
  {
    name: 'verify:code-trust',
    npmScript: 'verify:code-trust',
    category: 'unit',
    timeoutSec: 60,
    needs: [],
    why: 'Trust store + collector + ALLOWED_APPROVERS invariants',
  },
  {
    name: 'verify:silent-catch',
    npmScript: 'verify:silent-catch',
    category: 'unit',
    timeoutSec: 60,
    needs: [],
    why: 'No undocumented swallowing catch in src/ or sidecar-llm/',
  },
  {
    name: 'verify:silent-fixes',
    npmScript: 'verify:silent-fixes',
    category: 'unit',
    timeoutSec: 60,
    needs: [],
    why: 'Every documented silent-catch has a fix in code',
  },

  // ── contract ────────────────────────────────────────────────────────
  {
    name: 'verify:sidecar',
    npmScript: 'verify:sidecar',
    category: 'contract',
    timeoutSec: 120,
    needs: ['torch-env'],
    why: 'Torch sidecar request/response + activations',
  },
  {
    name: 'test:safe-load',
    npmScript: 'test:safe-load',
    category: 'contract',
    timeoutSec: 60,
    needs: ['torch-env'],
    why: 'Real malicious-pickle attempts vs safe_torch_load',
  },
  {
    name: 'test:scope',
    npmScript: 'test:scope',
    category: 'contract',
    timeoutSec: 60,
    needs: ['torch-env'],
    why: 'Python scope + hostile manifests + real HTTP',
  },
  {
    name: 'verify:ssh',
    npmScript: 'verify:ssh',
    category: 'contract',
    timeoutSec: 240,
    needs: ['cargo'],
    why: 'SSH failure classification (runs `cargo test ssh_failure_tests` + source checks, no live host)',
  },
  {
    name: 'verify:slurm',
    npmScript: 'verify:slurm',
    category: 'contract',
    timeoutSec: 240,
    needs: ['cargo'],
    why: 'SLURM reliability (runs the Rust reconcile/sbatch tests + source checks, no live cluster)',
  },
  {
    name: 'verify:credentials',
    npmScript: 'verify:credentials',
    category: 'contract',
    timeoutSec: 60,
    needs: [],
    why: 'Secret/credential scan across artifacts, logs, error sanitize',
  },
  {
    name: 'verify:submission',
    npmScript: 'verify:submission',
    category: 'contract',
    timeoutSec: 60,
    needs: [],
    why: 'Atomic remote run directory claim and retry idempotency',
  },
  {
    name: 'verify:recovery',
    npmScript: 'verify:recovery',
    category: 'contract',
    timeoutSec: 90,
    needs: [],
    why: 'Remote job recovery (live re-query, no localStorage cache)',
  },
  {
    name: 'verify:states',
    npmScript: 'verify:states',
    category: 'contract',
    timeoutSec: 60,
    needs: ['torch-env'],
    why: 'Run state machine transitions and invalid-transition guards',
  },
  {
    name: 'verify:events',
    npmScript: 'verify:events',
    category: 'contract',
    timeoutSec: 60,
    needs: [],
    why: 'Event ordering, terminal truncation, stale-read dropping',
  },

  // ── integration ─────────────────────────────────────────────────────
  {
    name: 'verify:codegen',
    npmScript: 'verify:codegen',
    category: 'integration',
    timeoutSec: 150,
    needs: ['torch-env'],
    why: 'Generator over 13 graphs + exec generated Python',
  },
  {
    name: 'verify:codegen-security',
    npmScript: 'verify:codegen-security',
    category: 'integration',
    timeoutSec: 90,
    needs: ['torch-env'],
    why: '5362 hostile-payload cases through Python ast/tokenize',
  },
  {
    name: 'verify:traingen',
    npmScript: 'verify:traingen',
    category: 'integration',
    timeoutSec: 150,
    needs: ['torch-env'],
    why: 'Training codegen (compile + multitask + eval-only)',
  },
  {
    name: 'test:robustness',
    npmScript: 'test:robustness',
    category: 'integration',
    timeoutSec: 90,
    needs: ['torch-env'],
    why: 'Torch sidecar robustness + crash recovery (isolated instance)',
  },
  {
    name: 'test:datasets',
    npmScript: 'test:datasets',
    category: 'integration',
    timeoutSec: 90,
    needs: ['torch-env'],
    why: 'Dataset handlers per-kind reliability matrix + hang regression',
  },
  {
    name: 'verify:concurrent',
    npmScript: 'verify:concurrent',
    category: 'integration',
    timeoutSec: 60,
    needs: [],
    why: 'Concurrent mutations (saveSeq+revAtStart dirty correction)',
  },
  {
    name: 'verify:graph-revision',
    npmScript: 'verify:graph-revision',
    category: 'integration',
    timeoutSec: 60,
    needs: [],
    why: 'Graph revision system (stale shape/dataset/smoke/refresh drop)',
  },
  {
    name: 'verify:ui-state',
    npmScript: 'verify:ui-state',
    category: 'integration',
    timeoutSec: 60,
    needs: [],
    why: 'UI state must not lie (training stale→unknown, save truthfulness)',
  },
  {
    name: 'verify:frontend-errors',
    npmScript: 'verify:frontend-errors',
    category: 'integration',
    timeoutSec: 60,
    needs: [],
    why: 'Frontend error states (loading/success/error/offline distinct)',
  },
  {
    name: 'verify:immutability',
    npmScript: 'verify:immutability',
    category: 'integration',
    timeoutSec: 60,
    needs: [],
    why: 'GraphStore immutability invariants',
  },
  {
    name: 'verify:code-trust-wiring',
    npmScript: 'verify:code-trust-wiring',
    category: 'integration',
    timeoutSec: 60,
    needs: [],
    why: 'Untrusted code never reaches /infer, smoke, startRun',
  },

  // ── e2e ─────────────────────────────────────────────────────────────
  {
    name: 'verify:smoke',
    npmScript: 'verify:smoke',
    category: 'e2e',
    timeoutSec: 150,
    needs: ['torch-env'],
    why: 'Scientific smoke (synthetic data, 5 epochs, checkpoint+metrics)',
  },
  {
    name: 'verify:failures',
    npmScript: 'verify:failures',
    category: 'e2e',
    timeoutSec: 150,
    needs: ['torch-env'],
    why: 'Training failure tests (9 failure modes incl. NaN input)',
  },
  {
    name: 'verify:checkpoint',
    npmScript: 'verify:checkpoint',
    category: 'e2e',
    timeoutSec: 200,
    needs: ['torch-env'],
    why: 'Checkpoint correctness (train/save/resume/cancel, full state)',
  },
  {
    name: 'verify:cancel',
    npmScript: 'verify:cancel',
    category: 'e2e',
    timeoutSec: 150,
    needs: ['torch-env'],
    why: 'SIGTERM/SIGINT signal handling, resume checkpoint, shielding',
  },
  {
    name: 'verify:metrics',
    npmScript: 'verify:metrics',
    category: 'e2e',
    timeoutSec: 90,
    needs: ['torch-env'],
    why: 'Metric correctness (weighted aggregation vs Python reference)',
  },
  {
    name: 'verify:manifest',
    npmScript: 'verify:manifest',
    category: 'e2e',
    timeoutSec: 300,
    needs: ['torch-env'],
    why: 'Manifest: git state, hashes, config identity, atomicity',
  },
  {
    name: 'verify:integrity',
    npmScript: 'verify:integrity',
    category: 'e2e',
    timeoutSec: 600,
    needs: ['torch-env'],
    why: 'Sabotaged checkpoints/metrics/manifest → never done',
  },

  // ── scientific ──────────────────────────────────────────────────────
  {
    name: 'verify:reference',
    npmScript: 'verify:reference',
    category: 'scientific',
    timeoutSec: 120,
    needs: ['torch-env'],
    why: 'Generated model == hand-written PyTorch (params/forward/loss/grads)',
  },
  {
    name: 'verify:reference-train',
    npmScript: 'verify:reference-train',
    category: 'scientific',
    timeoutSec: 300,
    needs: ['torch-env'],
    why: 'Reference experiments through the REAL trainer (same-seed rerun)',
  },
  {
    name: 'test:property',
    npmScript: 'test:property',
    category: 'scientific',
    timeoutSec: 120,
    needs: ['torch-env'],
    why: '200 random valid graphs vs an independent oracle',
  },
  {
    name: 'test:fuzz',
    npmScript: 'test:fuzz',
    category: 'scientific',
    timeoutSec: 120,
    needs: ['torch-env'],
    why: '948 invalid mutants must be rejected cleanly',
  },

  // ── remote ──────────────────────────────────────────────────────────
  {
    name: 'remote-live',
    category: 'remote',
    timeoutSec: 30,
    needs: ['ssh-host', 'slurm'],
    why: 'no live-cluster suite exists; needs an ssh alias and SLURM',
  },

  // ── hardware ────────────────────────────────────────────────────────
  {
    name: 'hardware-cuda',
    category: 'hardware',
    timeoutSec: 30,
    needs: ['cuda'],
    why: 'no CUDA device / no dedicated CUDA suite',
  },

  // ── infrastructure ──────────────────────────────────────────────────
  {
    name: 'build',
    npmScript: 'build',
    category: 'infrastructure',
    timeoutSec: 180,
    needs: [],
    why: 'tsc + vite build (catches type errors and bundle failures)',
  },
  {
    name: 'typecheck-scripts',
    npmScript: 'typecheck:scripts',
    category: 'infrastructure',
    timeoutSec: 120,
    needs: [],
    why: 'tsc over scripts/**/*.ts: tsx does not type-check',
  },
  {
    name: 'lint',
    command: ['npx', 'eslint', '.', "--ignore-pattern", "src-tauri/target/**"],
    category: 'infrastructure',
    timeoutSec: 120,
    needs: [],
    why: 'ESLint totals vs scripts/lint-baseline.json (gate no-regression)',
  },
  {
    name: 'cargo-check',
    command: ['sh', '-c', 'cd src-tauri && cargo check'],
    category: 'infrastructure',
    timeoutSec: 300,
    needs: ['cargo'],
    why: 'Rust workspace type-check',
  },
  {
    name: 'cargo-test',
    command: ['sh', '-c', 'cd src-tauri && cargo test'],
    category: 'infrastructure',
    timeoutSec: 600,
    needs: ['cargo'],
    why: 'Rust workspace unit tests',
  },
  {
    name: 'verify-opencode',
    npmScript: 'verify:opencode',
    category: 'infrastructure',
    timeoutSec: 90,
    needs: ['llm-key'],
    why: 'LLM sidecar + opencode CLI reachable + a free model works',
  },
]
