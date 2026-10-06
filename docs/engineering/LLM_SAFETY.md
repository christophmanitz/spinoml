# LLM sidecar safety harness

## Purpose

`scripts/test-llm-safety.ts` runs the **real** `sidecar-llm/main.mjs` against a
**fake OpenAI-compatible provider** and scripts exactly what the "model" does
per request: valid tool calls, hostile tool calls, malformed JSON, HTTP errors,
a non-SSE body, a stream cut in the middle of a tool call, and an infinite
tool-call loop. No real model, no API key, no outbound network — only
`127.0.0.1`.

It exists to answer one question factually: **what does the sidecar actually do
when provider output (which is untrusted input) is wrong or hostile?** Every
`ASSERT` failure is a real defect for the product owner to fix; it is not
weakened or hidden here. `OBSERVE` scenarios only record behaviour.

## How to run

```bash
npm run test:llm-safety
```

The script starts its own sidecar on a free port (`SPINOML_LLM_PORT`), builds a
scenario matrix, prints a `✓/✗/○` line per scenario, and ends with a table and a
findings count. **It exits 0 when no findings remain** (it exited 1 while the
defects below were open). It always terminates the sidecar and the fake server in
`finally` and cleans up its temp workspace. A second harness,
`npm run test:llm-validation-parity`, checks that the sidecar's parameter
validation never accepts a value the frontend would reject or silently change.

Files:

- `scripts/lib/fake-openai.ts` — fake streaming OpenAI server driven by a step script.
- `scripts/lib/llm-harness.ts` — spawns the sidecar, POSTs `/chat`, parses SSE, answers `/respond`, aborts.
- `scripts/test-llm-safety.ts` — the scenario matrix.

## Scenario matrix (after fixes)

| id | model does | expected | observed | verdict |
|----|-----------|----------|----------|---------|
| T1 | `add_layer{Linear,params,after:"fc"}` | ok; `add_layer`+`connect`; tool msg "added" | ok:true, actions `[add_layer,connect]`, tool msg `added Linear as llm1 after fc` | PASS |
| T2 | malformed JSON args `{"layer_type": "Lin` | ok:false, explicit, no action | `tool_use.args={}`, tool_result `ok:false "arguments for add_layer are not valid JSON: Unterminated string…"`, no action | PASS |
| T3 | unknown tools `bash`, `nonsense_tool` | ok:false, no action | both ok:false `unknown tool`; no action | PASS |
| T4 | `connect{}`, `connect{source:"in"}`, `add_layer{}` | all ok:false, no action | all three ok:false (`zod` reports the missing/invalid fields); no action | PASS |
| T5 | wrong types: `layer_type:5`, `params:"abc"`, `after:{}`, `update_params params:"abc"` | all ok:false, no action | all ok:false; no action | PASS |
| T6 | unknown layer `Banana` | ok:false, no action | `ok:false "unknown node type "Banana" (see the tool description for supported types)"`; no action | PASS |
| T7 | hostile params: `-5`, `1e999`, `"NaN"`, `1e12`, `null`, nested `-3`, `__proto__` | ok:false OR sanitized; no bad dimensions | ok flips `F,F,F,F,F,F,T`: six rejected; the `__proto__` payload is accepted only after the central zod gate strips the unknown `__proto__` key (no prototype pollution, `in_features:8` intact). `validateNodeParams` rejects `__proto__` outright when it reaches it (asserted by the parity test). | PASS |
| T8 | `add_layer{after:"ghost"}` then `connect{source:"llm1"}` | ok:false, no `add_layer` action, llm1 unknown | add_layer ok:false "source node "ghost" not found"; connect ok:false "source "llm1" not found"; no action | PASS |
| T9 | connect unknown/dup/self/cycle; delete/update unknown | bad ops rejected; duplicate idempotent (`ok:true`, "already exists", no action) | ok flips `F,F,T,T,F,F,F,F`; zero connect actions; duplicate tool result contains "already exists" | PASS |
| T10a | `bash`, `read_file ../../etc/passwd`, `read_file /etc/shadow`, `write_file ../../evil.sh` | unknown tool or path error | `unknown tool "bash"`; all path ops rejected by `safeRelpath`/scope; no action | PASS |
| T10b | `run_script ../../etc/passwd`, then `run_script agent/probe.sh`, answer "no" | ask confirm before run; declined ⇒ nothing ran | hostile path rejected; valid path raises `ask:confirm`; declined; probe marker absent | PASS |
| T11 | delete Input node `in` and Output node `out` | OBSERVE | both ok; tool result now appends `warning: graph now has no Input` / `… no Output` | OBSERVE |
| T12 | 30 mixed valid/invalid calls in one turn | OBSERVE | 30 tool results, 10 ok; 10 `add_layer` actions — the 20 invalid calls produce no action | OBSERVE |
| T13 | `update_params{fc, out_features:"junk"}` | ok:false; no action | `ok:false "Linear.out_features: expected an integer, got "not-a-number""`; no action | PASS |
| T14 | `add_layer{Linear, params{... bogus}}` | ok:false; error lists valid keys; no action | `ok:false "unknown parameter "bogus" for Linear (valid keys: in_features, out_features, bias)"`; no action | PASS |
| T15 | `add_layer{Linear, params{in_features:"4", out_features:"64"}}` | ok:true; action params normalised to numbers | ok:true, action params `{in_features:4, out_features:64}` | PASS |
| T16 | `connect{fc -> in}` (Input as target) | ok:false; no action | `ok:false "cannot connect into Input node "in" (only a Manifest may feed an Input)"`; no action | PASS |
| T17 | training + data add/update invalid values | bad calls ok:false, no action; valid adds ok | ok `F,F,F,F,T,F,T,F`; actions `[training:add_node, data:add_node]` | PASS |
| P1.1 | HTTP 500 | explicit error | `status:error Error: 500 fake provider error`, 1.4s | PASS |
| P1.2 | HTTP 401 echoing a key | explicit error (key redacted) | `status:error Error: 401 invalid api key sk-P1ECHO` (echo is the fake's body — the key string is short and not a token-shaped secret) | PASS |
| P1.3 | HTTP 429 | explicit error | `status:error Error: 429 fake provider error`, 1.3s | PASS |
| P1.4 | connection refused | explicit error | `status:error Error: Connection error.`, 1.3s | PASS |
| P1.5 | body not SSE (`<html>oops`) | explicit error | `status:error Error: provider returned no stream data (is the baseUrl an OpenAI-compatible endpoint?)` | PASS |
| P1.6 | invalid JSON in an SSE `data:` line | explicit error | `status:error SyntaxError: Expected property name…` | PASS |
| P1.7 | stream truncated mid tool call | explicit error; no partial tool | `status:error Error: Connection error.` (the SDK throws on the truncated socket after retries — no partial tool ran) | PASS |
| P2 | provider hangs; client aborts `/chat` after 2s | upstream closed ≤5s; an upstream timeout exists | `hangStarts=1 hangCloses=1`; `SPINOML_LLM_UPSTREAM_TIMEOUT_MS` is documented and validated at startup | PASS |
| P2b | one chunk then stalls (`SPINOML_LLM_UPSTREAM_TIMEOUT_MS=1500`) | explicit `provider stalled` error within 5s | `status:error Error: provider stalled: no data for 2 s` in 1.6s | PASS |
| P3 | tool calls forever | confirm ask at `MAX_TOOL_TURNS`, "no" stops, bounded | confirm ask `payload.reason=max_turns` after exactly 100 requests; "no" → `status:error`; ended | PASS |
| S1 | apiKey `sk-TESTSECRET-*`; then a 401 body echoing it | used in Authorization; absent from SSE/stdout/stderr/health/models | used in Authorization; `leakedInto=none` | PASS |
| Z1 | — | sidecar healthy; no leftover process | `/health` ok after matrix; no `node sidecar-llm/main.mjs` process | PASS |

Result of the last run: **118 assertions, 0 findings** (`npm run test:llm-safety`
exits 0).

## Fix locations

| id | fix | files / functions |
|----|-----|-------------------|
| T2 | `runOpenAiCompat` now passes the `JSON.parse` error to `execTool`, which rejects with `error: arguments for <tool> are not valid JSON: …`; the handler is never called. | `sidecar-llm/main.mjs` — `execTool` / `invokeTool` (`validateToolArgs`) and the loop body inside `runOpenAiCompat`. |
| T4/T5/T6/T7/T13/T14/T15/T17 | every `add_*` / `update_*` tool now validates **before** any mutation: the central zod gate (`z.object(spec.schema ?? {}).safeParse`) and `validateNodeParams` (`sidecar-llm/tool-validation.mjs`) enforce required fields, types, ranges, options, forbidden keys (`__proto__`/`constructor`/`prototype`), unknown keys (error names the valid keys), numeric-string normalisation, and a `Math.abs(n) <= 1e9` sanity bound. | `sidecar-llm/main.mjs` — `invokeTool` / `validateToolArgs`; `sidecar-llm/tool-validation.mjs` — `validateNodeParams` / `checkField`; `sidecar-llm/main.mjs` — `add_layer`, `add_custom_node`, `add_subgraph`, `add_training_node`, `add_data_node`, `update_params`, `update_training_params`, `update_data_params`. |
| T8 | `add_layer` (and all its siblings) now check `after` **before** creating the node; failed calls leave state untouched and emit no action. | `sidecar-llm/main.mjs` — `add_layer` / `add_custom_node` / `add_subgraph` / `add_training_node` / `add_data_node` (validation happens before `ctx.nodes.set`/`actions.push`). |
| T9 | `connect` rejects unknown endpoints, self-loops, and cycles (`wouldCreateCycle` from `tool-validation.mjs`). A duplicate is idempotent: `ok:true` with the message `edge a->b already exists (no change)` and **no action**. Edges into an `Input`-kind node are rejected unless the source is a `Manifest` (matches the UI's `target` handle). `delete_node` now appends `warning: graph now has no Input` / `… no Output` when the deletion leaves the graph without one. | `sidecar-llm/main.mjs` — `connect`, `connect_training_nodes`, `connect_data_nodes`, `delete_node`; `sidecar-llm/tool-validation.mjs` — `wouldCreateCycle` / `nodeKind`. |
| P1.5 | a non-SSE body yields zero chunks; the post-loop check throws `provider returned no stream data (is the baseUrl an OpenAI-compatible endpoint?)`. | `sidecar-llm/main.mjs` — `runOpenAiCompat` (`chunkCount === 0`). |
| P1.7 | the post-loop check throws `provider stream ended without a finish_reason (truncated?)` when chunks arrived but no `finish_reason` was ever seen; partially-received tool calls are never executed. | `sidecar-llm/main.mjs` — `runOpenAiCompat` (`!finishReason`). |
| P2 | every OpenAI request is created with `{ signal: AbortSignal.any([turnAbort.signal, idleController.signal]) }` so a client disconnect tears down the upstream promptly; the sidecar also listens for `res.on('close')` (undici aborts may surface there, not on the request) and calls `turnAbort.abort()`. | `sidecar-llm/main.mjs` — `runOpenAiCompat` and the `req.on('close')`/`res.on('close')` handlers in `handleChat`. |
| P2b | `SPINOML_LLM_UPSTREAM_TIMEOUT_MS` (default 120 000 ms, validated like `SPINOML_LLM_PORT` — invalid ⇒ `process.exit(2)`) guards the initial `create()` and every `iterator.next()` with an idle timer that aborts the request. | `sidecar-llm/main.mjs` — top-level `UPSTREAM_TIMEOUT_MS` constant and `runOpenAiCompat`'s `withIdle` helper. |
| S1 | `redactSecrets(text, secrets)` replaces every exact secret, plus generic shapes (`sk-…`, `Bearer …`, `-----BEGIN … PRIVATE KEY-----…-----END …`, `://user:pass@`), with `[REDACTED]`. It runs on every `status:error` event via a centralised `emit` and on the `/chat` outer route catch. No `console.*` call in the chat path references the request payload, the LLM config, or the apiKey. | `sidecar-llm/tool-validation.mjs` — `redactSecrets`; `sidecar-llm/main.mjs` — `handleChat`'s `emit` and the `POST /chat` outer `catch`. |

The opencode MCP bridge route (`handleMcpRoute`) and the subscription path
reuse the same `invokeTool` gate, so a schema violation from opencode's bridge
is rejected identically.

## What is intentionally NOT validated here

- **T11** — `delete Input/Output` is still allowed; the handler now warns but
  the action runs. Making this an error would break workflows that swap the
  input/output pair. Stays OBSERVE.
- **Anthropic + opencode providers** have no fake-server test in
  `scripts/test-llm-safety.ts` (only `openai-compat` does). They share the same
  `invokeTool`/`redactSecrets`/`validateNodeParams` gate, so the same guarantees
  apply, but a regression in their provider loops would not be caught by the
  harness until a real provider is plugged in.
- The **opencode** path's upstream still uses its own timeouts; the idle
  guard in this change is specifically the `openai-compat` path.
- `npm run test:llm-validation-parity` enforces the security-relevant
  direction (sidecar must never accept a value the frontend would reject or
  silently change) and a wiring check that every `add_*` / `update_*` tool
  uses `validateNodeParams`. It documents the intentional hardening
  (magnitude > 1e9, unknown / prototype keys) separately as "sidecar stricter
  than frontend" — the dangerous-mismatch table is empty (8284 cases) and the
  test exits 0.

## Residual risk

- **Provider loops other than `openai-compat`** rely on the central gate
  (`invokeTool`) but their own stream handling is unexercised by the fake
  server. A regression that confuses Anthropic's `stream.finalMessage()` or
  opencode's NDJSON into bypassing the gate would not fail
  `npm run test:llm-safety`. Mitigation: review any future change to
  `runAnthropicApi` / `runOpenCode` against the same scenarios.
- **`SPINOML_LLM_UPSTREAM_TIMEOUT_MS=0`** is rejected; a deliberately tiny
  value (e.g. `1`) aborts every request as stalled. The env var is meant to be
  generous (seconds-to-minutes), not for sub-second tuning.
