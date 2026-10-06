# SpinoML – Production and Scientific Reliability Hardening Plan

## Mission

You are a coding agent responsible for taking the existing **SpinoML** repository and hardening it for reliable scientific machine-learning work.

Your goal is **not** merely to make the application build.

Your goal is to make the repository sufficiently reliable that researchers can use it for experiments where:

* model architecture must be correct,
* generated PyTorch code must be correct,
* tensor shapes must be correct,
* training failures must never be reported as successful runs,
* datasets must be handled correctly,
* checkpoints must be reliable,
* experiment configurations must be reproducible,
* local and remote jobs must report their actual state,
* asynchronous operations must not corrupt state,
* scientific results must be traceable to the exact code/configuration/data used.

The existing architecture should be preserved wherever possible.

Do **not** rewrite the project from scratch.

---

# 0. CRITICAL FIRST PHASE – REPLACE/EXTEND CLAUDE CONTROL WITH OPENCode SUPPORT

## Objective

**This phase must be completed BEFORE downloading, installing, testing, or modifying the SpinoML codebase.**

The future development and later practical use of SpinoML will be performed primarily through **OpenCode**, not Claude Code.

The existing SpinoML architecture currently contains Claude/LLM-specific control mechanisms. These must therefore be extended so that **OpenCode is a first-class, supported coding/AI control path**.

Do not simply rename "Claude" to "OpenCode".

The implementation must introduce a clean abstraction that allows SpinoML to work with different AI coding/agent backends.

The desired direction is:

```text
                    SpinoML
                       |
                       v
              ┌─────────────────┐
              │ Agent Interface │
              └────────┬────────┘
                       |
              ┌────────┴─────────┐
              |                  |
              v                  v
       ┌─────────────┐    ┌─────────────┐
       │   OpenCode  │    │    Claude   │
       │   Backend   │    │   Backend   │
       └──────┬──────┘    └──────┬──────┘
              |                  |
              v                  v
        Selected model      Selected model
```

OpenCode must be the **default/primary integration**.

Claude support should remain available where practical, but the application must no longer be architecturally dependent on Claude Code.

---

## 0.1 OpenCode must be a first-class provider

Implement or adapt the existing LLM/agent control layer so that it supports:

```text
Provider
Model
Authentication/configuration
Capabilities
Request
Response
Tool/action handling
Errors
Timeouts
Cancellation
```

At minimum, the application must be able to distinguish:

```text
OpenCode
Claude
```

without duplicating the entire control architecture for each provider.

Prefer an interface similar in concept to:

```text
AgentProvider
    ├── OpenCodeProvider
    └── ClaudeProvider
```

Use the existing architecture and naming conventions where appropriate.

Do not blindly introduce a new abstraction if the repository already contains an equivalent mechanism.

---

# 0.2 OpenCode must be the default

The normal/default configuration must use:

```text
Provider: OpenCode
```

The system must not require Claude Code to be installed simply to use the normal AI-assisted SpinoML workflow.

The application should continue to work without either provider for core local functionality such as:

```text
Graph editing
Graph validation
Shape inference
Code generation
Local scientific workflows
```

where the architecture permits this.

---

# 0.3 OpenCode model selection

The user must be able to select the OpenCode model.

Do **not** hard-code a single model.

For example, the configuration concept should support:

```text
Provider:
    OpenCode

Model:
    Big Pickle
```

but the implementation must treat the model as configuration rather than as a special-case string.

The user is currently using:

```text
Big Pickle
```

as the preferred OpenCode model.

Therefore the initial/default OpenCode model should be configurable to **Big Pickle**, but the implementation must allow another available OpenCode model to be selected later.

The system should conceptually support:

```text
OpenCode
 ├── Big Pickle
 ├── Model B
 ├── Model C
 └── Future models
```

Do not assume that the currently selected model will remain available forever.

---

# 0.4 Model configuration must not be scattered through the code

Find every existing location where the current Claude model/provider is:

```text
hard-coded
assumed
constructed
validated
passed to a sidecar
stored in state
shown in the UI
```

Centralize provider/model configuration.

Prefer a single authoritative configuration structure.

For example:

```typescript
type AgentProvider = "opencode" | "claude";

interface AgentConfiguration {
  provider: AgentProvider;
  model: string;
}
```

Use the project's existing type/configuration system if one already exists.

Do not introduce unnecessary duplicate configuration sources.

---

# 0.5 OpenCode configuration UI

The user must be able to configure the AI control system from the application.

At minimum, the UI/configuration flow should make it possible to select:

```text
Provider
    OpenCode
    Claude

Model
    available/configured OpenCode model(s)
    available/configured Claude model(s)
```

The exact UI location should follow the existing SpinoML design.

Do not create a large new settings system if an existing settings/preferences mechanism can be extended.

---

# 0.6 Do not assume the OpenCode API

Before implementing the integration:

1. Inspect the existing SpinoML Claude integration.
2. Inspect how the current LLM sidecar communicates.
3. Inspect the installed OpenCode interface/documentation available in the development environment.
4. Determine the exact supported OpenCode invocation/API.
5. Determine how OpenCode exposes model selection.
6. Determine how authentication is handled.
7. Determine how tool calls/actions are represented.
8. Determine how errors and process termination are reported.

**Never invent an OpenCode API.**

If OpenCode is invoked as a CLI, use the actual CLI contract discovered during implementation.

If OpenCode exposes another supported integration mechanism, use that mechanism where appropriate.

---

# 0.7 OpenCode process lifecycle

Test the complete lifecycle:

```text
SpinoML starts
      ↓
OpenCode integration available
      ↓
Model selected
      ↓
Request sent
      ↓
Response received
      ↓
Action/tool result processed
```

Also test:

```text
OpenCode unavailable
OpenCode process fails
OpenCode times out
OpenCode exits unexpectedly
Invalid model
Invalid configuration
Malformed response
Request cancellation
```

Every failure must become an explicit application state.

Never silently fall back to Claude.

Never silently fall back to another model.

Never silently pretend that an AI operation succeeded.

---

# 0.8 OpenCode model selection must be observable

When an AI operation is executed, it must be possible to determine which provider and model were actually used.

For example:

```text
Provider: OpenCode
Model: Big Pickle
```

This information should be available to the relevant logs/metadata and, where appropriate, experiment artifacts.

Scientific experiments must not become ambiguous because an AI-generated graph was produced using an unknown model.

---

# 0.9 AI-generated changes must remain provider-independent

The following pipeline must be identical regardless of whether OpenCode or Claude is used:

```text
AI provider
      ↓
Structured action
      ↓
Schema validation
      ↓
Graph validation
      ↓
Commit
      ↓
Shape inference
      ↓
Code generation
      ↓
Scientific verification
```

Neither OpenCode nor Claude may bypass:

```text
Graph validation
Shape validation
Code generation validation
```

The provider is untrusted input.

---

# 0.10 OpenCode must not become a second source of truth

Do not allow:

```text
OpenCode state
```

and:

```text
SpinoML graph state
```

to become competing authoritative states.

SpinoML's graph store remains the authoritative representation of the model.

The AI provider may:

```text
propose
modify
query
```

but the resulting graph must always pass through SpinoML's own validation and commit mechanisms.

---

# 0.11 Preserve Claude compatibility

Do not remove Claude support merely because OpenCode becomes the primary provider.

Instead:

```text
Existing Claude functionality
        ↓
Provider abstraction
        ↓
Claude backend
```

and:

```text
OpenCode functionality
        ↓
Provider abstraction
        ↓
OpenCode backend
```

Both must use the same internal validation and graph mutation pipeline.

If the existing Claude integration cannot be cleanly retained, document the reason before removing or substantially changing it.

---

# 0.12 OpenCode tests

Before proceeding to the repository-hardening phases, create focused tests for:

### Configuration

```text
OpenCode selected
OpenCode model selected
Model persisted
Model changed
Invalid model handled
```

### Provider selection

```text
OpenCode → OpenCode backend
Claude → Claude backend
```

### Request handling

```text
Valid request
Invalid request
Timeout
Cancellation
Provider unavailable
```

### Model handling

```text
Big Pickle selected
Alternative model selected
Unknown model rejected or handled explicitly
```

### Failure handling

```text
OpenCode process failure
Malformed response
Provider unavailable
```

### Security

Verify that:

```text
Credentials
Tokens
Secrets
```

are not written to ordinary logs or experiment artifacts.

> **0.12 — partially implemented 2026-10-06 (Phase 77/78 session).** `npm run test:opencode-lifecycle`
> (68 rows) drives the real LLM sidecar and the REAL `mcp-bridge.mjs` against a fake `opencode`
> binary (`scripts/lib/fake-opencode`, `SPINOML_OPENCODE_BIN`): tool call through the bridge in
> token mode, no/wrong bridge secret rejected, master token and secret not in argv/env they must not
> be in, non-zero exit / `error` event / garbage lines / silent process (start timeout) / client
> abort (fake AND bridge child gone), invalid timeout env exits 2, no leftover processes or
> `spinoml-opencode-*` dirs. It found a real defect (the bridge always called port 7422, ignoring
> `SPINOML_LLM_PORT`) and an unawaited temp-dir cleanup — both fixed. The real opencode CLI
> (1.18.15) was exercised once by hand against a token-mode sidecar with a ScaDS model (the free
> `opencode/*` tier refuses non-OpenCode callers with 403): the `environment` secret reached the
> bridge and `add_layer` produced an `action`. NOT done: provider selection / model handling tests
> (default Big Pickle, unknown model), an automated real-CLI suite (needs a live model —
> `verify:opencode` stays BLOCKED), Anthropic/subscription paths.

---

# 0.13 OpenCode integration acceptance criteria

Do not proceed to the next phase until all applicable criteria are satisfied:

* [ ] OpenCode is a first-class provider.
* [ ] OpenCode is the default provider.
* [ ] The model is configurable.
* [ ] Big Pickle can be selected as the OpenCode model.
* [ ] Another OpenCode model can be selected without source-code changes.
* [ ] Provider/model configuration is centralized.
* [ ] The UI exposes provider/model selection where appropriate.
* [ ] OpenCode does not bypass graph validation.
* [ ] OpenCode failures are explicit.
* [ ] OpenCode timeouts are handled.
* [ ] OpenCode cancellation is handled.
* [ ] OpenCode process failure is handled.
* [ ] The actual provider/model used can be identified.
* [ ] Core SpinoML functionality does not require Claude Code.
* [ ] Existing Claude functionality remains available where intended.
* [ ] Provider-specific code does not leak throughout the graph/training architecture.
* [ ] Regression tests exist for the provider/model configuration.
* [ ] No credentials are leaked.

If an OpenCode dependency cannot be installed or accessed in the current environment, mark the relevant test:

```text
BLOCKED
```

with the exact reason.

Do not fake a successful OpenCode test.

---

# 0.14 Critical architectural rule

The final architecture should conceptually become:

```text
                 AI / CODING CONTROL
                         |
                         v
                ┌─────────────────┐
                │ Agent Provider  │
                │   Interface     │
                └────────┬────────┘
                         |
             ┌───────────┴───────────┐
             |                       |
             v                       v
      ┌──────────────┐       ┌──────────────┐
      │   OpenCode   │       │    Claude    │
      │   Provider   │       │   Provider   │
      └──────┬───────┘       └──────┬───────┘
             |                       |
             v                       v
        Selected model          Selected model
             |                       |
             └───────────┬───────────┘
                         v
                 Structured actions
                         |
                         v
                 Schema validation
                         |
                         v
                  Graph validation
                         |
                         v
                   Graph commit
                         |
                         v
                  Shape inference
                         |
                         v
                 Deterministic codegen
                         |
                         v
                  PyTorch validation
```

**OpenCode is the primary path. Claude is an optional compatible provider.**

Do not allow either AI provider to become a trusted authority over the scientific model state.

---

# 0.15 Only after this phase

Only after the OpenCode integration has been implemented, tested, and documented should the agent proceed to:

```text
PHASE 1 – DOWNLOAD, INSTALL AND VERIFY THE COMPLETE REPOSITORY
```

The repository-hardening process must then test the newly established OpenCode integration as part of the complete system.

---

# 1. DOWNLOAD, INSTALL AND VERIFY THE COMPLETE REPOSITORY

## Objective

Before changing the remaining application components, download the repository and install the **complete development/runtime environment** required by the project.

Do not assume the repository is already correctly installed.

---

## 1.1 Clone the repository

Use the official repository:

```bash
git clone https://github.com/christophmanitz/spinoml.git
cd spinoml
```

If the repository is already present:

```bash
git status
git remote -v
git log -1 --oneline
```

Record the exact commit being tested.

---

## 1.2 Inspect the repository before installation

Read:

```text
README.md
CLAUDE.md
package.json
package-lock.json
tsconfig*.json
vite.config.*
src/
src-tauri/
sidecar-torch/
sidecar-llm/
scripts/
workspace/
examples/
```

Do not make changes yet.

Determine:

```text
Frontend technology
Rust/Tauri version
Node requirements
Python requirements
PyTorch requirements
Sidecar architecture
LLM/agent requirements
Testing framework
Build commands
Training commands
Remote execution requirements
```

---

[Continue with the remainder of the existing plan unchanged, renumbering the existing phases by +1.]


## Rule 1 – Work sequentially

Complete one phase before starting the next.

Use:

```text
READ
→ UNDERSTAND
→ TEST
→ REPRODUCE
→ FIX
→ REGRESSION TEST
→ DOCUMENT
→ NEXT PHASE
```

Do not perform large unrelated changes simultaneously.

---

## Rule 2 – Never guess

If you do not understand a component:

1. Read the relevant source files.
2. Find existing tests.
3. Trace the data flow.
4. Reproduce the behavior.
5. Only then modify the code.

Never invent an API because you assume it exists.

---

## Rule 3 – Never hide errors

Do not make tests pass by hiding errors.

Never introduce code such as:

```python
try:
    ...
except Exception:
    pass
```

or:

```typescript
catch (_) {}
```

unless the exception is explicitly expected and safely handled.

Do not use:

```typescript
// @ts-ignore
```

to hide real problems.

Do not disable lint rules globally to make the build pass.

---

## Rule 4 – Never weaken tests

Do not modify a test merely because the implementation currently fails it.

The correct order is:

```text
Existing test fails
        ↓
Understand expected behavior
        ↓
Determine whether implementation or test is wrong
        ↓
Fix the actual problem
        ↓
Keep/add regression test
```

---

## Rule 5 – Every real bug gets a regression test

For every confirmed bug:

```text
1. Reproduce it.
2. Create a test that fails before the fix.
3. Fix the bug.
4. Verify the test passes.
5. Run related tests.
6. Document the fix.
```

---

## Rule 6 – Never claim a test passed if it could not run

Use explicit statuses:

```text
PASS
FAIL
SKIPPED
BLOCKED
NOT APPLICABLE
```

Never turn:

```text
CUDA unavailable
```

into:

```text
CUDA PASS
```

Never turn:

```text
SLURM unavailable
```

into:

```text
SLURM PASS
```

---

## Rule 7 – Scientific correctness has priority

Priority order:

```text
1. Data corruption
2. Scientifically incorrect results
3. Incorrect generated models
4. Incorrect training
5. Checkpoint corruption
6. Race conditions
7. Job/process state corruption
8. Security problems
9. Reproducibility
10. Performance
11. UI polish
```

Do not work on cosmetic UI improvements while critical scientific correctness problems remain.

---

# 1. PHASE 0 – DOWNLOAD, INSTALL AND VERIFY THE COMPLETE REPOSITORY

## Objective

Before changing anything, download the repository and install the **complete development/runtime environment** required by the project.

Do not assume the repository is already correctly installed.

---

## 1.1 Clone the repository

Use the official repository:

```bash
git clone https://github.com/christophmanitz/spinoml.git
cd spinoml
```

If the repository is already present:

```bash
git status
git remote -v
git log -1 --oneline
```

Record the exact commit being tested.

---

## 1.2 Inspect the repository before installation

Read:

```text
README.md
CLAUDE.md
package.json
package-lock.json
tsconfig*.json
vite.config.*
src/
src-tauri/
sidecar-torch/
sidecar-llm/
scripts/
workspace/
examples/
```

Do not make changes yet.

Determine:

```text
Frontend technology
Rust/Tauri version
Node requirements
Python requirements
PyTorch requirements
Sidecar architecture
LLM/Claude requirements
Testing framework
Build commands
Training commands
Remote execution requirements
```

---

## 1.3 Check the local environment

Record:

```bash
node --version
npm --version
python --version
python3 --version
rustc --version
cargo --version
git --version
```

If available:

```bash
nvidia-smi
```

Record:

```text
Operating system
CPU
RAM
GPU
GPU memory
CUDA version
Python version
Node version
Rust version
```

---

## 1.4 Install Node dependencies

Use the repository's lockfile.

Prefer:

```bash
npm ci
```

over:

```bash
npm install
```

when a valid `package-lock.json` exists.

Do not arbitrarily update dependencies.

---

## 1.5 Install Python dependencies

Inspect the repository first.

If it contains:

```text
requirements.txt
pyproject.toml
poetry.lock
uv.lock
```

follow the existing project mechanism.

Do not create a second competing dependency-management system unless necessary.

Use an isolated Python environment where appropriate:

```bash
python -m venv .venv
```

Then install the project's documented dependencies.

Verify PyTorch:

```bash
python -c "import torch; print(torch.__version__)"
```

If CUDA is expected:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 1.6 Install Rust dependencies

From the Rust/Tauri directory:

```bash
cargo check
```

Download/build all required dependencies.

Do not update the dependency versions unless required for a confirmed problem.

---

## 1.7 Install all project-specific tools

Inspect the documentation and package configuration for:

```text
Tauri
Claude CLI
MCP
PyTorch
SSH
SFTP
SLURM
system dependencies
```

Install everything required for normal local development.

If Claude is required for optional LLM functionality, verify whether the local environment has it.

Do not fabricate credentials.

---

## 1.8 Run the complete existing verification suite

Before modifying code, execute all documented checks.

At minimum, investigate and run:

```bash
npm run build
npm run lint
npm run verify:codegen
npm run verify:sidecar
npm run verify:traingen
```

Then:

```bash
cd src-tauri
cargo check
cd ..
```

If Python tests exist:

```bash
pytest
```

If JavaScript/TypeScript tests exist:

```bash
npm test
```

Use the actual repository scripts when they differ.

---

## 1.9 Create baseline report

Create:

```text
docs/engineering/BASELINE.md
```

Record:

```text
Repository commit
Operating system
Node version
npm version
Python version
PyTorch version
Rust version
Cargo version
CUDA version
GPU
```

Then:

```text
Build: PASS/FAIL
Lint: PASS/FAIL
Codegen verification: PASS/FAIL
Torch sidecar: PASS/FAIL
Training generation: PASS/FAIL
Rust: PASS/FAIL
Python tests: PASS/FAIL/SKIPPED
Other tests: PASS/FAIL/SKIPPED
```

For every failure, record:

```text
Command
Error
Relevant stack trace
Likely component
```

---

## 1.10 Important

Do not fix anything yet unless required simply to complete installation.

The purpose of this phase is to establish:

> **What exactly works before hardening begins?**

---

# 2. PHASE 1 – FULL REPOSITORY INVENTORY

Create:

```text
docs/engineering/ARCHITECTURE.md
docs/engineering/RISK_REGISTER.md
docs/engineering/TEST_MATRIX.md
```

Map:

```text
Frontend
GraphStore
Layer registry
Code generation
Shape inference
Dataset handling
Training
Persistence
Torch sidecar
LLM sidecar
MCP
Tauri
Rust commands
Filesystem
SSH
SFTP
PTY
SLURM
Workspace
```

For every component document:

```text
Inputs
Outputs
State
External dependencies
Failure modes
Existing tests
```

---

# 3. PHASE 2 – RISK REGISTER

Create a risk table.

Example:

| ID   | Area        | Risk                                    | Severity | Test   | Status |
| ---- | ----------- | --------------------------------------- | -------- | ------ | ------ |
| R001 | Graph       | Invalid graph can be committed          | CRITICAL | G001   | OPEN   |
| R002 | Codegen     | Generated code differs from graph       | CRITICAL | C001   | OPEN   |
| R003 | Shape       | Incorrect shape accepted                | CRITICAL | S001   | OPEN   |
| R004 | Training    | Failed training reported successful     | CRITICAL | T001   | OPEN   |
| R005 | Checkpoint  | Corrupted checkpoint                    | CRITICAL | T002   | OPEN   |
| R006 | Dataset     | Split leakage                           | CRITICAL | D001   | OPEN   |
| R007 | Async       | Stale response overwrites state         | HIGH     | A001   | OPEN   |
| R008 | Sidecar     | Crash leaves application inconsistent   | HIGH     | P001   | OPEN   |
| R009 | SSH         | Connection failure produces wrong state | HIGH     | R001   | OPEN   |
| R010 | SLURM       | Wrong remote job state                  | HIGH     | R002   | OPEN   |
| R011 | Persistence | Graph corruption                        | CRITICAL | P001   | OPEN   |
| R012 | Security    | Command/path injection                  | CRITICAL | SEC001 | OPEN   |

Update this throughout the project.

---

# 4. PHASE 3 – GRAPH STORE CORRECTNESS

The graph is the core model representation.

Treat it as a critical data structure.

---

## 4.1 Define graph invariants

At minimum:

```text
Node IDs are unique.
Edge IDs are unique.
Every edge references existing nodes.
Every edge references valid handles.
Layer types are known.
Layer parameters are valid.
Required parameters exist.
Dimensions are valid.
```

Determine whether cycles are allowed.

If cycles are not supported, explicitly reject them.

---

## 4.2 Test graph mutations

Test:

```text
Add node
Remove node
Update node
Add edge
Remove edge
Rewire edge
Change parameters
Change input
Change output
```

Test invalid cases:

```text
Duplicate node
Duplicate edge
Unknown node
Unknown edge
Unknown layer
Invalid parameter
Invalid connection
Deleted node with remaining edges
Invalid handle
Invalid dimension
```

---

## 4.3 Validate before commit

Preferred architecture:

```text
Mutation proposal
      ↓
Validation
      ↓
Commit
```

Not:

```text
Mutation
      ↓
Commit
      ↓
Validation
```

An invalid graph must never become the authoritative graph state.

---

# 5. PHASE 4 – GRAPH PERSISTENCE

Test:

```text
Create graph
Save graph
Close application
Reload graph
Compare graph
```

Test malformed files:

```text
Empty file
Invalid JSON
Truncated JSON
Missing fields
Unknown fields
Wrong types
Invalid values
Old schema
```

The application must fail safely.

It must never silently replace a corrupt graph with an empty graph.

---

## 5.1 Schema versioning

If not already present, introduce a version field such as:

```json
{
  "schemaVersion": 1
}
```

Future schema changes should use explicit migrations.

---

# 6. PHASE 5 – DETERMINISTIC CODE GENERATION

Code generation is CRITICAL.

For the same graph:

```text
generate(graph)
```

must produce deterministic output.

Run:

```text
generate(graph)
generate(graph)
generate(graph)
```

and compare outputs.

If some generated metadata legitimately varies, isolate that variation.

---

# 7. PHASE 6 – CODEGEN GOLDEN TESTS

Create fixtures for:

```text
Linear
MLP
CNN
Residual
Branch
Merge
Multi-input
Multi-output
Concat
Flatten
Reshape
Normalization
Pooling
Attention
Recurrent
GNN
```

For each:

```text
Graph fixture
Expected generated Python
```

Do not make the golden tests depend on timestamps or random identifiers.

> **Implemented 2026-10-06.** `npm run test:codegen-golden` (`scripts/test-codegen-golden.ts`,
> fixtures in `scripts/golden/`, builders in `scripts/lib/golden-cases.ts`): 40 cases — 30 model
> graphs (every listed category: linear, MLP, CNN, residual, branch/merge, multi-input,
> multi-output, concat, flatten, reshape, normalization, pooling, attention, transformer, recurrent,
> GNN, Custom, Subgraph, DataOp passthrough, ESPF/Sequence/manifest inputs, …), 4 training-graph and
> 6 data-canvas cases — each a committed `<case>.graph.json` + the byte-exact `<case>.expected.py`
> produced by the real generator (`--update` rewrites and lists what changed, `--list`). Per case:
> byte equality with a unified diff on mismatch, determinism (twice in-process + once in a fresh
> process), `ast.parse`, no volatile content (timestamps, absolute paths, addresses, uuids), and for
> model cases the generated module is imported and run once with the fixture input, asserting the
> output shape (PyG cases are SKIPPED, loudly, only when PyG is not importable). A **coverage guard**
> fails when any registry entry (55 layers, 13 training nodes, 12 data nodes) has no golden case and
> no explicit exclusion — adding a layer without a golden fails CI. Mutations proven red: one space of
> generator indentation, a deleted fixture, volatile content in the output.
> **Real bug found by writing the fixtures:** `MultiheadAttention` was emitted as
> `self.mha(x)`, but `nn.MultiheadAttention.forward` needs `(query, key, value)` and returns
> `(out, weights)` — the layer could never run (TypeError). Now `LayerSpec.selfAttention` makes the
> generator emit `self.mha(x, x, x, need_weights=False)[0]`; the `attention` golden case is the
> regression test (red without the fix, with the TypeError).

---

# 8. PHASE 7 – EXECUTE GENERATED CODE

A generated model is not considered valid merely because the Python parser accepts it.

For every important model:

```text
Graph
 ↓
Code generation
 ↓
Write temporary Python file
 ↓
Compile
 ↓
Import
 ↓
Instantiate
 ↓
Forward
 ↓
Loss
 ↓
Backward
```

Verify:

```python
loss.backward()
```

when the model is trainable.

---

# 9. PHASE 8 – SHAPE INFERENCE

Compare SpinoML's predicted shapes with real PyTorch execution.

For every test graph:

```text
SpinoML shape inference
        VS
Actual PyTorch tensor shapes
```

Use assertions.

Test:

```text
1D
2D
3D
4D
5D
Batch
Channels
Sequence
Convolution
Pooling
Flatten
Reshape
Transpose
Concat
Broadcasting
Multi-input
Multi-output
Attention
Recurrent
Normalization
```

---

# 10. PHASE 9 – SHAPE FAILURE MUST FAIL CLOSED

If shape inference fails:

Do not use an old shape silently.

Do not continue training with unknown dimensions.

Represent the state explicitly:

```text
VALID
INVALID
UNKNOWN
```

Training must refuse to start if the model is not known to be valid.

---

# 11. PHASE 10 – ASYNCHRONOUS SHAPE RACE CONDITIONS

Test:

```text
Graph A
request inference A

Graph B
request inference B

Response B arrives

Response A arrives later
```

Response A must not overwrite Graph B.

Use an appropriate mechanism such as:

```text
Graph revision
Request ID
Graph hash
```

The implementation must reject stale responses.

---

# 12. PHASE 11 – TORCH SIDECAR ROBUSTNESS

Test:

```text
Startup
Shutdown
Restart
Malformed request
Missing fields
Invalid dtype
Invalid shape
Unknown layer
Invalid graph
Timeout
Internal exception
Unexpected process termination
```

The API must return structured errors.

Example:

```json
{
  "error": {
    "code": "INVALID_GRAPH",
    "message": "Graph validation failed",
    "details": {}
  }
}
```

Use the project's existing API conventions if they already provide an error structure.

Do not create inconsistent error formats.

---

# 13. PHASE 12 – SIDECAR CRASH RECOVERY

Test:

```text
Start application
Start Torch sidecar
Send request
Kill Torch sidecar
Send request
Restart sidecar
Send request
```

The application must recover or clearly report that the service is unavailable.

It must never silently use stale data.

---

# 14. PHASE 13 – LOCAL PORT AND PROCESS MANAGEMENT

Test:

```text
Port already occupied
Sidecar startup failure
Sidecar shutdown
Repeated startup/shutdown
Application restart
```

Check for:

```text
Orphan processes
Zombie processes
Released ports
Incorrect service state
```

---

# 15. PHASE 14 – LLM / CLAUDE SAFETY

Claude must not be treated as a trusted compiler.

Preferred flow:

```text
Claude
 ↓
Structured graph action
 ↓
Schema validation
 ↓
Graph validation
 ↓
Graph commit
 ↓
Shape inference
 ↓
Code generation
```

Never rely on Claude producing correct Python code directly.

> **2026-10-06 — implemented for the model-graph/training/data tool surface; found with a fake-provider harness.**
> `npm run test:llm-safety` runs the REAL `sidecar-llm/main.mjs` against a fake OpenAI-compatible server that scripts
> exactly what the "model" does (no real model, no key, only 127.0.0.1). It first reported 11 genuine findings, all now
> fixed (124 assertions, 0 findings; `docs/engineering/LLM_SAFETY.md` has the scenario table and evidence):
> `execTool` called handlers WITHOUT validating arguments (the zod schema only described the tool to the model);
> malformed JSON arguments silently became `{}` and `add_layer` stored `layerType: undefined`; `add_layer` had no layer
> registry or value checks (unknown layers, negative/`Infinity`/`"NaN"` dimensions, wrong types all `ok:true` with an
> action); a failed `add_layer` (unknown `after`) had ALREADY created the node and emitted the action; `connect` accepted
> self-loops and cycles; a non-SSE or truncated provider stream ended as a silent `done`; there was no upstream timeout
> and a client abort did not close the provider connection; a provider error body that echoed the API key reached the
> SSE stream. A decisive extra reason: the frontend's `coerceParams` silently REPLACES junk values by defaults, so
> what the sidecar called "added" could be a different graph than the model believed.
> Fix: one central gate (`invokeTool`: zod validation + strict top-level arguments + explicit JSON-parse errors) shared
> by every provider path and the MCP bridge route; `sidecar-llm/tool-validation.mjs` validates node types and params
> against a COMMITTED catalog generated from the three frontend registries (`npm run gen:layer-catalog`, drift-tested)
> and accepts only values the frontend would store UNCHANGED (+ magnitude cap, prototype keys, unknown keys);
> handlers validate first and mutate only after every check passed (atomic); `connect` rejects self-loops, cycles and
> edges into Input nodes (except from a Manifest), duplicates are an explicit no-op; stream integrity (no data / no
> `finish_reason` → explicit error, no partial tool call runs); `AbortSignal` from the client plus an idle timeout
> (`SPINOML_LLM_UPSTREAM_TIMEOUT_MS`, default 120 s, invalid value refused at startup), SDK `maxRetries: 2`; every
> `status:error` goes through `redactSecrets`. `npm run test:llm-validation-parity` (8284 cases: the sidecar never
> accepts a value the frontend would change or reject; 274 deliberately stricter). Review caught one more hole: zod's
> default object STRIPS unknown top-level arguments (a hallucinated `activation` would vanish while the model believes
> it applied) → now an explicit error (T14b).

---

# 16. PHASE 15 – MCP VALIDATION

Test MCP tools with:

```text
Unknown node ID
Unknown edge ID
Invalid layer
Invalid parameter
Missing parameter
Wrong parameter type
Negative dimension
NaN
Infinity
Invalid connection
Duplicate edge
Invalid graph mutation
```

All invalid operations must be rejected cleanly.

> **2026-10-06 — implemented through the shared gate; MCP bridge not exercised end-to-end.** The inputs the plan
> lists are covered by `test:llm-safety` scenarios: unknown node id (T9), invalid layer (T6), invalid/unknown parameter
> (T14), missing parameter (T4), wrong parameter type (T5), negative dimension / NaN / Infinity (T7: `-5`, `1e999`,
> `"NaN"`, `1e12`, nested negatives), invalid connection (T9/T16), duplicate edge (T9: explicit no-op, no action),
> invalid mutation (T11: deleting Input/Output is allowed but the result now carries a warning). The opencode MCP bridge
> route (`/internal/mcp/<id>/call`) goes through the SAME `invokeTool`, but no test drives the stdio bridge or the real
> `opencode` CLI. There is no `disconnect` tool, so "unknown edge id" does not apply.

---

# 17. PHASE 16 – CLAUDE ADVERSARIAL TESTING

Send deliberately problematic instructions.

Examples:

```text
Connect this layer to a nonexistent node.
Delete the input node.
Create a negative tensor dimension.
Connect incompatible tensors.
Create duplicate edges.
Change output size to an invalid value.
```

Expected behavior:

```text
Claude may propose the operation.
Validator must reject invalid operation.
Invalid state must never be committed.
```

> **2026-10-06 — implemented for the openai-compat path.** The six adversarial instructions are scripted as model
> output: connect to a nonexistent node (T9), delete the input node (T11), negative dimension (T7), duplicate edges (T9),
> invalid output size (T7 `out_features: -3`) → rejected at the tool layer with no action, state unchanged; "connect
> incompatible tensors" is structurally valid, so it is caught one stage later: the semantic mutants of `test:fuzz`
> (wrong `in_features`, Concat/Add mismatch, conv kernel > input) come back from the real sidecar as a structured error
> that `verificationFromInferResult` classifies `invalid` and the training gate refuses. Deleting the Input node is
> accepted (with a warning) — the model is then UNKNOWN/invalid and cannot train, but the delete itself is not blocked.

---

# 18. PHASE 17 – DATASET RELIABILITY

Test all supported dataset formats documented by the project.

Test:

```text
Valid dataset
Empty dataset
Missing file
Corrupt file
Wrong dtype
NaN
Infinity
Single sample
Large dataset
Unicode filename
Spaces in path
Relative path
Absolute path
```

Errors must be explicit.

Never silently turn a dataset error into an empty dataset.

> **2026-09-14 — reliability slice landed.** `scripts/test-datasets.py`
> (`npm run test:datasets`) drives 84 checks over every kind
> (tabular/image_folder/graph_folder/tensor/molecule/protein/pyg/huggingface/
> manifest) across all listed scenarios. Invariant: ok:true ⇒ real non-empty
> tensor; else explicit `error` string — never a 500, never a silent empty,
> never a hang. Bugs it caught & fixed in `dataset_handlers.py`:
> - empty CSV + `features` option → infinite pad-loop that wedged the worker
>   (regression-locked in the harness);
> - `_sample_tabular` unguarded `read_csv` (corrupt file → crash) — now explicit;
> - `_stats_tabular` `describe()` on an empty frame (header-only CSV) → crash;
> - `_sample_tensor_file` 0-element archive → silent empty `ok:true`;
> - missing file → uniform "file not found" everywhere (inspect/stats/sample);
> - image-folder iteration had no OSError guard (unreadable dir → crash).
> Stats for protein/HF stay a "limited to inspect" note; pyproject-eager parquet
> untested here (pyarrow absent from env — flagged in TEST_MATRIX.md). Next:
> PHASE 18 fingerprinting.

---

# 19. PHASE 18 – DATASET FINGERPRINTING

Scientific experiments must identify the dataset used.

Create a stable dataset identifier using appropriate metadata and/or content hashing.

Store:

```text
Dataset source
Dataset configuration
Split configuration
Dataset hash/identifier
```

Avoid relying only on human-readable dataset names.

> **2026-09-14 — implemented.** Sidecar `inspect` now attaches a stable SHA-256
> `fingerprint` to every ok result. Modes: `content` (single-file kinds: tabular
> via `_table_path`, tensor, molecule, protein), `structure` (image/graph folders:
> canonical sorted relpath+size listing — copy/rename-stable, no mtimes),
> `config+content` (manifest + referenced table), `reference` (pyg/huggingface:
> pins the ref file only — remote data NOT pinned). Handlers: `_fingerprint_for` /
> `_sha256_file` / `_fingerprint_dir` / `_fingerprint_manifest`.
> The TS side caches it with the inspect (by relpath) and **freezes it into
> `run.json`** (`DatasetConfig.fingerprint`, both `startRun` and `startEvalRun`),
> alongside `path`/`relpath`/kind/columns (= source + config) and the Split node's
> `val_ratio`/`seed` in `training` (= split config). `train.py` emits
> `run.provenance` (dataset + split + fingerprint_id) and `dataset.fingerprint`,
> re-hashing the primary file just before loading (`_verify_fingerprint`) so a run
> launched later on changed bytes fails loudly. `test:datasets` extended with
> determinism / copy-stability / content-change / n_files assertions (124 checks).
> Known gap: remote workspaces can't inspect pre-12b → fingerprint absent there;
> `reference` mode cannot pin data the sidecar never sees.

---

# 20. PHASE 19 – TRAIN / VALIDATION / TEST INTEGRITY

Check for accidental overlap.

Test:

```text
Same sample in train and validation
Same sample in train and test
Duplicates across splits
```

Record split strategy:

```text
Random
Stratified
Grouped
Time-based
Predefined
```

Do not silently change a user's split strategy.

> **2026-09-14 — implemented.** Split node gains `strategy` (random/stratified/
> grouped/time-based/predefined), compiled → `TrainingConfig.split_strategy`,
> frozen into run.json at launch. `train.py` validates it up front: only 'random'
> is implemented; any other strategy fails loudly with an explicit "will NOT be
> silently changed to random" message (no silent fallback). After the split the
> trainer asserts disjoint train/val subsets (`split.integrity` event with
> overlap=0, `overlaps` preview, strategy/seed/val_split) and refuses to train
> on a leaking split (fail-closed gate for future custom split methods). Strategy
> is also recorded in `run.provenance.split.strategy`. verify:traingen covers
> integrity + a negative strategy-guard case.

---

# 21. PHASE 20 – TRAINING SNAPSHOT

At training start, create an immutable snapshot containing:

```text
Graph
Model configuration
Training configuration
Dataset configuration
Preprocessing configuration
Seed
```

The running experiment must not depend directly on mutable UI state.

If the user edits the graph after training starts, the running experiment must not change.

> **2026-09-14 — implemented.** The executor already froze run.json /
> model.spinoml / model.py / train.py into the run dir at launch (trainer runs
> detached — structurally cannot depend on mutable UI state). Phase 20 adds the
> explicit contract: `startRun`/`startEvalRun` build a `snapshot` section
> (src/training/snapshot.ts) — sha256 of graph + model_py (frontend
> crypto.subtle) plus the graph's DataOp preprocessing scripts — frozen into
> run.json. train.py re-hashes the RUN-DIR copies (`_verify_snapshot`) and
> emits `run.snapshot`; on any drift it FAILS LOUDLY before training. The
> verify:traingen snapshot section proves both the ok:true verify path and that
> a mutated model.py is caught with no training. (R026 → ADDRESSED.)

---

# 22. PHASE 21 – TRAINING CONFIGURATION

Store:

```text
Experiment ID
Git commit
Graph
Generated model
Dataset identifier
Dataset split
Preprocessing
Seed
Optimizer
Learning rate
Scheduler
Batch size
Epochs
Loss
Metrics
Device
dtype
Software versions
Hardware information
```

> **2026-09-14 — implemented.** The run-time configuration is emitted once
> `train.py` emits `config.env` after the model builds, with `python`,
> `torch`, `cuda`, `numpy` versions, `device`, `gpu`/`gpu_mem_mb`, `dtype`,
> `cpus`, `ram_bytes`, and best-effort `git_commit` (workspace root; absent
> when not a git repo). The same snapshot is written into `metrics.json` at
> `run.done`. verify:traingen now asserts `config.env` is present and carries
> `python`/`torch`/`device`/`dtype` strings. The full training configuration
> (graph/model/seed/optimizer/loss/metrics/split/preprocessing) was already
> frozen into `run.json` by Phases 18–20.

Use the actual fields supported by the project.

---

# 23. PHASE 22 – SEED AND DETERMINISM

Determine all random sources used by training.

At minimum investigate:

```text
Python random
NumPy
PyTorch CPU
PyTorch CUDA
DataLoader workers
```

Set seeds where appropriate.

If full determinism is impossible:

Document exactly why.

Do not claim:

```text
Fully reproducible
```

when the underlying operation is nondeterministic.

> **2026-09-14 — implemented.** `train.py` seeds all four random sources the
> auditor lists — Python `random`, NumPy, torch CPU (`torch.manual_seed`), and
> all CUDA devices (`torch.cuda.manual_seed_all`) — plus cuDNN (
> `deterministic=True`, `benchmark=False`) and `use_deterministic_algorithms(
> True, warn_only=True)` so nondeterministic ops log instead of silently
> diverging. DataLoader uses a seeded `torch.Generator` + `worker_init_fn`
> (per-worker salted) so multi-worker shuffling is reproducible. A
> `run.determinism` event records the seed + every backend state at runtime.
> CAVEAT documented, not claimed away: CUDA `atomicAdd` reductions are
> nondeterministic even with these flags → bit-level GPU reproducibility is NOT
> claimed; CPU runs are reproducible for equal seed+stack+inputs.
> verify:traingen asserts `run.determinism` (seed=7, cuDNN flags, python/numpy
> seeds) in the e2e classification run.

---

# 24. PHASE 23 – SCIENTIFIC SMOKE TEST

Create a tiny local synthetic experiment.

Example:

```text
100 samples
10 input features
2 classes
```

Model:

```text
Linear
ReLU
Linear
```

Training:

```text
2–5 epochs
```

Verify:

```text
Training starts
Loss is finite
Loss changes
Metrics are finite
Checkpoint exists
Logs exist
Metadata exists
Training exits successfully
```

This test must be fast enough for regular CI execution.

> **2026-09-14 — implemented.** `npm run verify:smoke` (`scripts/verify-smoke.ts`)
> generates a synthetic 100-sample, 10-feature, 2-class tabular CSV (seeded
> xorshift PRNG — reproducible), trains a Linear(10,64)→ReLU→Linear(64,2) model
> for 5 epochs (Adam, CrossEntropyLoss, accuracy metric, 0.2 random val split),
> runs the REAL `train.py` end-to-end, then asserts:
> - training exits `done`, 5 `epoch.end`s emitted
> - train → val loss finite; loss strictly decreases across epochs
> - accuracy finite and in [0,1]
> - `checkpoints/best.pt` exists; `metrics.json` present with
>   `status=done` + `best_val_loss` + `epochs` + `n_params`
> - `run.provenance` + `run.determinism` events emitted
>
> ~4s total in CI — fast enough for regular execution.

---

# 25. PHASE 24 – TRAINING FAILURE TESTS

Deliberately cause:

```text
Invalid dataset
Invalid model
Invalid optimizer
Invalid learning rate
Missing output directory
Unwritable output directory
NaN input
Training process termination
```

Expected result:

```text
FAILED
```

Never:

```text
SUCCESS
```

Never leave the job indefinitely in:

```text
RUNNING
```

> **2026-09-14 — implemented.** `npm run verify:failures`
> (`scripts/verify-failures.ts`) deliberately causes EACH listed failure mode
> against the REAL trainer and asserts FAILURE (exit≠0, `status`≠done,
> `run.failed` event with a stage), never SUCCESS and never stuck in RUNNING:
> - invalid dataset (deleted CSV) → `fail("dataset")`
> - invalid model (syntax error) → `fail("model")`
> - invalid optimizer (unknown kind) → `fail("model")`
> - invalid learning rate (negative lr) → torch ValueError → `fail("model")`
> - missing output dir (a FILE named `checkpoints`) → crash, no data
> - unwritable output dir (chmod 555) → crash, no data
> - NaN input → `fail("dataset")` (NEW hardening, below)
> - mid-training SIGKILL → process dies, no `run.done`, status≠done
>
> Hardening added: `load_tabular` in training_template.py previously
> `fillna(0.0)`'d non-finite feature cells SILENTLY — a NaN/inf dataset would
> train on an imputed zero matrix and produce believable-but-garbage metrics.
> It now raises a ValueError naming the affected columns, → `fail("dataset")`.
> 9 cases × ~2s in CI.

---

# 26. PHASE 25 – NUMERICAL FAILURE DETECTION

Monitor appropriate training values for:

```text
NaN
Infinity
```

At minimum investigate:

```text
Loss
Metrics
Gradients
```

If numerical instability makes the result unusable, the run must not be reported as successful.

Use an explicit failure reason where appropriate.

> **2026-09-14 — implemented.** The trainer monitors the values this phase
> lists and fails with the explicit stage `numeric` (run.failed + status
> failed, never success) on the first non-finite hit:
> - **train loss** — `require_finite("train loss", …)` per batch, before backward
> - **gradients** — every param's `.grad` checked before each optimizer step;
>   the failure names the first non-finite parameter
> - **val loss / val accuracy / per-head metrics** — checked after each
>   `evaluate()` pass, and on the eval-only path too
> - `_is_finite` handles scalars and tensors (`.isfinite().all()`)
>
> verify:failures gained a new case: a model that turns its output NaN after N
> forwards → `run.failed` with stage `numeric`, status failed, exit ≠ 0.

---

# 27. PHASE 26 – CHECKPOINT CORRECTNESS

Test:

```text
Train
Save checkpoint
Stop
Load checkpoint
Resume
```

Verify that checkpoints preserve the necessary state:

```text
Model state
Optimizer state
Scheduler state
Epoch
Global step
Random state where supported
Experiment configuration
```

> **2026-09-14 — implemented.** Checkpoints (best.pt + last.pt) now preserve
> the FULL state list: `model_state`, `optim_state`, `sched_state`, `epoch`,
> `global_step` (new — tracked across epochs, restored on resume), `rng`
> (torch CPU + all CUDA streams, NumPy, Python stdlib — captured via
> `_rng_state()` and restored via `_restore_rng()` on resume), `config` (the
> frozen run.json dict), plus `best_val`/`classes`/`head_classes` as before.
> A cancelled run now saves last.pt for the last COMPLETED epoch, so the
> stop → load → resume loop works.
> `npm run verify:checkpoint` (`scripts/verify-checkpoint.ts`) covers the
> whole audit: train → save (all keys asserted) → resume (start_epoch,
> global_step, scheduler state, epoch continuity) → cancel (resumable
> checkpoint at last completed epoch). 31 checks, ~30s in CI.

---

# 28. PHASE 27 – ATOMIC CHECKPOINT WRITES

Where appropriate, use an atomic write strategy:

```text
Write temporary checkpoint
        ↓
Flush/close
        ↓
Atomic rename
```

The goal is to avoid a partially written checkpoint replacing the last valid checkpoint.

> **2026-09-14 — implemented.** `_atomic_save()` in training_template.py: all
> three checkpoint writes (best.pt, last.pt, cancel-save) serialize to
> `<name>.tmp` in the same directory, fsync the file, `os.replace()` onto the
> final name, then best-effort fsync the directory so the rename itself
> survives a crash. A stray .tmp is cleaned up in a finally block.
> verify:checkpoint simulates a partial write + crash via a monkeypatched
> torch.save → the previous valid checkpoint still loads, no .tmp leftover.

---

# 29. PHASE 28 – CHECKPOINT CRASH TEST

Simulate:

```text
Training
 ↓
Checkpoint write
 ↓
Process termination
 ↓
Restart
```

Verify that the system either has:

```text
Previous valid checkpoint
```

or:

```text
New valid checkpoint
```

and never silently accepts a corrupted checkpoint.

> **2026-09-14 — implemented (Phases 27+28).** Checkpoint writes are atomic
> (Phase 27: `_atomic_save` — tmp file + fsync + `os.replace` + dir fsync), so
> a crash mid-save leaves either the previous or the new VALID checkpoint,
> never a truncated one. Phase 28 tests this end-to-end in
> `verify:checkpoint`: a SIGKILL right after a checkpoint event leaves every
> .pt on disk loadable with no .tmp leftovers, and a restart resuming from
> that checkpoint continues cleanly (`run.resumed`). A deliberately corrupted
> .pt is rejected loudly — `run.failed` stage `resume`, status failed — never
> silently accepted.

---

# 30. PHASE 29 – METRIC CORRECTNESS

Check:

```text
Batch loss
Epoch loss
Validation loss
Metric aggregation
```

Pay special attention to batches of different sizes.

Do not incorrectly average batch averages when weighted averaging is required.

Create explicit tests.

> **2026-09-14 — implemented.** The trainer was already weighted-correct
> (train `running += loss.item() * bs` / `n_seen += bs`; val `vrun += … * bs` /
> `vseen`; per-head loss/accuracy over LABELED rows) — Phase 29 makes it
> provable: `npm run verify:metrics` trains a FROZEN model (lr=0) on 97 rows
> with batch 32 (last batch = 1 sample) and compares every reported number
> against an independent per-sample Python reference over the identical split:
> epoch train loss, val loss, val accuracy all match within 1e-4, while the
> naive mean-of-batch-means provably differs (>1e-3) — the test would CATCH
> an unweighted regression.

---

# 31. PHASE 30 – LOCAL JOB STATE MACHINE

Define and test training states.

Example:

```text
CREATED
 ↓
STARTING
 ↓
RUNNING
 ├──> SUCCEEDED
 ├──> FAILED
 └──> CANCELLED
```

Prevent invalid transitions.

For example:

```text
SUCCEEDED → RUNNING
FAILED → RUNNING
CANCELLED → SUCCEEDED
```

must not happen accidentally.

> **2026-09-22 — implemented.** `transition_status()` in `training_template.py`
> gates trainer writes: queued→running at launch, running→done/failed/cancelled.
> A late write (done racing user cancel, running after crash) is REJECTED so
> CANCELLED→SUCCEEDED, FAILED→RUNNING, SUCCEEDED→RUNNING are prevented.
> Local `stop_training_run` and remote `ssh_stop_training_run` only flip
> non-terminal states. `npm run verify:states` covers all 43 state checks.

---

# 32. PHASE 31 – TRAINING EVENT ORDERING

Test out-of-order events.

Example:

```text
RUNNING
EPOCH 5
FAILED
EPOCH 6
```

A late event must not overwrite a final state.

Final states should be protected from stale asynchronous updates.

> **2026-09-22 — implemented.** Strict append-only `events.jsonl` with
> read-boundary protection in `src/training/events.ts`. `parseFinalEvents`
> truncates at the first terminal event (`run.done`, `run.failed`, `run.cancelled`),
> ensuring out-of-order lines (e.g. trailing EPOCH after FAILED) are dropped.
> `latestWinsGuard` drops older in-flight tail/read responses that resolve after
> a newer snapshot. Verified by `npm run verify:events` (26 checks).

---

# 33. PHASE 32 – CANCELLATION

Test:

```text
Cancel before start
Cancel during startup
Cancel during training
Cancel after completion
Double cancellation
```

The resulting state must be consistent.

> **2026-09-22 — implemented.** SIGTERM/SIGINT handled via `_Cancelled` exception
> unwinding in `training_template.py`. A cancelling run emits exactly one
> `run.cancelled`, updates `metrics.json`, saves a resumable `last.pt` at the last
> completed epoch (epoch -1 during startup/epoch 0), and shields completed runs
> from post-done cancellation writes. Verified by `npm run verify:cancel` (28 checks).

---

# 34. PHASE 33 – JOB SUBMISSION IDEMPOTENCY

Test:

```text
Submit job
Network timeout
Client retries
```

Ensure the system does not accidentally submit two jobs.

If perfect idempotency is impossible, use a submission identifier and verify the existing job before retrying.

> **2026-09-22 — implemented.** Remote SSH start claims the run directory atomically
> (`mkdir` claim). Concurrent starts have exactly one winner; duplicate launch attempts
> are blocked before start. Retries after a lost response detect the recorded PID and return
> success without re-executing. Verified by `npm run verify:submission` (7 checks).

---

# 35. PHASE 34 – SSH RELIABILITY

If SSH functionality is supported, test:

```text
Successful connection
Authentication failure
Host unavailable
Connection timeout
Connection loss
Reconnect
Missing remote directory
Permission denied
SFTP failure
Remote command failure
```

Errors must be explicit.

> **2026-09-22 — implemented.** `ssh_failure()` in `src-tauri/src/ssh.rs` maps exit
> codes and stderr patterns into distinct, explicit failure classifications:
> auth/host-key failure, DNS resolution failure, connection timeout, connection reset/loss,
> missing remote directory, permission denied, remote filesystem full, SFTP failure,
> and remote command error. Uses multiplexed ControlMaster sockets with keepalives
> (`ServerAliveInterval=20`, `ConnectTimeout=10`). Verified by `npm run verify:ssh` (17 checks)
> and Rust unit tests in `ssh_failure_tests`.

---

# 36. PHASE 35 – SSH CREDENTIAL SAFETY

Search logs and workspace files for:

```text
Passwords
Private keys
Tokens
OAuth secrets
SSH credentials
```

They must never be written to ordinary logs or experiment artifacts.

> **2026-09-22 — implemented.** Credential safety audit and enforcement. Connection
> store delegates authentication to system `~/.ssh/config` and `ssh-agent` without
> persisting passwords or keys. Runtime environment recording (`_env_info`) captures
> explicit platform/framework diagnostics without leaking `os.environ`. `sanitize_credentials`
> scrubs private keys, URL passwords, and API tokens from SSH diagnostic outputs.
> Verified by `npm run verify:credentials` (8 checks).

---

# 37. PHASE 36 – SLURM RELIABILITY

If SLURM support is available, test:

```text
Submit success
Submit failure
Pending
Running
Completed
Failed
Cancelled
Unknown
Communication failure
```

Persist the remote job ID.

> **2026-09-22 — implemented.** `build_sbatch` generates the correct `#SBATCH` header for every SlurmConfig field (partition/time/mem/cpus/gres/account/qos/modules/pre_run_script; job-name sanitized, defaults applied) and freezes it as `train.sbatch` alongside `pid` (`slurm:<jid>`) on successful `sbatch` (parsed via `Submitted batch job <id>`); `sbatch` failure (`MLF_SUBMIT_FAILED`) surfaces as `sbatch failed: <stderr>`. `reconcile_slurm_status` in `training.rs` maps all 8 states: `squeue %T` while queued (`PENDING`/`CONFIGURING`→`queued`, otherwise `running` unless terminal file already `done`/`failed`/`cancelled`), `sacct State` after queue (`COMPLETED`→`done`, `CANCELLED`→`cancelled` with `by <uid>` suffix stripped, `FAILED`/`TIMEOUT`/`OUT_OF_MEMORY`/`NODE_FAIL`→`failed`), and `UNKNOWN`/communication loss falls back to `reconcile_status` (`queued`/`running` without alive → `failed`, never silently `running`). SLURM stop uses `scancel <jid>` (direct uses `kill -TERM -<pid>`). Verified by `npm run verify:slurm` (33 checks: 4 sbatch + 9 reconciliation Rust unit tests + probe/persistence/marker string checks).

---

# 38. PHASE 37 – REMOTE JOB RECOVERY

Critical test:

```text
Submit remote training job
Close SpinoML
Wait
Restart SpinoML
Reconnect
Query actual remote job
```

The application must recover the actual remote state.

Do not rely only on UI state saved before shutdown.

> **2026-09-22 — implemented.** Recovery is file- and scheduler-backed, never `localStorage`. `useTrainingStore.runs` is memory-only (`training/store.ts:145` `[]`); `App.tsx:257` blanks `runs:[]` on every `workspaceRoot` reconnect and `ExperimentsExplorer:32` on `currentId` change, then both call live `training.list()` → `ssh_list_training_runs` which per-run cats `status`/`pid`/`run.json`/`metrics.json`/`events.jsonl` and probes liveness via `squeue -j <jid> -h -o '%T'` (SLURM) or `kill -0 <pid>` (direct) + `sacct State` fallback, reconciled by `training.rs:reconcile_status`/`reconcile_slurm_status` (stale `running` without alive → `failed`). Direct runs use `setsid ... & echo $! > pid` (brace-group) and remote direct adds `nohup setsid`; SLURM writes `slurm:<jid>` via `sbatch` — both survive app/ssh close (reparent to init). `connections/store.ts` persists only `alias/root` and `getCurrentConnection()`/`training/backend.ts` dispatch live per call. Lost-response retry is idempotent via the Phase-33 atomic `mkdir` claim. Verified by `npm run verify:recovery` (38 checks: store/App/connections/detached/live query/reconcile/atomic claim + Rust 9-case reuse).

---

# 39. PHASE 38 – UI STATE MUST NOT LIE

Examples:

If saving fails:

```text
Do not display "Saved".
```

If training fails:

```text
Do not display "Completed".
```

If shape inference fails:

```text
Do not display "Valid".
```

If SSH disconnects:

```text
Do not display "Connected".
```

If SLURM state is unknown:

```text
Do not display "Running" unless verified.
```

> **2026-09-22 — implemented.** `training/store.ts:155` `refresh()` now degrades any stale `running`/`queued` run to `unknown` + `alive:false` on `listError` (SSH transport loss) and clears the poll timer — the last successful `runs` no longer lingers as `running` (green pulse) while disconnected; `listError` and `listLoading` remain distinct (never just `loading=false`). `App.tsx:154` `ProjectHeader` derives SSH badge from live `listError` + `status.kind==='error'`: connected → violet `ssh · alias`, disconnected → rose `ssh · alias — nicht verbunden` (with error title) and an error badge when project load fails. `StatusPill.tsx:12` only pulses when `status==='running' && alive`; SLURM unknown is `reconcile_slurm_status` → `failed`/`unknown` never `running` (Phase 36). `inference/store.ts:70` `throw e` that left `inferring` hanging now sets `error`/`offline` + `clearShapesOnNodes()`; `InferenceBadge` shows `ok` only on `ok`, `verifier` is fail-closed (`offline`→`unknown`), `NewRunModal` blocks `invalid`/`unknown`. Save truthfulness: `workspace/store.ts:375` awaits `fs.write` before `dirty:false`, `data/graph/doc.ts:41` + `training/graph/doc.ts:33` set `saved` only on success / `error` on catch, `Toolbar.tsx:25` reports via dialog/alert. Verified by `npm run verify:ui-state` (33 checks).

---

# 40. PHASE 39 – FRONTEND ERROR STATES

External operations should distinguish:

```text
Loading
Success
Error
Timeout
Cancelled
Unavailable
```

Do not represent all failures simply as:

```text
loading = false
```

> **2026-09-22 — implemented (partial + caveat).** Loading/Success/Error/Unavailable are distinct everywhere: `inference/store.ts:7` `Status='idle'|'inferring'|'ok'|'error'|'offline'` with `inferring`→`ok`/`error`/`offline`, `offline in result` vs throw→`error`, `runCounter` stale guard + Phase-39 AbortError fix (latest inferring without successor → `idle` + `clearShapes`, not hang); `training/store.ts:76` `listLoading` vs `listError` + `runs` + `alive` + `StatusPill` 6 styles vs only `loading=false`; `datasets/store.ts:12` `Cached<T>={loading,data,error}` + `missing_dep`; `project/store.ts:16` `'loading'|'loaded'|'error'|'remote-missing'`; `chat/store.ts:32` `'streaming'|'idle'` vs `online` + `pendingAsk`. Timeout/Cancelled remain partly conflated: inference fetch timeout → `offline`/`error` string (no `AbortSignal.timeout`), training SSH `ConnectTimeout=10`/`ServerAlive` maps to `ssh_failure` classified string in `listError` then degraded to `unknown`, not a `timeout` enum; datasets post has no `AbortSignal`; chat `AbortError` swallows to `done` not `cancelled`. No false Success on Timeout/Cancelled — the gap is a missing `timeout`/`cancelled` enum, documented in `LIMITATIONS.md`. Verified by `npm run verify:frontend-errors` (42 checks) + `verify:ui-state` (33).

---

# 41. PHASE 40 – GRAPH REVISION SYSTEM

If not already present, introduce a reliable graph revision mechanism.

Example:

```text
Graph revision 101
```

Every asynchronous operation records the revision it belongs to.

When a response arrives:

```text
Response revision == current revision?
```

If not:

```text
Ignore as stale.
```

Use this for:

```text
Shape inference
Dataset loading
LLM actions where relevant
Other asynchronous graph operations
```

> **2026-09-22 — implemented.** `GraphStore.revision: number` (0→monotonic) bumped on every committed structural change (`addLayer`/`updateNodeParams`/`replaceNodeLayer`/`deleteNode`/`connectNodes`/`onEdgesChange`/`onConnect`/`loadSnapshot`/`resetGraph`); pure `onNodesChange` position/dimensions drags do NOT bump (no spurious invalidation). `inference/store.ts:43` captures `graphRev` + `runCounter` before the await and drops the response if `graphRev !== revision` or `runId !== runCounter` (stale inference after graph edit → no shape overwrite). `datasets/store.ts:14` per-dataset `inspectSeq`/`statsSeq`/`smokeSeq` + `smoke` also captures `graphRev` — a second inspect/smoke on the same dataset before the first returns sees the first dropped, and a graph edit kills a pending smoke. `training/store.ts:121` `refreshSeq` is the same `latestWinsGuard` for run list (stale list after quick workspace switch dropped). `events.ts:71` `latestWinsGuard` was already the reference impl for run detail. Verified by `npm run verify:graph-revision` (32 checks: revision increments, drag not bump, all guards). Training/data graph stores still lack revision (known gap, architecture only).

---

# 42. PHASE 41 – CONCURRENT OPERATIONS

Test:

```text
Save + edit
Edit + inference
Claude mutation + user mutation
Training start + graph edit
Dataset reload + dataset inspection
```

Define which operations are allowed concurrently.

Prevent silent state corruption.

> **2026-09-22 — implemented.** `workspace/store.ts:529` `saveActive` now holds `saveSeq`+`revAtStart`; concurrent `saveActive` → last writer wins via `seq !== saveSeq` check before the py-twin write and before `set dirty:false`; an edit that landed while writing is detected via `revAtStart !== revision` and `dirty` is corrected to `true` via `fingerprintCurrent()` vs stale file hash (no silent loss, dirty indicator truthful). `inference/store.ts:39` `Edit+inference` is debounced 200 ms + `runCounter`+`graphRev` + `AbortController`; stale shape is dropped before `applyShapesToNodes` (`test:races` proves flatten vs linear). `chat/store.ts:158` `if status==='streaming' return` blocks second turn while one streams; LLM `dispatchAction` goes through `GraphStore` `validateGuard`/`coerceParams`/`revision` so no invariant break (semantic last-wins, not torn). `training/store.ts:215` `startRun` freezes `modelContent`+`modelPy`+`buildRunSnapshot` sha256 — the run dir is immutable and `train.py` ` _verify_snapshot` fails loudly on drift, so `Training start+graph edit` cannot corrupt the running experiment (canvas vs run may diverge, but run's bytes are proven). `datasets/store.ts:57` `inspectSeq`/`statsSeq`/`smokeSeq` + `smoke` `graphRev` + `refreshSeq` (Phase 41) make `Dataset reload+inspect` latest-wins: stale `inspect`/`stats`/`smoke`/`refresh` after a newer one is dropped, and a graph edit kills a pending smoke. Verified by `npm run verify:concurrent` (44 checks) + `verify:graph-revision` + `test:races`.

---

# 43. PHASE 42 – TRAINING IMMUTABILITY

Once a training run begins:

```text
Graph
Dataset configuration
Training configuration
Generated code
```

must be tied to the run snapshot.

Later UI changes must not modify the running experiment.

> **2026-09-22 — implemented (already frozen, now regression-locked).** `training/store.ts:217` `startRun` freezes `modelContent` via `fs.read(modelRelpath)` + `modelPy=generateFromSnapshot(parseFile(content)).code` + `snapshot=buildRunSnapshot(modelContent,modelPy)` (`snapshot.ts:43` sha256 of exact strings) and hands all three plus `dataset{fingerprint}`+`training`+`backend` verbatim to `training.start`; `training.rs:447`/`ssh.rs:919` write `run.json`/`model.spinoml`/`model.py`/`train.py` (bundle copy) atomically into `experiments/runs/<id>/` (`already exists` if duplicate) and launch detached `setsid`/`sbatch` (reparent to init, survives app close) — the run never reads `useGraphStore` after launch. `training_template.py:342` `_verify_snapshot` re-hashes `RUN_DIR` copies vs `run.json.snapshot` before training and `fail("snapshot",...Refusing)` on drift; dataset fingerprint is re-emitted as `run.provenance`. Later `GraphStore.revision` bumps on `addLayer`/`updateNodeParams`/etc. never touch the run dir, so UI edits cannot mutate the running experiment (canvas vs run may diverge if user edited dirty canvas before saving, but run's bytes are proven). Verified by `npm run verify:immutability` (27 checks) + `verify:traingen` snapshot section.

---

# 44. PHASE 43 – GENERATED PYTHON SECURITY

Audit all values inserted into generated Python:

```text
Layer names
Parameters
Paths
Labels
Dataset information
User input
LLM-generated values
```

Avoid unsafe string interpolation.

Never allow user-controlled input to become arbitrary executable Python unintentionally.

> **2026-10-06 — implemented.** Audit first, with a tokenizer-based oracle: a hostile
> sentinel identifier must never become a Python NAME token and every output must
> `ast.parse`. Honest findings: the model generator's `dataset-ref`/`column-single`/
> `columns-multi` params only occur on Input nodes that never reach `serializeParam`
> (no injection reproduced there), but newline/NUL/line-separator characters in
> `name` params, the data/training generators' own `pyStr` (escaped only `\` and `'`,
> no `\n`/`\r`/NUL), comment sinks (`# ── … label ──`, `# Quelle: …`), numeric
> NaN/Infinity/string values and `Custom.init_args` produced uncompilable or
> injectable Python (620 failing cases before the fix). Fix: one shared, pure
> `src/codegen/pyLiteral.ts` (`pyStr` round-trips through `ast.literal_eval`,
> `pyComment`, `pyIdent`, `pyFloat`/`pyInt`/`pyIntList`) used by all three generators.
> `npm run verify:codegen-security` = 5362 cases, 0 failing; benign output is
> byte-identical (one deliberate change: `weight_decay=0` → `0.0`); `verify:codegen`,
> `test:determinism`, `verify:traingen`, `verify:smoke` unchanged.
> **Intentional code sinks** (`Custom.source`, `Custom.init_args`, `DataOp.script`,
> `CustomScript.code`) cannot be escaped — they ARE code. They are listed and asserted
> exactly by the harness and are covered by the **code-trust gate** instead: content-addressed
> store (`src/trust/`, sha256 of `kind\0source`), kept outside params/`.spinoml` because
> `coerceParams` lets unknown keys through. Unapproved code is never sent to `/infer`,
> activations, dataset smoke, the training verifier, `startRun` or `startEvalRun`
> (inference reports `status:'untrusted'`); only human UI events approve (code-field edit,
> eject, built-in template, dialog click). Review found what the council design and the
> first workers missed: `init_args` was executable but ungated (quote-free
> `exec(bytes([...]).decode())` passed a "safe characters" whitelist, which also dropped
> legitimate `activation='relu'` and so silently built a different model) → own
> `custom-init-args` blob; three self-approval paths (blur on a store-mirrored value,
> stale-buffer flush on unmount, modal close without typing) → explicit `userEdited` rule.
> Verified by `verify:code-trust` (110), `verify:code-trust-wiring` (78). **Caveat:** this is a
> frontend gate — the sidecars still accept direct HTTP calls (CORS `*`, no token: R013/R014,
> Phase 77/78); editing counts as review; `train.py` does not re-check `code_trust`
> (docs/engineering/LIMITATIONS.md §2/§3).

# 45. PHASE 44 – COMMAND INJECTION REVIEW

Search for:

```text
shell=True
eval
exec
dynamic shell strings
unsafe subprocess calls
```

Also inspect Rust command construction.

Especially review:

```text
SSH commands
SLURM commands
Python execution
filesystem commands
```

Use argument arrays / safe APIs where possible.

> **2026-10-06 — implemented (Rust part compiled and tested: `cargo check` + 22 `cargo test`s pass in the `mlforge-dev` conda env).** Search result: no `shell=True`/`os.system`
> anywhere; `exec(compile(...))` of model code in the torch sidecar is by design (now gated, see
> Phase 43). Real defects found and fixed: (1) LLM `run_script` `args` were concatenated unquoted
> into the shell command → `splitArgs`/`quoteArgv` (`sidecar-llm/shell-safety.mjs`), proven by
> round-tripping hostile strings through a real `sh`, and the confirmation text now shows the
> quoted argv that really runs; script targets get a `./` prefix so `--wrap=…` can't become an
> sbatch option. (2) torch `/run_script`: the fallback `["bash","-lc", relpath]` executed the
> FILE NAME as a shell command string (write `x; touch /tmp/pwn`, then run it) and `sbatch rel`
> accepted option-like names → relpath whitelist (`[\w.+@%,=/ -]`, no leading `-`, extension
> `.py/.sh/.sbatch/.slurm`), `./`-prefixed argv, nothing written when rejected. (3) `/deps/check`
> and `/deps/install` passed arbitrary pip arguments (`--index-url`, VCS URLs, local paths) →
> `sidecar-torch/deps_policy.py` (plain PEP 508 name+version only; `torch`/`pip`/`setuptools`
> refused; `--` before specs). (4) `download_to_datasets` SSRF + an exfiltration chain (internal
> URL → file → `read_file` → LLM) → `checkDownloadUrl`, fail-closed `isBlockedAddress` (IPv4/IPv6
> incl. mapped/NAT64/6to4/Teredo), local `safeFetch` resolving DNS and checking every redirect hop,
> curl `--proto/--max-redirs` on the remote branch. (5) ssh: `remote_sidecar.rs`
> `ensure_remote_sidecar` passed the webview-supplied alias to `ssh` UNVALIDATED (a single argv
> element like `-oProxyCommand=…` is local command execution) and every spawn lacked `--` →
> `validate_alias` (now rejects a leading `-`, shared by `ssh.rs`/`pty.rs`/`remote_sidecar.rs`),
> `--` before every target, `shell_quote` on `SPINOML_PYTHON` in the `sh -c` launcher.
> Verified by `verify:command-injection` (162), `test:deps-policy` (34), `test:run-script` (125,
> mocked `subprocess.run`), `verify:sidecar` (new rejects). **The Rust edits compile and their
> tests pass** (`conda run -n mlforge-dev cargo check && cargo test`: 22 passed, incl. the new `alias_validation_tests`;
> an earlier version of this note wrongly said no toolchain existed — it only checked the base shell PATH). Still open: path scoping (Phase 45/46),
> pickle (47), sidecar auth/CORS (77/78); the torch `/run_script` endpoint still takes `root` from
> the payload.

# 46. PHASE 45 – PATH SECURITY

Test paths containing:

```text
Spaces
Unicode
Relative paths
Absolute paths
..
Symlinks
Missing paths
Long paths
```

Ensure path operations behave predictably.

> **2026-10-06 — implemented (tests + resolver).** Path behaviour is now specified by two
> matrices that run the same hostile inputs through both resolvers: `npm run verify:paths`
> (Node `sidecar-llm/path-scope.mjs`, 48 checks) and `npm run test:scope` (Python
> `sidecar-torch/scope.py`, 88 checks) — spaces, unicode, emoji, 255/256-byte segments, 4097-char
> paths, empty, NUL, `..`/`a/../../x`, absolute, `~`, backslash names, `//`, `/proc/self/*`,
> sibling-prefix confusion (`ws` vs `ws-evil`), missing files/parents for writes, symlink chains,
> dangling links, loops (Python's `os.path.realpath` silently swallows loops, so `scope.py`
> resolves component-wise and raises `PATH_INVALID`). Relative paths are rejected where an
> absolute one is required.

---

# 47. PHASE 46 – PATH TRAVERSAL

Check all filesystem boundaries.

Prevent unintended access through:

```text
../
absolute paths
symlinks
```

where the application expects paths to remain inside a workspace/dataset directory.

> **2026-10-06 — implemented (mechanism complete; local default NOT yet enforcing).**
> Rule: the fully resolved path must lie under the realpath of an allowed root OR of a
> user-configured symlink target — NOT "no symlink may leave the root", because the user
> symlinks data onto cluster scratch (`/work2`). Torch sidecar: `sidecar-torch/scope.py`;
> every request path (`/dataset/inspect|stats|smoke`, `/activations` abspaths + checkpoint,
> `/run_script` root + script) and every path derived from FILE CONTENT (manifest table and
> branch sources incl. absolute / `../`, `contains`-matches that are symlinks, structure-path
> columns, prep cards, image/graph folder files, cache dirs) goes through `check_path` and uses
> the RESOLVED path. Roots come from `SPINOML_ALLOWED_ROOTS`, `SPINOML_SYMLINK_TARGETS` and
> `~/.cache/spinoml/scope.json` (regular file, owned by the uid, not group/world-writable,
> re-read on change). Errors: HTTP 403 `SCOPE_DENIED` / `PATH_SYMLINK_OUTSIDE` /
> `SCOPE_UNCONFIGURED` with a one-line fix; data-level `ok:false` + `error_code` inside handlers;
> `/health.scope` shows mode + counts, never paths. Node sidecar: `resolveInWorkspace` on all
> local `read_file`/`list_dir`/`write_file`/notes/dataset/download paths. Remote HPC sidecar:
> `remote_sidecar.rs` now exports `SPINOML_ALLOWED_ROOTS="$ROOT…"` (Rust edit; `cargo check` passes).
> Verified with a real sidecar over HTTP (16 hostile scenarios incl. symlinked files/dirs,
> hostile manifests, `/proc/self/environ`, `run_script` root `/`: no marker leaked, nothing
> written outside). **Honest status:** with no root configured the sidecar runs in the visible
> mode `unconfigured-open` (warning + `/health`), because Rust has no channel to the sidecar
> and Rust had no writer for it yet — the local app is NOT scoped until Rust writes `scope.json`
> (or the user sets `SPINOML_ALLOWED_ROOTS`); `SPINOML_REQUIRE_SCOPE=1` makes it fail closed.
> Scoping does not protect the exec endpoints (`/infer`, smoke, activations run model `code`)
> — that is the sidecar token (Phase 77/78). Rust `resolve()` is still lexical (R016, LIMITATIONS).

> **2026-10-06 — Rust half implemented (R016).** `src-tauri/src/scope_file.rs`: the shell writes
> `scope.json` (atomic, 0600, parent 0700, canonical roots, `symlink_targets` + unknown keys preserved,
> an untrusted existing file never trusted) whenever the local workspace is picked/opened and clears the
> roots on close, so the managed sidecars switch from `unconfigured-open` to `enforced` as soon as a
> workspace exists (before that they stay open — `SPINOML_REQUIRE_SCOPE` is deliberately not enabled
> because the GUI flows could not be tested here). `resolve()` now validates the FULLY RESOLVED path
> against the canonical root + configured symlink targets (`check_resolved`; dangling symlinks, loops
> and symlinks to outside are rejected; the lexical path is returned so delete/rename act on a symlink
> itself, never on the real dataset directory behind `datasets -> /work2/...`). `list_workspace`
> follows allowed out-of-tree links with cycle protection, `training.rs` run/checkpoint paths go through
> the same check. Review found and had fixed: a dangling-symlink escape, delete/rename acting on the
> symlink TARGET, `list_workspace` aborting for allowed out-of-tree links, and an unchecked trust
> predicate when reading `symlink_targets`. Verified: `cargo test` 86, `test:scope` 100 (new golden
> fixture shared with the Rust writer test). Not verified: the real GUI; Windows/macOS (unix-only code
> returns an error there).

---

# 48. PHASE 47 – UNSAFE DESERIALIZATION

Search for:

```text
pickle
torch.load
eval
exec
unsafe YAML
dynamic imports
```

Review every occurrence.

A dataset file must not automatically be considered trusted.

Where safe serialization is possible, use it.

> **2026-10-06 — implemented.** 14 `torch.load(..., weights_only=False)` sites (torch sidecar
> main.py, dataset_handlers.py ×7, training_template.py ×6) now call one safe loader
> (`sidecar-torch/safe_load.py`): `weights_only=True` plus an allow-list of the real artifact types
> (PyG `Data`/`HeteroData`/storages, numpy RNG-state arrays). Measured first: with
> `weights_only=True` alone the PyG `AF-*.pt` graphs from the user's pipeline FAIL to load while
> checkpoints load — after registering PyG's classes all 18 real artifacts under
> `examples/reaction-workspace` load. Anything else raises `UnsafePickleError` →
> `error_code: UNSAFE_PICKLE` (datasets), a visible note (activations checkpoint), `run.failed`
> (trainer). The trainer cannot import sidecar modules, so the loader is embedded as a marker-
> delimited block in `training_template.py`; `test:safe-load` enforces byte-equality plus a static
> audit (one `weights_only=False`, only in the escape hatch). Escape hatch:
> `SPINOML_ALLOW_UNSAFE_PICKLE=1` (process-wide, loud stderr warning, recorded in `config.env` and
> `run.provenance` as `unsafe_pickle`). `npm run test:safe-load` (85 checks) really tries to
> execute pickled `os.system`/`Popen`/`exec`/`eval` payloads, also hidden inside a PyG `Data`
> attribute, a numpy object array, a dict key and nested lists: sentinel never created. Other
> deserialization: only JSON (`json.load`, `JSON.parse`) and `np.load(allow_pickle=False)`; no yaml,
> joblib, dill, marshal. Not covered: the escape hatch trusts the whole process, torch < 2.4 has no
> allow-list API (plain tensors/dicts only).

---

# 49. PHASE 48 – RUST ERROR HANDLING

Audit:

```text
src-tauri
filesystem
SSH
SFTP
PTY
sidecars
process management
```

Look for:

```text
unwrap()
expect()
panic!
```

Determine whether each is safe.

Do not blindly remove them.

A crash caused by an unexpected external condition should generally become a controlled error.

---

# 50. PHASE 49 – TYPESCRIPT ERROR HANDLING

Review:

```text
any
unknown
null
undefined
async functions
promise rejection
```

Use strict typing where practical.

Do not perform a massive unrelated TypeScript rewrite.

Prioritize code involved in:

```text
Graph
Codegen
Training
Sidecars
Persistence
Remote jobs
```

---

# 51. PHASE 50 – SILENT EXCEPTION AUDIT

Search for patterns such as:

```text
catch {}
catch (_) {}
except: pass
except Exception: pass
console.log(error)
```

Review each one individually.

For each occurrence determine:

```text
Expected error?
or
Hidden real failure?
```

Fix hidden failures.

> **2026-10-06 — implemented for the TypeScript frontend (Python/Node/Rust still open).** An AST scan
> (not grep) found 80 swallowing handlers in `src/`; 56 are EXPECTED with a written one-sentence
> reason each (`docs/engineering/SILENT_EXCEPTIONS.md` is the allow-list), 24 were HIDDEN failures
> and are fixed so the failure becomes a visible, truthful state: `.py`-twin write/rename/remove/move
> errors (a save no longer reports clean while the generated script is missing; banner via
> `pyTwinError`), malformed `run.json`/events/per-file reads in run detail and compare, canvas bind
> failures, dataset history read errors, a failed remote capability probe (was silently shown as the
> local backend; the start is now BLOCKED until the probe works — found in review, the worker had
> only added a banner), remote-sidecar refresh, malformed SSE abort. `npm run verify:silent-catch`
> fails any new swallow without a ≥15-char prose reason or without an allow-list row;
> `npm run verify:silent-fixes` (33) asserts each fixed site. Lint unchanged at the 82-problem
> baseline. **Still open:** Python sidecars (`except …: pass`, broad `except Exception: return …`),
> the Node sidecar (16 empty catches) and 39 Rust `let _ =`/`.ok()`/`unwrap` sites (Phase 48; the toolchain exists in the conda env).

> **2026-10-06 — Node sidecar half implemented.** The guard now also scans `sidecar-llm/*.mjs`
> (`ts.ScriptKind.JS`; 132 src + 4 node files, 73 allow-listed sites). The AST scan found 27 Node sites:
> 17 EXPECTED (kept, with reasons) and 10 that became explicit handlers (16 related changes in total) so the
> model gets an explicit error instead of a default that looks like data: notes list (ENOENT vs
> unreadable), local/remote `list_dir`/`read_file` probes, run summary events (`{text,error}`), `run.json`/
> `metrics.json` readers (explicit corrupt/unreadable state + warnings), download size, opencode model
> listing exit/timeout, `slurmStatus` (a timeout, abort or ssh exit 255 is an error, never an invented
> `UNKNOWN`), mcp-bridge rejected handler → JSON-RPC error. Before-fix guard
> output: `docs/engineering/evidence/phase50-node-before.txt`. **Still open:** Python sidecars and 39 Rust sites.

> **2026-10-06 — Python half implemented.** `scripts/verify-silent-except-py.py` (stdlib `ast`; `npm run
> verify:silent-except-py`, self-test `test:silent-except-py` with 23 detect/not-detect cases) finds every
> swallowing `except`/`contextlib.suppress` in `sidecar-torch/*.py` (69 sites incl. the trainer) and fails
> an undocumented one (no ≥15-char prose reason, or no row in the Python allow-list of
> `docs/engineering/SILENT_EXCEPTIONS.md`, or a stale row). 62 sites remain, all EXPECTED with a one-sentence
> reason (optional imports, best-effort caches/cleanup, informational stats); **hidden failures made
> explicit:** an unreadable `status` file looked "not started" and could let a late write overwrite a
> CANCELLED run (now a sentinel that blocks transitions and fails the run), an empty training loader
> (`drop_last` with batch > rows) ended `done` with `train_loss=0.0` (now `fail("split")`), a corrupt
> manifest made `_compute_resumable` treat unknown run hashes as a match (now `resumable:false`), a
> missing ESPF codebook made `espf_vocab_size` return an invented `2`. **Found in review (the worker had
> allow-listed it):** four `except Exception: pass` around the RNG restore on resume — a resume whose
> random streams could not be restored continued with a fresh seed while claiming "same random streams";
> `_restore_rng` now returns a per-stream status recorded as `rng_restore` in `run.resumed`
> (`verify-checkpoint.ts [rng restore failure is recorded]`). Still open: the 39 Rust sites (Phase 48).
---

# 52. PHASE 51 – RESOURCE LEAK TESTING

Repeatedly perform:

```text
Open workspace
Close workspace
Start sidecar
Stop sidecar
Run inference
Restart sidecar
```

Check for:

```text
Increasing process count
Memory growth
Unreleased ports
Unclosed files
```

---

# 53. PHASE 52 – LONG-RUNNING TEST

Run a long test involving:

```text
Repeated inference
Repeated saves
Training progress
Metric updates
Sidecar communication
```

Observe:

```text
Memory
CPU
GPU memory
Process count
Errors
State consistency
```

---

# 54. PHASE 53 – REFERENCE SCIENTIFIC EXPERIMENTS

Create at least three small reference experiments.

## Experiment A – MLP

Synthetic classification.

```text
Input
 ↓
Linear
 ↓
ReLU
 ↓
Linear
```

> **2026-10-06 — implemented.** Three reference experiments with committed graph fixtures
> (`examples/reference-experiments/{mlp,cnn,multi-input}/model.spinoml`; the harness asserts the
> committed bytes equal a fresh serialization): A MLP (Linear-ReLU-Linear), B CNN (Reshape-Conv2d-
> ReLU-MaxPool-Flatten-Linear on 8×8 images stored as 64 tabular columns), C multi-input (two
> Linear branches → Concat → Linear; trained through a `.manifest` with two per-row `.pt` branches,
> an existing trainer feature). `npm run verify:reference-train` runs each through the REAL
> `train.py` (55 s) and asserts the full artifact set (graph, `model.py` byte-equal to fresh
> codegen, `run.json` with dataset fingerprint/seed/independently recomputed snapshot hashes,
> `best.pt`/`last.pt` loadable with all state keys, `metrics.json` `n_params`, `events.jsonl` with
> `config.env`/`run.provenance`/`run.determinism`/`run.snapshot`/N×`epoch.end`, stdout/stderr logs),
> loss decrease, val accuracy ≥ 0.75 (measured 0.95 / 1.0), checkpoint↔model consistency
> (`strict=True` load, re-evaluated val loss equal to the trainer's best), a second run with the
> same seed reproduces every per-epoch loss to 1e-6 (measured 0.0 on CPU) and a different seed
> differs. A first delegated attempt returned stub files that printed SKIPPED and exited 0; it was
> rejected and the work redone in two smaller blocks.

---

## Experiment B – CNN

Small image/synthetic image task.

```text
Input
 ↓
Conv
 ↓
Activation
 ↓
Pooling
 ↓
Linear
```

---

## Experiment C – Multi-input

```text
Input A ─┐
         ├→ Merge → Classifier
Input B ─┘
```

Each experiment must produce:

```text
Graph
Dataset configuration
Training configuration
Generated code
Checkpoint
Metrics
Logs
Metadata
```

---

# 55. PHASE 54 – HAND-WRITTEN PYTORCH COMPARISON

For at least one reference model, create an equivalent manually written PyTorch implementation.

Compare:

```text
Architecture
Parameter count
Input shape
Output shape
Forward behavior
Loss behavior
Gradient behavior
```

Where operations are mathematically equivalent, compare outputs using appropriate PyTorch numerical assertions.

> **2026-10-06 — implemented, no generator defect found.** `scripts/lib/reference_models.py` holds
> hand-written `RefMLP`/`RefCNN`/`RefMultiInput` (written as a person would, not derived from the
> generated code). `npm run verify:reference` copies the generated parameters pairwise (count and
> shapes asserted first) and compares, in `eval()` mode: parameter counts, forward outputs
> (float64 rtol 1e-10 / float32 rtol 1e-5), CrossEntropy loss (1e-12) and EVERY parameter gradient
> (float64) — all equal for the three graphs. A negative control perturbs one reference weight and
> must be detected; a mutation test (ReLU→Tanh in the reference) made MLP and CNN fail in forward,
> loss and gradients with exit 1, so the harness can see a real difference.

---

# 56. PHASE 55 – PARAMETER COUNT

For generated models:

```text
Total parameters
Trainable parameters
```

must be calculable.

Compare against known/reference implementations for test models.

> **2026-10-06 — implemented.** Total and trainable counts are asserted for generated AND
> hand-written models against analytic formulas (MLP 210 = 176+34, CNN 170 = 40+130,
> multi-input 130 = 56+40+34), and the trainer's `metrics.json` `n_params` must equal them
> (`verify:reference`, `verify:reference-train`).

---

# 57. PHASE 56 – DEVICE TESTS

Test CPU.

If CUDA hardware is available, test CUDA.

If unavailable:

```text
SKIPPED – CUDA unavailable
```

Do not claim CUDA compatibility merely because the code contains a CUDA option.

> **2026-10-06 — CPU PASS, CUDA SKIPPED.** CPU: every comparison and training run above;
> `config.env.device == "cpu"` is asserted. CUDA: `SKIPPED  CUDA — torch.cuda.is_available() is
> False` is printed explicitly by both harnesses (this machine has the CPU build `torch 2.12.0+cpu`);
> the CUDA branches (forward comparison, a CPU-vs-CUDA best-val-loss check within 1e-3) are
> implemented and run automatically where a GPU exists. No CUDA claim is made.

---

# 58. PHASE 57 – DTYPE TESTS

Where supported, test relevant dtypes such as:

```text
float32
float64
```

Add other dtypes only where the existing project intends to support them.

Do not promise unsupported dtype combinations.

> **2026-10-06 — implemented for float32 and float64.** The model-level comparison runs both
> dtypes with separate tolerances; the trainer run asserts `config.env.dtype` equals the dtype of
> the checkpointed floating-point parameters. No other dtype is claimed (the project does not
> intend to support them).

---

# 59. PHASE 58 – EXPERIMENT ARTIFACT STRUCTURE

A successful experiment should produce a reconstructable artifact.

Example:

```text
experiment/
├── metadata.json
├── graph.json
├── model.py
├── config.json
├── dataset.json
├── metrics.jsonl
├── logs/
└── checkpoints/
```

Adapt this to the existing project architecture.

Do not duplicate information unnecessarily.

> **2026-10-06 — documented, structure already existed.** The run directory
> (`experiments/runs/<id>/`: `run.json`, `model.spinoml`, `model.py`, `train.py`, `manifest.json`,
> `metrics.json`, `events.jsonl`, `checkpoints/`, logs) is the artifact; no second structure was
> invented. Each file, its writer and the test that proves it: `docs/engineering/REPRODUCIBILITY.md` §1/§2.

---

# 60. PHASE 59 – SCIENTIFIC RUN MANIFEST

Create a machine-readable manifest containing, where available:

```json
{
  "experimentId": "...",
  "gitCommit": "...",
  "graphHash": "...",
  "datasetHash": "...",
  "configHash": "...",
  "seed": 1234,
  "software": {},
  "hardware": {},
  "createdAt": "..."
}
```

Use the project's existing metadata structures if appropriate.

> **2026-10-06 — implemented.** `train.py` writes `manifest.json` (schema
> `spinoml.run-manifest/1`) atomically at start (before data loading), after the environment is known
> and at every terminal state (done incl. eval-only, failed with stage/message, cancelled): experiment id,
> created/finished, git, hashes (graph, generated model, `train.py`, dataset fingerprint, config
> identity), seed, dtype, device, software, hardware, dataset kind/fingerprint mode, split, approved-code
> count, `unsafe_pickle`, summary, `identity_fields` and explicit `notes`. No absolute paths, hostname,
> user or environment values (asserted). A manifest write failure never changes the training outcome and
> is recorded as `manifest.error`. `npm run verify:manifest` (76 checks, real `train.py` runs, ~50 s).

---

# 61. PHASE 60 – GIT STATE

Record:

```text
Git commit
Working tree clean/dirty
```

If the experiment uses uncommitted source changes, record that fact.

Never falsely claim an experiment is reproducible from a Git commit if uncommitted changes were involved.

> **2026-10-06 — implemented.** Full commit, branch (null when detached), tracked changes (staged or
> unstaged, ≤ 50 relative paths) and untracked count excluding the run's own directory, read with argv
> lists. `reproducible_from_git` is true ONLY if the commit is known and no tracked file is modified;
> not a repository, `git` missing/timed out, any failed git call or uncommitted changes give false plus a
> `reason`/note, and a failed call can never look clean (`dirty_tracked: null`). Tested in a real temp
> repo: clean, modified, staged-only, untracked-only, no repo, PATH without git, detached HEAD.
> Limit: older git (< 1.8.5, no `-C`) reports an explicit reason instead of a state.

---

# 62. PHASE 61 – GENERATED CODE ARTIFACT

Save the exact generated model code used by the experiment.

This is important because the code generator itself may change in the future.

The historical experiment must retain the exact generated code.

> **2026-10-06 — already satisfied, now cross-checked.** `model.py` is written into the run directory at
> launch and its sha256 is frozen in `run.json.snapshot` and in the manifest; `verify:reference-train` and
> `verify:manifest` assert the run-dir file is byte-equal to the freshly generated code and to an independent
> hash.

---

# 63. PHASE 62 – SOFTWARE ENVIRONMENT

Record:

```text
Operating system
Python
PyTorch
CUDA
Node
Rust
Relevant package versions
GPU
```

as available.

> **2026-10-06 — implemented (extended).** `config.env` and `manifest.software/hardware` record python,
> torch, **torch_geometric (version or explicit null)**, numpy, CUDA/cuDNN, **OS/platform string**, device,
> dtype, GPU name/memory, CPU count, RAM. No hostname/user/absolute path (asserted). Node and Rust versions
> are deliberately not recorded: a training run uses neither. The SpinoML app version is not recorded
> (gap, see LIMITATIONS).

---

# 64. PHASE 63 – HASHING

Use stable hashes for:

```text
Graph
Configuration
Dataset identity
Generated model
```

The purpose is to determine whether two experiments actually used identical inputs/configuration.

> **2026-10-06 — implemented.** Stable hashes: graph (`graph_sha256`, file bytes), generated model
> (`model_py_sha256`), dataset (fingerprint), and `config_identity_sha256` = SHA-256 of canonical JSON
> (sorted keys, no whitespace) over the manifest's `identity_fields`, excluding run id, label,
> timestamps, paths and submission settings. Equal for identical inputs from different directories;
> changes with lr, seed, dataset content or graph (`verify:manifest` case 7).

---

# 65. PHASE 64 – PROPERTY TESTING

Where practical, test properties rather than only individual examples.

Important properties:

```text
Valid graph → accepted
Invalid graph → rejected
Valid graph → generated code executes
Same graph → same generated code
Predicted shape == actual shape
Invalid mutation → no graph corruption
```

> **2026-10-06 — implemented.** `npm run test:property` (seeded, `PROPERTY_SEED`/`PROPERTY_N`, ~9 s) checks the
> properties on generated graphs: valid graph → accepted; generated code executes (forward AND backward, finite
> gradients for every parameter); predicted shape and parameter count == an INDEPENDENT oracle (the generator
> tracks tensor shapes and counts by hand — Linear in·out+out, Conv2d out·in·k²+out, BatchNorm 2F, Embedding v·d —
> and never calls the app's inference); same graph → identical code (3×); save → load → generate identical.
> `npm run test:fuzz` covers “invalid graph → rejected, no corruption” (Phase 66). Mutation-checked: skewing the
> MLP oracle made `test:property` fail with the exact shape, a no-op `cycle` operator was flagged by `test:fuzz`.

---

# 66. PHASE 65 – RANDOM GRAPH TESTING

Create controlled random valid graphs.

Generate hundreds of small cases if practical.

For every valid graph:

```text
Validate
 ↓
Infer shapes
 ↓
Generate code
 ↓
Execute model
 ↓
Forward
 ↓
Backward where applicable
```

Do not generate arbitrary impossible graphs and expect all of them to be valid.

> **2026-10-06 — implemented for six families.** 200 random VALID graphs per run (seed 1234): MLP stacks
> (optional Dropout/BatchNorm1d), CNN (Conv2d kernels 1/3/5, stride, BatchNorm2d, MaxPool2d, Flatten), residual
> (Add skip), branch+Concat, multi-input (2–3 Inputs merged), sequence (int64 → Embedding → Flatten → Linear).
> All 200 pass the python execution/shape/param/gradient checks; 40 (spread over the families) also go through the
> real torch sidecar `/infer` with the app's `inferShapes`: `n_params`, output shape and per-attribute shapes equal
> shapes captured by independent forward hooks. Not covered: attention, recurrent, GNN, 1-D/3-D convolutions and
> other pooling layers (LIMITATIONS).

---

# 67. PHASE 66 – INVALID GRAPH FUZZING

Generate intentionally invalid graphs:

```text
Unknown node
Unknown edge
Missing node
Duplicate edge
Invalid parameter
Negative dimension
Invalid handle
Incompatible shape
```

Expected:

```text
Clean rejection
```

Not:

```text
Crash
Hang
Memory leak
Silent acceptance
Incorrect model
```

> **2026-10-06 — implemented; found and fixed a real bug.** `npm run test:fuzz` applies 28 mutation operators to
> valid graphs (948 mutants/run): structural (dangling node, unknown edge endpoints, duplicate node id, unknown layer,
> self-loop, cycle, zero/negative dims), coerced (NaN/Infinity/string/object/out-of-range select), load-reject (null
> required param), raw-structural (duplicate edge id, empty / `__proto__` / `constructor` / unicode / very long ids),
> semantic (wrong `in_features`, huge dimension, bad reshape count, conv kernel larger than input, Concat/Add shape
> mismatch) and a 10 000-node valid-chain stress. Every structural mutant must be rejected by `validateGraphState`,
> `loadSnapshot` must return false leaving the store state deep-equal to before, `generate` must not throw; every
> semantic mutant must come back from the real sidecar as a STRUCTURED error that `verificationFromInferResult`
> classifies `invalid` (never valid, never unknown) with the sidecar still healthy. **Finding:** `validateGraphState`
> threw “Maximum call stack size exceeded” on a valid 10 000-node chain (recursive cycle DFS in
> `src/canvas/invariants.ts`) — a hostile or large imported `.spinoml` could crash the loader. Fixed with an
> iterative DFS; `wouldCreateCycle` (BFS) and `generate` were probed up to 30 000 nodes (1.2 s) and are fine.

---

# 68. PHASE 67 – TEST PYRAMID

Organize tests into:

```text
Unit
Contract
Integration
E2E
Scientific
Remote
Hardware
```

Prefer many fast unit tests.

Use fewer expensive E2E tests.

> **2026-10-06 — implemented.** `scripts/suites.ts` registers every suite (54) with a category, a hard timeout, the
> capabilities it needs and a one-line reason; `npm run suites` lists them, `npm run ci -- --check` fails when
> `package.json` and the registry drift apart (it did, mid-session, when a worker added a script). Categories: unit 12,
> contract 12, integration 11, e2e 7 (real trainer), scientific 4 (reference equivalence, random graphs, fuzz),
> infrastructure 6 (build, lint, typecheck, cargo ×2, opencode), plus `remote-live` and `hardware-cuda` which have NO suite and
> are therefore reported BLOCKED instead of being absent.

---

# 69. PHASE 68 – CONTRACT TESTS

Test boundaries:

```text
Frontend ↔ Torch sidecar
Frontend ↔ LLM sidecar
Frontend ↔ Rust
Rust ↔ filesystem
Rust ↔ SSH
Training ↔ generated code
```

For each:

```text
Valid request
Invalid request
Expected response
Expected error
Timeout
```

> **2026-10-06 — partly implemented.** Boundaries with a contract suite: frontend ↔ torch sidecar (`verify:sidecar`,
> `test:robustness`, `test:scope`, `test:safe-load`, `test:deps-policy`, `test:run-script`), frontend ↔ LLM sidecar
> (`test:llm-safety`, `test:llm-validation-parity`, `verify:command-injection`, `verify:paths`), training ↔ generated
> code (`verify:reference`, `verify:reference-train`, `test:property`), Rust ↔ ssh/slurm (`cargo test` via `verify:ssh`/
> `verify:slurm`, string contracts). Gaps: no contract test for the Tauri `invoke` command surface itself, none for
> Rust ↔ filesystem beyond unit tests, none for a live ssh host (BLOCKED).

---

# 70. PHASE 69 – CI

If the project does not already have sufficient CI, add a minimal reliable CI pipeline.

At minimum:

```text
Install
Build
Lint
Codegen verification
Sidecar verification
Training generation verification
Rust check
Python tests
```

CI must not require private credentials for ordinary tests.

> **2026-10-06 — defined, NOT RUN.** `.github/workflows/ci.yml` (push to main + pull_request, `contents: read`, no
> secrets anywhere): `node` job (build, script type-check, suites without Python), `python` job (CPU torch + pinned
> `sidecar-torch/requirements.txt` from `pip freeze` of the dev env, Python suites, `.test-results` artifact), `rust` job
> (Tauri apt dependencies, `cargo check`, `cargo test`), each with a timeout. It parses as YAML and uses the same runner
> as local runs (`npm run ci`), but it has never executed on a GitHub runner.

---

# 71. PHASE 70 – CI DETERMINISM

Tests should not depend unnecessarily on:

```text
Current time
Random values
External internet
Private accounts
Claude credentials
SSH credentials
SLURM
Private datasets
```

Integration tests requiring these resources must be clearly separated.

> **2026-10-06 — implemented in the runner.** Child processes get an allow-listed environment only (PATH, HOME, LANG,
> TMPDIR, CONDA_*, XDG_*…): 89 variables incl. API keys, `SSH_AUTH_SOCK` and proxy settings are removed (names printed,
> never values), so a suite that secretly needs credentials, ssh or the internet fails instead of passing by accident.
> Live ssh/SLURM/LLM/CUDA suites are separate and BLOCKED unless explicitly enabled. All sidecar tests use 127.0.0.1;
> randomised tests are seeded (`PROPERTY_SEED`). Not enforced: wall-clock independence (no test freezes time).

---

# 72. PHASE 71 – TEST TIMEOUTS

External operations must have timeouts.

Especially:

```text
HTTP
SSE
SSH
SFTP
SLURM
Sidecars
Training processes
```

No automated test should hang indefinitely.

> **2026-10-06 — implemented for tests; product timeouts audited.** Every suite has a hard timeout; the runner kills
> the whole process group (SIGTERM, then SIGKILL after 5 s) and reports TIMEOUT. Product side: ssh `ConnectTimeout=10` +
> keepalives (Phase 34), `curl --max-time 300`, `git` 5 s, torch sidecar socket timeout, LLM upstream idle timeout
> (new), `/deps/*` 240 s/1800 s. Known gaps: no overall timeout for `ssh_exec` in Rust (Phase 48), the frontend datasets
> client has no `AbortSignal` (Phase 39), the Anthropic/subscription providers have no idle timeout.

---

# 73. PHASE 72 – RETRY LOGIC

Retries must have:

```text
Maximum attempts
Backoff
Timeout
Final failure state
```

Never implement infinite retry loops.

> **2026-10-06 — audited; nothing to fix.** The only retry in the code base is the OpenAI SDK's own
> (`maxRetries: 2`, now explicit with a comment). Every other loop is bounded by EOF, a closed queue, date arithmetic or a
> user confirmation (`MAX_TOOL_TURNS` asks before continuing). Remote submission is idempotent through the atomic
> run-directory claim (Phase 33), so a client-side retry cannot double-submit. No retry loop without a maximum exists.

---

# 74. PHASE 73 – EXPERIMENT RESULT INTEGRITY

Only mark a run:

```text
SUCCESS
```

when appropriate required artifacts exist:

```text
Valid exit status
Metrics
Required metadata
Expected checkpoint
Logs
```

If training failed but generated partial artifacts:

```text
FAILED
```

not:

```text
SUCCESS
```

> **2026-10-06 — implemented; a real false-success bug found first.** Sabotage against the UNFIXED trainer
> (`docs/engineering/evidence/phase73-before.txt`): a run whose `best.pt` was never written, truncated to 0 bytes
> or overwritten with garbage, whose `last.pt` was missing, whose metrics held `best_val_loss: NaN` or whose
> `manifest.json` was corrupt STILL ended `status=done` with `run.done`. Now `_verify_run_integrity` runs after
> `metrics.json` and the final checkpoints are written and BEFORE `done`: (a) metrics.json readable with finite
> `best_val_loss`, `epochs ≥ 1`, `n_params ≥ 1` (eval form for eval-only runs); (b) `best.pt` and `last.pt` exist,
> non-empty, zip-valid and — up to 256 MB — load through `safe_torch_load` with `model_state, optim_state, epoch,
> global_step, config`; (c) `events.jsonl` holds `run.provenance`, `config.env`, `run.snapshot` and ≥ 1
> `epoch.end`; (d) `manifest.json` parses with the pinned schema; (e) `stdout.log`/`stderr.log` exist when the
> executor launched the run (detected by the `pid` file; otherwise a note). Any problem → `run.integrity ok:false`,
> `run.failed` stage `integrity`, status FAILED, never `run.done`. The gate fails closed: an exception inside it is a
> failure. Side effect, deliberate: a manifest that cannot be written now fails the run (missing required
> metadata); `verify:manifest` case 10 was updated to assert exactly that. `npm run verify:integrity` (real runs +
> python-wrapper sabotage, ~110 s).

---

# 75. PHASE 74 – FAILED RUNS AND RESUME

If a failed run has a valid checkpoint, distinguish:

```text
FAILED
```

from:

```text
RESUMABLE
```

where appropriate.

Do not automatically resume without explicit user intent.

> **2026-10-06 — implemented.** At every terminal write `_compute_resumable` records
> `resumable: {resumable, resume_from, epoch, reason}` in `metrics.json` and `manifest.json` (+ `run.resumable`
> event): true only for a failed/cancelled run whose `last.pt` exists, loads, and carries this run's graph/model
> hashes; reasons otherwise: "run completed", "no checkpoint", "checkpoint corrupt", "checkpoint belongs to a
> different model", "checkpoint load verification skipped (size)". The STATUS stays `failed`/`cancelled` — no new
> status value — and `RunDetailModal` shows an amber "Fortsetzbar: Checkpoint nach Epoche N vorhanden (Status bleibt
> FEHLGESCHLAGEN)" banner (or the reason; an unreadable manifest is shown explicitly). Nothing resumes by itself:
> a second launch without `resume_from` starts at epoch 0 with no `run.resumed`; with `resume_from` it emits
> `run.resumed` and the epoch continues (`verify:integrity` §10/§11). Verified: crash at epoch k+1 → resumable at k;
> SIGTERM → cancelled + resumable; crash before any checkpoint, corrupt `last.pt`, another model's `last.pt` → not
> resumable with the right reason.

---

# 76. PHASE 75 – OFFLINE CORE FUNCTIONALITY

Where the architecture allows it, verify that core functionality does not unnecessarily depend on Claude/network access.

If Claude is unavailable:

```text
Graph editing
Validation
Shape inference
Code generation
Local training
```

should remain available where designed to be local.

> **2026-10-06 — verified by construction and by test environment.** No core module imports the chat/LLM code:
> `GraphStore`, validation, `generator.ts`/`pyLiteral.ts`, inference, persistence, the trust store, `training/store.ts`,
> `snapshot.ts` and the datasets store are free of `src/chat/*` imports (only the optional "Modell erklären" feature,
> `codegen/ExplainModal.tsx` + `modelIntent.ts`, uses the chat client). The passing suites (graph editing, validation,
> shape inference, codegen, local training, checkpointing — all except the two that test the LLM sidecar itself) run with
> NO LLM sidecar and with the API-key variables scrubbed out of the environment by the runner. Not tested: the UI itself with the LLM sidecar offline (no render tests).

---

# 77. PHASE 76 – CLAUDE FAILURE

Test:

```text
Claude unavailable
Claude timeout
Malformed Claude response
Invalid Claude tool arguments
Claude proposes invalid graph
```

The application must fail safely.

> **2026-10-06 — implemented for the openai-compat provider (fake server).** Provider unavailable (connection
> refused P1.4), HTTP 500/401/429 (P1.1–3), body that is not SSE (P1.5), invalid JSON inside an SSE frame (P1.6), stream cut
> in the middle of a tool call (P1.7), a provider that never answers (P2: client abort closes the upstream connection;
> idle timeout `provider stalled`), a model that never stops calling tools (P3: confirm card after exactly 100 steps,
> "no" ends the turn, bounded request count), invalid tool arguments and invalid graph proposals (T2–T9). Every case ends
> in an explicit `status:error` (or a normal completed turn) followed by `done`, the sidecar stays healthy and serves the
> next chat, and no secret reaches the stream (S1). NOT tested: the Anthropic API path, the claude-agent-sdk subscription
> path and the opencode CLI path (process failure, invalid model, unexpected exit, cancellation — plan §0.7/§0.12 — would
> need a fake `SPINOML_OPENCODE_BIN`); the idle timeout/abort changes were made for `openai-compat` only.

---

# 78. PHASE 77 – SECURITY REVIEW

Perform a focused security review of:

```text
Command injection
Path traversal
Python code injection
Unsafe deserialization
Subprocess handling
SSH command construction
LLM/MCP tool access
Workspace access
Local HTTP sidecars
Secrets
Logs
```

> **Phase 77 — partially implemented 2026-10-06 (sidecar authentication; the rest of the
> review was done piecewise).** The "Local HTTP sidecars" item is closed by Phase 77/78 below
> (R013/R014). Command injection, path traversal, Python code injection, unsafe
> deserialization, subprocess/ssh construction, LLM tool access, secrets and logs were
> covered by phases 43–47 and 76 (R012, R015, R016, R039–R041, R048–R050). NOT done: a
> single end-to-end threat-model walkthrough; the webview trust boundary (CSP null, Monaco
> from a CDN — new R052) and workspace-level file permissions were only noted
> (docs/engineering/LIMITATIONS.md §2).

---

# 79. PHASE 78 – LOCALHOST SERVICES

Check all local HTTP services.

Verify:

```text
Bind address
CORS
Input validation
Authentication requirements where appropriate
```

Do not expose a service on:

```text
0.0.0.0
```

unless there is an explicit reason.

> **Phase 78 — implemented 2026-10-06.** Both sidecars bind `127.0.0.1` only (code review:
> `ThreadingHTTPServer(("127.0.0.1", …))`, `server.listen(PORT, '127.0.0.1')`; the bind itself is not
> asserted by a test — the auth tests assert that a non-loopback `Host` header is refused, DNS-rebinding style).
> **CORS** no longer sends `*`: only an exact-match Origin from the allow-list is echoed
> (`Vary: Origin`), a disallowed Origin gets 403 and no CORS headers; preflights need no token.
> **Authentication**: per-launch 256-bit token (`X-SpinoML-Token`, constant-time compare) on
> every endpoint except `OPTIONS` and `GET /health` (which without a token reveals only
> `{ok, auth, requiresAuth, tokenOk:false}`); generated in Rust (`getrandom`), passed to the
> managed sidecars by env (`Command::env`, never `set_var`/argv), to the webview via
> `sidecar_token`; the sidecar deletes it from its own env so children never inherit it.
> Remote sidecar: fresh token per session over ssh stdin, exported only after `env.sh` ran
> (so `env.sh` cannot replace it). LLM sidecar: `/respond` and `/chat` need the token, ids are
> random UUIDs, the opencode MCP bridge uses a per-turn session secret via the opencode
> `environment` (never argv, never the master token). Frontend: `src/sidecars/auth.ts`
> `sidecarFetch` (retry once on 401; distinct `auth failed` vs `offline` states);
> `verify:sidecar-fetch` forbids bare `fetch(` in `src/`. Spec:
> `docs/engineering/SIDECAR_AUTH.md`.
> **Review caught before commit** (the workers' tests were green): (1) the LLM sidecar's global
> gate demanded the master token on `/internal/mcp/*`, so in token mode every opencode tool call
> would have been rejected — the bridge never has the master token; fixed with an explicit route
> exemption + a real-bridge e2e test (`test:opencode-lifecycle`); (2) the probe stayed on
> `auth-failed` after a remote sidecar restart with a new token (health answers 200 +
> `tokenOk:false`, so the 401 retry never ran) → probe now refreshes the token and re-probes once;
> (3) `verify-remote-deploy-files` read paths out of Rust comments.
> **Found while designing the rollout (a regression of my own phases 45–47):** the remote deploy
> uploaded only `main.py` + `dataset_handlers.py`, so a remote sidecar would have died with
> `ModuleNotFoundError` (`scope`, `safe_load`, `deps_policy`; the ESPF codebook was never shipped
> either). One `SIDECAR_FILES` constant now drives `deploy()` and `verify:remote-deploy-files`
> checks it against the import closure (R053; not exercised against a real login node).
> **Not done / honest limits:** no CSP (needs a real-webview test and bundled Monaco — R052),
> browser-dev stays tokenless, no rotation, same-user processes can read the token
> (LIMITATIONS.md §2). Verified: `test:sidecar-auth-torch` (143 rows, token/Origin/Host mutations
> red), `test:sidecar-auth-llm` (125 rows, 4 mutations red), `test:sidecar-auth-frontend`,
> `verify:sidecar-fetch`, `verify:remote-deploy-files`, `cargo test` (37).

---

# 80. PHASE 79 – DEPENDENCY SECURITY

Review:

```bash
npm audit
```

and the equivalent security checks for Python/Rust dependencies.

Do not blindly upgrade every dependency.

For each security-relevant update:

```text
Update
→ run tests
→ verify compatibility
```

---

# 81. PHASE 80 – DOCUMENTATION

Create or update:

```text
docs/engineering/
```

with:

```text
ARCHITECTURE.md
BASELINE.md
RISK_REGISTER.md
TEST_MATRIX.md
REPRODUCIBILITY.md
FAILURE_RECOVERY.md
REMOTE_TRAINING.md
LIMITATIONS.md
```

Documentation must describe the actual behavior of the system.

Do not document intended behavior as if it were already implemented.

---

# 82. PHASE 81 – KNOWN LIMITATIONS

Create:

```text
docs/engineering/LIMITATIONS.md
```

Document honestly:

```text
Nondeterministic PyTorch operations
CUDA limitations
Unsupported layers
Unsupported dataset types
Remote execution limitations
SLURM limitations
Platform limitations
Claude dependency
Known performance limitations
```

---

# 83. PHASE 82 – FINAL FULL TEST SUITE

Run all available project verification commands.

At minimum investigate and execute:

```bash
npm run build
npm run lint
npm run verify:codegen
npm run verify:sidecar
npm run verify:traingen
```

and:

```bash
cd src-tauri
cargo check
cd ..
```

plus:

```text
All unit tests
All integration tests
All persistence tests
All scientific smoke tests
All recovery tests
All security tests
All available fuzz/property tests
```

---

# 84. FINAL RELEASE GATE

The repository may only be described as:

```text
Production Ready for Scientific Work
```

if all applicable CRITICAL requirements pass.

---

## Code

* [ ] TypeScript build passes
* [ ] Lint passes
* [ ] Rust checks pass
* [ ] Python tests pass
* [ ] No known CRITICAL bugs

---

## Graph

* [ ] Graph invariants are enforced
* [ ] Invalid mutations are rejected
* [ ] Persistence roundtrip works
* [ ] Corrupt files fail safely
* [ ] Concurrent mutations are safe

---

## Code Generation

* [ ] Code generation is deterministic
* [ ] Golden tests exist
* [ ] Generated code compiles
* [ ] Generated code imports
* [ ] Generated models instantiate
* [ ] Forward pass works
* [ ] Backward pass works
* [ ] Gradient tests exist

---

## Shape Inference

* [ ] Positive tests
* [ ] Negative tests
* [ ] Multi-input tests
* [ ] Multi-output tests
* [ ] Shape mismatch detection
* [ ] Stale-response protection
* [ ] Failed inference cannot silently produce a valid state

---

## Sidecars

* [ ] Startup tested
* [ ] Shutdown tested
* [ ] Restart tested
* [ ] Invalid requests tested
* [ ] Timeout tested
* [ ] Crash recovery tested
* [ ] Error states are visible

---

## Training

* [ ] Seed handling
* [ ] Training snapshot
* [ ] Checkpointing
* [ ] Resume
* [ ] NaN/Inf detection
* [ ] Failure state
* [ ] Success state
* [ ] Cancellation
* [ ] Metric correctness
* [ ] Artifact integrity

---

## Scientific Reproducibility

* [ ] Git commit recorded
* [ ] Dirty Git state recorded
* [ ] Graph stored
* [ ] Generated model stored
* [ ] Training configuration stored
* [ ] Dataset identity stored
* [ ] Dataset split stored
* [ ] Seed stored
* [ ] Software versions stored
* [ ] Hardware information stored
* [ ] Experiment manifest stored

---

## Remote Execution

* [ ] SSH failure handling
* [ ] SSH reconnect
* [ ] SFTP failure handling
* [ ] SLURM submission
* [ ] SLURM status tracking
* [ ] SLURM failure handling
* [ ] Job ID persistence
* [ ] Application restart recovery

---

## Security

* [ ] No known command injection
* [ ] No known path traversal
* [ ] No unsafe Python execution
* [ ] Unsafe deserialization reviewed
* [ ] No secrets in logs
* [ ] Local services are appropriately bound
* [ ] LLM/MCP actions are validated

---

# 85. CRITICAL STOP CONDITIONS

The agent must **not** declare the project production-ready if any of the following remain possible:

```text
Data can be silently corrupted.

A wrong model can be generated and accepted as valid.

Shape inference can silently be wrong.

A failed training run can be reported as successful.

A corrupted checkpoint can be silently accepted.

Dataset leakage can occur without detection where the system claims to manage the split.

A stale asynchronous response can overwrite current state.

An invalid graph can become the authoritative graph.

A remote job can be reported with an incorrect final status.

A running experiment can be silently changed by later UI edits.

An experiment cannot be reconstructed from its artifacts.

Critical security vulnerabilities remain unresolved.
```

---

# 86. BUG SEVERITY

## CRITICAL

Affects scientific correctness, data integrity, security, or experiment validity.

Examples:

```text
Wrong generated architecture
Wrong labels
Dataset leakage
Wrong metric
Corrupt checkpoint
False successful training
Silent graph corruption
Wrong experiment configuration
```

Must be fixed before release.

---

## HIGH

Can cause major operational failure.

Examples:

```text
Training lost
SSH recovery broken
SLURM state incorrect
Sidecar crash
Persistence failure
Race condition
```

Must normally be fixed before release.

---

## MEDIUM

Clear defect but unlikely to silently invalidate scientific results.

---

## LOW

Cosmetic or convenience issue.

May be deferred.

---

# 87. BUG FIX PROCEDURE

For every confirmed bug:

```text
1. Reproduce.
2. Add failing regression test.
3. Identify root cause.
4. Apply minimal correct fix.
5. Run regression test.
6. Run surrounding tests.
7. Run full relevant suite.
8. Document the fix.
9. Update RISK_REGISTER.md.
```

---

# 88. DO NOT OVER-REFACTOR

Prefer:

```text
Small fix
+
Regression test
```

over:

```text
Large rewrite
+
No clear proof of correctness
```

Preserve existing architecture unless a change is required for correctness, reliability, security, or reproducibility.

---

# 89. DO NOT ADD FEATURES DURING HARDENING

Do not add unrelated:

```text
New layer types
New UI systems
Major redesigns
Animations
Experimental features
Unnecessary dependency upgrades
```

during this hardening project.

A new feature is allowed only when directly required for:

```text
Correctness
Testing
Reproducibility
Security
Error handling
Recovery
```

---

# 90. FINAL RELIABILITY REPORT

Create:

```text
docs/engineering/FINAL_RELIABILITY_REPORT.md
```

The report must contain:

## 90.1 Executive Summary

Use exactly one:

```text
PASS
CONDITIONAL
FAIL
```

---

## 90.2 Environment

Record:

```text
OS
CPU
RAM
GPU
CUDA
Python
PyTorch
Node
Rust
Git commit
```

---

## 90.3 Test Results

Use a table:

| Test                | Result    | Duration | Notes |
| ------------------- | --------- | -------: | ----- |
| Build               | PASS/FAIL |          |       |
| Lint                | PASS/FAIL |          |       |
| Codegen             | PASS/FAIL |          |       |
| Sidecar             | PASS/FAIL |          |       |
| Training generation | PASS/FAIL |          |       |
| Rust                | PASS/FAIL |          |       |
| Unit                | PASS/FAIL |          |       |
| Integration         | PASS/FAIL |          |       |
| Scientific smoke    | PASS/FAIL |          |       |
| Persistence         | PASS/FAIL |          |       |
| Recovery            | PASS/FAIL |          |       |
| Security            | PASS/FAIL |          |       |
| Fuzz/property       | PASS/FAIL |          |       |

---

## 90.4 Bugs Fixed

For every bug:

```text
Bug ID
Description
Root cause
Fix
Regression test
Severity
```

---

## 90.5 Remaining Risks

List every known remaining risk.

Do not hide weaknesses.

---

## 90.6 Reproducibility Assessment

Explain:

```text
What is deterministic?
What is partially deterministic?
What is nondeterministic?
What metadata is stored?
What is required to reproduce an experiment?
```

---

# 91. FINAL ARCHITECTURE TARGET

The hardened architecture should follow this conceptual flow:

```text
                    USER
                     |
                     v
               ┌───────────┐
               │ Graph UI  │
               └─────┬─────┘
                     |
                     v
              ┌──────────────┐
              │ Graph Store  │
              └──────┬───────┘
                     |
                     v
              ┌──────────────┐
              │   Validate   │
              └──────┬───────┘
                     |
                     v
             ┌─────────────────┐
             │ Shape Inference │
             └───────┬─────────┘
                     |
                     v
             ┌─────────────────┐
             │ Deterministic   │
             │ Code Generation  │
             └───────┬─────────┘
                     |
                     v
             ┌─────────────────┐
             │ Real PyTorch    │
             │ Forward/Backward│
             └───────┬─────────┘
                     |
                     v
             ┌─────────────────┐
             │ Experiment      │
             │ Snapshot        │
             └───────┬─────────┘
                     |
                     v
             ┌─────────────────┐
             │ Training        │
             └───────┬─────────┘
                     |
              ┌──────┴──────┐
              v             v
        Checkpoints       Metrics
              |             |
              └──────┬──────┘
                     v
             ┌─────────────────┐
             │ Reproducible    │
             │ Experiment       │
             │ Artifacts        │
             └─────────────────┘
```

LLM interaction should follow:

```text
Claude
   |
   v
Structured action
   |
   v
Schema validation
   |
   v
Graph validation
   |
   v
Commit
   |
   v
Shape inference
   |
   v
Code generation
```

Claude must not bypass validation.

---

# 92. FINAL WORKFLOW FOR THE AGENT

For every phase use:

```text
PHASE START
    ↓
Read relevant code
    ↓
Identify existing behavior
    ↓
Run existing tests
    ↓
Write focused test
    ↓
Reproduce problem
    ↓
Implement minimal fix
    ↓
Run focused test
    ↓
Run related tests
    ↓
Run full regression suite
    ↓
Update documentation
    ↓
Update risk register
    ↓
PHASE COMPLETE
```

Never skip directly from:

```text
Read
```

to:

```text
Rewrite
```

---

# 93. PRIORITY IF COMPUTE/TIME IS LIMITED

If the environment is limited and not everything can be executed simultaneously, prioritize:

```text
1. Graph validation
2. Shape inference
3. Deterministic code generation
4. Generated-model execution
5. Training correctness
6. Checkpoint correctness
7. Dataset integrity
8. Experiment reproducibility
9. Async/race correctness
10. Sidecar reliability
11. Remote execution
12. Security
13. Performance
14. UI polish
```

The first eight are the most important for scientific correctness.

---

# 94. FINAL DEFINITION OF DONE

The project is complete only when an independent developer can:

```text
1. Clone the repository.
2. Install the documented environment.
3. Run the test suite.
4. Obtain passing results.
5. Create a reference experiment.
6. Run the experiment.
7. Obtain model code, configuration, metrics and checkpoint artifacts.
8. Identify the exact Git commit.
9. Identify the dataset/configuration used.
10. Reload the checkpoint.
11. Reconstruct the experiment.
12. Understand all remaining limitations.
```

The ultimate requirement is:

> **SpinoML must not silently produce scientifically invalid results while claiming that the experiment succeeded.**

A crash is preferable to a silent incorrect scientific result.

An explicit `FAILED` state is preferable to a false `SUCCESS`.

An explicit `UNKNOWN` state is preferable to an invented value.

An explicit reproducibility limitation is preferable to a false reproducibility claim.

---

# 95. FINAL INSTRUCTION TO THE CODING AGENT

Start with **PHASE 0**.

Do not begin by changing source code.

First:

```text
Clone/download the repository.
Install the complete environment.
Verify all dependencies.
Run the existing test and verification commands.
Record the exact baseline.
```

Then proceed through the phases sequentially.

After every phase:

```text
Run tests.
Fix failures.
Add regression tests.
Document results.
```

Do not declare success based on intuition.

Do not declare production readiness based only on the README.

Do not assume existing tests are sufficient.

Do not hide failures.

Do not remove functionality merely to make tests pass.

The final standard is:

> **Scientifically trustworthy, reproducible, fail-safe, test-backed software — not merely software that builds.**

