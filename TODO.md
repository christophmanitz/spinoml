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

---

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

---

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

---

# 56. PHASE 55 – PARAMETER COUNT

For generated models:

```text
Total parameters
Trainable parameters
```

must be calculable.

Compare against known/reference implementations for test models.

---

# 57. PHASE 56 – DEVICE TESTS

Test CPU.

If CUDA hardware is available, test CUDA.

If unavailable:

```text
SKIPPED – CUDA unavailable
```

Do not claim CUDA compatibility merely because the code contains a CUDA option.

---

# 58. PHASE 57 – DTYPE TESTS

Where supported, test relevant dtypes such as:

```text
float32
float64
```

Add other dtypes only where the existing project intends to support them.

Do not promise unsupported dtype combinations.

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

---

# 61. PHASE 60 – GIT STATE

Record:

```text
Git commit
Working tree clean/dirty
```

If the experiment uses uncommitted source changes, record that fact.

Never falsely claim an experiment is reproducible from a Git commit if uncommitted changes were involved.

---

# 62. PHASE 61 – GENERATED CODE ARTIFACT

Save the exact generated model code used by the experiment.

This is important because the code generator itself may change in the future.

The historical experiment must retain the exact generated code.

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

