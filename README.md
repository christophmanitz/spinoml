# MLForge

Drag-and-drop PyTorch architecture builder, steered by Claude.

Build `nn.Module` architectures by dragging layers onto a canvas. A connected LLM
(Claude, via the Agent SDK) observes the graph through MCP tools and can edit it
live or on demand. Live shape inference validates the graph against a sample input
shape and surfaces errors inline.

## Stack

- **Shell**: Tauri 2 (Rust + system webview)
- **Frontend**: React 19 + TypeScript + Vite, React Flow for the canvas,
  Zustand for graph state, Tailwind v4, Monaco for code preview
- **LLM bridge**: Node.js sidecar running `@anthropic-ai/claude-agent-sdk`
  (uses Claude Code's OAuth → consumes Max subscription, not pay-per-use)
- **Shape inference**: Python sidecar running PyTorch, HTTP on 127.0.0.1:7421

## Phase status

- [x] **Phase 0** — Scaffold: window, panels, dependencies wired
- [x] **Phase 1** — Layer palette, canvas with typed nodes, inspector forms
- [x] **Phase 2** — Graph → PyTorch code generator
- [x] **Phase 3** — Python sidecar with shape inference
- [x] **Phase 4** — Claude Agent SDK sidecar + chat UI with live graph mutation
- [x] **Phase 5** — Save/load `.mlforge` files, undo/redo, templates
- [x] **Phase 5.5** — Workspace file explorer (virtual FS in localStorage)
- [x] **Phase 6** — Tauri desktop shell with real filesystem workspace
- [x] **Phase 6.5** — Sidecar lifecycle (auto-spawn torch+llm from Rust)
- [x] **Phase 7** — Distribution build (`.deb`; AppImage TODO when linuxdeploy/libfuse2 is available)
- [x] **Phase 8** — Dataset overview + smoke test (tabular/image/tensor/pdb/SMILES/HF)

## Repo layout

```
src/                    React frontend
  palette/              Draggable layer types (Conv, Linear, …)
  canvas/               React Flow canvas + Zustand GraphStore
  inspector/            Per-node parameter editor
  chat/                 Claude chat panel
  codegen/              Graph → PyTorch nn.Module source
  datasets/             Dataset overview + smoke test (phase 8)
  workspace/            File explorer + Tauri FS bridge
src-tauri/              Rust shell, sidecar lifecycle
sidecar-llm/            Node.js, runs Claude Agent SDK (phase 4+)
sidecar-torch/          Python, shape inference + dataset handlers (phase 3+)
```

## Develop

Everything runs inside the `mlforge-dev` conda env (node 20, rust 1.96, python 3.12).

```bash
conda activate mlforge-dev
npm install
npm run tauri dev          # native window; Rust shell auto-spawns both sidecars

# or vite-only (browser dev, no Rust, manual sidecars):
npm run dev                # browser at http://localhost:5173
npm run sidecar:torch      # shape inference on 127.0.0.1:7421
npm run sidecar:llm        # Claude bridge      on 127.0.0.1:7422
```

In Tauri mode the header badges say `shapes (auto)` / `LLM (auto)` —
the sidecars are children of the app process and get killed cleanly on
window close. In browser mode you start them manually in extra
terminals; badges show `shapes` / `LLM` without the `(auto)` suffix.

## Distribution build

```bash
conda activate mlforge-dev
npm run tauri build      # writes .deb to
                         # src-tauri/target/release/bundle/deb/
sudo dpkg -i src-tauri/target/release/bundle/deb/mlforge_*_amd64.deb
mlforge                  # launch from terminal, or click the desktop entry
```

The `.deb` is ~80 MB. AppImage support is gated on `libfuse2` (and a
working `linuxdeploy` download); enable both `deb` and `appimage` in
`bundle.targets` once that's set up on the build host.

The bundle embeds `sidecar-torch/` and `sidecar-llm/` (including
`sidecar-llm/node_modules`). The Rust shell resolves the sidecar
location from the resource directory at runtime instead of the source
tree.

### Runtime requirements (installed machine)

After `sudo dpkg -i mlforge_*.deb`, the app needs the following
binaries discoverable on the desktop session's `PATH`:

- `python3` (or `python`) with `torch` installed (`pip install torch`
  for CPU, or use a system / conda Python that already has it)
- `node` (≥ 20)
- `claude` (the Claude Code CLI, authenticated via `claude setup-token`
  against your Max subscription)

If any of these is missing, the corresponding sidecar fails to spawn
and the badge stays on `offline`; the rest of the UI keeps working
(you can still build, save, and export `.py` files — the codegen is
client-side).

The `.deb` depends on `libwebkit2gtk-4.1-0`, `libgtk-3-0`, and
`libayatana-appindicator3-1`, which `dpkg`/`apt` will pull in
automatically on Ubuntu 24.04+ and Debian 13+.

The LLM sidecar spawns `claude` under the hood — make sure
`claude setup-token` was run once so it can hit your Max subscription.

### Datasets

Drop dataset files into `<workspace>/datasets/` and they show up in the
**Datasets** tab in the left sidebar. Supported kinds:

- `*.csv` / `*.tsv` / `*.parquet` — tabular (head, dtypes, per-column stats, correlations)
- folder with class subdirs — ImageFolder (class counts, thumbnails, sizes)
- `*.pt` / `*.pth` / `*.npy` / `*.npz` — tensors (shape/dtype/stats/histogram)
- `*.pdb` — protein structures (chains/residues/atoms, uses biopython if available)
- `*.smi` / `*.smiles` — SMILES files (MW/atom-count distributions if rdkit is available)
- `*.hf` text file containing `hf:dataset_name` — HuggingFace dataset reference

Optional Python deps (graceful fallback if missing): `pandas`, `Pillow`,
`rdkit`, `biopython`, `datasets`. The Datasets tab shows a hint with the
exact `pip install` command when a dep is missing.

Each detail view has a **Smoke test** tab that pulls one sample from the
dataset, feeds it through the current graph's generated model, and reports
the output shape + timings — useful to validate a model end-to-end against
real data without writing training code.

Verification harnesses:

```bash
npm run verify:codegen     # generates 4 graphs → writes .py → runs python on each
npm run verify:sidecar     # autostarts sidecar, posts generated code, checks shapes & errors
```

### Tauri Linux system deps (one-time)

```bash
sudo apt install -y \
  libwebkit2gtk-4.1-dev \
  libjavascriptcoregtk-4.1-dev \
  libsoup-3.0-dev \
  libgtk-3-dev \
  librsvg2-dev \
  libayatana-appindicator3-dev \
  pkg-config
```

## Claude integration

Requires Claude Code installed locally and authenticated via
`claude setup-token` (Max subscription). The Node sidecar spawned by Tauri
inherits that auth, so no API key is needed.
