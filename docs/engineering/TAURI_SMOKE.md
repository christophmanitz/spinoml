# Tauri/WebKitGTK window smoke

Source of truth for the manual checklist that closes `RELEASE_GATE.md`
condition 3 / `REMAINING_WORK.md` §2. Verified elsewhere (real Chromium for the CSP, unit
+ real-process tests for auth and scope). This window itself has never been exercised.

## Prerequisites

- [ ] Desktop session on Linux (X11/Wayland). **`.deb` is the only supported bundle
  target** (Linux only; `scope_file.rs` uses `std::os::unix`, see `LIMITATIONS.md` §7.7).
- [ ] Tauri build packages (`README.md` "Linux system dependencies"): `libwebkit2gtk-4.1-dev`,
  `libjavascriptcoregtk-4.1-dev`, `libsoup-3.0-dev`, `libgtk-3-dev`, `librsvg2-dev`,
  `libayatana-appindicator3-dev`, `pkg-config`. `cargo check` already compiles here,
  so most are present.
- [ ] `conda activate mlforge-dev` (Node 20, Rust, Python 3.12 with torch/pyg; the env kept its old
  name after the rename). Start `npm run tauri ...` AND the installed `spinoml` from a shell with
  this env active: the launcher needs `python3` (with torch) and `node` on `PATH`, otherwise the
  matching sidecar stays offline (known, see README); the rest of the UI still works.
- [ ] For step 10: the `opencode` CLI on `PATH` and a model with quota (default chat provider).
- [ ] Ports 7421/7422/7424 free. Stop any `npm run sidecar:*` first.
- [ ] ~10 GB free disk for `npm run tauri build`.
- [ ] A scratch workspace folder. For the offline check, ability to drop the network
  (`nmcli networking off` or unplug).

## Build / run

| Mode | Command | Notes |
|---|---|---|
| Dev (Tauri + Vite) | `npm run tauri dev` | `devCsp: null` — proves the app itself. Window opens after `vite` is ready. |
| Release (`.deb`)   | `npm run tauri build` | Output: `src-tauri/target/release/bundle/deb/spinoml_*_amd64.deb`. Install with `sudo dpkg -i <file>`, launch `spinoml` from the shell (see prerequisites). `sidecar-torch/` and `sidecar-llm/` are bundled as resources (`tauri.conf.json`). **A release build has no devtools** (the `devtools` feature is not enabled): for steps 2, 7 and 12 use a debug bundle, `npm run tauri build -- --debug`, which enables devtools; record in the results whether its CSP matched the release one. |
| Browser dev (no Rust) | `npm run dev` + `npm run sidecar:torch` + `npm run sidecar:llm` | Sidecars listen unauthenticated (`unauthenticated-dev`); the Tauri-specific checks below do NOT apply here. |

## Checklist

| # | Step | Expected | Evidence to paste | Result |
|---|---|---|---|---|
| 1 | `npm run tauri build -- --debug` (devtools) or `npm run tauri build`; install the `.deb`; run `spinoml`. | Window opens at 1400×900 (`tauri.conf.json` window config), no blank page. | First 5 lines of `src-tauri/target/release/bundle/deb/*.deb` install output. | |
| 2 | Debug bundle: right-click → Inspect Element → Console. | No line containing `Content Security Policy`, `Refused to`, `unsafe-inline`, or `unsafe-eval`. | `console.log` transcript (or `domcontentloaded` → first 30 lines). | |
| 3 | Header badges for torch and LLM sidecars. | Both show **online** with the `(auto)` suffix; no `ungesichert` chip. | Screenshot of header. | |
| 4 | Stop one sidecar (`kill <pid>` of the spawned torch/llm child). | The matching badge flips to **offline** within ~5 s. | `ps` before/after; badge screenshot. | |
| 5 | Restart that sidecar by hand without the token (e.g. `npm run sidecar:torch` after `pkill`). | Badge shows **auth failed** / `ungesichert` (a token-less sidecar in a Tauri build is the documented amber state; see `SIDECAR_AUTH.md` "Frontend"). | Badge tooltip text + stderr first 3 lines from the manual sidecar. | |
| 6 | Pick the scratch folder as workspace (toolbar `Open folder…`). | `~/.cache/spinoml/scope.json` (or `$XDG_RUNTIME_DIR/spinoml/scope.json` — see `scope_file.rs:39-51`) lists the canonical root, mode `0600`. `GET /health` with the token reports `scope.mode: enforced`. | `ls -l $(eval echo ~/.cache/spinoml/scope.json)` and `curl -H 'X-SpinoML-Token: …' http://127.0.0.1:7421/health` body. | |
| 7 | With network off (`nmcli networking off`), open the generated-code preview (Code panel) and the source field of a `Custom` layer in the Inspector. | A Monaco editor mounts **and highlights Python** without any external fetch. `Network` tab shows zero requests. | Screenshot; first 10 lines of the Network panel. | |
| 8 | Drag a `Linear` and `ReLU` from the palette onto the canvas, wire `Input → Linear → ReLU → Output`. Save (`Ctrl+S`), close the file, reopen, press `Ctrl+Z` immediately. | Nothing changes on undo (R060: first undo after open must NOT restore the previous file's graph). | Screenshot after `Ctrl+Z`. | |
| 9 | Datasets tab: open a dataset under `datasets/` that is a symlink to another directory. | If the target is in `SPINOML_SYMLINK_TARGETS` (or `scope.json` `symlink_targets`) — listed; otherwise refused with `PATH_SYMLINK_OUTSIDE` ("resolves outside the workspace through a symlink … add the TARGET directory to SPINOML_SYMLINK_TARGETS"). | Dataset explorer row + the one-line error if refused. | |
| 10 | Chat panel: opencode provider answers one turn ("Add one Linear layer"). Tool call mutates the canvas; `run_script` confirmation appears and a click approves. | A new `Linear` node is visible after the turn; the `run_script` confirm card appears and disappears on click. | Before/after canvas screenshots; chat transcript first 10 lines. | |
| 11 | Start a 2-epoch local training run via the New-run dialog (any small dataset under `datasets/`). | RunDetail modal shows epochs → `done`, integrity `ok`, metrics chart populated. | RunDetail modal screenshot. | |
| 12 | Devtools: `Promise.reject(new Error('x'))`. | The rose "Unbehandelte Fehler" diagnostics banner appears. | Screenshot. | |

## CSP fallback procedure

If any step in rows 2 or 7 shows a `Content Security Policy` / `Refused to` violation
(blank page, editor fails to mount, network requests blocked):

1. In `src-tauri/tauri.conf.json`, set `app.security.csp` to `null` (the previous,
   unrestricted behaviour). `devCsp` already stays `null` for `tauri dev`.
2. `npm run tauri build` again and reinstall the `.deb`.
3. Confirm the app works.
4. **Send back**: the exact console line(s) showing the violated directive (e.g.
   `Refused to load the script 'https://…' because it violates the following Content
   Security Policy directive: "script-src 'self'"`) and the failing URL/feature.
   The policy then needs a Tauri-specific directive (`ipc:`, `asset:`, or a nonce)
   and is fixed in code, not by leaving `csp: null`.

## Results template

```
Date:            YYYY-MM-DD
OS / kernel:     <uname -a>
WebKitGTK:       <pkg-config --modversion webkit2gtk-4.1>
Build type:      dev | release .deb
Commit:          <git rev-parse HEAD>
Sidecar token:   present (managed) | absent (manual)
Steps passed:    <n>/12
CSP fallback:    not needed | used (console output attached)
Notes:           <free text>
```

After this run is green, `RELEASE_GATE.md` §Security row for the window flips `[~] → [x]`
and `LIMITATIONS.md` §2 item 1 + §7.7 are updated with the actual WebKitGTK version and
any observed quirks.
