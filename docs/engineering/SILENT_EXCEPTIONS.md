# Silent exceptions — Phase 50

> Principle: *an explicit FAILED state is preferable to a false SUCCESS; an
> explicit UNKNOWN is preferable to an invented value.* A swallowed error is
> acceptable only when it cannot make the UI or stored data claim something
> untrue.

This document is the **allow-list** for `scripts/verify-silent-catch.ts`
(`npm run verify:silent-catch`). The guard uses the TypeScript AST to find every
handler that **swallows** an error — a `catch` block whose body has no statement
other than comments (or only `return` / `return <literal>`), or a
`.catch(<arrow with empty/literal body>)` — and fails unless (a) the handler
carries a comment of ≥ 15 characters of prose that is not merely
`ignore`/`noop`/…, and (b) it appears in the table below.

Matching is by **file + pattern text**, so line-number drift is tolerated; a
pattern that occurs more than once in a file must appear that many times.

## Allow-list — current swallowing sites (all EXPECTED)

| Location | Pattern | Class | Reason (why the swallow cannot lie) | Action |
|---|---|---|---|---|
| `src/canvas/layoutStore.ts:11` | `catch { return 'TB' }` | EXPECTED | localStorage feature detection: the default flow direction is not a stored-state claim. | keep |
| `src/canvas/layoutStore.ts:28` | `catch {}` | EXPECTED | localStorage quota/private mode: only cross-reload persistence is lost, the live value is applied. | keep |
| `src/canvasdoc/store.ts:43` | `catch { return {} }` | EXPECTED | localStorage feature detection: no persisted canvas bindings; each canvas starts on its chooser. | keep |
| `src/canvasdoc/store.ts:56` | `catch {}` | EXPECTED | localStorage quota: only cross-reload persistence of the binding is skipped. | keep |
| `src/chat/client.ts:70` | `catch {}` | EXPECTED | Best-effort `respond` POST; a dropped answer makes the sidecar turn time out loudly, not a false success. | keep |
| `src/chat/client.ts:79` | `catch { return false }` | EXPECTED | Health probe: unreachable sidecar renders as explicit *offline*, never as a verified model. | keep |
| `src/chat/client.ts:94` | `catch { return [] }` | EXPECTED | Model-list probe: empty list falls back to built-in suggestions, an explicit UI state. | keep |
| `src/chat/client.ts:115` | `.catch(() => '')` | EXPECTED | Only enriches the thrown HTTP error; the HTTP status is the real failure and is still thrown. | keep |
| `src/chat/providerStore.ts:126` | `catch {}` | EXPECTED | localStorage quota/private mode: only provider-pref persistence is skipped. | keep |
| `src/chat/store.ts:95` | `catch { return [] }` | EXPECTED | localStorage feature detection: empty browser-session chat; not a claim about model/runs. | keep |
| `src/chat/store.ts:105` | `catch {}` | EXPECTED | localStorage quota: only cross-reload persistence of the chat is skipped. | keep |
| `src/chat/store.ts:114` | `catch {}` | EXPECTED | Best-effort cleanup on explicit project close; worst case the old chat is restored. | keep |
| `src/chat/store.ts:375` | `catch {}` | EXPECTED | A single unreadable note is skipped; notes are optional LLM context, not user-facing. | keep |
| `src/chat/store.ts:380` | `catch {}` | EXPECTED | Notes listing is optional LLM context; its absence cannot make an answer untrue. | keep |
| `src/chat/uiStore.ts:26` | `catch { return 1 }` | EXPECTED | localStorage feature detection: default font scale. | keep |
| `src/chat/uiStore.ts:33` | `catch { return false }` | EXPECTED | localStorage feature detection: default auto-approve off (the safer value). | keep |
| `src/chat/uiStore.ts:43` | `catch { return 'verbose' }` | EXPECTED | localStorage feature detection: default documentation verbosity. | keep |
| `src/chat/uiStore.ts:69` | `catch {}` | EXPECTED | localStorage quota: only persistence of the font scale is skipped. | keep |
| `src/chat/uiStore.ts:79` | `catch {}` | EXPECTED | localStorage quota: only persistence of auto-mode is skipped. | keep |
| `src/chat/uiStore.ts:88` | `catch {}` | EXPECTED | localStorage quota: only persistence of doc-mode is skipped. | keep |
| `src/codegen/explain.ts:37` | `catch { return '' }` | EXPECTED | A decorative registry summary is omitted rather than invented. | keep |
| `src/connections/store.ts:98` | `catch {}` | EXPECTED | localStorage quota/private mode: only connection-list persistence is skipped. | keep |
| `src/datasets/store.ts:257` | `catch {}` | EXPECTED | Smoke history logging is secondary; the smoke result itself is already shown truthfully. | keep |
| `src/datasets/store.ts:271` | `catch {}` | EXPECTED | A torn/corrupt trailing line in the append-only smoke log is expected. | keep |
| `src/inference/client.ts:49` | `catch { return false }` | EXPECTED | Health probe: unreachable torch sidecar renders as explicit *offline*. | keep |
| `src/palette/Palette.tsx:26` | `catch {}` | EXPECTED | localStorage quota: only persistence of the collapse state is skipped. | keep |
| `src/persistence/file.ts:104` | `catch {}` | EXPECTED | localStorage quota/private mode: autosave write is best-effort by design. | keep |
| `src/persistence/file.ts:122` | `catch { return null }` | EXPECTED | Autosave recovery: missing and corrupt both mean nothing to restore; no restored-state claim. | keep |
| `src/persistence/file.ts:131` | `catch {}` | EXPECTED | Best-effort autosave cleanup; failure cannot create a false state. | keep |
| `src/project/store.ts:163` | `catch {}` | EXPECTED | Best-effort teardown while closing a remote project; the target is dropped regardless. | keep |
| `src/project/store.ts:168` | `catch {}` | EXPECTED | Best-effort sidecar stop on close; status flips to `stopped` and no stale running status is shown. | keep |
| `src/project/store.ts:174` | `catch {}` | EXPECTED | Best-effort local directory cleanup on close; the workspace binding is cleared regardless. | keep |
| `src/sidecars/remoteSidecar.ts:45` | `catch {}` | EXPECTED | Best-effort remote-sidecar teardown; status is set to `stopped` so no stale tunnel is shown. | keep |
| `src/terminal/Terminal.tsx:66` | `catch {}` | EXPECTED | First fit during initial layout may hit a 0×0 element; ResizeObserver refits later. | keep |
| `src/terminal/Terminal.tsx:87` | `.catch(() => {})` | EXPECTED | Spawn/unmount race: killing the just-spawned child is pure cleanup. | keep |
| `src/terminal/Terminal.tsx:112` | `.catch(() => {})` | EXPECTED | Best-effort PTY keystroke; if the child is gone the `pty:exit` listener already shows it. | keep |
| `src/terminal/Terminal.tsx:124` | `catch {}` | EXPECTED | Refit can fail while the element is hidden; terminal still works. | keep |
| `src/terminal/Terminal.tsx:130` | `.catch(() => {})` | EXPECTED | Best-effort PTY resize; the next refit retries. | keep |
| `src/terminal/Terminal.tsx:142` | `catch {}` | EXPECTED | Refit from IntersectionObserver can fail on a hidden element; no user-visible claim. | keep |
| `src/terminal/Terminal.tsx:148` | `.catch(() => {})` | EXPECTED | Best-effort PTY resize from the visibility path. | keep |
| `src/terminal/Terminal.tsx:167` | `.catch(() => {})` | EXPECTED | Best-effort PTY kill on unmount; component is going away. | keep |
| `src/training/charts/series.ts:15` | `catch {}` | EXPECTED | A partial trailing line in the append-only events log is expected (SIGKILL). | keep |
| `src/training/graph/autosave.ts:16` | `catch {}` | EXPECTED | localStorage quota/private mode: training-graph autosave is best-effort. | keep |
| `src/training/graph/autosave.ts:36` | `catch { return null }` | EXPECTED | Autosave recovery: missing and corrupt both mean nothing to restore. | keep |
| `src/workspace/FileViewerModal.tsx:67` | `catch {}` | EXPECTED | Clipboard API unavailable: only the "copied" feedback is skipped. | keep |
| `src/workspace/PyCodeModal.tsx:38` | `catch {}` | EXPECTED | Clipboard API unavailable: the generated code shown is unaffected. | keep |
| `src/workspace/recentWorkspaces.ts:24` | `catch { return [] }` | EXPECTED | localStorage feature detection: convenience recents list, not a claim about disk. | keep |
| `src/workspace/recentWorkspaces.ts:34` | `catch {}` | EXPECTED | localStorage quota: only persistence of the recents list is skipped. | keep |
| `src/workspace/recentWorkspaces.ts:63` | `catch { return null }` | EXPECTED | localStorage feature detection: no remembered workspace → accurate Welcome screen. | keep |
| `src/workspace/recentWorkspaces.ts:75` | `catch {}` | EXPECTED | localStorage quota: only auto-reopen of the last workspace is lost. | keep |
| `src/workspace/recentWorkspaces.ts:96` | `catch { return null }` | EXPECTED | localStorage feature detection: no remembered file → canvas chooser, never the wrong file. | keep |
| `src/workspace/recentWorkspaces.ts:108` | `catch {}` | EXPECTED | localStorage quota: only the remembered file binding is lost. | keep |
| `src/workspace/store.ts:100` | `catch { return false }` | EXPECTED | A `.spinoml` may legitimately have no `.py` twin; an unreadable twin is treated as absent. | keep |
| `src/workspace/store.ts:175` | `catch {}` | EXPECTED | localStorage quota in browser mode: in-memory workspace stays valid; disk (Tauri) unaffected. | keep |
| `src/workspace/store.ts:532` | `catch {}` | EXPECTED | Best-effort directory cleanup on close; the pointer is dropped regardless. | keep |
| `src/workspace/store.ts:612` | `catch { return null }` | EXPECTED | Unparseable content yields no dirty baseline; dirty stays false conservatively. | keep |

## Fixed hidden failures (no longer swallowing)

Each of these previously swallowed an error that could make the UI or a stored
artifact claim success/valid/empty-but-fine. They now end in an explicit
error/unknown value the UI renders.

| Location (before) | Was | Now |
|---|---|---|
| `src/workspace/store.ts` create/save/import `.py` twin | ignored twin-write failure | `pyTwinError` in the workspace store, shown as a red banner in `FileExplorer` |
| `src/workspace/store.ts` rename/remove/move `.py` twin | assumed "maybe absent" | probes presence; a real rename/remove/move failure is surfaced |
| `src/training/graph/doc.ts` `ensureTrainingBound` | left graph unbound silently | `setStatus('training', 'error', …)` on the canvas header |
| `src/data/graph/doc.ts` `ensureDataBound` | left graph unbound silently | `setStatus('data', 'error', …)` on the canvas header |
| `src/training/RunDetailModal.tsx` run.json parse | `parsedCfg = null` (silent) | `cfgError` banner "run.json beschädigt: …" |
| `src/training/RunDetailModal.tsx` `openOnCanvas` | button did nothing | explicit `actionError` |
| `src/training/RunDetailModal.tsx` `reload`/`tailReload` | fall back to empty/previous logs | per-file `readError` + `eventsError` banner |
| `src/training/RunDetailModal.tsx` gpu poll | `setGpu([])` → "no GPU" | `gpuError` banner (unknown ≠ none) |
| `src/training/RunDetailModal.tsx` isMultitask | second parse swallowed | reuses `parsed.cfg`; corrupt config already surfaced |
| `src/training/CompareModal.tsx` events/run.json | empty defaults | per-run `⚠ run.json` / `⚠ events` markers |
| `src/training/EvalRunModal.tsx` `srcManifest` | fake `{pairs,target}` default | throws a descriptive corruption error |
| `src/training/EvalRunModal.tsx` source inspect | swallowed | `srcWarn` banner (optional enrichment) |
| `src/training/EvalRunModal.tsx` caps probe | `setCaps(null)` | `capsError`: "Fähigkeiten unbekannt", backend not guessed |
| `src/training/NewRunModal.tsx` caps probe | `setCaps(null); setBackendKind('local')` | `capsError` unknown state; no silent fallback to local for a remote host |
| `src/sidecars/remoteSidecar.ts` `refresh` | kept previous status | explicit `error`/unknown status |
| `src/workspace/FileExplorer.tsx` `openGraphOnCanvas` | silent no-op | `alert()` with the read/parse error |
| `src/canvasdoc/CanvasFileGate.tsx` bound-file open | silently unbound to chooser | `loadError` panel with "Andere Datei wählen" |
| `src/canvasdoc/CanvasFileGate.tsx` chooser list | `setList([])` → "keine Dateien" | `listErr` message distinct from an empty directory |
| `src/canvasdoc/ReloadCanvasButton.tsx` reload | swallowed (wrong comment) | `alert()` with the failure |
| `src/training/graph/TrainingInspector.tsx` ModelRef list | empty dropdown | `listErr` message under the select |
| `src/visualization/LayerExplain.tsx` runs list | false "no trained run" | `runsErr` message (weights stay random) |
| `src/chat/client.ts` malformed SSE frame | `console.warn`, stream continues | throws → the turn surfaces an error instead of a truncated "done" |
| `src/datasets/store.ts` `loadHistory` | empty history on read error | `historyError` rendered in the smoke panel |

## Summary

- Sites detected by the AST scan before the fix: **80**.
- Swallowing sites remaining after the fix (all EXPECTED, documented above): **56**.
- Swallowing sites that became explicit handlers: **24**.
- Additional non-swallowing hidden handlers fixed (bodies had statements, so the
  guard does not flag them): **10** (`NewRunModal`/`EvalRunModal` caps,
  `CanvasFileGate` load+list, `ReloadCanvasButton`, `TrainingInspector` ModelRef,
  `LayerExplain` runs, `chat/client` SSE, `datasets/store` history,
  `EvalRunModal` source inspect).
- Evidence of the pre-fix scan: `docs/engineering/evidence/phase50-ts-before.txt`.

# Node sidecar (`sidecar-llm/*.mjs`)

The same guard and the same rules are applied to the Node LLM sidecar
(`sidecar-llm/main.mjs`, `sidecar-llm/mcp-bridge.mjs`; the pure helpers
`shell-safety.mjs` / `path-scope.mjs` are out of scope). A swallow here is
acceptable only when it cannot make a **tool result** or the **chat** claim
something untrue — the model reads these results as fact, so a false empty list
("no runs", "no notes", "no models") or an invented value is a hidden failure.

## Allow-list — Node sidecar swallowing sites (all EXPECTED)

| Location | Pattern | Class | Reason (why the swallow cannot lie) | Action |
|---|---|---|---|---|
| `sidecar-llm/main.mjs:512` | `catch { return false }` | EXPECTED | `wsPathExists`: an absent file feeds an explicit "does not exist" tool error, never a write claim. | keep |
| `sidecar-llm/main.mjs:1623` | `catch { return false }` | EXPECTED | `confirmContinue`: an unanswerable continue-prompt stops the turn; no tool result or artifact is affected. | keep |
| `sidecar-llm/main.mjs:559` | `catch {}` | EXPECTED | `onChunk` listener: a throwing consumer must not break the run; stdout/stderr are still captured and returned. | keep |
| `sidecar-llm/main.mjs:567` | `catch {}` | EXPECTED | `spawnCapture` timeout: SIGTERM on an already-exited child is pure cleanup. | keep |
| `sidecar-llm/main.mjs:568` | `catch {}` | EXPECTED | `spawnCapture` timeout: the SIGKILL backstop on an already-exited child is pure cleanup. | keep |
| `sidecar-llm/main.mjs:1485` | `catch {}` | EXPECTED | `run_script` confirm preview: a failed read shows the prompt without a preview, an explicit state. | keep |
| `sidecar-llm/main.mjs:1712` | `catch {}` | EXPECTED | OpenAI tool-call `JSON.parse`: malformed model args become `{}` and schema validation returns an explicit tool error. | keep |
| `sidecar-llm/main.mjs:1887` | `catch {}` | EXPECTED | OpenCode `killHard`: SIGTERM on a possibly-dead child is best-effort cleanup. | keep |
| `sidecar-llm/main.mjs:1889` | `catch {}` | EXPECTED | OpenCode `killHard`: the SIGKILL backstop is best-effort cleanup. | keep |
| `sidecar-llm/main.mjs:1981` | `.catch(() => {})` | EXPECTED | OpenCode close: dropping the disposable temp session dir is best-effort cleanup. | keep |
| `sidecar-llm/main.mjs:1993` | `.catch(() => {})` | EXPECTED | OpenCode error path: dropping the disposable temp session dir is best-effort cleanup. | keep |
| `sidecar-llm/main.mjs:2354` | `catch {}` | EXPECTED | SSE `emit`: a write after the client disconnected has no reader left to mislead. | keep |
| `sidecar-llm/main.mjs:2365` | `catch {}` | EXPECTED | `setNoDelay` on a non-TCP socket: the heartbeat still works and no result is claimed. | keep |
| `sidecar-llm/main.mjs:2367` | `catch {}` | EXPECTED | SSE heartbeat write: a write after the client disconnected has no reader left to mislead. | keep |
| `sidecar-llm/main.mjs:2416` | `catch {}` | EXPECTED | SSE action-pump heartbeat write: a write after the client disconnected has no reader left to mislead. | keep |
| `sidecar-llm/main.mjs:2601` | `catch {}` | EXPECTED | Final error frame when the client is already gone; already inside the explicit error path. | keep |
| `sidecar-llm/mcp-bridge.mjs:48` | `.catch(() => null)` | EXPECTED | A non-JSON sidecar body is turned into an explicit HTTP error just below, not a silent success. | keep |

## Fixed hidden failures (no longer swallowing)

| Location (before) | Was | Now |
|---|---|---|
| `sidecar-llm/main.mjs` `notesList` local readdir | `.catch(() => [])` → "no notes" on an unreadable dir | `readdirOptional`: ENOENT → `[]` (explicit empty), any other error propagates |
| `sidecar-llm/main.mjs` `notesList` stat | failed stat → invented `size: 0` | `size: null` + `stat_error`; `list_notes` renders "(size unknown)" |
| `sidecar-llm/main.mjs` `wsListDir` (local + ssh) | `.catch(() => [])` / `ls … \|\| true` → run listing false-empty | `readdirOptional` locally; remotely `[ -d ]` → `[]` then a bare `ls` so a read error propagates |
| `sidecar-llm/main.mjs` `wsReadFile` | `.catch(() => '')` / remote `cat … \|\| true` | ENOENT → `''`, permission/IO failure throws (remote probes `[ -e ]` first) |
| `sidecar-llm/main.mjs` `wsListDirDetailed` (local + ssh) | `.catch(() => [])` / `ls … 2>/dev/null \|\| true` → `list_dir` false-empty | `readdirOptional` locally; remotely `[ -d ]` then a bare `ls`, so a read error propagates |
| `sidecar-llm/main.mjs` `readSummaryEvents` | `catch { return '' }` → silent empty summary | returns `{ text, error }`; an unreadable events log surfaces as an explicit error |
| `sidecar-llm/main.mjs` `runsList` run.json/metrics.json | `catch { /* skip */ }` → `{}` | `readRunJson` state + per-run `warning` (missing/corrupt/unreadable) shown on the run line |
| `sidecar-llm/main.mjs` `runRead` run.json/metrics.json | `catch { /* skip */ }` → `{}` | `readRunJson` state reported as `warnings`; an unreadable events log throws |
| `sidecar-llm/main.mjs` `downloadToDatasets` remote size | `parseInt(...) \|\| 0` → invented 0 bytes | throws "size could not be determined" instead of a false byte count |
| `sidecar-llm/main.mjs` `listOpenCodeModels` | ignored exit/timeout → empty model list | throws on timeout/abort/non-zero exit; `/opencode/models` returns an explicit error |
| `sidecar-llm/main.mjs` `slurmStatus` | ignored exit/timeout → `UNKNOWN` | throws on timeout/abort/ssh-255; `UNKNOWN` only for a real scheduler answer |
| `sidecar-llm/mcp-bridge.mjs` stdin dispatch | `handle(msg).catch(() => {})` → request left unanswered (opencode hangs) | replies with an explicit JSON-RPC `INTERNAL_ERROR` |

## Summary — Node sidecar

- Sites detected by the Node AST scan before the fix: **27**.
- Swallowing sites remaining after the fix (all EXPECTED, documented above): **17**.
- AST-flagged sites that became explicit handlers: **10** (`notesList` readdir,
  `wsListDir`, `wsReadFile`, `wsListDirDetailed`, `readSummaryEvents`,
  `runsList` run.json + metrics.json, `runRead` run.json + metrics.json,
  mcp-bridge dispatch).
- Additional non-swallowing hidden handlers fixed (no empty/literal catch, so the
  guard does not flag them): **6** (`notesList` stat, `downloadToDatasets` size,
  `listOpenCodeModels`, `slurmStatus`, `wsListDir` remote `\|\| true`,
  `wsListDirDetailed` remote `\|\| true`).
- Total hidden failures fixed: **16**.
- Evidence of the pre-fix scan: `docs/engineering/evidence/phase50-node-before.txt`.
