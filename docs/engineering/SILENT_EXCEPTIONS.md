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
| `src/training/RunDetailModal.tsx:291` | `catch { return null }` | EXPECTED | Phase 74 manifest.json parse failure: null is the documented "unknown" state; the banner falls back to "Manifest nicht lesbar" via the separate `manifestReadProblem` channel so a corrupt manifest never claims "not resumable". | keep |
| `src/training/RunDetailModal.tsx:119` | `catch {}` | EXPECTED | `mkdir` for the figures export directory: already-exists is benign, the write that follows will still succeed or fail explicitly. | keep |
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

# Python sidecar + trainer (`sidecar-torch/*.py`)

The same guard and the same rules are applied to the Python sources
(`sidecar-torch/main.py`, `dataset_handlers.py`, `safe_load.py`, `scope.py`,
`deps_policy.py`, `auth.py`, `training_template.py`). A swallow here is
acceptable only when it cannot make a **status**, **result**, **metric** or
**dataset claim** untrue. The trainer and the dataset handlers are
paper-grade: a swallowed error that lets a run/dataset claim something untrue
is a correctness bug, not style.

`scripts/verify-silent-except-py.py` (`npm run verify:silent-except-py`) uses
the stdlib `ast` to find every handler whose body is only `pass` / `...` / a
docstring / `continue` / `break` / `return` / `return <literal>`, or a
`contextlib.suppress(...)`. Matching is by **file + normalised pattern text**,
so line-number drift is tolerated; a pattern occurring N times in a file must
appear N times here.

## Python allow-list — current swallowing sites (all EXPECTED)

| Location | Pattern | Class | Reason (why the swallow cannot lie) | Action |
|---|---|---|---|---|
| `sidecar-torch/auth.py:132` | `except Exception: pass` | EXPECTED | `scrub_environ`: a read-only environ mapping (test double) cannot be mutated; production `os.environ` supports `pop`, so the token is still scrubbed. | keep |
| `sidecar-torch/auth.py:196` | `except (UnicodeEncodeError, AttributeError): return False` | EXPECTED | `token_matches`: a malformed supplied token fails the constant-time compare (no match, fail closed). | keep |
| `sidecar-torch/dataset_handlers.py:95` | `except OSError: pass` | EXPECTED | `.txt` kind-sniff failure just leaves the kind `unknown`, never a wrong kind. | keep |
| `sidecar-torch/dataset_handlers.py:135` | `except OSError: return None` | EXPECTED | unreadable directory → no primary table found; the kind stays `unknown`, an explicit state. | keep |
| `sidecar-torch/dataset_handlers.py:174` | `except OSError: return []` | EXPECTED | unreadable directory → no `.pt` discovered; detection stays `unknown`, not a false empty dataset. | keep |
| `sidecar-torch/dataset_handlers.py:186` | `except OSError: return False` | EXPECTED | unreadable directory → not classified as an image folder; kind stays `unknown`. | keep |
| `sidecar-torch/dataset_handlers.py:195` | `except OSError: continue` | EXPECTED | an unreadable class subdir is skipped while probing; folder detection is heuristic and stays honest. | keep |
| `sidecar-torch/dataset_handlers.py:244` | `except ScopeError: continue` | EXPECTED | `_fingerprint_dir`: a file resolving outside the scope is never hashed. | keep |
| `sidecar-torch/dataset_handlers.py:277` | `except Exception: return None` | EXPECTED | `_fingerprint_manifest`: a fingerprint is optional provenance; `None` is the documented unknown, never a false identity. | keep |
| `sidecar-torch/dataset_handlers.py:307` | `except (OSError, ScopeError): return None` | EXPECTED | `_fingerprint_for`: same — an unfingerprintable source reports no id instead of inventing one. | keep |
| `sidecar-torch/dataset_handlers.py:415` | `except OSError: pass` | EXPECTED | `_describe_dir_bundle`: an unreadable side-file listing yields an empty summary; the primary table was already parsed and is the real claim. | keep |
| `sidecar-torch/dataset_handlers.py:449` | `except ScopeError: continue` | EXPECTED | a symlinked-out image is never opened or thumbnailed. | keep |
| `sidecar-torch/dataset_handlers.py:464` | `except Exception: continue` | EXPECTED | one failed thumbnail is skipped; the other thumbnails and `sample_size` stay valid. | keep |
| `sidecar-torch/dataset_handlers.py:466` | `except ImportError: pass` | EXPECTED | optional dependency: Pillow absent → no thumbnails, but the class/size structure is still shown. | keep |
| `sidecar-torch/dataset_handlers.py:645` | `except Exception: pass` | EXPECTED | tensor `min`/`max`/`mean` are informational; the tensor itself loaded and its shape/dtype are reported. | keep |
| `sidecar-torch/dataset_handlers.py:682` | `except ImportError: pass` | EXPECTED | optional dependency: Biopython absent → the line-based fallback parser handles the `.pdb`. | keep |
| `sidecar-torch/dataset_handlers.py:966` | `except Exception: pass` | EXPECTED | a corrupt cached mol `.pt` is rebuilt from the SMILES; the cache is derived data, never truth. | keep |
| `sidecar-torch/dataset_handlers.py:975` | `except Exception: pass` | EXPECTED | mol-graph disk caching is best-effort; sampling still works without the cache. | keep |
| `sidecar-torch/dataset_handlers.py:1123` | `except FileNotFoundError: return []` | EXPECTED | no ESPF codebook → no substructure labels; the token preview is still shown. | keep |
| `sidecar-torch/dataset_handlers.py:1153` | `except FileNotFoundError: return None` | EXPECTED | no ESPF codebook → nothing to cache; a later training run raises an actionable error, not a success. | keep |
| `sidecar-torch/dataset_handlers.py:1167` | `except Exception: return None` | EXPECTED | the `.espf` cache write is best-effort; the sidecar still tokenizes in memory. | keep |
| `sidecar-torch/dataset_handlers.py:1537` | `except Exception: pass` | EXPECTED | a column histogram is informational; the column summary is still emitted. | keep |
| `sidecar-torch/dataset_handlers.py:1574` | `except ScopeError: continue` | EXPECTED | `_stats_image_folder`: an image outside the scope is not counted. | keep |
| `sidecar-torch/dataset_handlers.py:1579` | `except Exception: continue` | EXPECTED | a failed image open is skipped; the other sampled sizes are still counted. | keep |
| `sidecar-torch/dataset_handlers.py:1623` | `except Exception: pass` | EXPECTED | a tensor histogram is informational; `std`/`zeros_frac` are still reported. | keep |
| `sidecar-torch/main.py:339` | `except Exception: pass` | EXPECTED | `infer` `n_params` recount is best-effort; the pre-forward count is retained and a param count is not a correctness claim. | keep |
| `sidecar-torch/main.py:455` | `except Exception: pass` | EXPECTED | `smoke_test` `n_params` recount, same as above. | keep |
| `sidecar-torch/main.py:572` | `except Exception: pass` | EXPECTED | ESPF substructure labels are a best-effort overlay for the Explain view; the raw token preview is still shown. | keep |
| `sidecar-torch/main.py:686` | `except Exception: pass` | EXPECTED | best-effort lazy (in_channels=-1) init probe; a real shape error surfaces at the real, hooked forward pass. | keep |
| `sidecar-torch/main.py:782` | `except Exception: pass` | EXPECTED | viz post-processing (input/output previews + weight snapshots) only; the captured activations are kept. | keep |
| `sidecar-torch/safe_load.py:65` | `except ImportError: pass` | EXPECTED | optional dependency: PyG absent → those globals are simply not registered. | keep |
| `sidecar-torch/safe_load.py:70` | `except ImportError: pass` | EXPECTED | optional dependency: PyG absent → `HeteroData` global not registered. | keep |
| `sidecar-torch/safe_load.py:78` | `except ImportError: pass` | EXPECTED | optional dependency: PyG absent → storage globals not registered. | keep |
| `sidecar-torch/safe_load.py:104` | `except ImportError: pass` | EXPECTED | optional: `numpy.dtypes` unavailable on this build → no extra dtype globals. | keep |
| `sidecar-torch/safe_load.py:108` | `except AttributeError: pass` | EXPECTED | torch < 2.4 has no safe-globals API; plain tensors/dicts still load, other globals are refused. | keep |
| `sidecar-torch/scope.py:69` | `except AttributeError: return None` | EXPECTED | platform has no `os.getuid` (Windows) → the uid is reported unknown, not invented. | keep |
| `sidecar-torch/scope.py:271` | `except ValueError: return False` | EXPECTED | `commonpath` mismatch (different drives / mixed abs-rel) → not inside the root (fail closed). | keep |
| `sidecar-torch/training_template.py:100` | `except ImportError: pass` | EXPECTED | synced safe-load block: optional PyG globals (mirror of `safe_load.py`). | keep |
| `sidecar-torch/training_template.py:105` | `except ImportError: pass` | EXPECTED | synced safe-load block: optional `HeteroData` global. | keep |
| `sidecar-torch/training_template.py:113` | `except ImportError: pass` | EXPECTED | synced safe-load block: optional PyG storage globals. | keep |
| `sidecar-torch/training_template.py:139` | `except ImportError: pass` | EXPECTED | synced safe-load block: optional `numpy.dtypes` globals. | keep |
| `sidecar-torch/training_template.py:143` | `except AttributeError: pass` | EXPECTED | synced safe-load block: torch < 2.4 has no safe-globals API. | keep |
| `sidecar-torch/training_template.py:232` | `except FileNotFoundError: return ''` | EXPECTED | an absent status file genuinely means "not started"; any other read error uses the explicit `_STATUS_UNREADABLE` sentinel instead. | keep |
| `sidecar-torch/training_template.py:354` | `except Exception: pass` | EXPECTED | RNG capture: no CUDA backend → nothing to capture (best-effort; determinism event documents the caveat). | keep |
| `sidecar-torch/training_template.py:359` | `except Exception: pass` | EXPECTED | RNG capture: NumPy absent → no NumPy stream to capture (best-effort). | keep |
| `sidecar-torch/training_template.py:364` | `except Exception: pass` | EXPECTED | RNG capture: stdlib random capture is best-effort. | keep |
| `sidecar-torch/training_template.py:416` | `except Exception: pass` | EXPECTED | `_atomic_save` directory fsync is durability polish; the rename has already happened. | keep |
| `sidecar-torch/training_template.py:422` | `except Exception: pass` | EXPECTED | `_atomic_save` temp-file cleanup is best-effort; the final file is already in place. | keep |
| `sidecar-torch/training_template.py:542` | `except Exception: pass` | EXPECTED | `_env_info`: a build without cuDNN omits the version (unknown, not faked). | keep |
| `sidecar-torch/training_template.py:544` | `except Exception: pass` | EXPECTED | `_env_info`: torch metadata unavailable → the field is omitted (unknown). | keep |
| `sidecar-torch/training_template.py:549` | `except Exception: pass` | EXPECTED | `_env_info`: NumPy absent → version field omitted (unknown). | keep |
| `sidecar-torch/training_template.py:564` | `except Exception: pass` | EXPECTED | `_env_info`: a failing CUDA query omits the GPU fields (unknown). | keep |
| `sidecar-torch/training_template.py:571` | `except Exception: pass` | EXPECTED | `_env_info`: `sysconf` unsupported → RAM field omitted (unknown). | keep |
| `sidecar-torch/training_template.py:581` | `except Exception: pass` | EXPECTED | `_env_info`: not a git repo / git absent → the commit field is omitted, never invented. | keep |
| `sidecar-torch/training_template.py:935` | `except (TypeError, ValueError): return False` | EXPECTED | `_is_finite_loss`: a non-numeric loss is not finite → integrity fails closed. | keep |
| `sidecar-torch/training_template.py:1042` | `except Exception: continue` | EXPECTED | a torn/corrupt event line is skipped; the required event kinds are still checked by name. | keep |
| `sidecar-torch/training_template.py:1438` | `except Exception: pass` | EXPECTED | mol-graph disk cache write is best-effort; the graph is returned from memory. | keep |
| `sidecar-torch/training_template.py:1559` | `except Exception: pass` | EXPECTED | ESPF token disk cache write is best-effort; the ids are returned from memory. | keep |
| `sidecar-torch/training_template.py:2186` | `except (TypeError, AttributeError): pass` | EXPECTED | older torch without `warn_only`: determinism is documented, not enforced (never a success claim). | keep |
| `sidecar-torch/training_template.py:2585` | `except Exception: pass` | EXPECTED | cancel-at-boundary `last.pt` save is best-effort; the cancellation is already recorded and `_compute_resumable` reports honestly. | keep |
| `sidecar-torch/training_template.py:2706` | `except Exception: pass` | EXPECTED | signal-cancel `last.pt` save is best-effort; the cancellation is already recorded. | keep |
| `sidecar-torch/training_template.py:2721` | `except Exception: pass` | EXPECTED | failure-path `last.pt` save is best-effort; the failure is already recorded and the run is not claimed resumable. | keep |

## Fixed hidden failures (no longer swallowing)

| Location (before) | Was | Now |
|---|---|---|
| `sidecar-torch/training_template.py` `_read_status` | `except Exception: return ""` — an unreadable status looked like "not started", so a late terminal write could overwrite a real terminal state (CANCELLED → SUCCEEDED). | `FileNotFoundError` → `""`; any other read error → the explicit `_STATUS_UNREADABLE` sentinel, which blocks every transition, and `main()` turns it into a failed run. |
| `sidecar-torch/training_template.py` dummy-forward `except StopIteration` | `pass` — an empty training loader (`drop_last=True` with `batch_size` > training rows) skipped lazy init and every epoch reported `train_loss=0.0`, ending `done`. | `fail("split", …)` — an untrained run can never be reported done. |
| `sidecar-torch/training_template.py` `_compute_resumable` | a corrupt/unreadable manifest left the run hashes unknown, which was treated as a match → `resumable: true` for a checkpoint that might belong to another model. | unknown run hashes → `resumable: false`, reason `run hashes unavailable (manifest unreadable)`. |
| `sidecar-torch/dataset_handlers.py` `espf_vocab_size` | a missing ESPF codebook returned `2` — an invented, unusably small `num_embeddings` while `tokenize_espf` actually fell back to the char-level tokenizer. | returns the fallback char-level sequence vocab size that `tokenize_espf` really uses, so the inspect note is true. |
| `sidecar-torch/training_template.py` `_restore_rng` | four `except Exception: pass` handlers — a resume whose torch/CUDA/NumPy/stdlib random streams could not be restored silently continued with a fresh-seed stream while the run claimed to continue "from the same random streams" (Phase 26). | `_restore_rng` returns `{stream: restored / absent in checkpoint / failed: <reason>}`, recorded as `rng_restore` in the `run.resumed` event (never fatal: a CUDA stream on a CPU-only resume is legitimately unrestorable, but it is now visible). Regression: `verify-checkpoint.ts [rng restore failure is recorded]`. |

## Summary — Python

- Sites detected by the AST scan before the fix: **69**.
- Swallowing sites remaining after the fix (all EXPECTED, documented above): **62**.
- AST-flagged sites that became explicit handlers: **3** (`_read_status`'s
  non-ENOENT branch, the empty-loader `StopIteration`, the corrupt-manifest
  resumable branch). `espf_vocab_size` changed from `return 2` to a call, so it
  stopped being a swallow.
- Additional non-swallowing hidden handlers reviewed (single literal
  assignment, so the guard does not flag them): **26**; the notable ones in the
  trainer/dataset paths are listed under NOTES of the Phase-50-Python report.
- Regression tests: `scripts/verify-failures.ts` (status file, empty loader),
  `scripts/verify-integrity.ts` (corrupt-manifest resumable),
  `scripts/test-sidecar-robustness.ts` (ESPF vocab fallback).

# Rust (`src-tauri/src/*.rs`)

The same rules are applied to the Rust shell. There is no `syn` available in the
script toolchain, so `scripts/verify-rust-panics.ts` (`npm run verify:rust-panics`,
self-test `test:rust-panics-selftest`) masks string / line / block / raw / char
literals and comments, excludes every `#[cfg(test)]` item by brace matching, and
then flags: `unwrap(` / `expect(` / `panic!` / `unreachable!` / `todo!` /
`unimplemented!` / `let _ =` / `.ok()` / `unwrap_or_default` / `unwrap_or(` /
`unwrap_or_else(`. The pattern key is the code **before any same-line comment**, so
an appended justification comment never changes the allow-list match. A swallow here
is acceptable only when it cannot make a **UI status**, **stored artifact** or
**command result** claim something untrue.

**Limits (manual scan, not automatic):** slice/array indexing, `as` casts, integer
arithmetic and `Duration`/`Instant` subtraction are not machine-scanned. They were
reviewed by hand for Phase 48: all non-test slices are at `find`/`char_indices`
boundaries or guarded by a `len()` check (`sanitize_credentials`, `strip_iso_nanos`,
`parse_gpu_stats`, the `parts[…]` parsers); the only external-arithmetic panic found
was the epoch `+ 1` fixed below.

## Rust allow-list

| Location | Pattern | Class | Reason (why the swallow cannot lie) | Action |
|---|---|---|---|---|
| `src-tauri/src/lib.rs:46` | `.unwrap_or(manifest)` | EXPECTED | a missing parent falls back to the manifest dir; never a false path claim | keep |
| `src-tauri/src/lib.rs:60` | `.unwrap_or(manifest)` | EXPECTED | a missing parent falls back to the manifest dir; never a false path claim | keep |
| `src-tauri/src/lib.rs:146` | `let _ = c.kill();` | EXPECTED | best-effort cleanup: the child may already have exited, so only the primary result matters | keep |
| `src-tauri/src/lib.rs:147` | `let _ = c.wait();` | EXPECTED | best-effort cleanup: the child may already have exited, so only the primary result matters | keep |
| `src-tauri/src/lib.rs:153` | `let _ = c.kill();` | EXPECTED | best-effort cleanup: the child may already have exited, so only the primary result matters | keep |
| `src-tauri/src/lib.rs:154` | `let _ = c.wait();` | EXPECTED | best-effort cleanup: the child may already have exited, so only the primary result matters | keep |
| `src-tauri/src/lib.rs:168` | `torch: sc.torch.lock().map(\|g\| g.is_some()).unwrap_or(false),` | EXPECTED | a poisoned lock reads as not-managed; the holder never panics on these paths | keep |
| `src-tauri/src/lib.rs:169` | `llm: sc.llm.lock().map(\|g\| g.is_some()).unwrap_or(false),` | EXPECTED | a poisoned lock reads as not-managed; the holder never panics on these paths | keep |
| `src-tauri/src/lib.rs:215` | `let _ = tx.send(result);` | EXPECTED | best-effort UI/channel send; a dropped receiver has nothing left to mislead | keep |
| `src-tauri/src/lib.rs:240` | `.ok()` | EXPECTED | an optional/fallible read yields None, the documented unknown rather than a false value | keep |
| `src-tauri/src/lib.rs:338` | `.unwrap_or(false);` | EXPECTED | the absence reads as not-true (the conservative direction) | keep |
| `src-tauri/src/lib.rs:443` | `let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(\|d\| d.as_secs()).unwrap_or(0);` | EXPECTED | clock-before-epoch fallback; pid and a counter keep generated ids unique | keep |
| `src-tauri/src/lib.rs:599` | `let canonical_root = fs::canonicalize(&root).unwrap_or_else(\|_\| root.clone());` | EXPECTED | root was already validated; the lexical fallback is defensive only | keep |
| `src-tauri/src/lib.rs:665` | `.ok()` | EXPECTED | an optional/fallible read yields None, the documented unknown rather than a false value | keep |
| `src-tauri/src/lib.rs:666` | `.and_then(\|t\| t.duration_since(UNIX_EPOCH).ok())` | EXPECTED | an optional/fallible read yields None, the documented unknown rather than a false value | keep |
| `src-tauri/src/lib.rs:668` | `.unwrap_or_default();` | EXPECTED | documented empty default; the surrounding status carries the real state | keep |
| `src-tauri/src/lib.rs:781` | `let canonical_dsdir = fs::canonicalize(&dsdir).unwrap_or_else(\|_\| dsdir.clone());` | EXPECTED | datasets dir was already validated by resolve(); the fallback is defensive only | keep |
| `src-tauri/src/lib.rs:799` | `let size = if meta.is_file() { meta.len() } else { dir_size(&entry_canonical).unwrap_or(0) };` | EXPECTED | display-only size; a failed stat shows 0 bytes, not a false listing | keep |
| `src-tauri/src/lib.rs:820` | `let read = fs::read_dir(p).ok()?;` | EXPECTED | a failed read yields None up the Option chain, the documented skip state | keep |
| `src-tauri/src/lib.rs:822` | `let meta = entry.metadata().ok()?;` | EXPECTED | a failed read yields None up the Option chain, the documented skip state | keep |
| `src-tauri/src/lib.rs:826` | `total += dir_size(&entry.path()).unwrap_or(0);` | EXPECTED | display-only size; a failed stat shows 0 bytes, not a false listing | keep |
| `src-tauri/src/lib.rs:938` | `let _ = remote_sidecar::stop_remote_sidecar(window.app_handle().clone());` | EXPECTED | documented fallback; the primary outcome is reported separately | keep |
| `src-tauri/src/lib.rs:1011` | `.expect("error while running tauri application");` | EXPECTED | startup failure of the whole app has no caller to return to (Phase 48) | keep |
| `src-tauri/src/pty.rs:34` | `.unwrap_or(0);` | EXPECTED | clock-before-epoch fallback; pid and a counter keep generated ids unique | keep |
| `src-tauri/src/pty.rs:62` | `rows: args.rows.unwrap_or(30),` | EXPECTED | UI socket-size fallback; the terminal resizes again on first layout | keep |
| `src-tauri/src/pty.rs:63` | `cols: args.cols.unwrap_or(100),` | EXPECTED | UI socket-size fallback; the terminal resizes again on first layout | keep |
| `src-tauri/src/pty.rs:109` | `let _ = app_handle.emit(&exit_event, ());` | EXPECTED | best-effort UI/channel send; a dropped receiver has nothing left to mislead | keep |
| `src-tauri/src/pty.rs:114` | `.and_then(\|s\| s.sessions.lock().ok().and_then(\|mut m\| m.remove(&id_for_thread)))` | EXPECTED | a poisoned lock reads as the explicit unknown state, never a false value | keep |
| `src-tauri/src/pty.rs:123` | `let _ = app_handle.emit(&data_event, s);` | EXPECTED | best-effort UI/channel send; a dropped receiver has nothing left to mislead | keep |
| `src-tauri/src/pty.rs:126` | `let _ = app_handle.emit(&exit_event, format!("read error: {e}"));` | EXPECTED | best-effort UI/channel send; a dropped receiver has nothing left to mislead | keep |
| `src-tauri/src/pty.rs:140` | `let shell = env::var("SHELL").unwrap_or_else(\|_\| "bash".into());` | EXPECTED | documented executable fallback; a missing binary surfaces as a spawn error | keep |
| `src-tauri/src/pty.rs:244` | `let _ = child.kill();` | EXPECTED | best-effort cleanup: the child may already have exited, so only the primary result matters | keep |
| `src-tauri/src/pty.rs:254` | `let _ = child.kill();` | EXPECTED | best-effort cleanup: the child may already have exited, so only the primary result matters | keep |
| `src-tauri/src/remote_sidecar.rs:113` | `let _ = app.emit("remote-sidecar:status", status);` | EXPECTED | best-effort UI/channel send; a dropped receiver has nothing left to mislead | keep |
| `src-tauri/src/remote_sidecar.rs:130` | `std::thread::spawn(move \|\| { let _ = s.write_all(&owned); });` | EXPECTED | best-effort side effect; the primary outcome is reported separately | keep |
| `src-tauri/src/remote_sidecar.rs:135` | `let code = out.status.code().map(\|c\| c.to_string()).unwrap_or_else(\|\| "?".into());` | EXPECTED | documented fallback; the primary outcome is reported separately | keep |
| `src-tauri/src/remote_sidecar.rs:401` | `let _ = tokens.clear_remote();` | EXPECTED | best-effort token clear; the tunnel is already gone regardless | keep |
| `src-tauri/src/remote_sidecar.rs:428` | `let force = force.unwrap_or(false);` | EXPECTED | the absence reads as not-true (the conservative direction) | keep |
| `src-tauri/src/remote_sidecar.rs:450` | `let _ = tokens.clear_remote();` | EXPECTED | best-effort token clear; the tunnel is already gone regardless | keep |
| `src-tauri/src/remote_sidecar.rs:553` | `let _ = child.kill();` | EXPECTED | best-effort cleanup: the child may already have exited, so only the primary result matters | keep |
| `src-tauri/src/remote_sidecar.rs:554` | `let _ = child.wait();` | EXPECTED | best-effort cleanup: the child may already have exited, so only the primary result matters | keep |
| `src-tauri/src/remote_sidecar.rs:558` | `let _ = child.kill();` | EXPECTED | best-effort cleanup: the child may already have exited, so only the primary result matters | keep |
| `src-tauri/src/remote_sidecar.rs:559` | `let _ = child.wait();` | EXPECTED | best-effort cleanup: the child may already have exited, so only the primary result matters | keep |
| `src-tauri/src/remote_sidecar.rs:584` | `let _ = tx.send(Ok(()));` | EXPECTED | best-effort UI/channel send; a dropped receiver has nothing left to mislead | keep |
| `src-tauri/src/remote_sidecar.rs:590` | `let _ = tx.send(Err("remote sidecar exited before announcing readiness".to_string()));` | EXPECTED | best-effort UI/channel send; a dropped receiver has nothing left to mislead | keep |
| `src-tauri/src/remote_sidecar.rs:593` | `let _ = app_for_thread.emit("remote-sidecar:status", RemoteSidecarStatus::Stopped);` | EXPECTED | best-effort UI/channel send; a dropped receiver has nothing left to mislead | keep |
| `src-tauri/src/remote_sidecar.rs:606` | `let _ = tx_err.send(Err(format!(` | EXPECTED | best-effort UI/channel send; a dropped receiver has nothing left to mislead | keep |
| `src-tauri/src/remote_sidecar.rs:620` | `let _ = child.kill();` | EXPECTED | best-effort cleanup: the child may already have exited, so only the primary result matters | keep |
| `src-tauri/src/remote_sidecar.rs:621` | `let _ = child.wait();` | EXPECTED | best-effort cleanup: the child may already have exited, so only the primary result matters | keep |
| `src-tauri/src/remote_sidecar.rs:640` | `let _ = child.kill();` | EXPECTED | best-effort cleanup: the child may already have exited, so only the primary result matters | keep |
| `src-tauri/src/remote_sidecar.rs:641` | `let _ = child.wait();` | EXPECTED | best-effort cleanup: the child may already have exited, so only the primary result matters | keep |
| `src-tauri/src/remote_sidecar.rs:666` | `let _ = child.kill();` | EXPECTED | best-effort cleanup: the child may already have exited, so only the primary result matters | keep |
| `src-tauri/src/remote_sidecar.rs:667` | `let _ = child.wait();` | EXPECTED | best-effort cleanup: the child may already have exited, so only the primary result matters | keep |
| `src-tauri/src/remote_sidecar.rs:707` | `let _ = Command::new("fuser")` | EXPECTED | documented fallback; the primary outcome is reported separately | keep |
| `src-tauri/src/remote_sidecar.rs:719` | `let _ = Command::new("pkill")` | EXPECTED | documented fallback; the primary outcome is reported separately | keep |
| `src-tauri/src/remote_sidecar.rs:731` | `let _ = rs.child.kill();` | EXPECTED | best-effort cleanup: the child may already have exited, so only the primary result matters | keep |
| `src-tauri/src/remote_sidecar.rs:732` | `let _ = rs.child.wait();` | EXPECTED | best-effort cleanup: the child may already have exited, so only the primary result matters | keep |
| `src-tauri/src/remote_sidecar.rs:737` | `let _ = tokens.clear_remote();` | EXPECTED | best-effort token clear; the tunnel is already gone regardless | keep |
| `src-tauri/src/scope_file.rs:226` | `.unwrap_or(false)` | EXPECTED | the absence reads as not-true (the conservative direction) | keep |
| `src-tauri/src/scope_file.rs:298` | `.unwrap_or(0);` | EXPECTED | clock-before-epoch fallback; pid and a counter keep generated ids unique | keep |
| `src-tauri/src/scope_file.rs:335` | `let _ = fs::remove_file(&tmp_path);` | EXPECTED | best-effort side effect; the primary outcome is reported separately | keep |
| `src-tauri/src/scope_file.rs:339` | `let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));` | EXPECTED | best-effort side effect; the primary outcome is reported separately | keep |
| `src-tauri/src/scope_file.rs:345` | `let _ = d.sync_all();` | EXPECTED | best-effort side effect; the primary outcome is reported separately | keep |
| `src-tauri/src/scope_file.rs:402` | `let scope_path = scope_file_path_impl().ok();` | EXPECTED | an optional/fallible read yields None, the documented unknown rather than a false value | keep |
| `src-tauri/src/scope_file.rs:454` | `let tail: PathBuf = p.strip_prefix(&cur).unwrap_or(p).into();` | EXPECTED | split/find always yields a value here; the fallback cannot invent data | keep |
| `src-tauri/src/ssh.rs:200` | `let _ = std::fs::create_dir_all(&dir);` | EXPECTED | best-effort side effect; the primary outcome is reported separately | keep |
| `src-tauri/src/ssh.rs:258` | `.unwrap_or(tail.len());` | EXPECTED | no match uses the full length; this cannot truncate or invent data | keep |
| `src-tauri/src/ssh.rs:299` | `let code = exit.map(\|v\| v.to_string()).unwrap_or_else(\|\| "signal".into());` | EXPECTED | documented fallback; the primary outcome is reported separately | keep |
| `src-tauri/src/ssh.rs:322` | `let _ = stdin.write_all(&owned);` | EXPECTED | best-effort side effect; the primary outcome is reported separately | keep |
| `src-tauri/src/ssh.rs:524` | `let g = state.current.lock().ok()?;` | EXPECTED | a poisoned lock reads as the explicit unknown state, never a false value | keep |
| `src-tauri/src/ssh.rs:555` | `let kind = it.next().unwrap_or("");` | EXPECTED | an absent optional field renders as empty, never an invented string | keep |
| `src-tauri/src/ssh.rs:556` | `let rel = it.next().unwrap_or("").to_string();` | EXPECTED | an absent optional field renders as empty, never an invented string | keep |
| `src-tauri/src/ssh.rs:561` | `let name = rel.rsplit('/').next().unwrap_or(&rel).to_string();` | EXPECTED | split/find always yields a value here; the fallback cannot invent data | keep |
| `src-tauri/src/ssh.rs:596` | `.unwrap_or_else(\|\| root.clone());` | EXPECTED | documented fallback; the primary outcome is reported separately | keep |
| `src-tauri/src/ssh.rs:645` | `.unwrap_or_else(\|\| root.clone());` | EXPECTED | documented fallback; the primary outcome is reported separately | keep |
| `src-tauri/src/ssh.rs:675` | `.unwrap_or(tail.len());` | EXPECTED | no match uses the full length; this cannot truncate or invent data | keep |
| `src-tauri/src/ssh.rs:706` | `let size = parts[1].parse::<u64>().unwrap_or(0);` | EXPECTED | an unparseable external numeric value shows the documented default, not a fabricated one | keep |
| `src-tauri/src/ssh.rs:867` | `let size = parts[2].parse::<u64>().unwrap_or(0);` | EXPECTED | an unparseable external numeric value shows the documented default, not a fabricated one | keep |
| `src-tauri/src/ssh.rs:999` | `let cfg: Value = serde_json::from_str(&run_json).unwrap_or(Value::Null);` | EXPECTED | a corrupt/absent config yields null fields; the run status is reported separately | keep |
| `src-tauri/src/ssh.rs:1004` | `.unwrap_or("local");` | EXPECTED | documented default backend when the frozen config omits it | keep |
| `src-tauri/src/ssh.rs:1055` | `let s = \|k: &str\| slurm.and_then(\|v\| v.get(k)).and_then(\|x\| x.as_str()).unwrap_or("").trim().to_string();` | EXPECTED | an absent optional sbatch field is omitted rather than invented | keep |
| `src-tauri/src/ssh.rs:1077` | `let cpus = n("cpus_per_task").unwrap_or(8);` | EXPECTED | documented sbatch default when the config omits the field | keep |
| `src-tauri/src/ssh.rs:1283` | `pid = p.parse::<i32>().ok();` | EXPECTED | an optional/fallible read yields None, the documented unknown rather than a false value | keep |
| `src-tauri/src/ssh.rs:1388` | `let _ = &root;` | EXPECTED | documented fallback; the primary outcome is reported separately | keep |
| `src-tauri/src/training.rs:127` | `let cfg: Value = serde_json::from_str(run_json).unwrap_or(Value::Null);` | EXPECTED | a corrupt/absent config yields null fields; the run status is reported separately | keep |
| `src-tauri/src/training.rs:128` | `let metrics: Value = serde_json::from_str(metrics_json).unwrap_or(Value::Null);` | EXPECTED | a corrupt/absent config yields null fields; the run status is reported separately | keep |
| `src-tauri/src/training.rs:129` | `let s = \|v: &Value, k: &str\| v.get(k).and_then(\|x\| x.as_str()).unwrap_or("").to_string();` | EXPECTED | an absent optional field renders as empty, never an invented string | keep |
| `src-tauri/src/training.rs:138` | `.unwrap_or(0) as u32;` | EXPECTED | a truncated/absent epoch field reads as 0, the documented unknown | keep |
| `src-tauri/src/training.rs:147` | `.unwrap_or("")` | EXPECTED | an absent optional field renders as empty, never an invented string | keep |
| `src-tauri/src/training.rs:159` | `events_max_epoch.map(\|e\| e.saturating_add(1)).unwrap_or(0)` | EXPECTED | documented fallback; the primary outcome is reported separately | keep |
| `src-tauri/src/training.rs:172` | `eval_only: cfg.get("eval_only").and_then(\|x\| x.as_bool()).unwrap_or(false),` | EXPECTED | the absence reads as not-true (the conservative direction) | keep |
| `src-tauri/src/training.rs:231` | `let num = \|s: &str\| s.trim().parse::<f64>().unwrap_or(0.0);` | EXPECTED | an unparseable external numeric value shows the documented default, not a fabricated one | keep |
| `src-tauri/src/training.rs:239` | `index: cols[0].parse::<u32>().unwrap_or(0),` | EXPECTED | an unparseable external numeric value shows the documented default, not a fabricated one | keep |
| `src-tauri/src/training.rs:271` | `.ok()` | EXPECTED | an optional/fallible read yields None, the documented unknown rather than a false value | keep |
| `src-tauri/src/training.rs:272` | `.and_then(\|s\| s.trim().parse::<i32>().ok())` | EXPECTED | an optional/fallible read yields None, the documented unknown rather than a false value | keep |
| `src-tauri/src/training.rs:284` | `.unwrap_or(false)` | EXPECTED | the absence reads as not-true (the conservative direction) | keep |
| `src-tauri/src/training.rs:290` | `.unwrap_or_else(\|_\| "unknown".into())` | EXPECTED | documented fallback; the primary outcome is reported separately | keep |
| `src-tauri/src/training.rs:298` | `let run_json = safe_read_text(dir, "run.json").unwrap_or_default();` | EXPECTED | an absent/unreadable optional run file is the documented empty state | keep |
| `src-tauri/src/training.rs:299` | `let metrics_json = safe_read_text(dir, "metrics.json").unwrap_or_default();` | EXPECTED | an absent/unreadable optional run file is the documented empty state | keep |
| `src-tauri/src/training.rs:300` | `let status_raw = safe_read_text(dir, "status").unwrap_or_default();` | EXPECTED | an absent/unreadable optional run file is the documented empty state | keep |
| `src-tauri/src/training.rs:305` | `.unwrap_or_default();` | EXPECTED | documented empty default; the surrounding status carries the real state | keep |
| `src-tauri/src/training.rs:306` | `let alive = pid_of(dir).map(is_alive).unwrap_or(false);` | EXPECTED | process-liveness probe defaults to not-alive (the safe direction) | keep |
| `src-tauri/src/training.rs:324` | `if real.parent().map(\|p\| p != dir).unwrap_or(true) {` | EXPECTED | documented fallback; the primary outcome is reported separately | keep |
| `src-tauri/src/training.rs:327` | `fs::read_to_string(&real).ok()` | EXPECTED | an unsafe/absent run file yields None, the documented skip state | keep |
| `src-tauri/src/training.rs:344` | `Ok(real) => real.parent().map(\|p\| p == dir).unwrap_or(false),` | EXPECTED | the absence reads as not-true (the conservative direction) | keep |
| `src-tauri/src/training.rs:457` | `let a = sacct_state.trim().split_whitespace().next().unwrap_or("").to_ascii_uppercase();` | EXPECTED | an absent optional field renders as empty, never an invented string | keep |
| `src-tauri/src/training.rs:476` | `let alive = pid.map(is_alive).unwrap_or(false);` | EXPECTED | process-liveness probe defaults to not-alive (the safe direction) | keep |
| `src-tauri/src/training.rs:543` | `let python = std::env::var("SPINOML_PYTHON").unwrap_or_else(\|_\| "python".into());` | EXPECTED | documented executable fallback; a missing binary surfaces as a spawn error | keep |
| `src-tauri/src/training.rs:581` | `let _ = fs::write(dir.join("status"), "cancelled\n");` | EXPECTED | cooperative cancel write; a forced SIGTERM follows as the fallback | keep |
| `src-tauri/src/training.rs:585` | `let _ = Command::new("kill")` | EXPECTED | documented fallback; the primary outcome is reported separately | keep |
| `src-tauri/src/training.rs:589` | `let _ = Command::new("kill").arg("-TERM").arg(pid.to_string()).status();` | EXPECTED | documented fallback; the primary outcome is reported separately | keep |
| `src-tauri/src/training.rs:644` | `if pid_of(&dir).map(is_alive).unwrap_or(false) {` | EXPECTED | process-liveness probe defaults to not-alive (the safe direction) | keep |

## Rust: fixed hidden failures (no longer panicking / swallowing falsely)

| Location (before) | Was | Now |
|---|---|---|
| `src-tauri/src/training.rs` `RunSummary::from_parts` epochs fallback | `epoch + 1` on an epoch read from an externally-written `events.jsonl`: overflow-panics in debug and wraps to `0` in release for `epoch = u32::MAX` (a wrong/invented epoch count). | `epoch.saturating_add(1)`; regression `training::tests::huge_event_epoch_saturates_instead_of_overflowing` (fails under the old `+ 1`). |
| `src-tauri/src/training.rs` `list_training_runs` | `fs::canonicalize` failure of an **existing** `experiments/runs/` returned `Ok(vec![])` — a false "no runs" list (e.g. EACCES on a parent). | propagates a descriptive `Err("read_dir …")` so the UI shows the failure instead of an empty list. `not unit-testable: needs a Tauri State + a permission-denied directory`. |
| `src-tauri/src/remote_sidecar.rs` run watcher | a dead `let _ = &line;` no-op (no error swallowed). | removed; the surrounding `eprintln!` + drain comment stay. |

## Summary — Rust

- Files scanned: **8** (`src-tauri/src/*.rs`).
- Sites detected by the line scan (non-test, after the fixes): **112**.
- EXPECTED (kept, each with an in-code `//` reason and an allow-list row): **112**.
- FIXED hidden failures / panic sources: **2** (`from_parts` epoch overflow,
  `list_training_runs` false-empty).
- Non-test `unwrap()` / `panic!` / `unreachable!` / `todo!` / `unimplemented!`: **0**.
- The single `expect()` is the Tauri builder in `run()`; a startup failure of the
  whole process has no caller to return to, so it is kept and documented.
- Class breakdown: `let _ =` 43, `unwrap_or(` 42, `.ok()` 12, `unwrap_or_else(` 9,
  `unwrap_or_default` 5, `expect(` 1.
