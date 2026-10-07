# TypeScript safety — Phase 49

> Principle: *trust boundaries must narrow from `unknown`; promise rejections
> must be captured explicitly at the call site; non-null assertions on values
> that can really be absent at runtime are forbidden.* The audit is ratcheted:
> every site in `src/` of an explicit `any`, `as any`, `as unknown as X`,
> non-null assertion, `@ts-ignore`/`@ts-nocheck`/`@ts-expect-error`,
> `eslint-disable`, or a floating Promise must appear in the table below AND
> carry a ≥ 15-char prose reason (where the construct allows a comment).

This document is the **allow-list** for `scripts/verify-ts-safety.ts`
(`npm run verify:ts-safety`). Matching is by **file + normalised pattern
text**, so line-number drift is tolerated; a pattern that occurs more than
once in a file must appear that many times; a stale row fails.

## Allow-list — current sites

| Location | Pattern | Class | Reason | Action |
|---|---|---|---|---|
| `src/canvas/invariants.ts:64` | `adj.get(e.source)!` | SAFE | same-loop invariant: `adj` is initialised from the same `edges` array the loop walks; every `e.source` was just `set` into `adj`. | keep |
| `src/canvas/invariants.ts:188` | `adj.get(e.source)!` | SAFE | same-loop invariant as line 64; `adj` and `ids` come from the same `nodes`/`edges` arrays. | keep |
| `src/canvas/layout.ts:61` | `succ.get(e.source)!` | SAFE | topological layout: `succ`/`pred` are initialised for every `n.id` in `nodes` before any edge is walked; `ids.has(e.source)` is the explicit guard one line above. | keep |
| `src/canvas/layout.ts:62` | `pred.get(e.target)!` | SAFE | same-loop invariant as line 61. | keep |
| `src/canvas/layout.ts:67` | `pred.get(n.id)!` | SAFE | `pred.set(n.id, [])` ran for every `n.id` in `nodes` (line 58). | keep |
| `src/canvas/layout.ts:74` | `succ.get(u)!` | SAFE | `u` was just pushed onto `queue`; `queue` is initialised from `nodes` so every `u` has a `succ` entry. | keep |
| `src/canvas/layout.ts:76` | `indeg.get(v)!` | SAFE | `v` is in `succ.get(u)` which is from `edges`, every endpoint of which was checked against `ids.has` (line 60). | keep |
| `src/canvas/layout.ts:84` | `pred.get(n.id)!` | SAFE | same as line 67. | keep |
| `src/canvas/layout.ts:89` | `rank.get(n.id)!` | SAFE | ranks are computed for every node; the `Math.max(...nodes.map(n => rank.get(n.id)!))` is the upper-bound before layer assignment. | keep |
| `src/canvas/layout.ts:91` | `rank.get(n.id)!` | SAFE | same as line 89. | keep |
| `src/canvas/layout.ts:98` | `neighbours.get(id)!` | SAFE | `neighbours` is initialised for every `id` in `nodes` at the top of the sweep. | keep |
| `src/canvas/layout.ts:100` | `orderIndex.get(x)!` | SAFE | `orderIndex` is reindexed for every layer's nodes at the start of each sweep. | keep |
| `src/canvas/scopeStore.ts:51` | `stack.pop()!` | SAFE | guarded by `while (stack.length > depth)` so a pop is only attempted when non-empty. | keep |
| `src/codegen/dataGenerator.ts:51` | `adj.get(e.source)!` | SAFE | graph-traversal invariant: `adj` is built from the same `edges` walk just above. | keep |
| `src/codegen/dataGenerator.ts:59` | `queue.shift()!` | SAFE | guarded by `while (queue.length)` immediately above. | keep |
| `src/codegen/dataGenerator.ts:60` | `byId.get(id)!` | SAFE | `byId` is initialised from `nodes`; `id` is from a node-iteration immediately above. | keep |
| `src/codegen/generator.ts:122` | `ch.codePointAt(0)!` | SAFE | `for (const ch of s)` yields a non-empty string for every iteration, so `codePointAt(0)` is defined. | keep |
| `src/codegen/generator.ts:254` | `queue.shift()!` | SAFE | guarded by `while (queue.length)` immediately above. | keep |
| `src/codegen/generator.ts:299` | `byId.get(id)!` | SAFE | `byId` is built from `nodes` two lines above; `id` is iterated over `nodes`. | keep |
| `src/codegen/generator.ts:309` | `attrName.get(id)!` | SAFE | same-loop invariant: `attrName.set(id, ...)` ran for every `id` in the iteration just above. | keep |
| `src/codegen/generator.ts:312` | `attrName.get(id)!` | SAFE | same as line 309. | keep |
| `src/codegen/generator.ts:330` | `byId.get(id)!` | SAFE | `byId` is initialised from `nodes` at the top of the function; `id` is iterated over `nodes`. | keep |
| `src/codegen/generator.ts:414` | `byId.get(id)!` | SAFE | same as line 330. | keep |
| `src/codegen/generator.ts:514` | `byId.get(id)!` | SAFE | same as line 330. | keep |
| `src/codegen/pyLiteral.ts:23` | `ch.codePointAt(0)!` | SAFE | for-of string iteration always yields a non-empty code-unit sequence. | keep |
| `src/codegen/pyLiteral.ts:45` | `ch.codePointAt(0)!` | SAFE | same as line 23. | keep |
| `src/codegen/trainingGenerator.ts:189` | `stack.pop()!` | SAFE | guarded by `while (stack.length)` immediately above. | keep |
| `src/training/graph/autosave.ts:37` | `o as unknown as TrainingGraphSnapshot` | SAFE | cast is preceded by `Array.isArray(o.nodes) && Array.isArray(o.edges)`; the runtime shape is the only thing `TrainingGraphSnapshot` ever actually reads on the next line. | keep |
| `src/training/parseRunConfig.ts:20` | `raw as unknown as RunConfig` | SAFE | cast is preceded by an `isObj(raw)` + `dataset.relpath` narrow check; the runtime shape is the only thing the next eval-launch dialog reads. | keep |
| `src/visualization/store.ts:113` | `queue.shift()!` | SAFE | guarded by `while (queue.length)` immediately above. | keep |
| `src/workspace/FileExplorer.tsx:244` | `stack.pop()!` | SAFE | guarded by `while (stack.length)` at the top of the recursion. | keep |
| `src/workspace/store.ts:355` | `stack.pop()!` | SAFE | guarded by `while (stack.length)` immediately above. | keep |
| `src/workspace/tauri-fs.ts:16` | `window as unknown as { __TAURI_INTERNALS__?: unknown }` | SAFE | structural test for the Tauri runtime; no runtime data crosses the cast, only the static type widening does. | keep |

## Fixed (no longer in `src/`)

| Before | Was | Now |
|---|---|---|
| `src/datasets/DatasetDetail.tsx` `normalizeManifest(raw: any)` | `any` parameter let a torn/hostile JSON manifest mis-type every field downstream | signature narrowed to `unknown` + a small `isObj`/`typeof` ladder that never throws; the editor still defaults sensibly on junk. |
| `src/training/charts/Evaluation.tsx` `return e as unknown as EvalSummary` | unvalidated cast on a sidecar `eval.summary` event | added `isChatEvent`/`isEvalSummary` type guards; the function returns `null` for unknown `type` strings. |
| `src/workspace/store.ts` `JSON.parse(raw) as Persisted` | silent `as Persisted` after `JSON.parse` of `localStorage` | `normalizePersisted(raw: unknown)` + `normalizeEntry(...)`; corrupt entries drop out instead of polluting the entry map. |
| `src/workspace/store.ts` `{content} as unknown as File` | fake `File` only existed to call `fingerprintFile(content)` | call passes a structurally-valid `File` literal instead. |
| `src/training/EvalRunModal.tsx` `JSON.parse(...) as RunConfig` | unvalidated cast on `run.json` | goes through `parseRunConfig(JSON.parse(...))` (a new module); explicit "run.json ist beschädigt" thrown on a fundamental drift. |
| `src/training/RunDetailModal.tsx` `JSON.parse(runJson) as RunConfig` | same cast, same risk | routed through `parseRunConfig`. |
| `src/training/store.ts` `JSON.parse(...) as RunConfig` | same cast, same risk | routed through `parseRunConfig`. |
| `src/training/graph/autosave.ts` `JSON.parse(text) as TrainingGraphSnapshot` | cast came BEFORE the `Array.isArray` check | order swapped: narrow `unknown` first, cast last. |
| `src/chat/store.ts` `JSON.parse(raw) as ChatMessage[]` | cast before the `Array.isArray` check | order swapped; `unknown` first, `ChatMessage[]` last. |
| `src/canvasdoc/store.ts` `JSON.parse(localStorage.getItem(KEY) || '{}')` | no validation at all | narrow `unknown` + reject non-objects. |
| `src/palette/Palette.tsx` `JSON.parse(raw) as string[]` | cast on a localStorage read | narrow `unknown` + filter to strings. |
| `src/workspace/recentWorkspaces.ts` `JSON.parse(raw) as { root?: string; relpath?: string }` | cast on a localStorage read | narrow `unknown` + structural checks. |
| `src/training/EvalRunModal.tsx` `useTrainingStore((s) => s.evalSourceId)!` | non-null assertion in the render path | new `EvalRunModalInner({ sourceRunId })` wrapper; the outer returns `null` when `sourceRunId` is missing. |
| `src/inspector/Inspector.tsx` `byBranch.get(br)!` after `byBranch.has(br)` | TypeScript couldn't narrow across the has/get pair | captured the result in `const arr = ...; if (arr) arr.push(...)`. |
| `src/inspector/Inspector.tsx` `field.datalist!` | assertion duplicated a check that `listId` already gated | captured `const datalist = field.datalist` and narrowed through `datalist && ...`. |
| `src/datasets/DatasetDetail.tsx` `byBranch.get(branch)!` after `byBranch.has(branch)` | same TypeScript narrowing gap | captured + guarded. |
| `src/training/RunDetailModal.tsx` `predHeads!.map(...)` | could throw when `evalHeads` was truthy but `predHeads` was null | rewritten as `(evalHeads ?? (predHeads ?? []).map(...))`. |
| `src/training/NewRunModal.tsx` `cfg.heads!`, `chosen!`, `SWEEP_FIELDS.find(...)!` | three non-null assertions on values that can really be absent (no `Head` row, no resume run, unknown sweep axis) | narrowed to `cfg.heads && ...` / `chosen && ...` / `field ? field.label : a.key`. |
| `src/training/graph/TrainingGraphBar.tsx` `plan.modelRelpath.split('/').pop()!` | `pop()` returns `undefined` for empty/single-segment paths | captured `const basename = ... ?? plan.modelRelpath`. |
| `src/training/NewRunModal.tsx` `modelRelpath.split('/').pop()!` | same | captured + `?? modelRelpath`. |
| `src/project/store.ts` `(await tauriFs.currentDir())!` (× 2) | `tauriFs.currentDir()` returns `string \| null`; the `!` would have crashed when Tauri was mid-startup | explicit `if (!root) { set({ status: { kind: 'error', error: ... } }); return }` branch. |
| `src/project/Welcome.tsx` `initial!.id` (in the edit-connection onClick) | the disabled-button check used `!!initial` but the closure lost the narrowing | explicit `if (!initial) return` guard. |
| `src/layers/registry.ts` `sg!.nodes!.length` (Subgraph summary) | two `!` stacked on an `Array.isArray(sg?.nodes)` check | collapsed to `sg && Array.isArray(sg.nodes) ? sg.nodes.length : 0`. |
| `src/codegen/trainingCodegen.ts` `callbacks.find((c) => c.kind === 'EarlyStopping')!` | could throw if `hasEarlyStop` was set but no such callback existed | explicit `if (!es) throw new Error('codegen: hasEarlyStop true but no EarlyStopping callback')`. |
| `src/canvasdoc/CanvasFileGate.tsx` `adapter.save!()` | assertion duplicated an `adapter.save &&` check | captured `const save = adapter.save; if (save) void fireAndForget('canvas.save', save())`. |
| `src/main.tsx` `document.getElementById('root')!` | bootstrap crash, never reached the React error boundary | explicit `if (!rootEl) throw new Error('SpinoML bootstrap failed: #root element missing from index.html')`. |
| `src/training/EvalRunModal.tsx` `manifestMode = !!srcManifest && !externalIsManifest` + `useDatasetsStore.getState().inspects[cfg.dataset.relpath]?.data ?? null` access without a fallback | the `cfg.dataset.relpath` access was unguarded | the new `parseRunConfig` requires `dataset.relpath` to be a string; a malformed manifest fails fast. |
| `src/inspector/Inspector.tsx` `void inspectAction(datasetRel)` × 3 | fire-and-forget without a rejection handler | routed through `fireAndForget('inspectAction', inspectAction(...))`. |
| `src/inspector/Inspector.tsx` `void refresh()` / `void inspect(value)` | fire-and-forget | routed through `fireAndForget`. |
| `src/datasets/DatasetDetail.tsx` `void loadStats(relpath)` / `void loadHistory()` / `void inspectAction(...)` × 2 | fire-and-forget | routed through `fireAndForget`. |
| `src/datasets/DatasetExplorer.tsx` `void refresh()` × 2 | fire-and-forget | routed through `fireAndForget`. |
| `src/datasets/store.ts` `void get().inspect(relpath)` | fire-and-forget | routed through `fireAndForget`. |
| `src/data/graph/DataInspector.tsx` `void refresh()` | fire-and-forget | routed through `fireAndForget`. |
| `src/training/graph/TrainingInspector.tsx` `void refresh()` / `void inspect(...)` × 4 | fire-and-forget | routed through `fireAndForget`. |
| `src/training/ExperimentsExplorer.tsx` `void refresh()` × 2 | fire-and-forget | routed through `fireAndForget`. |
| `src/training/RunDetailModal.tsx` `void reload()` / `void tick()` | fire-and-forget | routed through `fireAndForget`. |
| `src/training/store.ts` `void get().refresh()` | fire-and-forget | routed through `fireAndForget`. |
| `src/visualization/store.ts` `get().run()` as an expression statement | fire-and-forget | routed through `void fireAndForget('viz.setWeightsRun.run', get().run())`. |
| `src/workspace/FileExplorer.tsx` `void useWorkspaceStore.getState().{openDirectory,openFile,move,importFromText,rename,remove}(...)` × many | fire-and-forget | routed through `fireAndForget`. |
| `src/workspace/FileExplorer.tsx` `importFromText(parentId, file.name, await file.text())` inside `try { … } catch { alert }` | sync `try/catch` does NOT catch a Promise rejection | made the call `await`-ed so the `catch` now actually fires. |
| `src/workspace/FileExplorer.tsx` `importFromText(ROOT_ID, file.name, text)` inside `try { … } catch { alert }` | same | added `.catch((err) => alert(...))`. |
| `src/canvasdoc/CanvasFileGate.tsx` `void save()` inside an arrow | fire-and-forget | routed through `fireAndForget`. |
| `src/trust/trustStore.ts` `void fireAndForget('trustStore.load', load())` | the detector mistook the bare `load()` for a Promise because `load` is in the heuristic name list; in fact `load()` is sync — the helper was unnecessary | reverted to `load()`; the trust store keeps its plain module-init shape. |
| `src/trust/ApproveCodeDialog.tsx` / `src/Toolbar.tsx` `trust.approve(...)` | the detector's heuristic thought `trust.approve` was a Promise; in fact the trust store is sync (all methods return `void`/boolean) | detector's `isPromiseReceiver` learned that `trust.*` is sync; no source change needed. |
| `src/visualization/store.ts` `get().run()` | bare Promise call | wrapped as above. |

## New infrastructure

- `src/errors/report.ts` — `reportError(context, err)`, `fireAndForget(context, promise)`, `errMessage(e)`, `isFiniteNumber(n)`, plus the tiny `useDiagnostics` zustand store (last 20 entries, dedupes with a counter).
- `src/errors/globalHandlers.ts` — installs `window.onerror` + `unhandledrejection` listeners, imported exactly once from `src/main.tsx` (last import line). Both feed `useDiagnostics`.
- `src/errors/DiagnosticsBanner.tsx` — German rose banner mounted once in `src/App.tsx`; hidden when empty; per-entry close button + "Alle schließen" bulk action; collapses duplicate (context, message) pairs into a `×N` counter.
- `src/training/parseRunConfig.ts` — trust boundary for `run.json`: takes `unknown`, throws a German "run.json ist beschädigt: …" on fundamental drift, returns the narrowed `RunConfig` for downstream accessors that already gate field-by-field.
