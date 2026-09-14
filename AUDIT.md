# SpinoML — Software-Audit & Release-Readiness-Review

**Stand:** 2026-09-07 · **Geprüft:** `~/projects/mlforge` (Git-Remote `christophmanitz/spinoml.git`)
**Umfang:** Frontend (React 19/TS, ~21,5 k LOC), Rust-Teil (`src-tauri`, ~3,7 k LOC),
Python-Sidecar (`sidecar-torch`, ~4 k LOC), Node-Sidecar (`sidecar-llm`, ~2,2 k LOC),
Verify-Skripte, Dokumentation, Governance & Packaging.

**Verifikationen ausgeführt:** `npm run build` ✅ grün (tsc + vite) · `npm run lint` ❌ **53 Probleme**
(49 errors / 4 warnings — komplette Liste in § A2) · `cargo check` ⚠️ nicht ausführbar (cargo nicht im PATH, unbelegt).

> **Legende:** Alle Belege sind als `Datei:Zeile` angegeben und beziehen sich auf diesen Stand.
> „→ Fix" verweist auf die Maßnahme in § 4.

---

## Gesamturteil

Funktional und technisch ein **starkes, ambitioniertes Projekt** (~32 k LOC, 82 sauber
dokumentierte Commits). Der Code ist im Kern erheblich besser als der Durchschnitt:
typsichere Zustand-Architektur, konsequent umgesetzter Dispatch-Layer (local ↔ SSH),
saubere Rust-Seite (`resolve()`-Pfadgate `lib.rs:144-154`, SSH-Quoting `ssh.rs:110-136`,
PDEATHSIG `lib.rs:66-77`), durchdachte Guards gegen Store-Endlosschleifen.

Es ist aber **noch kein veröffentlichbares Produkt**:

- **kritische Sicherheitshebel** (Code-Ausführung über den Loopback) — Blocker,
- **keine Tests, kein CI**, Lint nicht grün,
- **Governance/Metadaten nicht release-fähig** (Versionen überall verschieden,
  `Cargo.toml`-Scaffold-Placeholder, README behauptet Falsches),
- Docs und Realität driften teilweise auseinander.

---

## 1. Kritisch — echte Sicherheitslücken (Release-Blocker)

### 1.1 CSRF / CORS `*` auf beiden Sidecars — beliebige Website führt Code aus

Beide lokalen Server setzen **`Access-Control-Allow-Origin: *`** und haben **keinen
Origin-/Host-/Token-Check**. Jede Webseite, die im Browser des Nutzers offen ist, kann per
`fetch()` gegen `127.0.0.1` schreiben **und Antworten lesen** (DNS-Rebinding umgeht
zusätzlich Host-Checks). Aktuell sind damit alle untenstehenden Exec-/Lese-endpunkte von
außen erreichbar.

| Beleg | Stelle |
|-------|--------|
| CORS `*` + Methoden/Header im LLM-Sidecar | `sidecar-llm/main.mjs:1571-1573` (`Access-Control-Allow-Origin: *`) |
| CORS `*` im Torch-Sidecar | `sidecar-torch/main.py:793-795` |
| Falscher Anti-CSRF-Kommentar im Docstring | `sidecar-torch/main.py:15-17` („CORS is permissive because … bind address is 127.0.0.1 so no external host can reach it" — sachlich falsch) |
| Beide Server binden korrekt nur Loopback | `sidecar-torch/main.py:958` · `sidecar-llm/main.mjs:2156` (bindet `127.0.0.1` — das schützt aber nicht vor dem Browser-CSRF-Vektor) |

### 1.2 LLM-Sidecar: Bestätigungs-Endpunkt und Auto-Approve ohne Auth

| Problem | Stelle |
|---------|--------|
| `POST /respond` löst `askUser`-Bestätigungen ohne jede Auth — `askId`-Format sequenziell erratbar; ein lokaler Prozess **oder eine Website** (via CORS `*`) kann gefälschte Antworten liefern und damit Shell-Ausführung freigeben | `sidecar-llm/main.mjs:2136-2143` (Route), `:559` (`pendingAsks`-Map), `:568-581` (Vergabe), `:580-586` (`resolveAsk`) |
| `autoMode: true` kommt ungeprüft aus dem JSON-Body und schaltet die Shell-Genehmigung ab | `sidecar-llm/main.mjs:1906` (`const autoApproveShell = autoMode === true`) · Verwendung `:1346` (`autoApproveShell && m === 'shell'`) |

### 1.3 Torch-Sidecar: beliebige Code-/Befehl-Ausführung

| Problem | Stelle |
|---------|--------|
| `exec()` von LLM-generiertem Model-Code in `/infer` | `sidecar-torch/main.py:125, 285, 482` |
| `pip install` mit beliebigen Package-Specs | `sidecar-torch/main.py:721-737` (`/deps/install`, `subprocess.run(… pip install …, timeout=1800)` Z. 731) |
| `run_workspace_script`: schreibt beliebige `code`-Datei an beliebiges `root`/`relpath` und führt sie mit `bash`/`python`/`sbatch` aus | `sidecar-torch/main.py:740-788` (Ausführung 768 sbatch/120 s, 779 shell/600 s) |
| LLM-Sidecar: `args` von `run_script` werden **nicht shell-gequotet**, direkt in den Kommandostring konkateniert — Shell-Injection-Kanal | `sidecar-llm/main.mjs:518` (`const argStr = args ? \` ${args}\` : ''`), `:526` (sbatch), `:534` (exec) |
| SSRF in `downloadToDatasets`: nur `^https?://` geprüft — `http://169.254.169.254/…` (Cloud-Metadata) u. Ä. erlaubt | `sidecar-llm/main.mjs:304` (Regex), `:316` (`curl -fsSL --max-time 300 …`) |

### 1.4 `torch.load(weights_only=False)` — Pickle-RCE durch fremde `.pt`-Dateien

Ein untrusted `.pt`-Checkpoint/-Dataset (HuggingFace-Download, Kollegen-Datei) kann beliebigen
Python-Code ausführen — das läuft noch dazu auf den unvalidierten HTTP-Pfaden aus § 1.5.

| Datei | Zeilen |
|-------|--------|
| `sidecar-torch/main.py` | 569 (Checkpoint `/activations`) |
| `sidecar-torch/dataset_handlers.py` | 382, 413, 427 (tensor/graph), 765 (Manifest-Graphcache), 1021 (Manifest-Branch), 1361 (Stats), 1551 (Sample) |
| `sidecar-torch/training_template.py` | 249, 375, 503, 969, 1165 |

(`np.load(…, allow_pickle=False)` ist an den korrekten Stellen: `dataset_handlers.py:433, 436, 1364`.)

### 1.5 Kein Pfad-Scoping im Sidecar — beliebige Dateien lesen/schreiben/ausführen

Die Rust-Seite hat einen korrekten Pfad-Gate (`resolve()` in `lib.rs:144-154`, genutzt z. B.
`lib.rs:256, 267, 277, 292, 679, 690-691`). **Der HTTP-Sidecar hat kein Äquivalent.**

| Problem | Stelle |
|---------|--------|
| `/dataset/inspect`, `/dataset/stats` akzeptieren beliebige `abspath` | `sidecar-torch/main.py:822-834` (Routing), Handler `dataset_handlers.py:193-748` |
| CSV-`head`-Leak (erste 10 Zeilen + Kolumnennamen) | `dataset_handlers.py:209` (`_inspect_tabular`) |
| `prep_card.json` wird komplett zurückgegeben | `dataset_handlers.py:247` |
| Base64-Thumbnails beliebiger Bildordner | `dataset_handlers.py:255-300` (`_inspect_image_folder`, `Image.open` Z. 276) |
| Tensor-Stats beliebiger `.pt/.npy/.npz` (min/max/mean/Histogramm) | `dataset_handlers.py:422-466` (`_inspect_tensor`) |
| `/activations` lädt beliebigen `checkpoint`-Pfad | `sidecar-torch/main.py:884-885` (→ `569`) |
| `/run_script`: `root` komplett aus dem Payload, nur `..`-Check auf `relpath` | `sidecar-torch/main.py:748-756` |

### 1.6 Weitere Folgeprobleme desselben Musters

| Problem | Stelle |
|---------|--------|
| **Unbounded Input-Shapes** → `torch.zeros([1, 10⁹, 10⁹])` killt per OOM den **gesamten** Sidecar-Prozess | `main.py:182-193` (infer), `:546-553` (activations) |
| Unbounded Batch-Replikation/Protein-Tensor | `dataset_handlers.py:1502-1503` (`_sample_tabular`), `:1727-1731` (`_sample_protein`) |
| Thread-per-Connection ohne Limit, `daemon_threads` | `main.py:958` (`ThreadingHTTPServer`) |
| Blockierende Langläufer ohne Gesamt-Limit: `/deps/check` 240 s, `/deps/install` **1800 s**, `run_script` 600 s/120 s | `main.py:696`, `:731`, `:779`, `:768` |
| Kein Socket-Timeout, kein Request-Cap → Slowloris | `main.py:958` (Klassen-Definition) |
| Body ohne Größenlimit komplett in den RAM geladen | `main.py:811` (`Content-Length` außerhalb try) · `sidecar-llm/main.mjs:1894-1895` (`for await (const chunk of req) body += chunk`) |
| **Tracebacks in HTTP-Antworten** (mit absoluten Pfaden) bei gleichzeitig CORS `*` → Infoleak/exfiltrierbar | `main.py:135, 151, 205, 290, 303, 327, 485, 493, 556, 623` (alle `"trace": traceback.format_exc(…)`), generischer 500-Branch `:907-912, 935-941` |
| JSON-Serialisierung ohne `default=` (`Path`/NaN → 500) | `main.py:944-951` (`_json`) |
| **Ganz-Datei-Loads** großer CSVs/Parquet in den RAM | `dataset_handlers.py:193-223` (`_inspect_tabular`), `:1261-1316` (`_stats_tabular`) |
| `flatten().float()` kopiert den kompletten Tensor | `dataset_handlers.py:1371` (`_stats_tensor`) |
| **Netzwerk-Downloads ohne Timeout** (HuggingFace/PyG) blockieren Request-Thread | `dataset_handlers.py:582` (`load_dataset_builder`), `:603-642` (`_load_pyg`) |
| Parallele Smoke-Requests schreiben gleichzeitig `.graphcache`/`.espf` (Race) | `dataset_handlers.py:758-776`, `:942-965` |
| `deps_install` kann `torch` zur Laufzeit upgraden → laufender Server crasht am nächsten Forward | `main.py:731` |

**Empfohlene Gegenmaßnahmen (ein Paket, ~1 Tag):**
1. Per-Launch-Zufallstoken (Rust erzeugt, Frontend übergibt, beide Sidecars prüfen) + Origin-/Host-Allowlist statt `*`.
2. `weights_only=True` + `torch.serialization.add_safe_globals()` für PyG-Typen.
3. Workspace-Root-Whitelisting im Sidecar (Spiegel des Rust-`resolve()`-Gates).
4. Element-Cap (z. B. 10⁷ pro Tensor) + saubere Fehlerantwort statt OOM.
5. Body-Size-Limit, Request-Cap, Tracebacks nur ins stderr.

---

## 2. Falsch — konkrete Bugs (priorisiert)

| # | Schwere | Problem | Beleg |
|---|---------|---------|-------|
| B1 | hoch | **`currentDir()!` → Crash beim App-Start.** `tauriFs.currentDir()` ist `invoke<string|null>`; liefert es `null` (kein Workspace), wirft `init()/migrate()` — und `root: null` wird in `status.kind:'loaded'` als `root: string` *typlügen* gespeichert → später `workspaceRoot.split(...)`-TypeError → ErrorBoundary-Blank-Screen. | `src/project/store.ts:107`, `:119` (`!`), `:120` (Setter); Typ-Definition `:19-21`; direkte (`tauriFs`/`tauriSsh`)-Imports `:2`, `:11` |
| B2 | hoch | **Terminal schluckt Fehler still.** Jede `pty_write`/`pty_resize`/`pty_kill`-Fault → `catch(()=>{})`; Reconnect hängt am `\r`-Check (nur Enter triggert) → Eingaben in tote PTY verschwinden kommentarlos, Nutzer „tippt ins Leere". | `src/terminal/Terminal.tsx:87` (kill), `:110` (write), `:126`/`:140` (resize), `:156` (kill), Reconnect über `\r` `:114` |
| B3 | mittel | **Inference kann ewig „inferring" kleben.** Nicht-Abort-Fehler wird im `setTimeout`-Callback re-thrown, ohne äußeren Catch → Status bleibt hängen. | `src/inference/store.ts:41` (Timer), `:73-75` (`catch (e) { … throw e }` bei `:23` `timer`-Scope) |
| B4 | mittel | **Edge-id-Kollision.** Neu erzeugte Kanten `e${cur.length+1}` können mit geladenen Snapshot-Kanten (`e${i+1}`) kollidieren → React-Flow-Zustandskorruption. | `src/canvas/GraphStore.ts:150` (connect) vs. `:180` (loadSnapshot) |
| B5 | mittel | **Dirty-Flap ohne Writeback-Guard.** Der Graph-Subscriber schreibt `dirty` bei jedem `nodes/edges`-Wechsel; der im `CLAUDE.md` verbriefte `applyingShapes`-äquivalente Guard fehlt dort. | `src/workspace/store.ts:567-574` (`setState({ dirty: isDirty })` Z. 574), Basis `:527-541` |
| B6 | mittel | **`project/store.ts` umgeht den Dispatch-Layer.** Direkte `tauriFs`/`tauriSsh`-Imports verletzen die dokumentierte Architektur-Regel („**Do NOT import** `tauri-fs`/`tauri-ssh` directly outside `connections/backend.ts`"). | `src/project/store.ts:2, 11`; Anwendungen `:71, 80, 107, 119, 137, 146, 162` (letzteres `tauriSsh.close()` in `closeProject`) |
| B7 | niedrig | **Build-Warnungen:** 1,36-MB-Hauptchunk (389 KB gzip) + 6× `INEFFECTIVE_DYNAMIC_IMPORT`. | `vite build`-Log (Chunk `index-RkhiQI3n.js`); betroffen u. a. `src/connections/backend.ts` (importiert von `src/datasets/DatasetDetail.tsx` + statisch von `src/canvas/doc.ts`, `src/chat/store.ts`, …) |
| B8 | niedrig | **Verify-Skripte ohne Timeout.** Ein hängender Trainer hängt CI/Dev endlos auf; `python` kommt vom PATH, nicht vom conda-Interpreter; auf Fehlerpfaden bleibt der Sidecar-Prozess inkonsistent. | `scripts/verify-traingen.ts:132, 182, 249, 251` (`execSync('python -u train.py', …)` ohne timeout) · `scripts/verify-sidecar.ts:172` (Spawn `python`), Cleanup-Lücken auf `process.exit`-Pfaden |
| B9 | niedrig | **GPU-Poll blendet Fehler als leere Liste aus**, ohne Nutzerhinweis. | `src/training/RunDetailModal.tsx:154` (`catch { setGpu([]) }`) |
| B10 | niedrig | Sweep-Limit 64 hart, ohne UI-Erklärung. | `src/training/NewRunModal.tsx:187` (`sweepCount <= 64`) |

---

## 3. Unsauber — Qualität, Architektur, Governance

### 3.1 Lint ist nicht grün (53 Probleme / 49 errors, 4 warnings)

**Zusatzbefund:** `eslint .` lintet auch **Build-Artefakte** — 13 der 53 Probleme sind
`Parsing error` in generiertem, minifiziertem JS unter `src-tauri/target/release/build/…/tauri-codegen-assets/*.js`
(9 Dateien). Der `globalIgnores` deckt nur `dist` ab (`eslint.config.js:9`), nicht
`src-tauri/target` — `.gitignore` wird von ESLint im Flat-Config-Modus nicht automatisch angewendet.

#### Komplette Problemliste (`npm run lint`, 49 errors / 4 warnings):

| Stelle | Regel | Problem |
|--------|-------|---------|
| `scripts/verify-codegen.ts:264:15` | `@typescript-eslint/no-explicit-any` | Unexpected `any` |
| `src-tauri/target/release/build/app-*/out/tauri-codegen-assets/*.js` (9 Dateien) | — | 13× `Parsing error: Unexpected character` (minifizierte Artefakte) |
| `src/canvas/subgraphPorts.ts:52:13` und `:52:23` | `no-unused-vars` | `_proxyOf`, `_autoCreated` zugewiesen, nie genutzt |
| `src/canvasdoc/CanvasFileGate.tsx:19:17` | `react-refresh/only-export-components` | Nicht-Komponenten-Export |
| `src/canvasdoc/CanvasFileGate.tsx:37:5` | `react-hooks/set-state-in-effect` | `setLoading(true)` direkt im Effekt |
| `src/chat/store.ts:356:7` | `prefer-const` | `recent_notes` nie reassigniert |
| `src/data/graph/persist.ts:28:46` | `preserve-caught-error` | `throw new Error(...)` ohne `{ cause }` |
| `src/datasets/DatasetDetail.tsx:153:40, 189:38, 226:39, 284:40, 307:41` | `no-unused-vars` | `_relpath` definiert, nie genutzt |
| `src/datasets/DatasetDetail.tsx:478:33, 479:91` | `no-explicit-any` | `normalizeManifest(raw: any)` |
| `src/inspector/CodeField.tsx:45:3` | `react-hooks/refs` | `valueRef.current = …` während Render |
| `src/inspector/Inspector.tsx:832:21` | `set-state-in-effect` | Draft-Sync `setDraft` im Effekt |
| `src/inspector/Inspector.tsx:1137:21`, `:1165:21` | `set-state-in-effect` | Zahl-Drafts |
| `src/inspector/Inspector.tsx:1194:21` | `set-state-in-effect` | Array-Draft-Sync |
| `src/inspector/Inspector.tsx:1194:61`, `:1194:62` | `exhaustive-deps` (warn) | `arr` fehlt / komplexer Dep-Ausdruck |
| `src/inspector/Inspector.tsx:1228:21`, `:1282:21` | `set-state-in-effect` | Shape-/IntList-Drafts |
| `src/layers/registry.ts:863:36` | `no-useless-escape` | `\[` — unnötiges Escape |
| `src/persistence/file.ts:27:15` | `preserve-caught-error` | ohne `{ cause }` |
| `src/training/EvalRunModal.tsx:118:7` | `set-state-in-effect` | `setBranches/setTargetCol/setTable` direkt im Effekt |
| `src/training/ExperimentsExplorer.tsx:192:21` | `set-state-in-effect` | `setDraft` im Effekt (mit `eslint-disable-line`) |
| `src/training/NewRunModal.tsx:140:6` | `exhaustive-deps` (warn) | `inspectDataset`, `prefill.datasetRelpath` fehlen |
| `src/training/NewRunModal.tsx:154:5` | `set-state-in-effect` | `setCaps(null)` direkt im Effekt |
| `src/training/RunDetailModal.tsx:128:10` | `set-state-in-effect` | `void reload()` direkt im Effekt |
| `src/training/RunDetailModal.tsx:159:6` | `exhaustive-deps` (warn) | `runId` fehlt |
| `src/training/RunDetailModal.tsx:164:7` | `set-state-in-effect` | `setPromoteName(...)` direkt im Effekt |
| `src/training/charts/Evaluation.tsx:18:17`, `:32:17` | `react-refresh/only-export-components` | Konstanten/Helper neben Komponenten |
| `src/training/graph/persist.ts:28:46` | `preserve-caught-error` | ohne `{ cause }` |
| `src/visualization/primitives.tsx:14:17, 22:17, 28:17, 269:14, 273:14` | `react-refresh/only-export-components` | 5× Helper/Exporte neben Komponenten |
| `src/visualization/primitives.tsx:85:28` | `no-useless-assignment` | `w` zugewiesen, nie verwendet |
| `src/workspace/store.ts:68:61` | `no-useless-escape` | `\/` in Regex |

Davon größtenteils mit `--fix` behebbar oder kleine Refactors; die `set-state-in-effect`-
Fehler betreffen hauptsächlich `src/inspector/Inspector.tsx` (draft-buffer-Pattern, bewusst, aber
gegen die aktuelle React-Hooks-Regel).

### 3.2 Monolithen / Duplikation

| Befund | Stelle |
|--------|--------|
| `sidecar-llm/main.mjs` 2.158 Zeilen; **kein Linter, kein Typecheck, kein `devDependencies`** | Dateistatistik, `sidecar-llm/package.json:1-40` |
| 3× fast identische Kontext-Fabriken (`makeGraphContext`/`makeTrainingContext`/`makeDataContext`) | `main.mjs:38-70`, `:74-100`, `:104-130` |
| Tool-Specs mit struktureller Duplikation (alle Graph-Tools 3×) | `main.mjs:686-1406` (`buildToolSpecs`) |
| System-Prompt als 250-Zeilen-String im Code | `main.mjs:1629-1879` |
| `sidecar-torch/dataset_handlers.py` 1.745 Zeilen | Dateistatistik |
| `sidecar-torch/main.py` 968 Zeilen (HTTP + Exec + Aktivierungs-Viz + pip in einer Datei) | Dateistatistik |
| **Duplikation Python-Sidecar ↔ `training_template.py`** — ESPF-Tokenizer, Manifest-Resolve, MolGraph, Lookup (mit „Kept in sync"-Kommentaren) | `dataset_handlers.py` (✓ `_load_espf_codebook` `:844-874`, `_espf_encode` §) ↔ `training_template.py` (`_load_espf_codebook` `:307`, `_espf_encode` `:332`, `_SEQ_VOCABS` `:278`, `tokenize_sequence` `:291`, Lookup `:304`/`394`) |
| `_graph_info()` doppelter Aufruf pro Sample | `dataset_handlers.py:359` (Zeile mit `_graph_info(d)` 2×) |
| Tote Molecule-Dict-Branche (nicht mehr erzeugtes Format) | `dataset_handlers.py:1031-1065` (`_branch_field`/`_branch_slots`) |
| Doppelter `import subprocess` | `main.py:25` und `:746` |

### 3.3 Fehlende deklarierte Abhängigkeiten

| Befund | Stelle |
|--------|--------|
| **Kein `requirements.txt` / `pyproject.toml` / `environment.yml` im gesamten Repo** | Repo-Glob, keine Treffer |
| Benötigt (lazy-importiert): `torch`, `torch_geometric`, `pandas`, `numpy`, `Pillow`, `rdkit`, `biopython`, `datasets`, `python-dateutil` | Import-Stellen in `dataset_handlers.py` (u. a. `:578` HF, `:603` PyG) |
| Remote-Deploy installiert nur `numpy pandas pillow python-dateutil` (+ `torch`) — **`rdkit`, `biopython`, `torch_geometric`, `datasets` fehlen → Molecule/Protein/PyG/HF-Kinds degradieren am HPC** | `src-tauri/src/remote_sidecar.rs:160` (pip-Zeile), `:117` (check), `:351` (Meldung) |
| `sidecar-torch/espf/`-README (BSD-3, Codebooks) korrekt lizenziert, aber als Paket nicht versionspezifisch deklariert | `sidecar-torch/espf/` |

### 3.4 Tote Exporte / tote Funktionen

| Befund | Stelle |
|--------|--------|
| `getActiveRemote()` — 0 Referenzen | `src/connections/store.ts:187` |
| `remoteTrainingBlocked()` — 0 Referenzen | `src/training/backend.ts:19` |
| `columnsFor` — ausdrücklich tot (via `void columnsFor` unterdrückt) | `src/inspector/Inspector.tsx:866`/`:884` |
| `bash -lc rel`-Fallback: Nicht-`.py/.sh`-relpaths werden als Kommandostring ausgeführt (vermutlich funktionslos) | `main.py:773-778` (`run_workspace_script`) |
| hart kodierte `WORKSPACE_ROOT.parents[2]`-Annahme („run dir 3 Ebenen tief") | `training_template.py:44-46` |
| Verwaistes `mlforge/pid` im Root (gitignored, liegt herum) | `~/projects/mlforge/pid` |

### 3.5 Governance / Metadaten

| Befund | Beleg |
|--------|-------|
| **Versionen dreimal verschieden:** `package.json` **0.0.0** · `tauri.conf.json` **0.1.0** · `Cargo.toml` **0.1.0** | `package.json:4` · `src-tauri/tauri.conf.json:4` · `src-tauri/Cargo.toml:3` |
| `private: true` in `package.json` | `package.json:3` |
| **`Cargo.toml` = Scaffold-Placeholder:** `name="app"`, `description="A Tauri App"`, `authors=["you"]`, `license=""`, `repository=""` (Binary korrekt: `[[bin]] name = "spinoml"`, `Cargo.toml:17-19`) | `src-tauri/Cargo.toml:2-8` |
| Zeit-Pin dokumentiert, aber Schuld: `time = "=0.3.46"` wegen veraltetem `cookie` 0.18 | `Cargo.toml:33-35` |
| **README-Badge „TypeScript-5" falsch** — real `typescript ~6.0.2` | `README.md:15` vs. `package.json:49` |
| **README/CLAUDE.md nennen Env `spinoml-dev`, das existiert nicht** — auf diesem Rechner heißt es `mlforge-dev` | `README.md:109, 112` · `CLAUDE.md:228` vs. `~/anaconda3/envs/mlforge-dev` |
| README: „bundle embeds both sidecars" — gebündelt wird nur der **Quellcode**; die App ruft `python`/`node` vom PATH | `README.md:143-145` vs. `src-tauri/tauri.conf.json:41-44` (resources) und `src-tauri/src/lib.rs:733` (python), `:741` (node) |
| `sidecar-torch/README.md` dokumentiert nur `/infer` + `/health` („Phase 3"), real gibt es `/dataset/*`, `/activations`, `/deps/*`, `/run_script` — auch Docstring in `main.py:12-13` unvollständig | `sidecar-torch/README.md`, `main.py:12-13` |
| Unterbudgetiertes Release-Setup: `targets` nur `deb`, **kein AppImage, kein Windows/macOS**, `security.csp: null` | `src-tauri/tauri.conf.json:30`, `:25` |

**Lizenz:** proprietäre eigene „Noncommercial No-Derivatives Source-Available License" (NC-ND-Variante, kein OSI/FSF-Approval). Für Veröffentlichung legal ok, aber:
- **Contributions-Klausel** (irrevocable Lizenz an den Autor, „confers no rights on You") mit einem echten Community-Prozess unvereinbar → Entscheidung treffen und klar kommunizieren | `LICENSE:83-86` (Abschnitt 5, Zeilen 84-85) |
- Laufzeit-/Nutzungs-Blocker bleiben „run+read only" | `LICENSE:39-45, 52-68` |
- Keine OSI-Kennzeichnung; README sollte trennen: „Source-Code-Zugang ≠ Open Source" | `README.md:219-227` |

**Fehlende Governance-Dateien:** `CHANGELOG.md` · `SECURITY.md` · `CONTRIBUTING.md` · **kein `.github/`** (kein CI/CD) | Repo-Glob

### 3.6 Git-Hygiene — getrackte Artefakte, die nicht ins Repo gehören

| Artefakt | Befund |
|----------|--------|
| Python-Bytecode im Git | `examples/reaction-workspace/experiments/runs/…/__pycache__/model.cpython-312.pyc` (vom Audit bestätigt getrackt) |
| Runtime-PID-Datei im Git | `examples/…/experiments/runs/…/pid` |
| Binäres PDF (2,6 MB) | `docs/iris-tutorial.pdf` |
| Workspace-Verzeichnis im Repo-Root | `workspace/datasets/iris.csv` |
| 18 `.pt`-Modelldateien (1,6 MB) ohne LFS | `examples/reaction-workspace/**` (inkl. `.graphcache/*.pt`, `graphs/*.pt`, `experiments/**/best.pt`, `models/best/dual.pt`) |
| `.gitignore` deckt nur `sidecar-torch/__pycache__` ab, nicht `**/__pycache__/`, `**/pid`, `workspace/` | `.gitignore:23-24, 28` |
| Gesamt: 185 getrackte Dateien | `git ls-files \| wc -l` |

---

## 4. Maßnahmenliste für die Veröffentlichung (priorisiert)

### Stufe 0 — Blocker, schon vor einer privaten Beta

| # | Maßnahme | Referenzen |
|---|----------|------------|
| 1 | **Sidecars absichern:** Launch-Token + Origin-Allowlist statt CORS `*`; `/respond` + `autoMode` authentifizieren; `args` shell-quoten; SSRF auflösen. | `main.mjs:1571, 2136, 1906, 518, 534, 304` · `main.py:793` |
| 2 | **Pickle-RCE entfernen:** `weights_only=True` (+ Safe-Globals für PyG) an allen 12 Stellen. | `main.py:569` · `dataset_handlers.py:382, 413, 427, 765, 1021, 1361, 1551` · `training_template.py:249, 375, 503, 969, 1165` |
| 3 | **Pfad-Scoping** im Sidecar (Workspace-Root-Whitelist, Spiegel von `lib.rs:144`). | `main.py:822-888, 748-756` |
| 4 | **Element-Cap + Request-Cap + Body-Limit**; Tracebacks nur ins stderr. | `main.py:182-193, 546-553, 958, 811` · `dataset_handlers.py:1502-1503, 1727-1731` |
| 5 | **Lint grün** (53 Punkte); `src-tauri/target` in `globalIgnores` | § 3.1 |
| 6 | **Bugs B1–B4 fixen.** | § 2 |

### Stufe 1 — Pflicht für eine öffentliche Release

| # | Maßnahme | Referenzen |
|---|----------|------------|
| 7 | Versionen synchronisieren (`package.json` → 0.1.0), `Cargo.toml`-Metadaten richtig setzen. | `package.json:4` · `Cargo.toml:2-8` |
| 8 | **CI** (`.github/workflows/ci.yml`): `npm run build`, `npm run lint`, `cargo check`, `verify:codegen`, `verify:traingen`, `verify:sidecar`. | — (fehlt komplett) |
| 9 | Tests für die reinen Kerne (`generator.ts`, `coerceParams`) + `verify:sidecar` um `/dataset/*` erweitern; `execSync`-Timeouts. | § B8 |
| 10 | Doku berichtigen (Env-Name, TS-Badge, Sidecar-Runtime-Anforderungen, Endpunkte). | `README.md:15, 109, 143` · `CLAUDE.md:228` · `sidecar-torch/README.md` |
| 11 | Repo bereinigen: `__pycache__`, `pid`, PDF, `.pt` (LFS), `workspace/`; `.gitignore` erweitern; `CHANGELOG.md` anlegen. | § 3.6 |
| 12 | CSP setzen (`default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self' http://127.0.0.1:*`). | `tauri.conf.json:25` |
| 13 | Bundling-Targets entscheiden (AppImage zusätzlich; Plattformstrategie). | `tauri.conf.json:30` |
| 14 | Abhängigkeiten deklarieren (`requirements.txt` / `pyproject.toml`), Remote-Deploy-Installliste vervollständigen. | § 3.3 |

### Stufe 2 — für „rund" empfehlenswert

- `SECURITY.md`, `CONTRIBUTING.md`; Lizenzmodel-Entscheidung transparent machen. — `LICENSE:83-86`
- `engines`-Feld + Node-Version, Pin-Strategie für `@anthropic-ai/claude-agent-sdk` (0.x!), Linter im LLM-Sidecar. — `sidecar-llm/package.json`
- Duplikation Sidecar ↔ `training_template.py` in ein geteiltes Modul; `main.mjs`-Fabriken faktorisieren; System-Prompt auslagern. — `main.mjs:38-130, 686-1406, 1629-1879`
- `time`-Pin samt veraltetem `cookie` aufräumen. — `Cargo.toml:33-35`
- UI-Sprache vereinheitlichen (DE/EN-Mix, teils im selben Panel), Accessibility-Grundlage (`aria-label`, `<label>` für Textareas, Fokus-Ringe).
- 512×512-Icon aus 1024er-Master (`tauri icon`); `ssh_exec`-Gesamt-Timeout. — `src-tauri/src/ssh.rs:167-202`

---

## Kontext & Abgrenzung

- **Repo bereinigt? nein.** Diese Datei ist eine Momentaufnahme des Stands von 2026-09-07.
- **Was absichtlich NICHT Teil dieses Audits ist:** Funktions-Weiterentwicklung, Backlog (siehe `TODO.md`), Design-Branding (siehe `BRAND.md`/`docs/FEATURES.md`).
- **Zustand der Codebasis:** Frontend & Rust sind überdurchschnittlich sauber und gut dokumentiert;
  die kritischen Punkte liegen in der Loopback-Absicherung, den fehlenden Tests/CI und der
  Release-Hygiene (Metadaten, Doku-Realitätsdrift, getrackte Artefakte).