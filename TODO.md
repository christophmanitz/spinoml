# TODO — capability gaps

Missing capabilities that block real model-building, found during an
ML-engineer evaluation of the codegen + shape-inference pipeline (2026-06-14).
The architecture/codegen core is sound (branching DAGs, residual skips,
multi-input, live ground-truth shape inference); these are the gaps that keep
it a *model designer* rather than a *model builder*.

Ordered roughly by leverage. Each layer-add follows the
"Add a new layer type" recipe in CLAUDE.md (registry.ts + generator.ts
serializeParam + an Inspector FixHint + verify:codegen/verify:sidecar cases).

## Layer catalog gaps

- [ ] **`Reshape` / `View` / `Permute` / `Transpose`** — cheapest win, highest
      unlock. Only `Flatten` exists today, so you cannot go from conv feature
      maps to a token sequence. Blocks ViT, CRNN, any Conv→Seq bridge.
      Needs a functional/merge-style node (forwardExpr emits `x.reshape(...)`
      / `x.permute(...)`), not an `nn.Module`.
- [ ] **`Embedding`** (+ Positional Encoding) — the Transformer path is
      effectively unusable for NLP without it. The current transformer template
      feeds raw `[1,16,512]` floats, never token IDs. Requires integer-tensor
      inputs (today Input shapes assume float `torch.zeros`).
- [ ] **Sequence models: `LSTM` / `GRU` / `RNN`** — none exist. Closes the
      entire recurrent / sequence-modeling gap. Note: these return
      `(output, (h, c))` tuples — codegen + shape-inference hooks assume a
      single tensor output, so this touches generator.ts and the sidecar hook.
- [ ] **GNNs** — graph neural network layers (e.g. `GCNConv`, `GATConv`,
      `GraphSAGE`, `MessagePassing`). Bigger lift: depends on
      `torch_geometric` (new optional sidecar dep), edge-index inputs (a new
      input modality beyond dense tensors), and a graph-batch concept. Likely
      its own phase, not a single layer add.

## Smaller catalog gaps

- [ ] **`BatchNorm1d`** — only `BatchNorm2d` exists; needed for MLP/tabular nets.
- [ ] **`Conv3d`** — only 1d/2d; blocks volumetric / video models.
- [ ] **`Softmax` / `LogSoftmax`** — no explicit output activation (loss
      usually handles it, but needed for inference heads / attention from scratch).
- [ ] **`TransformerEncoder` stack** — only the single `TransformerEncoderLayer`
      exists; multi-layer stacking is manual. Add a `num_layers` wrapper.
- [ ] **`Add` skip shape adaptation** — the residual `Add` does no shape
      matching, so downsample residuals (1×1 conv in the skip branch) must be
      wired by hand or PyTorch throws. Consider a FixHint that detects the
      mismatch and suggests the projection conv.

## Beyond architecture (larger, separate decision)

- [~] **Training** — siehe ausführlicher Plan unten in "Phase 13–18: Training-System".
      **Phase 13 (Foundation) + 14 (Trainings-Graph) + 15 (Live-Tracking-UI)
      + 16 (Remote-Direct via ssh+nohup) + 17 (SLURM-Submit) sind umgesetzt.**
      18 offen (Sweeps + Compare-Polish + Export).

# Phase 13–18: Training-System — ausführlicher Plan

Stand der Architektur, an die wir andocken:

- `models/*.mlforge` ist der Modellgraph (Source of Truth)
- `models/*.py` wird via `generator.ts` aus dem Graph emittiert
  (pure `nn.Module`, kein Training)
- `datasets/` enthält die geladenen Datensätze (6 Formate)
- `experiments/` ist der Run-Output-Bucket (bisher nur
  `smoke-results.jsonl`)
- Workspace kennt zwei Modi: **local** und **remote-ssh** (Phase 12)
- Sidecars: `torch` (Shape-Inferenz, Smoke-Test) und `llm` (Chat)
- Connection-Dispatch in `src/connections/backend.ts` ist die einzige
  Stelle die zwischen lokalem FS und SSH unterscheidet

Das Trainings-System soll **drei orthogonale Achsen** sauber bedienen:

1. **WO läuft das Training** — local Python / HPC direkt (Login-Knoten oder
   ssh-spawned shell) / **SLURM-Job auf Compute-Knoten**
2. **WIE wird Training konfiguriert** — separater visueller
   Trainings-Graph (eigene Canvas-Ansicht), nicht Formulare
3. **WIE überlebt der Run einen MLForge-Close** — alle Trainings laufen
   detached; MLForge ist nur eine Sicht auf den Status

Diese drei Achsen geben die Phasen-Reihenfolge.

## Phase 13: Foundation — Trainings-Config, Run-Verzeichnis, Local-Executor  ✅ ERLEDIGT (2026-06-14)

Das eigentliche Fundament. Wenn 13 steht, kann man von Hand einen Run
starten, ihn überleben den App-Restart sehen, Logs lesen.

**Umsetzung** (Branch `phase-13-training-foundation`):
- `sidecar-torch/training_template.py` — pure-python Trainer (tabular),
  liest `run.json`, importiert `model.py`, schreibt events.jsonl +
  checkpoints + status + metrics.json. Klassifikation (CrossEntropy/BCE) +
  Regression (MSE/L1); Optimizer Adam/AdamW/SGD/RMSprop; Scheduler
  Step/Cosine/Plateau.
- `src-tauri/src/training.rs` — Local-Executor: `start_training_run`
  (detached `setsid`, schreibt Run-Dir + pid), `stop_training_run`
  (kooperativ via status-File + SIGTERM auf Prozessgruppe),
  `list_training_runs`, `training_run_status`, `read_training_run_file`,
  `delete_training_run`. Stale „running" wird zu „failed" reconciled, wenn
  pid weg ist.
- Frontend `src/training/` — `types.ts`, `tauri-training.ts`,
  `backend.ts` (Dispatch; remote-ssh in Phase 13 geblockt), `store.ts`
  (Zustand + Poller alle 2s solange ein Run lebt), `ExperimentsExplorer`,
  `NewRunModal` (manuelle Config), `RunDetailModal` (Overview/Events/Logs),
  `StatusPill`. Experiments-Tab in `LeftSidebar`, Modals in `App.tsx`.
- Noch NICHT in 13 (kommt später): Live-Charts (Phase 15), visueller
  Trainings-Graph (Phase 14), Remote/SLURM (16/17). Run-Tailing ist
  Polling, nicht fs-watch/Tauri-Events.

### 13.1 Run-Verzeichnis-Format

```
experiments/
  runs/
    2026-06-15T12-30-00_iris-mlp_a8f3/
      run.json              ← frozen config + meta (s.u.)
      model.mlforge         ← snapshot des Architektur-Graphs
      model.py              ← snapshot des generierten Codes
      train.py              ← snapshot des Trainings-Codes
      events.jsonl          ← append-only event stream (eine Zeile / Event)
      metrics.json          ← finale Zusammenfassung (best metric, total time, …)
      stdout.log            ← prozess-stdout
      stderr.log            ← prozess-stderr
      checkpoints/
        epoch_0.pt
        epoch_5.pt
        best.pt             ← symlink auf den besten checkpoint
      pid                   ← lokaler PID oder SLURM-jobid (s.u.)
      status                ← single-line: queued|running|done|failed|cancelled
      slurm-12345.out       ← (nur bei SLURM, zusätzlich zu stdout/err)
```

`run.json` enthält:
- run_id, run_label, created_at, ended_at?, status
- model_path (Pfad relativ zum Workspace)
- training_graph_path (Pfad zum visuellen Training-Graph, s. Phase 14)
- backend: `{kind: "local" | "ssh-direct" | "slurm", ... }`
- frozen Konfiguration (Optimizer, Loss, LR-Schedule, Batch-Size, …)
- final metrics snapshot

Naming: `<iso-timestamp>_<slug-from-label>_<short-rand>` macht
Sortierung im FS = Sortierung in der UI = chronologisch.

### 13.2 events.jsonl Schema

Streamendes Event-Log, eine JSON-Zeile pro Event:

```json
{"t":"2026-06-15T12:30:01.123Z","kind":"epoch.start","epoch":0}
{"t":"2026-06-15T12:30:02.456Z","kind":"batch","epoch":0,"step":10,"loss":2.31,"lr":0.001}
{"t":"2026-06-15T12:31:15.789Z","kind":"epoch.end","epoch":0,"train_loss":1.87,"val_loss":1.92,"val_acc":0.43}
{"t":"2026-06-15T12:45:01.000Z","kind":"checkpoint","epoch":5,"path":"checkpoints/epoch_5.pt","val_loss":0.34,"is_best":true}
{"t":"2026-06-15T13:00:00.000Z","kind":"run.done","total_seconds":1800,"best_val_loss":0.21}
```

Wichtig: **append-only**, atomic per-line write (POSIX guarantee bis 4KB).
Frontend tailt die Datei (lokal: `fs.watchFile`; remote: `ssh + tail -f`).

### 13.3 Trainings-Skript-Template

Pure-Python, **kein MLForge-Runtime-Dep** außer torch:

```
sidecar-torch/training_template.py
```

Liest `run.json` aus dem Run-Dir, importiert `model.py` (das macht
Codegen schon), baut Optimizer/Loss/Scheduler aus der Config,
fährt die Training-Loop, schreibt nach `events.jsonl` +
`checkpoints/`. Crash-Verhalten: `events.jsonl` letzte Zeile
`run.failed` mit Traceback, dann `status=failed`.

Detach-Mechanismus lokal: `setsid` + `nohup` + stdout/stderr in
die Run-Dir-Files umgeleitet. PID nach `pid`-Datei.

### 13.4 Local-Executor (Rust)

`src-tauri/src/training/local.rs`:

- `start_run(run_id, train_config) -> RunHandle`
- `stop_run(run_id) -> ()` (kill PID)
- `run_status(run_id) -> Status` (lese status-File + ggf. `kill -0 <pid>`)
- `tail_events(run_id) -> Stream` (Tauri-Event-Channel,
  emittiert Zeilen aus `events.jsonl` ab Position X)

### 13.5 UI: Run-Liste + Run-Detail

Neuer „Experiments"-Tab in der linken Sidebar (neben Files +
Datasets). Zeigt:

- Liste aller Runs in `experiments/runs/`, sortiert neueste oben
- Pro Run: Status-Pille, Label, Modell-Name, kurzes Metric-Tag
- Klick → Run-Detail-Modal (analog zum DatasetDetail) mit
  Logs, finalen Metriken, Config-Diff zu anderen Runs

In 13 noch **keine** Live-Charts, noch **kein** visueller
Trainings-Graph. Nur „kann ich einen Run starten und ihn nach
App-Restart wieder sehen".

## Phase 14: Visueller Trainings-Graph (eigene Canvas-Ansicht)  ✅ ERLEDIGT (2026-06-14)

Wenn Phase 13 steht, kommt die UX-Innovation: Training wird nicht
über Formulare konfiguriert, sondern wie das Modell selbst als
Graph geknüpft.

**Umsetzung** (14a Engine / 14b Editor / 14c Wiring):
- 14a — `src/training/graph/{registry,store,persist}.ts` +
  `src/codegen/trainingGenerator.ts` (PURE Graph→run.json-Compiler) +
  Template-Erweiterungen (Metrics, EarlyStopping, GradClip, AMP). Harness
  `scripts/verify-traingen.ts` (`npm run verify:traingen`) — Compiler +
  echtes Training der kompilierten Config.
- 14b — Mode-Toggle Architektur↔Training (`viewMode` Store + `ModeToggle`),
  `TrainingCanvas`/`TrainingPalette`/`TrainingNode`/`TrainingInspector`
  parallel zur Architektur-Canvas; App.tsx swappt Palette/Canvas/Inspector.
- 14c — „▶ Run starten" direkt aus dem Graph (Inspector-CompilePanel →
  `useTrainingStore.startRun`), Save/Load `.mltrain` unter
  `experiments/training-graphs/` (`TrainingGraphBar` + `graph/files.ts`).
- Abweichung vom Plan: KEIN Rename `GraphStore`→`useArchitectureGraphStore`
  (zu invasiv für den Nutzen); stattdessen ein parallel lebender
  `useTrainingGraphStore`. Der Trainings-Graph kompiliert in die Phase-13
  `run.json` (nicht in eine eigene `train.py`) — der konfig-getriebene
  Trainer aus Phase 13 wird wiederverwendet statt ein zweiter Codepfad.
- Offen: Edge-Validierung/Smoke-Test des Trainings-Graphs; `.mltrain` im
  FileExplorer-Baum (aktuell eigener Save/Load-Bar).

### 14.1 Mode-Switch im Canvas

`Canvas.tsx` bekommt einen Mode-Toggle oben: **Architecture** ↔
**Training**. Beide nutzen die gleiche React-Flow-Engine, aber:

- Architecture-Mode arbeitet auf `models/*.mlforge` (heute)
- Training-Mode arbeitet auf `experiments/training-graphs/<name>.mltrain`
  — gleiches Persistenz-Schema, anderes Suffix

`GraphStore` wird zu `useArchitectureGraphStore` umbenannt und ein
parallel-lebender `useTrainingGraphStore` daneben gebaut. Beide
teilen die Generator/Codegen-Infrastruktur über ein gemeinsames
`canvas/`-Modul.

### 14.2 Trainings-Knoten-Palette

Neue Layer-Kategorien (gleicher Mechanismus wie `layers/registry.ts`,
aber separate Datei `training/registry.ts`):

**Source/Sink**:
- `DatasetSource` — Reference auf `datasets/<name>`, mit Field
  `dataset-ref` (das gibt's schon aus Phase 11)
- `ModelSource` — Reference auf `models/<name>.mlforge`, mit
  Live-Param-Count im Inspector

**Data-Pipeline**:
- `Split` — train/val/test ratios, seed
- `DataLoader` — batch_size, num_workers, shuffle, pin_memory,
  drop_last
- `Augment` (für Image): RandomCrop, HorizontalFlip,
  Normalize — Sub-Pipeline mit Drag-Drop in eigenes Sub-Panel

**Training-Komponenten**:
- `Loss` — kind-Select aus {CrossEntropyLoss, BCEWithLogitsLoss,
  MSELoss, L1Loss, KLDivLoss, custom-Python} mit kind-spezifischen
  Params (label_smoothing, pos_weight, …)
- `Optimizer` — kind aus {Adam, AdamW, SGD, Lion, RMSprop} mit
  lr, weight_decay, betas/momentum
- `Scheduler` — kind aus {None, StepLR, MultiStepLR, CosineAnnealingLR,
  CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau, custom}
- `Metric` — Accuracy, F1, Precision/Recall, ConfusionMatrix, MSE,
  R2 — kann mehrfach gewählt werden

**Callbacks** (mehrfach, alle parallel an TrainLoop verdrahtet):
- `EarlyStopping` — monitor metric, patience, mode (min/max)
- `ModelCheckpoint` — every_n_epochs, save_top_k, monitor
- `ReduceLROnPlateau` — wenn nicht im Scheduler-Node
- `GradientClipping` — max_norm, norm_type
- `MixedPrecision` — AMP an/aus, dtype (fp16/bf16)
- `WandBLogger` (optional, eigene Phase) — entity, project, run_name

**Orchestrator**:
- `TrainLoop` — der zentrale Knoten. Hat Inputs für
  data (DataLoader), model (Model), loss (Loss),
  optimizer (Optimizer), scheduler (Scheduler optional), metrics
  (Metric[]), callbacks (Callback[]). Params: epochs,
  gradient_accumulation_steps, val_every_n_epochs, log_every_n_steps,
  seed, deterministic.

### 14.3 Visualisierung — Beispiel-Layout

```
┌──────────────┐    ┌──────┐    ┌────────────┐
│ DatasetSource│───→│Split │───→│ DataLoader │───┐
│ iris.csv     │    │70/15/│    │ batch=32   │   │
│ features:[…] │    │  15  │    │ workers=4  │   │
└──────────────┘    └──────┘    └────────────┘   │
                                                 ↓
┌──────────────┐                          ┌──────────────┐
│ ModelSource  │─────────────────────────→│              │
│ iris-mlp     │                          │  TrainLoop   │
│ 67 params    │                          │              │
└──────────────┘                          │  epochs=200  │
                                          │  seed=42     │
┌──────────────┐                          │              │
│   Loss       │─────────────────────────→│              │
│ CrossEntropy │                          │              │
└──────────────┘                          │              │
                                          │              │
┌──────────────┐                          │              │
│  Optimizer   │─────────────────────────→│              │
│ Adam lr=1e-3 │                          │              │
└──────────────┘                          │              │
                                          │              │
┌──────────────┐                          │              │
│  Scheduler   │─────────────────────────→│              │
│ Cosine 200ep │                          │              │
└──────────────┘                          │              │
                                          │              │
┌──────────────┐                          │              │
│ EarlyStopping│─────────────────────────→│              │
│ patience=20  │                          └──────────────┘
└──────────────┘
```

Edge-Validation: Loss erwartet `(prediction, target)`-Konvention,
Metric ebenso. Inspector erkennt Mismatch und schlägt Reshape vor.

### 14.4 Codegen — Training-Graph → train.py

Analog zu Architecture-Codegen (`codegen/generator.ts`) gibt's
`codegen/trainingGenerator.ts`. Nimmt den Training-Graph, emittiert:

```python
import torch
from torch.utils.data import DataLoader, random_split

from model import Model
from sidecar_torch.dataset_handlers import load_dataset

def run():
    ds = load_dataset("datasets/iris.csv", features=[...], target="species")
    train_ds, val_ds, test_ds = random_split(ds, [0.7, 0.15, 0.15],
                                              generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)

    model = Model()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    callbacks = [EarlyStopping(...)]

    for epoch in range(200):
        emit({"kind":"epoch.start","epoch":epoch})
        ...
        emit({"kind":"epoch.end", ...})
```

Smoke-Test-Modus für den Trainings-Graph: gleicher Trick wie heute
beim Modell-Smoke-Test — 1 Batch durchschicken, prüfen ob Loss
endlich ist, ohne den ganzen Run zu starten.

## Phase 15: Live-Tracking-UI  ✅ ERLEDIGT (2026-06-14)

**Umsetzung** (15a Charts / 15b Compare):
- 15a — `src/training/charts/LineChart.tsx` (pure-SVG Multi-Series-Chart,
  keine Lib, log-Y, Crosshair + Hover-Readout, Gaps bei null-y) +
  `charts/series.ts` (leitet loss/lr/metric-Serien aus events.jsonl ab,
  stabile Palette). `RunDetailModal` bekommt Tab „charts" (Loss mit
  log-Toggle, Metriken, LR) + Live-Progress-Bar in Overview mit
  epoch x/N + ETA (aus beobachteter Epoch-Kadenz).
- 15b — Multi-Run-Compare: Checkboxen pro Run in `ExperimentsExplorer`,
  „Vergleichen"-Bar, `CompareModal` (Loss-Overlay aller gewählten Runs,
  finale-Metriken-Tabelle, Config-Diff der training-Sektionen). Compare-
  State (`compareIds`/`compareOpen`) im `useTrainingStore`.
- Abweichung vom Plan: Tailing bleibt Polling (alle 2s solange ein Run
  lebt), KEIN `fs::watch`/Tauri-`training:event` — für Epoch-Granularität
  ausreichend; fs-watch verschoben (relevant erst bei Batch-Live-Charts).
  Kein eigener Bottom-Panel-Tab „Training" — Charts leben im Run-Detail-
  Modal (weniger invasiv, gleiche Daten). Hardware-Strip + Sample-Preds
  noch offen (brauchen optionale Sidecar-Polls bzw. sample-Callback).

Sobald Phase 13+14 stehen, baue Live-Visualisierung:

### 15.1 Run-Tab im Bottom-Panel

Zusätzlich zu Code+Terminal: dritter Tab **Training**. Wenn ein
Run aktiv ist:

- Status-Bar oben: `running · epoch 47/200 · 14m elapsed · ETA 38m`
- **Loss-Chart**: train + val, log-scale-Toggle, smoothing
- **Metric-Chart**: pro Metric eine Linie, Best-marker
- **LR-Chart**: lr über Steps
- **Logs**: stdout-Tail mit Filter (ERROR/WARN/INFO)
- **Sample-Preds**: für Klassifikation ein paar val-Beispiele mit
  Prediction vs. Truth (kommt aus optionalem `sample` Callback)
- **Hardware-Strip**: GPU util/mem/temp, CPU%, RAM (kommt aus
  parallel-laufendem `nvidia-smi`/`top`-Polling, optional)

### 15.2 Tailing-Mechanismus

Lokal: Rust `fs::watch` auf `events.jsonl`, neue Zeilen werden
geparst und als Tauri-Event `training:event` emittiert. Frontend
hat eine Zustand-Store `useTrainingRunStore` der die Event-Liste
hält + aktive Charts berechnet.

Remote: ssh + `tail -F` als langlebige Session (wie unser
remote-sidecar tunnel). Output wird Zeile für Zeile als Tauri-Event
weitergereicht.

SLURM: same wie remote, aber Quelle ist `slurm-<jobid>.out`
(oder besser: events.jsonl, denn das Training-Skript schreibt
unabhängig von SLURMs stdout-Buffering direkt in die Datei mit
`flush=True`).

### 15.3 Multi-Run-Vergleich

Im Experiments-Tab: mehrere Runs auswählen → Compare-Modus:
- Alle ausgewählten Loss-Kurven übereinander
- Tabelle final-metrics (val_loss, val_acc, train_time, n_params)
- Config-Diff highlights nur veränderte Felder

Best-of-N: schnellste Wahl welches Modell ins `models/best/`
verlinkt wird.

## Phase 16: Remote-Direct-Training (kein SLURM)  ✅ ERLEDIGT (2026-06-14)

**Umsetzung**:
- `src-tauri/src/ssh.rs` — sechs `ssh_*`-Mirror der lokalen Training-Commands
  (`ssh_start_training_run` / `ssh_list_training_runs` /
  `ssh_training_run_status` / `ssh_read_training_run_file` /
  `ssh_stop_training_run` / `ssh_delete_training_run`). Start schreibt das Run-
  Dir auf den Host (run.json/model.mlforge/model.py + train.py aus dem lokalen
  Bundle) und launcht detached via `nohup setsid <python> -u train.py
  > stdout 2> stderr < /dev/null & echo $! > pid` — überlebt ssh-Session UND
  App-Close. Status = `kill -0 <pid>` + status-File über ssh; Liste in EINEM
  Round-Trip (marker-delimitierter Shell-Loop, in Rust geparst).
- `src-tauri/src/training.rs` — `RunSummary::from_parts` / `RunStatus::new` /
  `reconcile_status` / `validate_run_id` / `READABLE` als `pub(crate)`
  herausgezogen, damit local + ssh exakt dieselbe Summarize-/Reconcile-Logik
  teilen.
- Frontend — `connections/store.ts` bekommt `python?` pro Remote-Connection
  (+ `remotePython()`), `tauri-ssh.ts` die sechs Wrapper,
  `training/backend.ts` dispatcht jetzt local↔remote (statt zu blocken),
  `ExperimentsExplorer` zeigt einen Remote-Strip mit Host + Python-Pfad-Input
  (persistiert via `updateRemote`).
- Abweichung vom Plan: KEIN rsync — Run-Dateien gehen per `cat >`/stdin über
  ssh (klein, kein extra Tool nötig); Dataset bleibt auf dem Host (abspath =
  `<root>/datasets/...`). Backend-Detection (sbatch/nvidia-smi/module,
  RemoteCapabilities) ist Vorbereitung für Phase 17 und hier noch NICHT dabei.
  Remote-Python muss eine Umgebung mit torch (+ pandas für tabular) sein — der
  Strip weist darauf hin; fehlende Deps landen sichtbar in stderr.log.

Manche User wollen testweise auf dem Login-Knoten trainieren oder
auf einem Compute-Knoten den sie sich vorher mit `salloc` geholt
haben. Das ist der einfachere Fall, weil keine Queue dazwischen ist.

### 16.1 Backend-Detection

Beim Open eines Remote-Workspace: zusätzlich zur torch-Sidecar-
Probe (Phase 12b) prüfe:

```bash
command -v sbatch    # → SLURM verfügbar?
command -v salloc    # → interaktive Reservierung möglich?
nvidia-smi -L 2>/dev/null    # → GPUs erreichbar?
command -v module    # → Module-System?
```

Ergebnis in `RemoteCapabilities` festhalten: `{has_slurm,
has_gpu, has_modules, gpu_names: [...]}`. UI bekommt das via
Tauri-Event.

### 16.2 Detached-Run via ssh+nohup+setsid

`start_remote_run(run_id, ...)`:
1. ssh: rsync run.json + model.py + train.py + dataset (oder Pfad
   wenn schon da) nach `<root>/experiments/runs/<run_id>/`
2. ssh + spawn:
   ```bash
   cd <run_dir>
   nohup setsid <venv>/bin/python -u train.py \
     > stdout.log 2> stderr.log < /dev/null &
   echo $! > pid
   disown
   ```
3. ssh-Verbindung darf nach diesem Command sterben — `setsid` löst
   den Python-Prozess von der Session, `nohup` blockt HUP.

Lebenszeichen-Check: regelmäßig (z.B. alle 5 s) `ssh + kill -0 <pid>`
+ tail von events.jsonl. Wenn `kill -0` fehlschlägt → Run gilt als
beendet, lese status-File für final-state.

Cleanup-Lesson aus Phase 12b: systemd-logind reapt User-Prozesse
**nicht** auf RHEL/Rocky (KillUserProcesses=no Default). Deshalb
funktioniert nohup+setsid hier zuverlässig — das war beim
sidecar-torch der bug, hier wird es zum Feature.

## Phase 17: SLURM-Integration  ✅ ERLEDIGT (2026-06-14)

**Umsetzung**:
- `src-tauri/src/ssh.rs` — `ssh_start_training_run` verzweigt nach
  `backend.kind` aus run.json: `slurm` → `build_sbatch()` schreibt
  `train.sbatch` (#SBATCH-Header aus partition/time/mem/cpus/gres/account/qos
  + `module load`-Zeilen + pre_run_script + `python -u train.py`), submittet
  via `sbatch`, parsed die Jobid und friert sie als `pid = slurm:<jobid>`
  ein. Liveness/Status/Stop/Delete sind slurm-aware: `slurm:`-pids gehen über
  `squeue`/`scancel` statt `kill -0`/`kill`. Neuer Probe-Command
  `ssh_remote_training_capabilities` (sbatch? nvidia-smi? `sinfo`-Partitionen).
- Frontend — `RunBackend`/`SlurmConfig` in types; `NewRunModal` zeigt einen
  Backend-Block (Direkt ↔ SLURM) NUR wenn die Probe `has_slurm` meldet, mit
  Partition-Dropdown (aus `sinfo`), time/mem/cpus/gres/account, module-Liste,
  pre-run-script. SlurmConfig wird auf der Connection gemerkt (`slurm?`).
- Abweichung vom Plan: KEIN `SlurmConfig`-Graph-Node (17.1) — SLURM wird beim
  Run-Start gewählt statt als Trainings-Graph-Knoten (deutlich weniger
  invasiv, gleiche Capability). KEIN rsync (wie Phase 16). Resume aus
  Checkpoint (17.3) + sacct-Final-State-Parsing (17.2) noch offen: Status
  kommt aus dem status-File des Trainers + squeue-Liveness; ein per SLURM
  gekillter Job (OOM/Timeout) wird über `reconcile_status` als failed
  angezeigt, sobald er aus squeue fällt.

Der eigentliche HPC-Use-Case. Voraussetzung: Phase 16.

### 17.1 sbatch-Generator

Neuer Trainings-Graph-Node **`SlurmConfig`** (alternativ zum
`LocalConfig` für `TrainLoop.backend`). Felder:

- `partition` (Dropdown, befüllt aus `sinfo -h -o '%P'`)
- `time` (HH:MM:SS, default 04:00:00)
- `mem` (z.B. 32G)
- `cpus_per_task` (default 8)
- `gres` (z.B. `gpu:a100:1`)
- `account` (optional)
- `qos` (optional)
- `modules`: Liste von `module load …` (Multi-Select aus
  `module avail` Output, im Erst-Setup gecached)
- `pre_run_script`: Freitext (wird vor python ausgeführt — z. B.
  `export OMP_NUM_THREADS=8`)

Codegen-Snippet erweitert um Header:

```bash
#!/bin/bash
#SBATCH --job-name=mlforge-iris-mlp-a8f3
#SBATCH --partition=gpu-l40
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0
export OMP_NUM_THREADS=8

cd $SLURM_SUBMIT_DIR
<venv>/bin/python -u train.py
```

### 17.2 Job-Lifecycle

`submit_slurm(run_id) -> jobid`:
1. ssh: rsync Run-Dir wie 16.1
2. ssh: `cd <run_dir> && sbatch train.sbatch` → parse jobid
   aus `Submitted batch job 12345`
3. Schreibe jobid in `pid`-Datei (mit `slurm:` prefix damit Status-
   Check weiß: SLURM, nicht direkter PID)

Status-Polling: `squeue -j <jobid> --noheader -o '%T,%S,%e,%R'`
gibt State (PENDING/RUNNING/COMPLETED/FAILED/CANCELLED), Start-/End-
Zeit, Reason. Wenn `squeue` leer → Job nicht mehr in Queue →
`sacct -j <jobid> -o State,ExitCode --parsable2 --noheader` für
finalen State.

UI in der Run-Liste: PENDING-Pille mit ETA aus `--start-time`
(SLURM schätzt), RUNNING-Pille mit GPU-Anzeige aus events.jsonl,
COMPLETED/FAILED entsprechend.

### 17.3 Cancel + Resume

`scancel <jobid>` cancellt. Resume: aus letztem Checkpoint einen
neuen Run mit `resume_from=<checkpoint-path>` starten —
`TrainLoop`-Knoten bekommt optional `resume_from` Field.

### 17.4 Log-Streaming durch die Queue

Während PENDING gibt's keine Logs. Während RUNNING: `events.jsonl`
existiert sobald Python startet → tail via ssh wie in 15.2.

`slurm-<jobid>.out` wird per SLURM aktualisiert (buffered, evtl.
bis Job-Ende kaum sichtbar). Deshalb verlassen wir uns auf
`events.jsonl` + `print(..., flush=True)` im Training-Skript.

## Phase 18: Polish + Multi-Run

Letzte Phase macht's komfortabel:

- **Hyperparam-Sweeps**: Trainings-Graph-Knoten mit
  Range/Choice-Feld (`lr: [1e-2, 1e-3, 1e-4]`), startet pro
  Kombination einen Run. SLURM-Array-Jobs wenn verfügbar.
- **Auto-Tag von Runs** durch den Chat: „warum ist Run #47
  schlechter als #46?" → Claude diff'd die configs + metrics.
- **Export**: nach W&B / TensorBoard / CSV.
- **`best.pt`-Promotion**: bestes Modell aus N Runs nach
  `models/best/<name>.pt`, von wo aus es als Pretrained in eine
  neue Architektur-Pipeline kommt.

## Übergreifende Querschnittsthemen

### Resilienz — wirklich entkoppelt vom UI

Der harte Test für „läuft weiter wenn MLForge zu":

| Backend | Mechanismus | Recovery beim Re-Open |
|-|-|-|
| **local** | `setsid` + `nohup` + PID-File | `kill -0 <pid>` + `status`-File |
| **ssh-direct** | gleicher Trick remote | ssh + `kill -0` |
| **SLURM** | Job in der Queue | `squeue` + `sacct` |

In allen drei Fällen ist `events.jsonl` die single source of
truth — wenn die Datei wächst, läuft was, egal ob mit PID-File
oder ohne. Beim App-Open scant ein Background-Job alle
`experiments/runs/*` und re-attached an noch laufende Runs.

### Code-Generation-Konsistenz

Beide Generatoren (Architektur + Training) müssen reproduzibel
sein (CLAUDE.md invariant): gleicher Graph → gleicher Code. Keine
Date.now() im Generator. Trainings-Code referenziert das Modell
über relativen Pfad, nicht über sys.path-Hacks.

### Sidecar-Reuse

Der Phase-12b-torch-Sidecar (Shape-Inferenz + Datensätze) und
das Training nutzen die **gleiche venv**. Heißt: Trainings-Deps
(`tqdm`, ggf. `accelerate`, `safetensors`) kommen in den
gleichen `install()`-Step. Spart Bootstrap-Zeit auf HPC.

### Workspace-Backend bleibt agnostisch

Local-vs-Remote-Trennung läuft weiter über
`src/connections/backend.ts`. Trainings-Lifecycle-Calls (start,
status, cancel) bekommen ihren eigenen Dispatcher in
`src/training/backend.ts` analog. KEIN if-remote im Component-
Code.

### Datenschutz

Trainings-Logs (events.jsonl, stdout.log) können sensitive
Informationen enthalten (Sample-Inputs, Modell-Outputs).
Standardmäßig nur 127.0.0.1, nie nach außen. Wenn W&B-Logger
aktiv, expliziter Consent + Token-Storage in
`~/.config/mlforge/secrets` (nicht in `mlforge.project.json`).

## Phase-Reihenfolge — Empfehlung

```
13 (Foundation: Run-Format + Local-Executor + Liste)
↓
14 (Training-Graph + Codegen)
↓
15 (Live-UI: Charts + Tail)
↓
16 (Remote-Direct via ssh+nohup)
↓
17 (SLURM-Submit + Status-Polling)
↓
18 (Sweeps + Compare + Export)
```

13–15 sind nutzbar ohne 16/17 — local-only Trainings sind ein
sinnvoller MVP. 16/17 öffnen den HPC, 18 ist Komfort.

Jede Phase eigene Commit-Serie wie bei Phase 12 (a/b/c). Vor
13 sollten die Layer-Catalog-Lücken (oben in diesem File) gefüllt
sein, sonst trainiert man modelle die ein paar Schichten zu wenig
haben.
