# Handoff — affbind 4-Branch Dual-Encoder/Dual-Head Session (2026-06-19)

> **Zweck dieses Dokuments:** Vollständige Übergabe an eine spätere Claude-Session zum **Bug-Patchen**.
> Es dokumentiert (A) was gebaut wurde, (B) ALLE gefundenen Bugs mit Reproduktion + Ursache + Fix-Status,
> (C) die offenen Bugs, die noch im SpinoML-Code (nicht im Workspace) gepatcht werden müssen.
> Workspace: REMOTE HPC `zw93onug@login01.sc.uni-leipzig.de:~/spinoml_gnn`. Modell: `affbind.spinoml`.

---

## A. Was in dieser Session gebaut wurde

**Ziel des Users:** Modell auf `binder_decoy.csv` umbauen mit **4 simultanen Inputs** —
SMILES-Sequenz + 3D-Struktur des Liganden, dasselbe für das Protein. Vorhersage:
**Affinität (`value`) per Regression** + **Bindung (`label`) per Klassifikation**.

**Architektur `affbind.spinoml` (4 Inputs → 4 Encoder → Fusion → 2 Köpfe):**
- `llm5` Graph `lig_seq` → `llm7` Custom `SeqEncoder` (vocab 72) → [B,128]
- `ligand` Graph `lig_3d` → `lig_enc` Subgraph `LigandEncoder` (GCN-Stack) → [B,128]
- `llm6` Graph `prot_seq` → `llm8` Custom `SeqEncoder` (vocab 27) → [B,128]
- `protein` Graph `prot_3d` → `llm14` Subgraph `ProteinEncoder` (GCN-Stack) → [B,128]
- `merge` Concat(512) → `fc1`(512→256) → `act` ReLU → `llm9` Dropout(0.2) → `llm10`(256→128) → `llm11` ReLU
- **cls-Kopf:** `llm12`(128→2) → `out` Output  (AKTIV, Klassifikation auf `label`)
- **reg-Kopf:** `llm13`(128→1)  (gebaut, NICHT verdrahtet — Regression auf `value` "später")
- `llm1` Manifest-Node (gebunden an `binder_decoy.manifest`) → Kanten zu allen 4 Graph-Inputs.

**SeqEncoder (Custom, llm7/llm8):** DeepDTA-Stil. Nimmt PyG-Batch, `to_dense_batch` → [B,Lmax],
Embedding(padding_idx=0) → 3× Conv1d(kernel 7) → masked GlobalMaxPool → [B,out_dim].
Wichtig: `x.long().clamp(0, vocab-1)` vor dem Embedding (s. Bug #4).

**Daten (alle unter `datasets/`):**
- `ligand_3d/` — 4572 `.pt`, 3D-Molekülgraph (x=[N,8], pos=[N,3], Bindungskanten), key=`substrate_inchikey`
- `ligand_seq_tokens/` — 4572 `.pt`, SMILES-Token-Chain-Graph (x=[N,1] int), key=`substrate_inchikey`, vocab 66
- `protein_seq_tokens/` — AA-Token-Chain-Graph (x=[N,1] int, IDs 0–22), key=`uniprot`, vocab 27
- rankbind cache `cache_brenda_sabio_proteins` — prot_3d Residuen-Kontaktgraph (x=[N,33], edge_attr=[E,7]), key=`uniprot`
- `binder_decoy.manifest` — 4 Branches mit Keys `lig_seq/lig_3d/prot_seq/prot_3d`, target `label` (classification)

**Scripts in `agent/`:**
- `prep_ligand_branches.py` — baut ligand_3d + ligand_seq_tokens (lief als SLURM `22377045`, COMPLETED, 4571 echte 3D + 1 2D-Fallback, 0 Fehler)
- `convert_prot_seq_to_graph.py` — konvertierte 901 prot_seq `.pt` von plain-Tensor → PyG Data (s. Bug #2)
- `smoke_*.py` — Manifest-Pairing-Smoke (9398/9632 Zeilen = 97.6% volle 4-Branch-Coverage)
- `train_affbind.py` + sbatch — Trainingsskript (spiegelt den Training-Graph), GPU-Job

**Training-Graph (visuell):** Manifest-DatasetSource → Split(0.2,seed42) → DataLoader(32) →
TrainLoop(40ep); + ModelSource(affbind.spinoml), Loss(CrossEntropyLoss), Optimizer(AdamW lr1e-3 wd1e-4),
Scheduler(ReduceLROnPlateau patience4 γ0.5), Metric(accuracy,f1), EarlyStopping(val_loss patience8).

**Laufender Trainingslauf:** SLURM `22377455`, GPU RTX 2080 Ti auf `clara` (account scads), RUNNING.
Run-Ordner `experiments/runs/affbind-20260619-104617/`. Stand bei letzter Prüfung: ep5/40,
train 0.206 / val 0.218 / val_acc 90.4% / val_f1 0.845. Gesund, kein Overfit. 885.571 Params.

---

## B. Gefundene Bugs — mit Reproduktion, Ursache, Fix-Status

### Bug #1 — Manifest baute lig_3d aus SMILES statt aus 3D-Graphen  [GEFIXT im Workspace]
- **Symptom:** User bemerkte: "der 3D-Input zieht dieselben Sequenzen". Stimmte teils.
- **Ursache:** `binder_decoy.manifest` war alte Version: Branch `ligand` nutzte `kind: molecule`
  → baute Graph INLINE aus dem SMILES-String (2D, ohne `pos`), ignorierte das berechnete `datasets/ligand_3d/`.
  Außerdem Keys (`ligand/ligand_seq/protein/protein_seq`) ≠ Architektur-Branches (`lig_seq/lig_3d/prot_seq/prot_3d`),
  und referenziertes `smiles_seq_tokens/` war LEER.
- **Fix:** Manifest neu geschrieben — alle 4 Branches als `.pt`-`dir`-Quellen mit Keys passend zur Architektur,
  zeigend auf die real existierenden Verzeichnisse. Persistiert + zurückgelesen.
- **Lehre für künftige Sessions:** Manifest-Keys MÜSSEN exakt den Graph-Node `branch`-Params entsprechen,
  sonst matcht das Pairing nicht. `kind: molecule` = inline-2D, NICHT die vorberechneten 3D-Graphen.

### Bug #2 — prot_seq .pt waren plain Tensors statt PyG Data → Collate-Crash  [GEFIXT im Workspace]
- **Symptom:** Manifest-Collate brach mit `'TensorBatch' object has no attribute 'stores_as'`.
- **Ursache:** SpinoMLs `graph_collate` ruft für JEDE Branch `torch_geometric Batch.from_data_list` auf.
  Die alten prot_seq-Dateien waren reine 1D-Token-Tensoren, kein PyG `Data`.
- **Fix:** `agent/convert_prot_seq_to_graph.py` — alle 901 Dateien → `Data` Chain-Graph (x=[L,1], bidir chain edges).
- **Lehre:** Über das Manifest sind NUR PyG-Data/Graph-Branches möglich. Plain-Tensor-Branches gehen NICHT.
  (Bereits in decisions.md notiert; trat hier erneut auf, weil alte Dateien übrig waren.)

### Bug #3 — 3D-Embedding schlug fehl (alles 2D-Fallback)  [GEFIXT im Workspace]
- **Ursache:** eigenes `maxIterations`-Argument störte RDKit ETKDG.
- **Fix:** bewährter ETKDG-Aufruf + `useRandomCoords`-Fallback. Danach 4571/4572 echtes 3D.

### Bug #4 — IndexError im SeqEncoder beim Architektur-Validator (Custom #llm7)  [GEFIXT im Workspace]
- **Symptom:** Canvas zeigte `Custom #llm7 → IndexError: index out of range in self`. Echte Daten liefen aber.
- **Ursache:** Der Architektur-**Forward-Validator** speist SYNTHETISCHE Dummy-Daten in Graph-Inputs.
  Für Token-Branches erzeugt er zufällige `x`-Werte NICHT begrenzt auf den Vokabularbereich →
  `nn.Embedding(vocab,...)` bekommt Index ≥ vocab → IndexError.
- **Fix (Workspace-seitig):** beide SeqEncoder klemmen `x.long().clamp(0, self.vocab-1)` vor dem Lookup.
- **>>> POTENZIELLER SpinoML-BUG zum Patchen:** Der Dummy-Daten-Generator des Architektur-Validators
  sollte für Embedding/Token-Graph-Inputs IDs im gültigen Bereich [0, num_embeddings-1] erzeugen,
  statt beliebiger Floats/Ints. Sonst muss jeder User seinen Encoder manuell clampen. Siehe C-2.

### Bug #5 — Node-Positionen / Canvas-Layout ließen sich nicht via Tools ändern  [TEIL-WORKAROUND, SpinoML-Bug offen]
- **Symptom:** User: "im canvas hat sich nichts geändert" / "layout wieder unübersichtlich".
- **Befund:** `update_params({position})` schrieb nach `params.position`, die Canvas liest aber das
  **Top-Level-`position`**-Feld pro Node. Direktes Schreiben der `.spinoml`-Datei mit Top-Level-`position`
  funktioniert NUR, wenn der User danach das Modell NEU LÄDT (nicht speichert — Speichern überschreibt
  die Datei mit dem In-Memory-Layout). Auto-Layout beim Speichern stapelt Nodes ID-weise in 2 Spalten
  statt entlang der Datenfluss-Logik.
- **>>> SpinoML-BUGS zum Patchen:** siehe C-1 (kein Tool für Layout; Auto-Layout unbrauchbar;
  Save überschreibt externe Layout-Edits; keine Live-Reload der Positionen aus dem Modell-Graph).

### Bug #6 — Extern (sbatch) gestartete Runs erscheinen NICHT im GUI Run-Viewer  [SpinoML-Bug OFFEN — Hauptproblem]
- **Symptom:** User: "ich seh nichts in der GUI" für den laufenden Run.
- **Befund (faktisch verifiziert):**
  - `run.json` ist wohlgeformt; `read_run(run_id)` liest sie sauber (Epochen, Metriken, best_val_loss).
  - `list_runs` FINDET alle Runs unter `experiments/runs/`, liest `status` + `model_path`, zeigt aber
    `loss=—` / `epochs=?` — auch nach Ergänzen aller gängigen Feld-Alias-Namen in run.json.
  - → Loss/Epochen werden NICHT aus run.json gelesen. Der Torch-Sidecar listet Runs gar nicht;
    der Run-Viewer ist Frontend-Logik.
  - Keine Index-/Registry-Datei in `experiments/`. Kein GUI-nativer Referenz-Run zum Schema-Abgleich.
  - Zusätzlich: Live-Job überschreibt run.json jede Epoche → externe Schema-Patches haben keinen Bestand.
- **Vermutete Ursache:** Die GUI speist ihre Run-Liste aus ihrem EIGENEN "Train"-Start-Flow
  (interner Index/Watcher). Extern per sbatch angelegte Runs werden dort nie registriert.
  Mit Workspace-Tools ist dieser GUI-interne Index nicht von außen befüllbar.
- **>>> SpinoML-BUGS zum Patchen:** siehe C-3 (run.json-Schema dokumentieren/parsen; Run-Verzeichnis
  watchen statt internen Index; `device`-Feld korrekt setzen). DIES ist der wichtigste Patch-Punkt.

### Bug #7 (kosmetisch) — `run.json.device = "cpu"` obwohl GPU  [GEFIXT für nächstes Skript]
- Hartkodiertes Label im Config-Dict des train-Skripts; Job lief real auf RTX 2080 Ti. Skript korrigieren.

### Kleinere Umgebungs-Stolpersteine (kein Bug, für Reproduktion wichtig)
- Login-Node-Python hat KEIN pandas/rdkit/torch_geometric/sklearn → Skripte müssen unter
  `.spinoml/venv/bin/python` re-exec'en (String-Check auf `os.getcwd()`, wie im funktionierenden Test).
- `sklearn` fehlt in der venv → Accuracy/F1 mit numpy selbst rechnen.
- Login-Node shell cap ~2 min → Heavy compute (3D-Embed, Training) via SLURM.

---

## C. OFFENE SpinoML-Bugs zum Patchen (NICHT im Workspace lösbar — App-/Sidecar-Code)

> Der SpinoML-App/Frontend/Sidecar-Code liegt NICHT in diesem Workspace, daher hier nur präzise
> Problembeschreibung + Reproduktion + Lösungsskizze für die patchende Session.

### C-1. Canvas-Layout (aus Bug #5)
- **(a)** Es gibt kein MCP/Tool, um Node-Positionen der Canvas zu setzen. `update_params` schreibt nach
  `params.position`, die Canvas rendert aber Top-Level-`node.position`. → Tool ergänzen, das Top-Level-position
  setzt UND einen Live-Reload triggert.
- **(b)** Auto-Layout beim Speichern stapelt Nodes ID-weise in 2 Spalten (ignoriert Edges/Datenfluss).
  → durch ein DAG-Layout (z.B. layered/Sugiyama links→rechts entlang Topologie) ersetzen.
- **(c)** Speichern überschreibt extern editierte Top-Level-`position` mit dem In-Memory-Layout.
  → entweder externe Edits beim Fokus-Wechsel reloaden, oder Positionen nicht beim Save überschreiben.

### C-2. Architektur-Validator Dummy-Daten für Token/Embedding-Inputs (aus Bug #4)
- Der Forward-Validator erzeugt Dummy-`x` für Graph-Inputs ohne Rücksicht auf nachfolgende `nn.Embedding`.
- **Fix-Skizze:** Wenn ein Graph/Sequence-Input in ein Embedding mündet (oder als int/long typisiert ist),
  Dummy-IDs in `randint(0, num_embeddings)` erzeugen statt beliebiger Werte. Alternativ Dummy-IDs generell
  klein halten (z.B. 0). Verhindert IndexError ohne dass User manuell clampen müssen.

### C-3. GUI Run-Viewer zeigt extern erzeugte Runs nicht / nur teilweise  (HAUPTPROBLEM, aus Bug #6)
- **Reproduktion:** Run-Ordner mit gültiger `run.json` unter `experiments/runs/<id>/` anlegen (per sbatch,
  nicht über den GUI-"Train"-Button). → erscheint nicht im Viewer; `list_runs` zeigt `loss=—`/`epochs=?`.
- **Zu klären im App-Code:**
  1. Woher liest der Run-Viewer seine Liste? (interner Index/State vs. Verzeichnis-Scan)
  2. Welches exakte `run.json`-Schema parst `list_runs` für `loss`/`epochs`? (welche Keys?)
- **Fix-Skizze:**
  - Run-Viewer + `list_runs` sollten `experiments/runs/` per Verzeichnis-Scan + Filewatcher lesen,
    damit auch extern (sbatch) erzeugte Runs erscheinen und live aktualisieren.
  - Das erwartete `run.json`-Schema dokumentieren (ein JSON-Schema in repo ablegen), damit externe
    Trainingsskripte kompatibel schreiben können. Aktuell unbekannt, welche Keys für loss/epochs gelesen werden.
  - Robust gegen das Überschreiben jede Epoche (atomar schreiben: temp + rename).

### C-4. (verifizieren) Multi-Task / Dual-Head im visuellen Training-Graph nicht möglich
- Der visuelle Training-Graph erlaubt nur 1 Loss / 1 Target. Das Modell hat aber 2 Köpfe
  (cls `label` + reg `value`). Aktuell wird nur der cls-Kopf trainiert; das externe Skript könnte
  später MSE(value)+BCE/CE(label) gewichtet kombinieren.
- **Feature-Wunsch:** Training-Graph um mehrere (Loss, Target, Head)-Tripel + Gewichte erweitern.

---

## D. Nächste konkrete Schritte (für die nächste Session)
1. Trainingslauf `22377455` zu Ende beobachten: `slurm_status` + `read_run affbind-20260619-104617`
   (bzw. Job-Log `agent/…22377455.out`). Auf Plateau/EarlyStopping (val_loss 8 Ep. ohne Verbesserung) achten.
2. Falls gewünscht: Regression auf `value` aktivieren → reg-Kopf `llm13` verdrahten + eigenes
   Multi-Task-Skript (gewichtete MSE+CE), da der visuelle Graph nur 1 Loss kann (C-4).
3. SpinoML-App-Bugs C-1…C-3 im App-Repo patchen (dieser Workspace enthält den App-Code NICHT).
4. `device`-Feld im train-Skript korrekt aus `torch.cuda.is_available()` setzen (Bug #7).

## E. Schlüssel-Dateien (Pfade)
- Modell: `affbind.spinoml`
- Manifest: `datasets/binder_decoy.manifest`  ·  Tabelle: `datasets/binder_decoy.csv` (9632 Zeilen)
- Daten: `datasets/ligand_3d/`, `datasets/ligand_seq_tokens/`, `datasets/protein_seq_tokens/`, rankbind prot-cache
- Scripts: `agent/prep_ligand_branches.py`, `agent/convert_prot_seq_to_graph.py`, `agent/train_affbind.py` (+ sbatch), `agent/smoke_*.py`
- Runs: `experiments/runs/affbind-20260619-104617/run.json` (+ best.pt)
- Notizen: `notes/lab-notebook.md`, `notes/session-2026-06-19.md`, `notes/decisions.md`, DIESES Dokument
