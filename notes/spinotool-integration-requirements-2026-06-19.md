# SpinoTool — Integrations-, UX- & Konsistenz-Anforderungen + App-Bugs

> **Zweck:** Übergabe an eine spätere Claude-Session, die Zugriff auf das **SpinoTool-App-Repo**
> (Frontend + Sidecar + Chatbot-SDK-Glue) hat — NICHT nur auf diesen HPC-Workspace.
> Dieses Dokument bündelt die vom User am 2026-06-19 gemeldeten **Inkonsistenzen zwischen
> Chatbot und GUI** und die gewünschten **neuen Features**. Es ergänzt
> `handoff-2026-06-19-bugs-and-session.md` (dort stehen die modell-/datenseitigen Bugs);
> HIER geht es um die **Tool-/GUI-/Integrations-Ebene**.
>
> **Leitprinzip des Users (wörtlich sinngemäß):**
> „Es soll quasi alles im *Kosmos* von SpinoTool ablaufen. Der Chatbot soll zwar Zugriff
> auf das System haben, aber **vor allem IN SpinoTool arbeiten**, und ich sollte **alles über
> die GUI beobachten** können." → Jede Chatbot-Aktion muss sich in einem GUI-Artefakt
> spiegeln (Canvas-Node, Datei im Explorer, Run im Viewer, Job im Job-Panel). Kein „unsichtbarer"
> Seiteneffekt im Dateisystem, den die GUI nicht kennt.

---

## 0. Architektur-Kontext (was wo lebt)

- **Workspace** (HPC, `zw93onug@login01.sc.uni-leipzig.de:~/spinoml_gnn`): Datasets, Modelle
  (`*.spinoml`), `agent/`-Skripte, `experiments/runs/`, `notes/`. Hierauf hat der Chatbot via
  MCP-Tools (read/write/run_script/slurm) direkten Zugriff.
- **SpinoTool-App** (lokal, NICHT im Workspace): Electron/Webview-Frontend (3 Canvases:
  Architektur/Training/Daten), File-Explorer, Datasets-Tab, Run-Viewer, Sidecar-Anbindung.
- **Sidecars** (conda `mlforge-dev`): Torch-Sidecar (Shape-Inference/Smoke) + Chatbot-SDK-Bridge.
- **Kernproblem aller u.g. Bugs:** Chatbot-Aktionen treffen das **Dateisystem/Modell-JSON**,
  aber die **GUI-State-Stores** (canvasdoc-Layout, Explorer-Watcher, Run-Index, Sidecar-Auswahl)
  werden nicht invalidiert/neu gelesen → GUI und Wahrheit driften auseinander.

---

## 1. GEMELDETE BUGS / INKONSISTENZEN

### BUG-A — Canvas aktualisiert Node-Positionen nicht zuverlässig, wenn der Bot Nodes verschiebt
- **Symptom:** Bot setzt Positionen (per `update_params` und später per direktem Schreiben der
  `position`-Felder in `affbind.spinoml`), aber die Canvas zeigt weiter das alte/zerfallene Layout.
- **Beobachtete Ursachen (in dieser Session empirisch ermittelt):**
  1. `update_params` schrieb Positionen nach `params.position`, die Canvas liest aber das
     **Top-Level-`position`**-Feld pro Node → kein Effekt.
  2. Selbst nach Schreiben des Top-Level-`position` zeigt die Canvas das alte Layout, weil sie
     ihren **eigenen In-Memory-canvasdoc-Layout-State** hält und die Datei nicht live nachlädt.
  3. Auto-Layout der Canvas stapelt beim **Speichern** Nodes stumpf nach ID-Reihenfolge in
     2 Spalten und überschreibt damit ein vom Bot geschriebenes sauberes Layout.
- **Fix-Richtung (App):**
  - Ein MCP-Tool / Sidecar-Event `canvas.layout.set(nodeId → {x,y})` einführen, das den
    **canvasdoc-Store** mutiert (nicht nur die Datei) und ein Re-Render triggert.
  - File-Watcher auf `*.spinoml`: bei externer Änderung Modell **+ Layout** neu in den Store laden
    (mit „externe Änderung erkannt — neu laden?"-Hinweis statt stiller Divergenz).
  - Auto-Layout beim Speichern NICHT erzwingen, wenn bereits gültige Positionen existieren.
  - Optional: ein deterministischer **„Auto-Anordnen nach Datenfluss"**-Button (Topo-Sort
    Links→Rechts, Branches als Zeilen) — das wollte der Bot faktisch tun, kann es aber nicht.

### BUG-B — Trainings-/Run-Dateien in `experiments/` erscheinen NICHT im File-Explorer und nicht im Run-Viewer
- **Symptom:** sbatch-gestarteter Lauf schreibt `experiments/runs/affbind-20260619-104617/run.json`
  (+ `best.pt`). `read_run`/`list_runs` (MCP) finden ihn, aber **die GUI zeigt ihn nicht** — weder
  im File-Explorer noch im Run-Viewer. User kann den laufenden Run nicht sehen.
- **Beobachtete Ursachen:**
  1. **File-Explorer** filtert/zeigt `experiments/` nicht (oder watcht den Pfad nicht) → für den
     User „unsichtbar". Er kann die Datei gar nicht öffnen.
  2. **Run-Viewer** speist seine Liste vermutlich aus dem **GUI-internen Train-Start-Flow**
     (interner Index/Watcher), in den **extern per sbatch** angelegte Runs nie eingetragen wurden.
  3. **Schema-Drift:** `list_runs` liest `status`/`model_path`, aber NICHT `best_val_loss`/`epochs`
     aus der vom Bot geschriebenen `run.json` (zeigt `loss=—`, `epochs=?`) → der striktere
     GUI-Viewer rendert deshalb evtl. gar nichts. (Selbst nach Hinzufügen aller Feld-Aliasse blieb
     `loss=—` → Loss/Epochen werden offenbar NICHT aus `run.json` gelesen.)
- **Fix-Richtung (App) — das ist das HAUPTPROBLEM:**
  - **Run-Discovery auf Dateisystem umstellen:** Run-Viewer + `list_runs` sollen `experiments/runs/*/run.json`
    **scannen** (Watcher), statt sich auf einen GUI-internen Index zu verlassen. Dann erscheinen
    auch extern (sbatch) gestartete Läufe automatisch.
  - **`run.json`-Schema offiziell dokumentieren** (ein JSON-Schema im Repo ablegen) und Viewer +
    `list_runs` daran ausrichten: kanonische Felder `best_val_loss`, `epochs_run`/`epochs`,
    `history[]` mit `{epoch, train_loss, val_loss, val_acc, val_f1, lr}`, `status`, `model_path`,
    `device`. Der Chatbot kann dann garantiert kompatible Dateien schreiben.
  - **File-Explorer:** `experiments/` standardmäßig sichtbar/expandierbar machen (oder einen
    dedizierten „Runs"-Tree-Knoten), mit Live-Watcher.

### BUG-C — Data-/Manifest-Inputs sind in den Input-Nodes „falsch", werden aber erkannt; Sidecar sagt „noch keine Auswahl"
- **Symptom:** Die Graph-Input-Nodes referenzieren ein Dataset/Manifest + Branch; das Pairing
  funktioniert (Smoke-Test grün, 9398/9632 Zeilen), aber in der **Sidecar-Auswahl** steht „es gibt
  noch keine Auswahl" — d.h. die GUI-Dropdowns für Dataset/Branch sind leer/nicht synchron mit dem,
  was der Bot in die Node-Params geschrieben hat.
- **Beobachtete Ursachen:**
  1. **Zwei konkurrierende Konventionen:** (a) Manifest **inline** in jedem Graph-Node
     (`dataset`+`branch` in params, ältere Form) vs. (b) dedizierter **Manifest-Node**, der per
     Kante die typed Inputs speist (aktuelles Datenmodell laut Memory „manifest-node-model").
     Der Bot hat beide gemischt → GUI-Auswahl-Logik erkennt den vom Bot gesetzten Zustand nicht
     als „gültige Auswahl".
  2. Die Sidecar/GUI-Auswahl wird vermutlich nur durch **User-Interaktion im Dropdown** befüllt,
     nicht aus den Node-Params rückwärts rekonstruiert → vom Bot gesetzte Werte „existieren", sind
     aber im UI-State nicht als Selection registriert.
  3. **Branch-Sampling für Nicht-Graph-Branches** war laut Memory ohnehin noch offen (Sidecar).
- **Fix-Richtung (App):**
  - **Eine** Konvention kanonisieren (Manifest-Node-Modell) und die GUI so bauen, dass sie die
    aktive Auswahl **aus den Node-Params/Kanten rekonstruiert** (Single Source of Truth = Modell-JSON),
    statt aus separatem UI-State. Wenn ein Manifest-Node mit gültigem `.manifest` verbunden ist,
    müssen die Branch-Dropdowns der Inputs automatisch befüllt + als „ausgewählt" markiert sein.
  - Validierung im UI: „Manifest gebunden, Branches X/Y/Z aufgelöst, N Zeilen vollständig" als
    sichtbarer Status am Manifest-Node (statt „keine Auswahl").

### BUG-D — Generelle Inkonsistenz Chatbot ↔ Workspace ↔ GUI
- **Symptom (Sammelposten):** Der Bot ändert Dateien/Modell, die GUI weiß nichts davon, bis
  manuell neu geladen wird; teils überschreibt ein GUI-Speichern wiederum Bot-Änderungen.
- **Fix-Richtung (App):** **Bidirektionale Sync-Schicht.** Modell-JSON / Dataset-Ordner /
  `experiments/` als **Single Source of Truth** mit File-Watchern; jede Chatbot-Mutation feuert
  ein Sidecar-Event, das den passenden GUI-Store invalidiert + neu lädt; Konflikt-Erkennung
  (Datei extern geändert) statt stillem Überschreiben. Ziel: **kein** manuelles „schließen & neu
  öffnen" mehr nötig.

---

## 2. GEWÜNSCHTE NEUE FEATURES

### FEAT-1 — Alles im „Kosmos" von SpinoTool / Chatbot arbeitet sichtbar IN der GUI
- **Anforderung:** Der Chatbot soll Systemzugriff behalten, aber jede Aktion muss sich in einem
  **GUI-Artefakt** materialisieren, das der User live beobachtet. Keine „unsichtbaren" FS-Effekte.
- **Umsetzung:** siehe BUG-D Sync-Schicht + pro Tool ein GUI-Spiegel:
  - `write_file`/`download_to_datasets` → Datei poppt im Explorer/Datasets-Tab auf (existiert teils schon).
  - `run_script` → erscheint im **Job-/Terminal-Panel** mit Live-Log (siehe FEAT-2).
  - Architektur-/Training-/Daten-Mutationen → Canvas-Re-Render (siehe BUG-A).

### FEAT-2 — Live-Ansicht laufender SLURM-Jobs in der GUI
- **Anforderung:** Ein **Job-Panel**, das laufende/historische SLURM-Jobs zeigt: Job-ID, Name,
  Status (PENDING/RUNNING/COMPLETED/FAILED), Partition/Node, **Live-stdout/stderr-Tail**, und
  Verknüpfung zum zugehörigen Run (`experiments/runs/...`).
- **Umsetzung:**
  - Sidecar pollt `squeue`/`sacct` für die Jobs des Users + tail't die `--output`-Logdatei.
  - Pro Job eine Karte mit Live-Log-Stream (gleiche SSE-Heartbeat-Technik wie Chat, wg.
    WebKitGTK-Buffering — siehe Memory „webview-sse-buffering").
  - Direkter Sprung Job → Run-Viewer (Lernkurve) sobald `run.json` existiert.
  - Aktionen: `scancel` aus der GUI, Log in Editor öffnen.

### FEAT-3 — Auto-Modus für den Chatbot
- **Anforderung:** Ein Schalter, der den Chatbot eine Aufgabe **autonom** durchziehen lässt
  (mehrere Tool-Schritte ohne Einzel-Bestätigung), statt jeden `run_script` per Klick zu bestätigen.
- **Umsetzung / Design-Fragen für die nächste Session:**
  - Toggle „Auto-Modus" im Chat-Panel; im Auto-Modus werden `run_script`-Freigaben automatisch
    erteilt (mit klarer optischer Kennzeichnung + jederzeit „Stop").
  - **Sicherheitsgrenzen:** weiterhin auf den Workspace beschränkt; destruktive Aktionen
    (Löschen, Überschreiben großer Dateien) evtl. trotzdem bestätigen oder per Allow/Deny-Liste.
  - Achtung Memory „chatbot-sdk-tool-gating": `bypassPermissions` ließ in der Vergangenheit
    eingebaute Bash/Write-Tools durch → beim Auto-Modus genau auf die kuratierten MCP-Tools
    beschränken, NICHT generisches bypass nutzen.
  - Fortschritts-/Stop-UI + Schritt-Log, damit der User den Auto-Lauf in der GUI mitverfolgt.

### FEAT-4 — Ausführliche Doku als Standard, per Button abschaltbar
- **Anforderung:** Die kontinuierliche, ausführliche Reproduzierbarkeits-Doku (record_step →
  `notes/lab-notebook.md`) soll **standardmäßig AN** sein, aber über einen **GUI-Button**
  (Toggle) abschaltbar (z.B. für schnelles Experimentieren ohne Notebook-Spam).
- **Umsetzung:**
  - Toggle „Ausführliche Doku" (default ON) im Chat-/Settings-Panel; State an den Chatbot-System-
    prompt/Tool-Gating durchreichen (bei OFF: record_step/record-Verpflichtung aussetzen, nur auf
    explizite Anfrage dokumentieren).
  - Optional Stufen: „Aus / Kompakt / Ausführlich".
  - Notebook in der GUI sichtbar/rendern (Markdown-Viewer im Notes-Tab) + Button „Eintrag hinzufügen".

---

## 3. PRIORISIERUNG (Vorschlag für die nächste Session)

1. **BUG-B (Run-Discovery + Explorer-Sichtbarkeit von `experiments/`)** — Hauptproblem, blockiert
   das Beobachten von Trainings. FS-basierte Run-Discovery + `run.json`-Schema fixen.
2. **BUG-D / FEAT-1 (Sync-Schicht)** — Fundament; behebt BUG-A, BUG-C teils gleich mit.
3. **FEAT-2 (Live-SLURM-Panel)** — hoher User-Wert, baut auf der Sync-/Sidecar-Event-Schicht auf.
4. **BUG-A (Canvas-Layout-Tooling)** — Layout-Set-Tool + kein Auto-Layout-Overwrite.
5. **BUG-C (Manifest-Auswahl aus Modell rekonstruieren)** — eine Konvention kanonisieren.
6. **FEAT-3 (Auto-Modus)** + **FEAT-4 (Doku-Toggle)** — UX-Schalter, abschließend.

---

## 4. WICHTIG: Repo-Zugriff nötig
Alle BUG-A…D und FEAT-1…4 betreffen **SpinoTool-App-/Sidecar-/Frontend-Code, der NICHT in
diesem HPC-Workspace liegt**. Diese Session kann sie daher nur präzise spezifizieren, nicht
patchen. Die nächste Session braucht das **SpinoTool-App-Repo** (lokal) als Workspace.
Referenz-Artefakte aus DIESEM Workspace zum Abgleich:
- vom Bot geschriebene `run.json`: `experiments/runs/affbind-20260619-104617/run.json`
- Modell mit Layout-Feldern: `affbind.spinoml`
- Manifest (kanonische 4-Branch-Form): `datasets/binder_decoy.manifest`
- frühere Bug-Doku: `notes/handoff-2026-06-19-bugs-and-session.md`

---

### BUG-E — Datenaufbereitung läuft komplett am Daten-Canvas vorbei (kein `.spinodata` erzeugt)
- **Symptom:** In dieser Session wurde **real Daten preprocessiert** — aber der **Daten-Canvas
  blieb durchgehend leer** (`{nodes: [], edges: []}`). Eine GUI-Beobachterin sieht von der
  gesamten Datenpipeline **nichts**; es existiert kein Daten-Canvas-Artefakt, das den Ablauf
  dokumentiert oder reproduzierbar macht. Genau der Verstoß gegen das Leitprinzip („alles im
  Kosmos von SpinoTool, alles über die GUI beobachtbar").
- **Was faktisch preprocessiert wurde (alles via `agent/`-Skripte + SLURM, NICHT via Canvas):**
  1. **Ligand-3D-Graphen** — `datasets/binder_decoy.csv` (9632 Zeilen) → dedupliziert auf
     **4572 eindeutige Liganden** (per `substrate_inchikey`, SMILES-Spalte `substrate_smiles_canon`)
     → RDKit ETKDG-3D-Konformer → PyG-Graph mit `pos` → `datasets/ligand_3d/<inchikey>.pt`
     (4571 echtes 3D, 1× 2D-Fallback, 0 Fehler). Skript: `agent/prep_ligand*.py`, SLURM `22377045`.
  2. **Ligand-SMILES-Token-Graphen** — derselbe Lauf → `datasets/ligand_seq_tokens/<inchikey>.pt`
     (PyG-Chain-Graph, vocab≈66). 4572 Dateien.
  3. **prot_seq-Konvertierung** — `datasets/protein_seq_tokens/*.pt` waren reine 1D-Token-Tensoren
     (alte Form) → zu PyG-Chain-Graphen (`x=[L,1]`, bidirektionale Kanten) konvertiert. Skript:
     `agent/convert_prot_seq_to_graph.py` (901 Dateien).
  4. **Manifest** — `datasets/binder_decoy.manifest` mehrfach umgeschrieben (4-Branch-Form).
- **Beobachtete Ursachen:**
  1. Der Chatbot hat den **direkten Skript-Weg** (`write_file`+`run_script`/sbatch) genommen, weil
     (a) RDKit-3D-Embedding + SLURM jenseits dessen liegt, was typed Daten-Canvas-Nodes aktuell
     abdecken, und (b) es kein Tool gibt, das eine Skript-Pipeline **rückwirkend als Daten-Canvas-
     Graph materialisiert**. Ergebnis: voll reproduzierbar im Lab-Notebook, aber **unsichtbar in der GUI**.
  2. Der Daten-Canvas kompiliert zu **einer linearen pandas-Pipeline (ein geteiltes `df`)** — der
     RDKit/SLURM-Schwergewichts-Schritt (4572 Moleküle, GPU/Cluster) passt nicht ins
     Login-Node-`run_script`-Modell des Canvas und wurde deshalb umgangen.
- **Fix-Richtung (App):**
  - **Daten-Canvas und tatsächliche Datenaufbereitung koppeln:** Wenn der Chatbot Daten
    preprocessiert, soll **immer** ein Daten-Canvas-Graph (`.spinodata`) entstehen — entweder
    (a) der Bot baut die Pipeline mit `add_data_node`/`connect_data_nodes` (typed Nodes:
    TableSource → SmilesToGraph(embed_3d) → WriteDataset) UND führt sie aus, oder (b) ein
    `CustomScript`-Node kapselt das Schwergewichts-Skript, sodass der **Canvas die Single Source
    of Truth** ist und der Explorer/Datasets-Tab das Ergebnis live zeigt.
  - **SLURM-fähige Daten-Nodes:** dem Daten-Canvas einen Ausführungsmodus geben, der einen
    Schritt (z.B. SmilesToGraph über 4572 Moleküle) als sbatch-Job startet (statt nur Login-Node),
    mit Anbindung ans Live-Job-Panel (FEAT-2). Dann ist auch schwere Datenaufbereitung GUI-sichtbar.
  - **Reverse-Materialisierung (optional):** ein Tool/Feature, das ein vorhandenes `agent/`-Prep-
    Skript als Daten-Canvas-`CustomScript`-Node importiert, damit nachträglich ein GUI-Artefakt
    entsteht.
  - **Konsistenz-Hinweis im Chatbot-Prompt:** Datenaufbereitung soll bevorzugt über den
    Daten-Canvas laufen (typed Nodes wo möglich, sonst CustomScript), NICHT über freie `agent/`-
    Skripte ohne Canvas-Spiegel — analog zu FEAT-1.

**Priorisierung:** einordnen direkt nach BUG-C bzw. zusammen mit FEAT-1/BUG-D (Sync-Schicht),
da es dasselbe Grundprinzip betrifft: jede Chatbot-Aktion braucht ein GUI-Artefakt. Konkret
sollte der Daten-Canvas die **dritte Säule** der FS-basierten Discovery werden (neben Modell-JSON
und `experiments/`-Runs).


---

### BUG-F — Canvas unterscheidet Sequenz-Input (Token-Chain-Graph) nicht vom echten Geometrie-Graph; beide sind „Graph"-Nodes
- **Symptom (User-Wahrnehmung):** „Die Sequence-Nodes haben auch einen Graph-Input-Node, und die
  Graph-Branches sind laut Manifest auch nur die Sequenzen — irgendwas stimmt am Modell nicht."
  Auf der Canvas sehen **alle vier Inputs identisch aus** (`Graph`-Node), obwohl zwei davon
  Token-Sequenzen und zwei davon 3D-Geometrie-Graphen sind → der User kann Seq- und 3D-Branch
  visuell nicht auseinanderhalten und vermutet einen Modellfehler.
- **Faktenlage (per `agent/inspect_branches.py` aus den echten `.pt` verifiziert — Modell ist KORREKT):**

  | Branch | x | Geometrie | Typ |
  |---|---|---|---|
  | lig_seq | `[N,1]` int64 (Token-IDs 23–29) | chain-edges | SMILES-Token-Sequenz |
  | lig_3d | `[N,8]` float32 | `pos=[N,3]` | echter 3D-Molekülgraph |
  | prot_seq | `[N,1]` int64 (Token-IDs 0–20) | chain-edges | AA-Token-Sequenz |
  | prot_3d | `[N,33]` float32 | `edge_attr=[N,7]` | Residuen-Kontaktgraph |

  Die vier Branches sind also **strukturell verschieden** (int-Token-Kette vs. float-Geometrie).
  Das Trainingsergebnis (93,9 % Acc / 0,905 F1 auf der Minderheitsklasse, Run
  `affbind-20260619-104617`) beweist, dass alle vier unterschiedliches Signal tragen — wären
  Seq und 3D identisch, gäbe es kein Zusatzsignal.
- **Ursache:** SpinoML speichert eine **Token-Sequenz als PyG-Chain-Graph** (`x` = Token-IDs
  `[N,1]`), weil das **Manifest-Collate** pro Branch `Batch.from_data_list` aufruft und PyG-`Data`
  braucht. Der dedizierte **`Sequence`-Input-Node-Pfad ist im Sidecar für Manifest-Sampling noch
  nicht implementiert** (Memory „manifest-node-model": „non-graph branch sampling still pending").
  Deshalb laufen aktuell alle vier Branches als `Graph`-Nodes — funktional korrekt (der
  `SeqEncoder` baut via `to_dense_batch` die Sequenz zurück und nutzt Conv1d), aber auf der Canvas
  **nicht unterscheidbar**.
- **Fix-Richtung (App):**
  - **Visuelle Differenzierung:** Graph-Nodes nach Inhalt typisieren/labeln — z.B. ein
    „Sequence (chain-graph)"-Badge wenn `x` int + letzte Dim == 1 + reine Chain-Edges, vs.
    „3D Graph" wenn float-Features + `pos`/`edge_attr`. Mindestens die `name`-/`shape`-Info
    (`[L,1]` vs. `[N,F]`) prominent am Node anzeigen.
  - **`Sequence`-Input-Node-Pfad im Sidecar fertigstellen** (Manifest-Sampling für Nicht-Graph-
    Branches), damit Token-Sequenzen als echte `Sequence`-Nodes (statt als getarnte Graphen)
    modelliert werden können — dann ist die Modalität schon am Node-Typ ablesbar.
  - **Manifest-Branch-Inspektion in der GUI:** pro Branch anzeigen, was die `.pt` real enthalten
    (dtype, x-Shape, hat pos/edge_attr) — damit „Sequenz vs. 3D" sichtbar wird, statt nur „Graph".

**Einordnung:** Reine GUI-Darstellungslücke (kein Modell-/Datenfehler), aber wiederkehrende
User-Verwirrung → zusammen mit BUG-C (Manifest/Input-Darstellung) behandeln.


---

### BUG-B — HARTER BEWEIS (2026-06-19, nach Run-Abschluss)
User-Wunsch: den fertigen Run `affbind-20260619-104617` in der **GUI-Run-Übersicht** sehen
(nicht als CSV). Test durchgeführt: `run.json` mit ALLEN plausiblen Feld-Aliassen gleichzeitig
geschrieben — `best_val_loss`, `loss`, `val_loss`, `best_val_acc`, `val_acc`, `epochs`,
`epochs_run`, `best_epoch`, zusätzlich ein verschachteltes `metrics:{best_val_loss,...,epochs}`
sowie `model` neben `model_path`. Ergebnis von `list_runs` UNVERÄNDERT:
`affbind-20260619-104617 [completed] loss=— model=affbind.spinoml epochs=?`.
→ `status` + `model_path` WERDEN aus `run.json` gelesen (leerer Ordner `…104330` zeigt
`model=?`/unknown), aber **`loss`/`epochs` NICHT** — egal welche Feldnamen. Damit ist bewiesen:
Loss/Epochen (und die GUI-Run-Liste) stammen aus einem **App-internen Index/Parser**, der NICHT
aus `experiments/runs/*/run.json` befüllt wird und NICHT vom Workspace aus beschreibbar ist.
**Konsequenz für die App-Session:** Run-Discovery zwingend auf FS-Scan von `experiments/runs/`
umstellen (Watcher), `run.json`-Schema im Repo dokumentieren, und extern (sbatch) gestartete
Runs in denselben Index aufnehmen wie GUI-gestartete. Bis dahin gibt es KEINEN Workspace-seitigen
Weg, einen sbatch-Run in die GUI-Übersicht zu bekommen.
