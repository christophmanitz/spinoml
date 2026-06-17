

## 2026-06-16 — Training des Dual-Encoders (Affinity)

Der visuelle Trainings-Graph (Phase 13/14) unterstützt nur tabellarische
Datensätze, nicht den `rxn.manifest` (Ligand-Molekülgraph + Protein-.pt).
Lösung: eigenständiges Skript `train_dualencoder.py` im Workspace-Root.

- Featurisierung identisch zum Sidecar: Ligand = 5 RDKit-Atom-Features
  (AtomicNum, Degree, FormalCharge, IsAromatic, TotalNumHs);
  Protein = `.pt` Data(x=[N,12], edge_index), Match per "uniprot im Dateinamen".
- Modell inline mit KORRIGIERTEN in_channels (Ligand 5, Protein 12) — die
  generierte `models/dualencoder.py` ist veraltet (noch 9/20).
- MSELoss, Adam(lr=1e-3), ReduceLROnPlateau, 80/20-Split, seed=42.
- Checkpoint → experiments/dualencoder/best.pt, Verlauf → history.json.
- Läuft durch; Train-Loss konvergiert. Val ist instabil/negatives R², weil
  der Demo-Datensatz nur 10 Zeilen (2 Val) hat — kein Algorithmus-Bug.
