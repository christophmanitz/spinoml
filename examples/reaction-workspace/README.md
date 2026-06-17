# Beispiel-Workspace: Reaktions-Affinität (Dual-Encoder)

Ein minimaler, lauffähiger Workspace, der zeigt, wie du **gepaarte Graph-Daten**
(Ligand + Enzym pro Reaktion) für einen Dual-Encoder anlegst — genau die Struktur
deines `reactionDataFiltering`-Setups, nur klein (10 Reaktionen).

## Layout

```
datasets/
  reactions.csv                       ← die Tabelle: eine Zeile = eine Reaktion
  graphs/proteins/                    ← 3D-Struktur-Graphen (graphein-Output), 1 .pt pro Enzym
      AF-P0A6F5-F1-model_v4.pt            (AlphaFold-Dateiname; UniProt-ID steckt drin)
      AF-P9WPE7-F1-model_v4.pt
      AF-Q9X0E6-F1-model_v4.pt
      AF-P00918-F1-model_v4.pt
  rxn.manifest                        ← die KLAMMER: koppelt Tabelle ↔ Graphen ↔ Target
```

`reactions.csv`:

| smiles | uniprot | affinity | ec_class |
|--------|---------|----------|----------|
| CCO | P0A6F5 | 5.42 | 0 |
| CC(=O)Oc1ccccc1C(=O)O | P9WPE7 | 7.31 | 1 |
| … | … | … | … |

Die Protein-`.pt` sind PyG-`Data`-Objekte mit `x` `[n_residues, 12]` (physikochemische
Features **+ 3D-Koordinaten** in den letzten 3 Spalten), `edge_index` (Kontaktkarte
< 8 Å) und `pos` `[n_residues, 3]`.

## Die Manifest-Datei erklärt

`rxn.manifest` ist reines JSON. Jeder Schlüssel:

```jsonc
{
  "table": "reactions.csv",        // Pfad RELATIV zum Manifest
  "pairs": {
    // Jeder Branch = ein Encoder-Input deines Modells.
    "ligand": {
      "column": "smiles",          // welche CSV-Spalte
      "kind": "molecule"           // → Graph LIVE aus SMILES bauen (RDKit), kein Verzeichnis nötig
    },
    "protein": {
      "column": "uniprot",         // welche CSV-Spalte (hier UniProt-IDs)
      "dir": "graphs/proteins",    // Verzeichnis mit den .pt (relativ zum Manifest)
      "match": "contains",         // ID muss im DATEINAMEN vorkommen: P0A6F5 → AF-P0A6F5-F1-model_v4.pt
      "ext": ".pt"
    }
  },
  "target": {
    "column": "affinity",          // welche Spalte vorhergesagt wird
    "type": "regression"           // "regression" (float) oder "classification" (int-Klasse)
  }
}
```

### Auflösungs-Varianten (je Branch frei wählbar)

| willst du … | schreib im Branch |
|---|---|
| Graph live aus einer SMILES-Spalte | `"kind": "molecule"` (kein `dir`) |
| ID → Datei, **ID im Dateinamen** | `"dir": "...", "match": "contains", "ext": ".pt"` |
| ID → exakt `dir/<id>.pt` | `"dir": "...", "match": "exact", "ext": ".pt"` |
| Spalte enthält direkt den Pfad | nur `"column": "..."` (kein `dir`) |

### Auf EC-Klassifikation umstellen

Nur das `target` tauschen:

```json
"target": { "column": "ec_class", "type": "classification" }
```

…und im Modell den letzten `Linear`-Head auf `out_features = <Anzahl EC-Klassen>` setzen.

## So öffnest du ihn in SpinoML

1. **File → Workspace öffnen** → diesen Ordner (`reaction-workspace`) wählen.
2. Links **Datasets** → `rxn.manifest` anklicken → du siehst 10 Paare + die Slots
   (`ligand.x`, `protein.x [n,12]`, `protein.pos [n,3]`, `target`).
3. **Templates → „Dual-Encoder GNN (Ligand + Protein)"** laden.
4. `ligand`-Graph-Knoten → Inspector → **dataset** = `rxn.manifest`, **Branch** = `ligand`.
   `protein`-Graph-Knoten → **dataset** = `rxn.manifest`, **Branch** = `protein`.
5. **in_channels** der ersten GCNConv anpassen: Ligand = **5**, Protein = **12**
   (steht in den Slots `ligand.x` / `protein.x`).
6. **SmokeTest** (im Dataset-Detail) bzw. **Explain ▶** — fährt echte gepaarte Daten durch.

Die generierte `.py` (Code-Preview) ist dein trainierbares Modell:
`forward(self, ligand, protein)` → ein Affinity-Score pro Paar.
