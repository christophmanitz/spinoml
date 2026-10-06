# Auto-generiert aus dem Daten-Canvas (SpinoML).
# Reproduzierbare Daten-Pipeline — läuft im Workspace-Root.
# Schwere Schritte (Graph-Bau, große Downloads) ggf. via SLURM ausführen.

import pandas as pd

df = None  # wird vom Quell-Knoten gesetzt

# ── TableSource (src) ──
df = pd.read_csv('datasets/raw.csv')
print('loaded', df.shape, 'from', 'datasets/raw.csv')

# ── CustomScript (cs): dedup ──
df = df.drop_duplicates()
print("after dedup", df.shape)

# ── RenameColumns (ren) ──
df = df.rename(columns={'old': 'new'})

# ── WriteDataset (out) ──
import os
os.makedirs(os.path.dirname('datasets/processed.csv') or '.', exist_ok=True)
df.to_csv('datasets/processed.csv', index=False)
print('wrote', 'datasets/processed.csv')

print('pipeline done')
