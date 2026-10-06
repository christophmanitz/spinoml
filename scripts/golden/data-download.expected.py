# Auto-generiert aus dem Daten-Canvas (SpinoML).
# Reproduzierbare Daten-Pipeline — läuft im Workspace-Root.
# Schwere Schritte (Graph-Bau, große Downloads) ggf. via SLURM ausführen.

import pandas as pd

df = None  # wird vom Quell-Knoten gesetzt

# ── TableSource (src) ──
df = pd.read_csv('datasets/raw.csv')
print('loaded', df.shape, 'from', 'datasets/raw.csv')

# ── DownloadColumn (dl) ──
import os, urllib.request
os.makedirs('datasets/raw', exist_ok=True)
_paths = {}
for _id in df['uniprot'].dropna().astype(str).unique():
    _url = 'https://files.rcsb.org/download/{id}.pdb'.replace('{id}', _id)
    _dest = os.path.join('datasets/raw', '{id}.pdb'.replace('{id}', _id))
    if not os.path.exists(_dest):
        try:
            urllib.request.urlretrieve(_url, _dest); print("downloaded", _dest)
        except Exception as _e:
            print("FAILED", _id, _e)
    _paths[_id] = _dest
df['file_path'] = df['uniprot'].astype(str).map(_paths)

# ── WriteDataset (out) ──
import os
os.makedirs(os.path.dirname('datasets/processed.csv') or '.', exist_ok=True)
df.to_csv('datasets/processed.csv', index=False)
print('wrote', 'datasets/processed.csv')

print('pipeline done')
