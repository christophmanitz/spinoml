# Auto-generiert aus dem Daten-Canvas (SpinoML).
# Reproduzierbare Daten-Pipeline — läuft im Workspace-Root.
# Schwere Schritte (Graph-Bau, große Downloads) ggf. via SLURM ausführen.

import pandas as pd

df = None  # wird vom Quell-Knoten gesetzt

# ── TableSource (src) ──
df = pd.read_csv('datasets/pdbs.csv')
print('loaded', df.shape, 'from', 'datasets/pdbs.csv')

# ── StructureToGraph (s2g) ──
from Bio.PDB import PDBParser
import torch, os
from torch_geometric.data import Data
_parser = PDBParser(QUIET=True)
_graphs = []
for _p in df['file_path'].dropna().astype(str):
    try:
        _s = _parser.get_structure("s", _p)
    except Exception as _e:
        print("skip", _p, _e); continue
    _ca = [a for a in _s.get_atoms() if a.get_id() == "CA"]
    if not _ca:
        continue
    _coords = torch.tensor([list(a.get_coord()) for a in _ca], dtype=torch.float)
    _d = torch.cdist(_coords, _coords)
    _mask = (_d < 8) & (_d > 0)
    _edge_index = _mask.nonzero(as_tuple=False).t().contiguous()
    _graphs.append(Data(x=torch.ones((_coords.shape[0], 1)), pos=_coords, edge_index=_edge_index))
os.makedirs(os.path.dirname('datasets/graphs/protein_graphs.pt') or '.', exist_ok=True)
torch.save(_graphs, 'datasets/graphs/protein_graphs.pt')
print('built', len(_graphs), 'protein graphs ->', 'datasets/graphs/protein_graphs.pt')

print('pipeline done')
