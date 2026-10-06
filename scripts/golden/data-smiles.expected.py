# Auto-generiert aus dem Daten-Canvas (SpinoML).
# Reproduzierbare Daten-Pipeline — läuft im Workspace-Root.
# Schwere Schritte (Graph-Bau, große Downloads) ggf. via SLURM ausführen.

import pandas as pd

df = None  # wird vom Quell-Knoten gesetzt

# ── TableSource (src) ──
df = pd.read_csv('datasets/mols.csv')
print('loaded', df.shape, 'from', 'datasets/mols.csv')

# ── SmilesToGraph (s2g) ──
from rdkit import Chem
from rdkit.Chem import AllChem
import torch, os
from torch_geometric.data import Data
_HYB = {Chem.HybridizationType.SP: 1, Chem.HybridizationType.SP2: 2, Chem.HybridizationType.SP3: 3, Chem.HybridizationType.SP3D: 4, Chem.HybridizationType.SP3D2: 5}
# Physicochemical ATOM features → node feature matrix x.
def _atom_features(a):
    f = [float(a.GetAtomicNum())]
    f += [float(a.GetDegree()), float(a.GetFormalCharge()), float(a.GetTotalNumHs()), float(_HYB.get(a.GetHybridization(), 0)), float(a.GetIsAromatic()), float(a.IsInRing())]
    return f
_BT = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}
# BOND features → edge_attr: one-hot bond type (4) + conjugated + in_ring.
def _bond_features(b):
    oh = [0.0, 0.0, 0.0, 0.0]
    oh[_BT.get(b.GetBondType(), 0)] = 1.0
    return oh + [float(b.GetIsConjugated()), float(b.IsInRing())]
_graphs = []
for _smi in df['smiles'].dropna().astype(str):
    _m = Chem.MolFromSmiles(_smi)
    if _m is None:
        continue
    _m = Chem.AddHs(_m)
    if AllChem.EmbedMolecule(_m, randomSeed=0) != 0:
        continue
    try:
        AllChem.MMFFOptimizeMolecule(_m)
    except Exception:
        pass
    _x = torch.tensor([_atom_features(a) for a in _m.GetAtoms()], dtype=torch.float)
    _ei = []
    _ea = []
    for _b in _m.GetBonds():
        _i, _j = _b.GetBeginAtomIdx(), _b.GetEndAtomIdx()
        _ei += [[_i, _j], [_j, _i]]
        _bf = _bond_features(_b)
        _ea += [_bf, _bf]
    _edge_index = torch.tensor(_ei, dtype=torch.long).t().contiguous() if _ei else torch.empty((2, 0), dtype=torch.long)
    _data = Data(x=_x, edge_index=_edge_index)
    _data.edge_attr = torch.tensor(_ea, dtype=torch.float) if _ea else torch.empty((0, 6), dtype=torch.float)
    if _m.GetNumConformers() > 0:
        _conf = _m.GetConformer()
        _data.pos = torch.tensor([list(_conf.GetAtomPosition(i)) for i in range(_m.GetNumAtoms())], dtype=torch.float)
    _graphs.append(_data)
os.makedirs(os.path.dirname('datasets/graphs/mol_graphs.pt') or '.', exist_ok=True)
torch.save(_graphs, 'datasets/graphs/mol_graphs.pt')
print('built', len(_graphs), 'molecule graphs ->', 'datasets/graphs/mol_graphs.pt')

print('pipeline done')
