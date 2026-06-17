"""Training-Algorithmus für den Dual-Encoder (Affinity-Regression).

Der visuelle Trainings-Graph in SpinoML (Phase 13/14) unterstützt nur
tabellarische Datensätze — dieses Modell verarbeitet aber Graph-Inputs
(Ligand-Molekülgraph + Protein-.pt-Graph) aus `datasets/rxn.manifest`.
Dieses Skript ist daher das eigenständige Pendant: gleiche Featurisierung
wie der Torch-Sidecar, gleiche Modellstruktur wie `models/dualencoder.spinoml`.

Aufruf (im Workspace-Root examples/reaction-workspace/):
    conda activate spinoml-dev
    python train_dualencoder.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

# ----------------------------------------------------------------------------
# Konfiguration (entspricht dem visuellen Trainings-Graphen, auf Regression)
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "datasets" / "rxn.manifest"

EPOCHS = 200
BATCH_SIZE = 4
LR = 1e-3
WEIGHT_DECAY = 0.0
VAL_RATIO = 0.2
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)


# ----------------------------------------------------------------------------
# Featurisierung — 1:1 wie sidecar-torch/dataset_handlers.py
# ----------------------------------------------------------------------------
def mol_to_graph(smiles: str) -> Data:
    """SMILES → PyG Data(x=[N,5], edge_index). Atome=Knoten, Bindungen=Kanten."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit konnte SMILES nicht parsen: {smiles!r}")
    feats = [
        [
            float(a.GetAtomicNum()),
            float(a.GetDegree()),
            float(a.GetFormalCharge()),
            float(int(a.GetIsAromatic())),
            float(a.GetTotalNumHs()),
        ]
        for a in mol.GetAtoms()
    ]
    src, dst = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]
    x = torch.tensor(feats, dtype=torch.float32) if feats else torch.zeros((1, 5))
    edge_index = (
        torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    )
    return Data(x=x, edge_index=edge_index)


_protein_cache: dict[str, Data] = {}


def load_protein_graph(uniprot: str, graphs_dir: Path, ext: str) -> Data:
    """Lädt den passenden Protein-.pt-Graphen (Match: Dateiname enthält uniprot)."""
    if uniprot in _protein_cache:
        return _protein_cache[uniprot]
    matches = [p for p in graphs_dir.glob(f"*{ext}") if uniprot in p.name]
    if not matches:
        raise FileNotFoundError(f"Kein Protein-Graph für {uniprot!r} in {graphs_dir}")
    d = torch.load(matches[0], map_location="cpu", weights_only=False)
    # Nur die für GCNConv nötigen Felder behalten (x + edge_index).
    data = Data(x=d.x.float(), edge_index=d.edge_index.long())
    _protein_cache[uniprot] = data
    return data


# ----------------------------------------------------------------------------
# Dataset aus dem Manifest
# ----------------------------------------------------------------------------
class ReactionDataset(Dataset):
    def __init__(self, manifest_path: Path):
        import csv

        man = json.loads(manifest_path.read_text())
        ds_dir = manifest_path.parent
        table = ds_dir / man["table"]
        lig_col = man["pairs"]["ligand"]["column"]
        prot_col = man["pairs"]["protein"]["column"]
        graphs_dir = ds_dir / man["pairs"]["protein"]["dir"]
        ext = man["pairs"]["protein"].get("ext", ".pt")
        tgt_col = man["target"]["column"]

        self.samples: list[tuple[Data, Data, float]] = []
        with table.open() as f:
            for row in csv.DictReader(f):
                lig = mol_to_graph(row[lig_col])
                prot = load_protein_graph(row[prot_col], graphs_dir, ext)
                y = float(row[tgt_col])
                self.samples.append((lig, prot, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        lig, prot, y = self.samples[idx]
        return lig, prot, torch.tensor([y], dtype=torch.float32)


def collate(batch):
    ligs, prots, ys = zip(*batch)
    return Batch.from_data_list(ligs), Batch.from_data_list(prots), torch.cat(ys)


# ----------------------------------------------------------------------------
# Modell — entspricht models/dualencoder.spinoml (korrigierte in_channels: 5 / 12)
# ----------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.g1 = GCNConv(in_channels, 64)
        self.act = nn.ReLU()
        self.g2 = GCNConv(64, 128)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.act(self.g1(x, edge_index))
        x = self.g2(x, edge_index)
        return global_mean_pool(x, batch)


class DualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ligand_encoder = Encoder(in_channels=5)
        self.protein_encoder = Encoder(in_channels=12)
        self.fc1 = nn.Linear(256, 64)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, ligand: Data, protein: Data) -> torch.Tensor:
        z = torch.cat([self.ligand_encoder(ligand), self.protein_encoder(protein)], dim=-1)
        return self.fc2(self.act(self.fc1(z)))


# ----------------------------------------------------------------------------
# Metriken
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn) -> dict[str, float]:
    model.eval()
    preds, targets, total_loss, n = [], [], 0.0, 0
    for lig, prot, y in loader:
        lig, prot, y = lig.to(DEVICE), prot.to(DEVICE), y.to(DEVICE)
        out = model(lig, prot).view(-1)
        total_loss += loss_fn(out, y).item() * y.size(0)
        n += y.size(0)
        preds.append(out.cpu())
        targets.append(y.cpu())
    p = torch.cat(preds)
    t = torch.cat(targets)
    mse = float(((p - t) ** 2).mean())
    mae = float((p - t).abs().mean())
    ss_res = float(((t - p) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum()) or 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {"loss": total_loss / max(n, 1), "rmse": math.sqrt(mse), "mae": mae, "r2": r2}


# ----------------------------------------------------------------------------
# Trainings-Loop
# ----------------------------------------------------------------------------
def main() -> None:
    print(f"Device: {DEVICE}")
    full = ReactionDataset(MANIFEST)
    print(f"Datensatz: {len(full)} Reaktionen geladen")

    n_val = max(1, int(round(len(full) * VAL_RATIO)))
    n_train = len(full) - n_val
    gen = torch.Generator().manual_seed(SEED)
    train_set, val_set = torch.utils.data.random_split(full, [n_train, n_val], generator=gen)
    print(f"Split: {n_train} train / {n_val} val")

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    model = DualEncoder().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modell: {n_params:,} Parameter\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )
    loss_fn = nn.MSELoss()

    out_dir = HERE / "experiments" / "dualencoder"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss, n = 0.0, 0
        for lig, prot, y in train_loader:
            lig, prot, y = lig.to(DEVICE), prot.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(lig, prot).view(-1)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * y.size(0)
            n += y.size(0)
        train_loss = epoch_loss / max(n, 1)

        val = evaluate(model, val_loader, loss_fn)
        scheduler.step(val["loss"])
        history.append({"epoch": epoch, "train_loss": train_loss, **val})

        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save(model.state_dict(), out_dir / "best.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | train {train_loss:6.3f} | "
                f"val {val['loss']:6.3f} | RMSE {val['rmse']:5.3f} | "
                f"MAE {val['mae']:5.3f} | R² {val['r2']:6.3f}"
            )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nBestes Val-Loss: {best_val:.4f}")
    print(f"Checkpoint:      {out_dir / 'best.pt'}")
    print(f"Verlauf:         {out_dir / 'history.json'}")


if __name__ == "__main__":
    main()
