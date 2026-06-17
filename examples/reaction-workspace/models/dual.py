import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data


class LigandEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn_conv = GCNConv(in_channels=5, out_channels=64, improved=False, cached=False, add_self_loops=True, bias=True)
        self.re_lu = nn.ReLU(inplace=False)
        self.gcn_conv_2 = GCNConv(in_channels=64, out_channels=128, improved=False, cached=False, add_self_loops=True, bias=True)

    def forward(self, ligand):
        x, edge_index, batch = ligand.x, ligand.edge_index, ligand.batch
        gcn_conv = self.gcn_conv(x, edge_index)
        re_lu = self.re_lu(gcn_conv)
        gcn_conv_2 = self.gcn_conv_2(re_lu, edge_index)
        fx_global_mean_pool = global_mean_pool(gcn_conv_2, batch)
        return fx_global_mean_pool


class ProteinEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn_conv = GCNConv(in_channels=12, out_channels=64, improved=False, cached=False, add_self_loops=True, bias=True)
        self.re_lu = nn.ReLU(inplace=False)
        self.gcn_conv_2 = GCNConv(in_channels=64, out_channels=128, improved=False, cached=False, add_self_loops=True, bias=True)

    def forward(self, protein):
        x, edge_index, batch = protein.x, protein.edge_index, protein.batch
        gcn_conv = self.gcn_conv(x, edge_index)
        re_lu = self.re_lu(gcn_conv)
        gcn_conv_2 = self.gcn_conv_2(re_lu, edge_index)
        fx_global_mean_pool = global_mean_pool(gcn_conv_2, batch)
        return fx_global_mean_pool


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ligand_encoder = LigandEncoder()
        self.protein_encoder = ProteinEncoder()
        self.linear = nn.Linear(in_features=256, out_features=64, bias=True)
        self.re_lu = nn.ReLU(inplace=False)
        self.linear_2 = nn.Linear(in_features=64, out_features=1, bias=True)

    def forward(self, ligand, protein):
        ligand_encoder = self.ligand_encoder(ligand)
        protein_encoder = self.protein_encoder(protein)
        m_concat = torch.cat([ligand_encoder, protein_encoder], dim=-1)
        linear = self.linear(m_concat)
        re_lu = self.re_lu(linear)
        linear_2 = self.linear_2(re_lu)
        return linear_2


if __name__ == "__main__":
    model = Model()
    ligand = Data(x=torch.zeros((3, 5)), edge_index=torch.zeros((2, 4), dtype=torch.long), batch=torch.zeros((3,), dtype=torch.long))
    protein = Data(x=torch.zeros((120, 12)), edge_index=torch.zeros((2, 792), dtype=torch.long), batch=torch.zeros((120,), dtype=torch.long))
    out = model(ligand, protein)
    # Count params AFTER a forward so lazy (in_channels=-1) layers are initialized.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    if isinstance(out, tuple):
        print("Output shapes:", [tuple(o.shape) for o in out])
    elif isinstance(out, dict):
        print("Output shapes:", {k: tuple(v.shape) for k, v in out.items()})
    else:
        print(f"Output shape: {tuple(out.shape)}")
