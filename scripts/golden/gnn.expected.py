import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn_conv = GCNConv(in_channels=-1, out_channels=16, improved=False, cached=False, add_self_loops=True, bias=True)
        self.re_lu = nn.ReLU(inplace=False)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        gcn_conv = self.gcn_conv(x, edge_index)
        re_lu = self.re_lu(gcn_conv)
        fx_global_mean_pool = global_mean_pool(re_lu, batch)
        return fx_global_mean_pool


if __name__ == "__main__":
    model = Model()
    data = Data(x=torch.zeros((32, 9)), edge_index=torch.zeros((2, 64), dtype=torch.long), batch=torch.zeros((32,), dtype=torch.long))
    out = model(data)
    # Count params AFTER a forward so lazy (in_channels=-1) layers are initialized.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    if isinstance(out, tuple):
        print("Output shapes:", [tuple(o.shape) for o in out])
    elif isinstance(out, dict):
        print("Output shapes:", {k: tuple(v.shape) for k, v in out.items()})
    else:
        print(f"Output shape: {tuple(out.shape)}")
