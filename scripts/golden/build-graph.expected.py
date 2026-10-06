import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_max_pool


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat_conv = GATConv(in_channels=4, out_channels=4, heads=2, concat=True, dropout=0.0, bias=True)
        self.re_lu = nn.ReLU(inplace=False)

    def forward(self, batch, x):
        fx_build_graph_ei = (lambda _nbr: torch.stack([torch.arange(x.size(0), device=x.device).repeat_interleave(_nbr.size(1)), _nbr.reshape(-1)]))(torch.cdist(x, x).topk(min(3 + 1, x.size(0)), largest=False).indices[:, 1:])
        fx_build_graph = x
        gat_conv = self.gat_conv(fx_build_graph, fx_build_graph_ei)
        re_lu = self.re_lu(gat_conv)
        fx_global_max_pool = global_max_pool(re_lu, batch)
        return fx_global_max_pool


if __name__ == "__main__":
    model = Model()
    batch = torch.zeros((8,), dtype=torch.long)
    x = torch.zeros((8, 4))
    out = model(batch, x)
    # Count params AFTER a forward so lazy (in_channels=-1) layers are initialized.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    if isinstance(out, tuple):
        print("Output shapes:", [tuple(o.shape) for o in out])
    elif isinstance(out, dict):
        print("Output shapes:", {k: tuple(v.shape) for k, v in out.items()})
    else:
        print(f"Output shape: {tuple(out.shape)}")
