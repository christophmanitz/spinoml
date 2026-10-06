import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv, SAGEConv, TransformerConv, global_add_pool


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sage_conv = SAGEConv(in_channels=6, out_channels=5, aggr='mean', normalize=False, bias=True)
        self.graph_conv = GraphConv(in_channels=5, out_channels=4, aggr='add', bias=True)
        self.graph_transformer = TransformerConv(in_channels=4, out_channels=4, heads=2, concat=True, beta=False, dropout=0.0, bias=True)

    def forward(self, batch, edge_index, x):
        sage_conv = self.sage_conv(x, edge_index)
        graph_conv = self.graph_conv(sage_conv, edge_index)
        graph_transformer = self.graph_transformer(graph_conv, edge_index)
        fx_global_add_pool = global_add_pool(graph_transformer, batch)
        return fx_global_add_pool


if __name__ == "__main__":
    model = Model()
    batch = torch.zeros((8,), dtype=torch.long)
    edge_index = torch.zeros((2, 12), dtype=torch.long)
    x = torch.zeros((8, 6))
    out = model(batch, edge_index, x)
    # Count params AFTER a forward so lazy (in_channels=-1) layers are initialized.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    if isinstance(out, tuple):
        print("Output shapes:", [tuple(o.shape) for o in out])
    elif isinstance(out, dict):
        print("Output shapes:", {k: tuple(v.shape) for k, v in out.items()})
    else:
        print(f"Output shape: {tuple(out.shape)}")
