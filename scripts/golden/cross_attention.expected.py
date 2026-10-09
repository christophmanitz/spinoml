import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=16, num_heads=4, dropout=0.0, batch_first=True)

    def forward(self, query, context):
        cross_attention = self.cross_attention(query, context, context, need_weights=False)[0]
        return cross_attention


if __name__ == "__main__":
    model = Model()
    query = torch.zeros((2, 5, 16))
    context = torch.zeros((2, 7, 16))
    out = model(query, context)
    # Count params AFTER a forward so lazy (in_channels=-1) layers are initialized.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    if isinstance(out, tuple):
        print("Output shapes:", [tuple(o.shape) for o in out])
    elif isinstance(out, dict):
        print("Output shapes:", {k: tuple(v.shape) for k, v in out.items()})
    else:
        print(f"Output shape: {tuple(out.shape)}")
