import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, dropout=0.0, batch_first=True)

    def forward(self, x):
        multihead_attention = self.multihead_attention(x, x, x, need_weights=False)[0]
        return multihead_attention


if __name__ == "__main__":
    model = Model()
    x = torch.zeros((1, 10, 32))
    out = model(x)
    # Count params AFTER a forward so lazy (in_channels=-1) layers are initialized.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    if isinstance(out, tuple):
        print("Output shapes:", [tuple(o.shape) for o in out])
    elif isinstance(out, dict):
        print("Output shapes:", {k: tuple(v.shape) for k, v in out.items()})
    else:
        print(f"Output shape: {tuple(out.shape)}")
