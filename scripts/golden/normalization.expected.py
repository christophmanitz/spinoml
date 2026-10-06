import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm1d = nn.BatchNorm1d(num_features=16, eps=1e-5, momentum=0.1)
        self.layer_norm = nn.LayerNorm(normalized_shape=(16,), eps=1e-5)
        self.group_norm = nn.GroupNorm(num_groups=4, num_channels=16)

    def forward(self, x):
        batch_norm1d = self.batch_norm1d(x)
        layer_norm = self.layer_norm(batch_norm1d)
        group_norm = self.group_norm(layer_norm)
        return group_norm


if __name__ == "__main__":
    model = Model()
    x = torch.zeros((1, 16))
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
