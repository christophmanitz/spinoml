import torch
import torch.nn as nn
import torch.nn.functional as F


class Scale(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = Scale(factor=2.0)

    def forward(self, x):
        scale = self.scale(x)
        return scale


if __name__ == "__main__":
    model = Model()
    x = torch.zeros((1, 8))
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
