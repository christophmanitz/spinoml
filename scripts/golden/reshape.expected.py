import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        fx_reshape = x.reshape(x.shape[0], 2, 12)
        return fx_reshape


if __name__ == "__main__":
    model = Model()
    x = torch.zeros((1, 24))
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
