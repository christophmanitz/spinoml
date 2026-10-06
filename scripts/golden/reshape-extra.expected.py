import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        fx_permute = x.permute(0, 2, 1, 3)
        fx_transpose = fx_permute.transpose(1, 2)
        fx_view = fx_transpose.reshape(fx_transpose.shape[0], -1).contiguous()
        return fx_view


if __name__ == "__main__":
    model = Model()
    x = torch.zeros((1, 2, 3, 4))
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
