import torch
import torch.nn as nn


class MlpBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=64, out_features=32, bias=True)
        self.re_lu = nn.ReLU(inplace=False)

    def forward(self, x):
        linear = self.linear(x)
        re_lu = self.re_lu(linear)
        return re_lu


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_block = MlpBlock()

    def forward(self, x):
        mlp_block = self.mlp_block(x)
        return mlp_block


if __name__ == "__main__":
    model = Model()
    x = torch.zeros((1, 64))
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
