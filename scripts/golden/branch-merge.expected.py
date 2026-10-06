import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=12, out_features=8, bias=True)
        self.linear_2 = nn.Linear(in_features=12, out_features=8, bias=True)
        self.re_lu = nn.ReLU(inplace=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        linear = self.linear(x)
        linear_2 = self.linear_2(x)
        re_lu = self.re_lu(linear)
        tanh = self.tanh(linear_2)
        m_add = re_lu + tanh
        return m_add


if __name__ == "__main__":
    model = Model()
    x = torch.zeros((1, 12))
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
