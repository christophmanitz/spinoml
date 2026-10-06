import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=6, out_features=8, bias=True)
        self.linear_2 = nn.Linear(in_features=4, out_features=8, bias=True)
        self.re_lu = nn.ReLU(inplace=False)
        self.re_lu_2 = nn.ReLU(inplace=False)
        self.linear_3 = nn.Linear(in_features=16, out_features=2, bias=True)

    def forward(self, a, b):
        linear = self.linear(a)
        linear_2 = self.linear_2(b)
        re_lu = self.re_lu(linear)
        re_lu_2 = self.re_lu_2(linear_2)
        m_concat = torch.cat([re_lu, re_lu_2], dim=1)
        linear_3 = self.linear_3(m_concat)
        return linear_3


if __name__ == "__main__":
    model = Model()
    a = torch.zeros((1, 6))
    b = torch.zeros((1, 4))
    out = model(a, b)
    # Count params AFTER a forward so lazy (in_channels=-1) layers are initialized.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    if isinstance(out, tuple):
        print("Output shapes:", [tuple(o.shape) for o in out])
    elif isinstance(out, dict):
        print("Output shapes:", {k: tuple(v.shape) for k, v in out.items()})
    else:
        print(f"Output shape: {tuple(out.shape)}")
