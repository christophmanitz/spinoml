import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=10, out_features=4, bias=True)
        self.linear_2 = nn.Linear(in_features=10, out_features=5, bias=True)
        self.linear_3 = nn.Linear(in_features=10, out_features=6, bias=True)

    def forward(self, x):
        linear = self.linear(x)
        linear_2 = self.linear_2(x)
        linear_3 = self.linear_3(x)
        m_concat = torch.cat([linear, linear_2, linear_3], dim=1)
        return m_concat


if __name__ == "__main__":
    model = Model()
    x = torch.zeros((1, 10))
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
