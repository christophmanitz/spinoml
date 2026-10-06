import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, a, b):
        m_add = a + b
        m_multiply = a * b
        m_stack = torch.stack([a, b], dim=0)
        return { 'mu': m_add, 'aux': m_multiply, 'sigma': m_stack }


if __name__ == "__main__":
    model = Model()
    a = torch.zeros((1, 4))
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
