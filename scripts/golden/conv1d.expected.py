import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        conv1d = self.conv1d(x)
        return conv1d


if __name__ == "__main__":
    model = Model()
    x = torch.zeros((1, 1, 16))
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
