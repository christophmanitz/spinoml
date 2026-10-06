import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

    def forward(self, x):
        conv_transpose2d = self.conv_transpose2d(x)
        return conv_transpose2d


if __name__ == "__main__":
    model = Model()
    x = torch.zeros((1, 4, 8, 8))
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
