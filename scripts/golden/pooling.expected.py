import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.avg_pool2d = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d(output_size=(2, 2))

    def forward(self, x):
        max_pool2d = self.max_pool2d(x)
        avg_pool2d = self.avg_pool2d(max_pool2d)
        adaptive_avg_pool2d = self.adaptive_avg_pool2d(avg_pool2d)
        return adaptive_avg_pool2d


if __name__ == "__main__":
    model = Model()
    x = torch.zeros((1, 1, 16, 16))
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
