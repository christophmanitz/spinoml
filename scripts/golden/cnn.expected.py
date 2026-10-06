import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.batch_norm2d = nn.BatchNorm2d(num_features=8, eps=1e-5, momentum=0.1)
        self.re_lu = nn.ReLU(inplace=False)
        self.dropout2d = nn.Dropout2d(p=0.1)
        self.max_pool2d = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear = nn.Linear(in_features=512, out_features=10, bias=True)

    def forward(self, x):
        conv2d = self.conv2d(x)
        batch_norm2d = self.batch_norm2d(conv2d)
        re_lu = self.re_lu(batch_norm2d)
        dropout2d = self.dropout2d(re_lu)
        max_pool2d = self.max_pool2d(dropout2d)
        flatten = self.flatten(max_pool2d)
        linear = self.linear(flatten)
        return linear


if __name__ == "__main__":
    model = Model()
    x = torch.zeros((1, 3, 16, 16))
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
