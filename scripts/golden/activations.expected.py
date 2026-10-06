import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.re_lu = nn.ReLU(inplace=False)
        self.gelu = nn.GELU()
        self.si_lu = nn.SiLU(inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        re_lu = self.re_lu(x)
        gelu = self.gelu(re_lu)
        si_lu = self.si_lu(gelu)
        sigmoid = self.sigmoid(si_lu)
        tanh = self.tanh(sigmoid)
        softmax = self.softmax(tanh)
        log_softmax = self.log_softmax(softmax)
        return log_softmax


if __name__ == "__main__":
    model = Model()
    x = torch.zeros((1, 8))
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
