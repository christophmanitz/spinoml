import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.0, activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0, activation='gelu', batch_first=True), num_layers=2)

    def forward(self, x):
        transformer_encoder_layer = self.transformer_encoder_layer(x)
        transformer_encoder = self.transformer_encoder(transformer_encoder_layer)
        return transformer_encoder


if __name__ == "__main__":
    model = Model()
    x = torch.zeros((1, 16, 32))
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
