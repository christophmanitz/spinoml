import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=50, embedding_dim=16)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear = nn.Linear(in_features=128, out_features=4, bias=True)

    def forward(self, seq):
        embedding = self.embedding(seq)
        flatten = self.flatten(embedding)
        linear = self.linear(flatten)
        return linear


if __name__ == "__main__":
    model = Model()
    seq = torch.zeros((1, 8), dtype=torch.long)
    out = model(seq)
    # Count params AFTER a forward so lazy (in_channels=-1) layers are initialized.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    if isinstance(out, tuple):
        print("Output shapes:", [tuple(o.shape) for o in out])
    elif isinstance(out, dict):
        print("Output shapes:", {k: tuple(v.shape) for k, v in out.items()})
    else:
        print(f"Output shape: {tuple(out.shape)}")
