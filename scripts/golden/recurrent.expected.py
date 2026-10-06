import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=16, hidden_size=16, num_layers=1, batch_first=True, bidirectional=False, dropout=0.0)
        self.gru = nn.GRU(input_size=16, hidden_size=16, num_layers=1, batch_first=True, bidirectional=False, dropout=0.0)
        self.rnn = nn.RNN(input_size=16, hidden_size=16, num_layers=1, nonlinearity='tanh', batch_first=True, bidirectional=False)

    def forward(self, x):
        lstm, _ = self.lstm(x)
        gru, _ = self.gru(lstm)
        rnn, _ = self.rnn(gru)
        return rnn


if __name__ == "__main__":
    model = Model()
    x = torch.zeros((1, 8, 16))
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
