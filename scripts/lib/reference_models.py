"""Hand-written reference PyTorch modules for SpinoML codegen equivalence checks.

These modules are written the way a person would write them by hand — NOT derived
from the generated code — so a numerical match between a SpinoML-generated
``Model`` and these references proves the generator emits an equivalent model.

They mirror the three reference graphs in ``scripts/lib/reference-graphs.ts``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

BATCH_SIZE = 8
NUM_CLASSES = 2


# ─── Analytic parameter counts ────────────────────────────────────────────────

def linear_params(in_features: int, out_features: int, bias: bool = True) -> int:
    return in_features * out_features + (out_features if bias else 0)


def conv2d_params(in_channels: int, out_channels: int, kernel: int, bias: bool = True) -> int:
    return out_channels * in_channels * kernel * kernel + (out_channels if bias else 0)


def mlp_expected_params() -> int:
    return linear_params(10, 16) + linear_params(16, 2)


def cnn_expected_params() -> int:
    return conv2d_params(1, 4, 3) + linear_params(64, 2)


def multi_input_expected_params() -> int:
    return linear_params(6, 8) + linear_params(4, 8) + linear_params(16, 2)


# ─── Hand-written reference modules ───────────────────────────────────────────

class RefMLP(nn.Module):
    """Input x [B, 10] → Linear(10, 16) → ReLU → Linear(16, 2)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class RefCNN(nn.Module):
    """Input x [B, 64] → Reshape [B,1,8,8] → Conv2d(1,4,3,p=1) → ReLU →
    MaxPool2d(2) → Flatten → Linear(64, 2)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], 1, 8, 8)
        x = self.conv(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)


class RefMultiInput(nn.Module):
    """Inputs a [B, 6], b [B, 4] → Linear(6,8)+ReLU and Linear(4,8)+ReLU →
    Concat(dim=1) → Linear(16, 2)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc_a = nn.Linear(6, 8)
        self.fc_b = nn.Linear(4, 8)
        self.fc = nn.Linear(16, 2)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ha = torch.relu(self.fc_a(a))
        hb = torch.relu(self.fc_b(b))
        return self.fc(torch.cat([ha, hb], dim=1))


REFERENCE_MODELS: dict[str, type[nn.Module]] = {
    'mlp': RefMLP,
    'cnn': RefCNN,
    'multi-input': RefMultiInput,
}

EXPECTED_PARAMS: dict[str, object] = {
    'mlp': mlp_expected_params,
    'cnn': cnn_expected_params,
    'multi-input': multi_input_expected_params,
}


# ─── Seeded input builder ─────────────────────────────────────────────────────

def build_inputs(
    name: str,
    batch: int,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Return the named forward inputs for a reference experiment.

    Keys match the generated `forward` argument names, so the comparison feeds
    each model by name (order comes from the generated signature).
    """
    g = torch.Generator().manual_seed(seed)
    if name == 'mlp':
        return {'x': torch.randn(batch, 10, generator=g, dtype=dtype)}
    if name == 'cnn':
        return {'x': torch.randn(batch, 64, generator=g, dtype=dtype)}
    if name == 'multi-input':
        return {
            'a': torch.randn(batch, 6, generator=g, dtype=dtype),
            'b': torch.randn(batch, 4, generator=g, dtype=dtype),
        }
    raise ValueError(f'unknown reference experiment: {name}')
