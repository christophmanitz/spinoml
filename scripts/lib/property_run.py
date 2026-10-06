"""Batch python runner for the property test (``scripts/test-property.ts``).

One process, one JSON job on stdin, one JSON result on stdout. For every case it

  * exec()s the SpinoML-generated model code,
  * instantiates ``Model`` and runs a real forward pass on seeded random inputs
    at the requested batch size (float32, int64 for token inputs),
  * checks the output is finite and has the shape the TS oracle predicted,
  * counts parameters and compares against the oracle,
  * runs ``sum().backward()`` and checks every grad-requiring parameter got a
    finite gradient,
  * and re-runs a batch-1 forward with forward hooks so the per-module output
    shapes can be compared against the torch sidecar's ``/infer`` shapes.

This file never talks to the app; it is the independent execution half of the
oracle. Any per-case failure is reported as data (``ok: false`` + ``error``),
never raised, so one bad model cannot abort the batch.
"""

from __future__ import annotations

import json
import sys
import traceback

import torch
import torch.nn as nn


def _load_generated(code: str) -> type[nn.Module]:
    ns: dict[str, object] = {"__name__": "<spinoml-model>"}
    exec(compile(code, "<spinoml-model>", "exec"), ns)  # noqa: S102 - by design
    model_cls = ns.get("Model")
    if isinstance(model_cls, type) and issubclass(model_cls, nn.Module) and model_cls is not nn.Module:
        return model_cls
    for value in ns.values():
        if isinstance(value, type) and issubclass(value, nn.Module) and value is not nn.Module:
            return value
    raise RuntimeError("no nn.Module subclass found in generated code")


def _min_embedding_vocab(model: nn.Module) -> int:
    vocabs = [m.num_embeddings for m in model.modules() if isinstance(m, nn.Embedding)]
    return min(vocabs) if vocabs else 8


def _normalize_out(out: object) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, tuple) and out:
        for item in out:
            if isinstance(item, torch.Tensor):
                return item
    if isinstance(out, dict):
        tensors = [v for v in out.values() if isinstance(v, torch.Tensor)]
        if len(tensors) == 1:
            return tensors[0]
        if tensors:
            return sum(tensors[1:], tensors[0])
    raise RuntimeError(f"model returned a non-tensor: {type(out).__name__}")


def _build_inputs(model: nn.Module, inputs: list[dict], batch: int, seed: int) -> list[torch.Tensor]:
    vocab = _min_embedding_vocab(model)
    gen = torch.Generator().manual_seed(seed)
    tensors: list[torch.Tensor] = []
    for i, spec in enumerate(inputs):
        shape = list(spec.get("shape", [1]))
        if len(shape) > 0:
            shape[0] = batch
        dtype = str(spec.get("dtype", "float32"))
        if dtype in ("int64", "long"):
            t = torch.randint(0, max(vocab, 1), tuple(shape), generator=gen, dtype=torch.long)
        else:
            t = torch.randn(tuple(shape), generator=gen, dtype=torch.float32)
        tensors.append(t)
    return tensors


def _hook_shapes(model: nn.Module, inputs: list[torch.Tensor]) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}

    def make(name: str):
        def hook(_m: nn.Module, _inp: object, out: object) -> None:
            if isinstance(out, torch.Tensor):
                shapes[name] = list(out.shape)
            elif isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
                shapes[name] = list(out[0].shape)
        return hook

    handles = []
    for name, mod in model.named_modules():
        if name == "":
            continue
        handles.append(mod.register_forward_hook(make(name)))
    try:
        with torch.no_grad():
            _ = model(*inputs)
    finally:
        for h in handles:
            h.remove()
    return shapes


def _run_case(case: dict, batch: int) -> dict:
    result: dict[str, object] = {
        "ok": False,
        "error": None,
        "output_shape": None,
        "n_params": None,
        "grads_ok": False,
        "grads_detail": "",
        "finite": False,
        "hook_shapes": {},
        "output_shape_b1": None,
    }
    try:
        model_cls = _load_generated(str(case["code"]))
    except Exception as exc:
        result["error"] = f"load: {type(exc).__name__}: {exc}"
        return result

    try:
        model = model_cls()
        model.eval()
    except Exception as exc:
        result["error"] = f"construct: {type(exc).__name__}: {exc}"
        return result

    inputs = [i for i in case.get("inputs", []) if isinstance(i, dict)]
    # Forward with the requested batch (3 by default) for shape/param/grad checks.
    try:
        xs = _build_inputs(model, inputs, batch, seed=int(case.get("index", 0)) + 17)
    except Exception as exc:
        result["error"] = f"input: {type(exc).__name__}: {exc}"
        return result

    try:
        out = model(*xs)
        out_t = _normalize_out(out)
    except Exception as exc:
        result["error"] = f"forward: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"
        return result

    result["finite"] = bool(torch.isfinite(out_t).all().item())
    result["output_shape"] = list(out_t.shape)
    result["n_params"] = int(sum(p.numel() for p in model.parameters()))

    # Backward: every grad-requiring parameter must get a finite gradient.
    try:
        model.zero_grad()
        loss = out_t.float().sum()
        loss.backward()
        grad_missing: list[str] = []
        grad_bad: list[str] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                grad_missing.append(name)
            elif not torch.isfinite(p.grad).all().item():
                grad_bad.append(name)
        result["grads_ok"] = not grad_missing and not grad_bad
        if grad_missing:
            result["grads_detail"] = f"missing grad: {grad_missing[:4]}"
        elif grad_bad:
            result["grads_detail"] = f"non-finite grad: {grad_bad[:4]}"
        else:
            result["grads_detail"] = f"all {sum(1 for _ in model.parameters())} tensors finite"
    except Exception as exc:
        result["grads_detail"] = f"backward: {type(exc).__name__}: {exc}"
        result["grads_ok"] = False

    # Batch-1 hook pass for per-module shape comparison against the sidecar.
    try:
        model.zero_grad(set_to_none=True)
        xs1 = _build_inputs(model, inputs, 1, seed=int(case.get("index", 0)) + 17)
        with torch.no_grad():
            out1 = _normalize_out(model(*xs1))
        result["output_shape_b1"] = list(out1.shape)
        result["hook_shapes"] = _hook_shapes(model, xs1)
    except Exception as exc:
        result["error"] = f"hook-pass: {type(exc).__name__}: {exc}"
        return result

    result["ok"] = True
    return result


def main() -> int:
    raw = sys.stdin.read()
    job = json.loads(raw)
    batch = int(job.get("batch", 3))
    cases = job.get("cases", [])
    results: dict[str, dict] = {}
    for case in cases:
        idx = str(case.get("index", len(results)))
        try:
            results[idx] = _run_case(case, batch)
        except Exception as exc:  # defensive: never abort the batch
            results[idx] = {
                "ok": False,
                "error": f"runner: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}",
                "output_shape": None,
                "n_params": None,
                "grads_ok": False,
                "grads_detail": "",
                "finite": False,
                "hook_shapes": {},
                "output_shape_b1": None,
            }
    json.dump({"results": results}, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
