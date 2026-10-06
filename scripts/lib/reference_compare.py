"""Compare a SpinoML-generated PyTorch model against a hand-written reference.

Reads ONE JSON job on stdin::

    {"experiments": [{"name": "mlp", "model_py": "<path to generated model.py>"}, ...]}

and prints ONE JSON result on stdout::

    {"results": {"<name>": [{"check": "...", "ok": true, "detail": "..."}, ...], ...}}

Loaded the way the sidecar does (``exec(compile(code, "<spinoml-model>",
"exec"), ns)``). Each experiment yields the checks defined in the harness:

- ``param_count``         total and trainable parameter counts match expected
- ``param_shapes``        per-tensor shapes match between generated and reference
- ``copy_weights``        generated weights copied into the reference pairwise
- ``forward_f64``         outputs equal in float64 (rtol 1e-10, atol 1e-12)
- ``forward_f32``         outputs equal in float32 (rtol 1e-5, atol 1e-6)
- ``loss_f64``            CrossEntropyLoss on seeded targets equal in float64
- ``grads_f64``           per-parameter gradients equal after loss.backward()
- ``negative_control``    +1e-3 to one weight of the reference makes outputs
                          diverge from the generated model
- ``cuda_forward_f32``    repeated on CUDA if available (else printed as SKIPPED)
"""

from __future__ import annotations

import copy
import inspect
import json
import sys
import traceback

import torch
import torch.nn as nn

from reference_models import (
    BATCH_SIZE,
    NUM_CLASSES,
    REFERENCE_MODELS,
    EXPECTED_PARAMS,
    build_inputs,
)


SEED = 1729
TARGET_SEED = 1730


def _load_generated(model_py: str) -> type[nn.Module]:
    with open(model_py, 'r', encoding='utf-8') as fh:
        code = fh.read()
    ns: dict[str, object] = {'__name__': '<spinoml-model>'}
    exec(compile(code, '<spinoml-model>', 'exec'), ns)
    model_cls = ns.get('Model')
    if isinstance(model_cls, type) and issubclass(model_cls, nn.Module) and model_cls is not nn.Module:
        return model_cls
    # Fallback: any user-defined nn.Module subclass defined in this namespace.
    for value in ns.values():
        if isinstance(value, type) and issubclass(value, nn.Module) and value is not nn.Module:
            return value
    raise RuntimeError('no nn.Module subclass found in generated code')


def _forward_inputs(model: nn.Module, inputs: dict[str, torch.Tensor]) -> list[object]:
    """Return the positional argument list to call `model.forward` with, in the
    order of its declared parameters (excluding `self`). Looks up by name first,
    falls back to the positional order of `inputs`."""
    sig = inspect.signature(model.forward)
    params = [name for name in sig.parameters if name != 'self']
    values = list(inputs.values())
    args: list[object] = []
    for i, name in enumerate(params):
        if name in inputs:
            args.append(inputs[name])
        else:
            args.append(values[i])
    return args


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _shape_list(m: nn.Module) -> list[tuple[int, ...]]:
    return [tuple(p.shape) for p in m.parameters()]


def _named_shape_list(m: nn.Module) -> list[str]:
    return [f'{name}{tuple(p.shape)}' for name, p in m.named_parameters()]


def _run_one(name: str, model_py: str) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []

    def add(check: str, ok: bool, detail: str = '', skipped: bool = False) -> None:
        payload: dict[str, object] = {'check': check, 'ok': bool(ok), 'detail': detail}
        if skipped:
            payload['skipped'] = True
        checks.append(payload)

    expected_fn_obj = EXPECTED_PARAMS.get(name)
    ref_cls = REFERENCE_MODELS.get(name)
    if expected_fn_obj is None or ref_cls is None:
        add('experiment_known', False, f'unknown reference experiment {name!r}')
        return checks

    expected = int(expected_fn_obj())  # type: ignore[call-arg]

    try:
        gen_cls = _load_generated(model_py)
    except Exception as exc:
        add('generated_load', False, f'{type(exc).__name__}: {exc}')
        return checks
    add('generated_load', True, f'class={gen_cls.__name__}')

    try:
        gen = gen_cls().eval()
        ref = ref_cls().eval()
    except Exception as exc:
        add('generated_construct', False, f'{type(exc).__name__}: {exc}')
        return checks
    add('generated_construct', True, 'eval() ok for both models')

    # (a) parameter counts
    gen_total = sum(p.numel() for p in gen.parameters())
    ref_total = sum(p.numel() for p in ref.parameters())
    gen_train = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    ref_train = sum(p.numel() for p in ref.parameters() if p.requires_grad)
    counts_ok = (
        gen_total == expected
        and ref_total == expected
        and gen_train == gen_total
        and ref_train == ref_total
    )
    add(
        'param_count',
        counts_ok,
        f'expected={expected} generated={gen_total}/{gen_train} trainable reference={ref_total}/{ref_train}',
    )

    # parameter shapes + weight copy
    gshapes = _shape_list(gen)
    rshapes = _shape_list(ref)
    if gshapes != rshapes:
        add(
            'param_shapes',
            False,
            f'generated={_named_shape_list(gen)} reference={_named_shape_list(ref)}',
        )
        add('copy_weights', False, 'skipped: shape mismatch')
        return checks
    add('param_shapes', True, f'{len(gshapes)} tensors: {_named_shape_list(gen)}')
    with torch.no_grad():
        for g_p, r_p in zip(gen.parameters(), ref.parameters()):
            r_p.copy_(g_p)
    add('copy_weights', True, 'generated weights copied into reference pairwise')

    # (b) forward f64
    gen.double()
    ref.double()
    inputs64 = build_inputs(name, BATCH_SIZE, torch.float64, SEED)
    try:
        with torch.no_grad():
            g_out = gen(*_forward_inputs(gen, inputs64))
            r_out = ref(*_forward_inputs(ref, inputs64))
    except Exception as exc:
        add('forward_f64', False, f'{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}')
        return checks
    if not isinstance(g_out, torch.Tensor) or not isinstance(r_out, torch.Tensor):
        add('forward_f64', False, f'non-tensor output: gen={type(g_out).__name__} ref={type(r_out).__name__}')
        return checks
    fwd_shape_ok = tuple(g_out.shape) == (BATCH_SIZE, NUM_CLASSES) and tuple(r_out.shape) == (BATCH_SIZE, NUM_CLASSES)
    fwd_val_ok = torch.allclose(g_out, r_out, rtol=1e-10, atol=1e-12)
    add(
        'forward_f64',
        fwd_shape_ok and fwd_val_ok,
        f'shape gen={tuple(g_out.shape)} ref={tuple(r_out.shape)} max_abs_diff={_max_abs_diff(g_out, r_out):.3e}',
    )

    # (b') forward f32
    gen.float()
    ref.float()
    inputs32 = build_inputs(name, BATCH_SIZE, torch.float32, SEED)
    try:
        with torch.no_grad():
            g_out32 = gen(*_forward_inputs(gen, inputs32))
            r_out32 = ref(*_forward_inputs(ref, inputs32))
    except Exception as exc:
        add('forward_f32', False, f'{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}')
        return checks
    fwd32_ok = (
        tuple(g_out32.shape) == (BATCH_SIZE, NUM_CLASSES)
        and tuple(r_out32.shape) == (BATCH_SIZE, NUM_CLASSES)
        and torch.allclose(g_out32, r_out32, rtol=1e-5, atol=1e-6)
    )
    add(
        'forward_f32',
        fwd32_ok,
        f'shape gen={tuple(g_out32.shape)} ref={tuple(r_out32.shape)} max_abs_diff={_max_abs_diff(g_out32, r_out32):.3e}',
    )

    # (c) + (d) loss and gradients in float64
    gen.double()
    ref.double()
    inputs64 = build_inputs(name, BATCH_SIZE, torch.float64, SEED)
    targets = torch.randint(
        low=0,
        high=NUM_CLASSES,
        size=(BATCH_SIZE,),
        generator=torch.Generator().manual_seed(TARGET_SEED),
    )
    criterion = nn.CrossEntropyLoss()
    gen.zero_grad()
    ref.zero_grad()
    g_logits = gen(*_forward_inputs(gen, inputs64))
    r_logits = ref(*_forward_inputs(ref, inputs64))
    if not isinstance(g_logits, torch.Tensor) or not isinstance(r_logits, torch.Tensor):
        add('loss_f64', False, f'non-tensor logits: gen={type(g_logits).__name__} ref={type(r_logits).__name__}')
        return checks
    g_loss = criterion(g_logits, targets)
    r_loss = criterion(r_logits, targets)
    loss_ok = torch.allclose(g_loss.detach(), r_loss.detach(), rtol=1e-10, atol=1e-12)
    add(
        'loss_f64',
        loss_ok,
        f'gen={g_loss.item():.6e} ref={r_loss.item():.6e} abs_diff={abs(g_loss.item() - r_loss.item()):.3e}',
    )

    g_loss.backward()
    r_loss.backward()
    grad_ok = True
    grad_detail: list[str] = []
    for idx, (g_p, r_p) in enumerate(zip(gen.parameters(), ref.parameters())):
        if g_p.grad is None or r_p.grad is None:
            grad_ok = False
            grad_detail.append(f'[{idx}] None grad')
            continue
        if not torch.isfinite(g_p.grad).all() or not torch.isfinite(r_p.grad).all():
            grad_ok = False
            grad_detail.append(f'[{idx}] non-finite')
            continue
        if not torch.allclose(g_p.grad, r_p.grad, rtol=1e-10, atol=1e-12):
            grad_ok = False
            grad_detail.append(
                f'[{idx}] diff max_abs={_max_abs_diff(g_p.grad, r_p.grad):.3e}'
            )
    add(
        'grads_f64',
        grad_ok,
        ('all ' + str(len(list(gen.parameters()))) + ' tensors equal' if grad_ok else '; '.join(grad_detail)),
    )

    # (b'') CUDA float32 — only if available
    if torch.cuda.is_available():
        try:
            gen.to('cuda')
            ref.to('cuda')
            inputs_c = build_inputs(name, BATCH_SIZE, torch.float32, SEED)
            inputs_c = {k: v.cuda() for k, v in inputs_c.items()}
            with torch.no_grad():
                gc = gen(*_forward_inputs(gen, inputs_c))
                rc = ref(*_forward_inputs(ref, inputs_c))
            cuda_ok = (
                tuple(gc.shape) == (BATCH_SIZE, NUM_CLASSES)
                and tuple(rc.shape) == (BATCH_SIZE, NUM_CLASSES)
                and torch.allclose(gc, rc, rtol=1e-5, atol=1e-6)
            )
            add(
                'cuda_forward_f32',
                cuda_ok,
                f'shape gen={tuple(gc.shape)} ref={tuple(rc.shape)} max_abs_diff={_max_abs_diff(gc, rc):.3e}',
            )
        except Exception as exc:
            add('cuda_forward_f32', False, f'{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}')
        finally:
            gen.to('cpu')
            ref.to('cpu')
    else:
        add(
            'cuda_forward_f32',
            True,
            'SKIPPED  CUDA — torch.cuda.is_available() is False',
            skipped=True,
        )

    # (e) negative control: perturb reference weight by +1e-3, expect divergence
    try:
        gen.double()
        ctrl = copy.deepcopy(ref).double()
        ctrl_inputs = build_inputs(name, BATCH_SIZE, torch.float64, SEED)
        with torch.no_grad():
            first = next(ctrl.parameters())
            first.add_(1e-3)
            g_perturbed = gen(*_forward_inputs(gen, ctrl_inputs))
            c_perturbed = ctrl(*_forward_inputs(ctrl, ctrl_inputs))
        detected = not torch.allclose(g_perturbed, c_perturbed, rtol=1e-10, atol=1e-12)
        add(
            'negative_control',
            detected,
            f'perturbation detected: max_abs_diff={_max_abs_diff(g_perturbed, c_perturbed):.3e}',
        )
    except Exception as exc:
        add('negative_control', False, f'{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}')

    return checks


def main() -> int:
    raw = sys.stdin.read()
    job = json.loads(raw)
    experiments = job.get('experiments', [])
    results: dict[str, list[dict[str, object]]] = {}
    for exp in experiments:
        name = str(exp.get('name', ''))
        model_py = str(exp.get('model_py', ''))
        results[name] = _run_one(name, model_py)
    json.dump({'results': results}, sys.stdout, sort_keys=False)
    sys.stdout.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
