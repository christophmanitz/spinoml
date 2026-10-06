# multi-input reference experiment

Hand-written reference module: `RefMultiInput` in
`scripts/lib/reference_models.py`. Graph:
`multi-inputSnapshot()` in `scripts/lib/reference-graphs.ts`.

- Input a [1,6] + Input b [1,4] → Linear(6,8)+ReLU and Linear(4,8)+ReLU → Concat(dim=1) → Linear(16,2) → Output
- Expected parameters: 130 (trainable 130)

`model.spinoml` is the serialized graph. `scripts/verify-reference.ts`
regenerates the PyTorch model from it and numerically compares the generated
`Model` against the hand-written reference (parameters, forward passes, loss,
gradients, and a negative control).

## How to run this experiment

Train the committed graph end-to-end through the real trainer:

```bash
npm run verify:reference-train
```

The two-input model is fed through a `.manifest` dataset with two file branches
(`a` [6], `b` [4]) pointing at per-row `.pt` tensors — the trainer's supported
way to provide two numeric inputs without touching product code. The synthetic
label depends mostly on `a`; 6 epochs with Adam(0.01). The harness asserts the
full artifact set, checkpoint↔model consistency, loss behaviour, CPU
reproducibility and device/dtype. Model-level equivalence is covered separately
by `npm run verify:reference`.
