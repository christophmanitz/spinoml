# mlp reference experiment

Hand-written reference module: `RefMLP` in
`scripts/lib/reference_models.py`. Graph:
`mlpSnapshot()` in `scripts/lib/reference-graphs.ts`.

- Input x [1,10] float32 → Linear(10,16) → ReLU → Linear(16,2) → Output
- Expected parameters: 210 (trainable 210)

`model.spinoml` is the serialized graph. `scripts/verify-reference.ts`
regenerates the PyTorch model from it and numerically compares the generated
`Model` against the hand-written reference (parameters, forward passes, loss,
gradients, and a negative control).

## How to run this experiment

Train the committed graph end-to-end through the real trainer:

```bash
npm run verify:reference-train
```

The harness builds a synthetic seeded CSV (200 rows, 10 float features `f0..f9`
+ integer label `y`) and trains this graph for 4 epochs with Adam(0.01). It
then asserts the full artifact set, checkpoint↔model consistency, loss
behaviour, CPU reproducibility and device/dtype. Model-level equivalence is
covered separately by `npm run verify:reference`.
