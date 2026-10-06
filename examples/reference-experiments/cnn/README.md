# cnn reference experiment

Hand-written reference module: `RefCNN` in
`scripts/lib/reference_models.py`. Graph:
`cnnSnapshot()` in `scripts/lib/reference-graphs.ts`.

- Input x [1,64] float32 → Reshape [1,8,8] → Conv2d(1,4,k3,p1) → ReLU → MaxPool2d(2) → Flatten → Linear(64,2) → Output
- Expected parameters: 170 (trainable 170)

`model.spinoml` is the serialized graph. `scripts/verify-reference.ts`
regenerates the PyTorch model from it and numerically compares the generated
`Model` against the hand-written reference (parameters, forward passes, loss,
gradients, and a negative control).

## How to run this experiment

Train the committed graph end-to-end through the real trainer:

```bash
npm run verify:reference-train
```

The harness builds a synthetic seeded CSV (200 rows, 64 float features
`p0..p63` = an 8×8 image, + integer label `y`; class 1 has a bright 4×4 block
in the top-left quadrant, class 0 in the bottom-right, background N(0,0.3)) and
trains this graph for 6 epochs with Adam(0.01). It then asserts the full
artifact set, checkpoint↔model consistency, loss behaviour, CPU reproducibility
and device/dtype. Model-level equivalence is covered separately by
`npm run verify:reference`.
