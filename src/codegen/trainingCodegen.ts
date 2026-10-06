// Training-graph → readable PyTorch training script. PURE (same plan → same
// code, no I/O) like generator.ts. This is the inspectable counterpart to the
// architecture CodePreview: it turns the compiled TrainingPlan into the loop
// the Phase-13 trainer (sidecar-torch/training_template.py) actually runs —
// dataset load, model, optimizer/loss/scheduler, the epoch loop and callbacks.
// It is illustrative (the live run uses the sidecar template) but faithful to
// the plan's config, so the user can read exactly what their graph will do.

import type { TrainingPlan } from './trainingGenerator'
import { pyStr, pyComment, pyInt, pyFloat } from './pyLiteral'

const LOSS_KINDS = ['CrossEntropyLoss', 'BCEWithLogitsLoss', 'MSELoss', 'L1Loss']
const OPTIMIZER_KINDS = ['Adam', 'AdamW', 'SGD', 'RMSprop']

/** Validate a user/LLM-supplied enum against the known list; anything else
 *  falls back to the safe default so it can never reach `torch.nn. …` /
 *  `torch.optim. …` as an injected attribute name. */
function safeLoss(kind: unknown): string {
  const k = String(kind)
  return LOSS_KINDS.includes(k) ? k : 'CrossEntropyLoss'
}

function safeOptimizer(kind: unknown): string {
  const k = String(kind)
  return OPTIMIZER_KINDS.includes(k) ? k : 'Adam'
}

function optimizerLine(o: TrainingPlan['training']['optimizer']): string {
  const kind = safeOptimizer(o.kind)
  const base = `lr=${pyFloat(o.lr, 1e-3)}, weight_decay=${pyFloat(o.weight_decay, 0)}`
  if (kind === 'SGD') return `torch.optim.SGD(model.parameters(), ${base}, momentum=${pyFloat(o.momentum, 0.9)})`
  return `torch.optim.${kind}(model.parameters(), ${base})`
}

function schedulerBlock(s: TrainingPlan['training']['scheduler'], epochs: number): string[] {
  const get = (k: string, d: number) => (typeof s[k] === 'number' ? (s[k] as number) : d)
  switch (s.kind) {
    case 'StepLR':
      return [`scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=${pyInt(get('step_size', 30), 30)}, gamma=${pyFloat(get('gamma', 0.1), 0.1)})`]
    case 'CosineAnnealingLR':
      return [`scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=${pyInt(get('t_max', epochs), epochs)})`]
    case 'ReduceLROnPlateau':
      return [`scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=${pyInt(get('patience', 10), 10)})`]
    default:
      return ['scheduler = None']
  }
}

const LOSS_TASK: Record<string, string> = {
  CrossEntropyLoss: 'classification',
  BCEWithLogitsLoss: 'binary classification',
  MSELoss: 'regression',
  L1Loss: 'regression',
}

export function generateTrainingCode(plan: TrainingPlan | null): string {
  if (!plan) {
    return '# Trainingsgraph noch nicht startklar.\n# Es fehlt mindestens: DatasetSource(+target), ModelSource(+model), Loss, Optimizer, TrainLoop.\n# Verdrahte die Komponenten in den TrainLoop, dann erscheint hier das Skript.'
  }

  const t = plan.training
  const lossKind = safeLoss(t.loss.kind)
  const task = LOSS_TASK[lossKind] ?? 'training'
  const callbacks = t.callbacks ?? []
  const hasEarlyStop = callbacks.some((c) => c.kind === 'EarlyStopping')
  const clip = callbacks.find((c) => c.kind === 'GradientClipping')
  const amp = callbacks.some((c) => c.kind === 'MixedPrecision')

  const L: string[] = []
  L.push('# Auto-generiert aus dem Trainingsgraphen (SpinoML).')
  L.push(`# ${pyComment(`Aufgabe: ${task} · Loss ${lossKind} · ${pyInt(t.epochs, 50)} Epochen · Batch ${pyInt(t.batch_size, 32)}`)}`)
  L.push('#')
  L.push('# Hinweis: Der echte Lauf nutzt den Sidecar (training_template.py); dies ist')
  L.push('# das lesbare Äquivalent des kompilierten Plans.')
  L.push('')
  L.push('import torch')
  L.push('from torch.utils.data import DataLoader, random_split')
  L.push('')
  L.push('# ── Konfiguration (aus dem Graphen) ──')
  L.push(`MODEL_PATH   = ${pyStr(plan.modelRelpath)}`)
  L.push(`DATASET      = ${pyStr(plan.datasetRelpath)}`)
  L.push(`TARGET       = ${pyStr(plan.target)}`)
  L.push(`FEATURES     = ${plan.features ? `[${plan.features.map((f) => pyStr(f)).join(', ')}]` : 'None  # alle Spalten außer target'}`)
  L.push(`EPOCHS       = ${pyInt(t.epochs, 50)}`)
  L.push(`BATCH_SIZE   = ${pyInt(t.batch_size, 32)}`)
  L.push(`VAL_SPLIT    = ${pyFloat(t.val_split, 0.2)}`)
  L.push(`SEED         = ${pyInt(t.seed, 42)}`)
  L.push('')
  L.push('torch.manual_seed(SEED)')
  L.push("device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
  L.push('')
  L.push('# ── Daten ──')
  L.push(`# ${pyComment(`Quelle: ${plan.datasetRelpath} (vom DatasetSource-Knoten)`)}`)
  L.push('dataset = load_dataset(DATASET, target=TARGET, features=FEATURES)  # siehe Sidecar-Loader')
  L.push('n_val = int(len(dataset) * VAL_SPLIT)')
  L.push('train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])')
  const dlExtra = `${t.num_workers ? `, num_workers=${pyInt(t.num_workers, 0)}` : ''}${t.drop_last ? ', drop_last=True' : ''}`
  L.push(`train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=${t.shuffle === false ? 'False' : 'True'}${dlExtra})`)
  L.push('val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)')
  L.push('')
  L.push('# ── Modell · Loss · Optimizer · Scheduler ──')
  L.push(`model = build_model_from(${pyStr(plan.modelRelpath)}).to(device)  # die generierte nn.Module`)
  const lossArgs = (lossKind === 'CrossEntropyLoss' && t.loss.label_smoothing) ? `label_smoothing=${pyFloat(t.loss.label_smoothing, 0)}` : ''
  L.push(`criterion = torch.nn.${lossKind}(${lossArgs})`)
  L.push(`optimizer = ${optimizerLine(t.optimizer)}`)
  for (const line of schedulerBlock(t.scheduler, t.epochs)) L.push(line)
  if (amp) L.push('scaler = torch.cuda.amp.GradScaler()')
  L.push('')
  if (hasEarlyStop) {
    const es = callbacks.find((c) => c.kind === 'EarlyStopping')!
    const patience = typeof es.patience === 'number' ? es.patience : 10
    L.push(`# ${pyComment(`EarlyStopping: stoppt wenn val_loss ${pyInt(patience, 10)} Epochen nicht besser wird`)}`)
    L.push('best_val, bad_epochs = float(\'inf\'), 0')
    L.push(`PATIENCE = ${pyInt(patience, 10)}`)
    L.push('')
  }
  L.push('# ── Trainingsschleife ──')
  L.push('for epoch in range(EPOCHS):')
  L.push('    model.train()')
  L.push('    for x, y in train_loader:')
  L.push('        x, y = x.to(device), y.to(device)')
  L.push('        optimizer.zero_grad()')
  if (amp) {
    L.push('        with torch.cuda.amp.autocast():')
    L.push('            out = model(x)')
    L.push('            loss = criterion(out, y)')
    L.push('        scaler.scale(loss).backward()')
    if (clip) {
      const mx = typeof clip.max_norm === 'number' ? clip.max_norm : 1.0
      L.push('        scaler.unscale_(optimizer)')
      L.push(`        torch.nn.utils.clip_grad_norm_(model.parameters(), ${pyFloat(mx, 1.0)})`)
    }
    L.push('        scaler.step(optimizer); scaler.update()')
  } else {
    L.push('        out = model(x)')
    L.push('        loss = criterion(out, y)')
    L.push('        loss.backward()')
    if (clip) {
      const mx = typeof clip.max_norm === 'number' ? clip.max_norm : 1.0
      L.push(`        torch.nn.utils.clip_grad_norm_(model.parameters(), ${pyFloat(mx, 1.0)})`)
    }
    L.push('        optimizer.step()')
  }
  L.push('')
  L.push('    # Validierung')
  L.push('    model.eval()')
  L.push('    val_loss = 0.0')
  L.push('    with torch.no_grad():')
  L.push('        for x, y in val_loader:')
  L.push('            x, y = x.to(device), y.to(device)')
  L.push('            val_loss += criterion(model(x), y).item()')
  L.push('    val_loss /= max(1, len(val_loader))')
  if (t.scheduler.kind === 'ReduceLROnPlateau') {
    L.push('    scheduler.step(val_loss)')
  } else if (t.scheduler.kind !== 'none') {
    L.push('    scheduler.step()')
  }
  if ((t.metrics ?? []).length) {
    L.push(`    # ${pyComment(`zusätzliche Metriken: ${(t.metrics ?? []).join(', ')}`)}`)
  }
  L.push("    print(f'epoch {epoch}: val_loss={val_loss:.4f}')")
  if (hasEarlyStop) {
    L.push('    if val_loss < best_val:')
    L.push('        best_val, bad_epochs = val_loss, 0')
    L.push("        torch.save(model.state_dict(), 'best.pt')")
    L.push('    else:')
    L.push('        bad_epochs += 1')
    L.push('        if bad_epochs >= PATIENCE:')
    L.push("            print('early stopping'); break")
  }
  L.push('')
  return L.join('\n')
}
