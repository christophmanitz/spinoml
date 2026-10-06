# Auto-generiert aus dem Trainingsgraphen (SpinoML).
# Aufgabe: classification · Loss CrossEntropyLoss · 5 Epochen · Batch 32
#
# Hinweis: Der echte Lauf nutzt den Sidecar (training_template.py); dies ist
# das lesbare Äquivalent des kompilierten Plans.

import torch
from torch.utils.data import DataLoader, random_split

# ── Konfiguration (aus dem Graphen) ──
MODEL_PATH   = 'models/dual.spinoml'
DATASET      = 'datasets/pairs.csv'
TARGET       = ''
FEATURES     = None  # alle Spalten außer target
EPOCHS       = 5
BATCH_SIZE   = 32
VAL_SPLIT    = 0.2
SEED         = 42

torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Daten ──
# Quelle: datasets/pairs.csv (vom DatasetSource-Knoten)
dataset = load_dataset(DATASET, target=TARGET, features=FEATURES)  # siehe Sidecar-Loader
n_val = int(len(dataset) * VAL_SPLIT)
train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ── Modell · Loss · Optimizer · Scheduler ──
model = build_model_from('models/dual.spinoml').to(device)  # die generierte nn.Module
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
scheduler = None
scaler = torch.cuda.amp.GradScaler()

# ── Trainingsschleife ──
for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()

    # Validierung
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            val_loss += criterion(model(x), y).item()
    val_loss /= max(1, len(val_loader))
    # zusätzliche Metriken: accuracy
    print(f'epoch {epoch}: val_loss={val_loss:.4f}')
