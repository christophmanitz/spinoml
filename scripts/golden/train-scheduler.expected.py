# Auto-generiert aus dem Trainingsgraphen (SpinoML).
# Aufgabe: regression · Loss MSELoss · 30 Epochen · Batch 32
#
# Hinweis: Der echte Lauf nutzt den Sidecar (training_template.py); dies ist
# das lesbare Äquivalent des kompilierten Plans.

import torch
from torch.utils.data import DataLoader, random_split

# ── Konfiguration (aus dem Graphen) ──
MODEL_PATH   = 'models/mlp.spinoml'
DATASET      = 'datasets/tab.csv'
TARGET       = 'y'
FEATURES     = None  # alle Spalten außer target
EPOCHS       = 30
BATCH_SIZE   = 32
VAL_SPLIT    = 0.2
SEED         = 42

torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Daten ──
# Quelle: datasets/tab.csv (vom DatasetSource-Knoten)
dataset = load_dataset(DATASET, target=TARGET, features=FEATURES)  # siehe Sidecar-Loader
n_val = int(len(dataset) * VAL_SPLIT)
train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ── Modell · Loss · Optimizer · Scheduler ──
model = build_model_from('models/mlp.spinoml').to(device)  # die generierte nn.Module
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# ── Trainingsschleife ──
for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    # Validierung
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            val_loss += criterion(model(x), y).item()
    val_loss /= max(1, len(val_loader))
    scheduler.step()
    print(f'epoch {epoch}: val_loss={val_loss:.4f}')
