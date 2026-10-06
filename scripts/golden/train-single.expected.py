# Auto-generiert aus dem Trainingsgraphen (SpinoML).
# Aufgabe: classification · Loss CrossEntropyLoss · 10 Epochen · Batch 16
#
# Hinweis: Der echte Lauf nutzt den Sidecar (training_template.py); dies ist
# das lesbare Äquivalent des kompilierten Plans.

import torch
from torch.utils.data import DataLoader, random_split

# ── Konfiguration (aus dem Graphen) ──
MODEL_PATH   = 'models/mlp.spinoml'
DATASET      = 'datasets/features.csv'
TARGET       = 'label'
FEATURES     = ['a', 'b']
EPOCHS       = 10
BATCH_SIZE   = 16
VAL_SPLIT    = 0.2
SEED         = 42

torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Daten ──
# Quelle: datasets/features.csv (vom DatasetSource-Knoten)
dataset = load_dataset(DATASET, target=TARGET, features=FEATURES)  # siehe Sidecar-Loader
n_val = int(len(dataset) * VAL_SPLIT)
train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ── Modell · Loss · Optimizer · Scheduler ──
model = build_model_from('models/mlp.spinoml').to(device)  # die generierte nn.Module
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
scheduler = None

# EarlyStopping: stoppt wenn val_loss 5 Epochen nicht besser wird
best_val, bad_epochs = float('inf'), 0
PATIENCE = 5

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
    # zusätzliche Metriken: accuracy
    print(f'epoch {epoch}: val_loss={val_loss:.4f}')
    if val_loss < best_val:
        best_val, bad_epochs = val_loss, 0
        torch.save(model.state_dict(), 'best.pt')
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print('early stopping'); break
