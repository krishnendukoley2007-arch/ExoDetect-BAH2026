import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# ════════════════════════════════════════════════════════════
# DATASET — with light augmentation for the TRAIN split only
# ════════════════════════════════════════════════════════════
class LightCurveDataset(Dataset):
    def __init__(self, fluxes, labels, augment=False):
        self.fluxes = fluxes
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.fluxes)

    def __getitem__(self, idx):
        flux = self.fluxes[idx].copy().astype(np.float32)

        if self.augment:
            # Random circular phase shift — epoch jitter. Keeps the
            # transit SHAPE intact but moves its position slightly,
            # forcing the model to learn shape, not exact pixel position.
            shift = np.random.randint(-8, 9)
            flux = np.roll(flux, shift)
            # Small Gaussian noise — simulates photometric scatter.
            # np.random.normal() defaults to float64, so we force
            # float32 back to match what PyTorch's conv layers expect.
            noise = np.random.normal(0, 0.01, size=flux.shape).astype(np.float32)
            flux = flux + noise
            flux = np.clip(flux, 0.0, 1.0).astype(np.float32)

        flux_t = torch.from_numpy(flux.astype(np.float32)).unsqueeze(0)
        label_t = torch.tensor(self.labels[idx], dtype=torch.long)
        return flux_t, label_t


# ════════════════════════════════════════════════════════════
# SIMPLIFIED CNN — far fewer parameters than before, to avoid
# memorizing a 144-sample dataset instead of learning from it.
# ════════════════════════════════════════════════════════════
class ExoplanetCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, padding=3),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def train_one_fold(X_train, y_train, X_val, y_val, device,
                    epochs=30, patience=6):
    train_ds = LightCurveDataset(X_train, y_train, augment=True)
    val_ds = LightCurveDataset(X_val, y_val, augment=False)

    # drop_last=True avoids a final batch of size 1, which would
    # crash BatchNorm1d during training mode.
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = ExoplanetCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for flux_batch, label_batch in train_loader:
            flux_batch, label_batch = flux_batch.to(device), label_batch.to(device)
            optimizer.zero_grad()
            output = model(flux_batch)
            loss = criterion(output, label_batch)
            loss.backward()
            optimizer.step()

        # ── Validation pass — used for early stopping ──
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for flux_batch, label_batch in val_loader:
                flux_batch, label_batch = flux_batch.to(device), label_batch.to(device)
                output = model(flux_batch)
                val_loss_total += criterion(output, label_batch).item()
        val_loss = val_loss_total / max(len(val_loader), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for flux_batch, label_batch in val_loader:
            output = model(flux_batch.to(device))
            preds.extend(output.argmax(dim=1).cpu().numpy())
            trues.extend(label_batch.numpy())

    return preds, trues


print("Loading raw light curves...")
curve_dir = "raw_curves"
fluxes_list = []
labels_list = []

for fname in os.listdir(curve_dir):
    if not fname.endswith('.npy'):
        continue
    fpath = os.path.join(curve_dir, fname)
    if 'planet' in fname and 'false' not in fname:
        labels_list.append(1)
    elif 'false_positive' in fname:
        labels_list.append(0)
    else:
        continue
    fluxes_list.append(np.load(fpath).astype(np.float32))

fluxes = np.array(fluxes_list, dtype=np.float32)
labels = np.array(labels_list)

print(f"Total curves: {len(fluxes)}")
print(f"Planets: {(labels==1).sum()} | False Positives: {(labels==0).sum()}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

# ════════════════════════════════════════════════════════════
# 5-FOLD STRATIFIED CROSS-VALIDATION
# This replaces a single lucky/unlucky 80-20 split with FIVE
# independent train/test splits, then reports the average — a
# scientifically honest measure of how well the model generalizes.
# ════════════════════════════════════════════════════════════
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_preds = []
all_trues = []
fold_accs = []

print("\nRunning 5-fold cross-validation with early stopping + augmentation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(fluxes, labels)):
    X_train, X_val = fluxes[train_idx], fluxes[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    preds, trues = train_one_fold(X_train, y_train, X_val, y_val, device)

    fold_acc = float(np.mean(np.array(preds) == np.array(trues)) * 100)
    fold_accs.append(fold_acc)
    print(f"  Fold {fold+1}: accuracy = {fold_acc:.1f}%")

    all_preds.extend(preds)
    all_trues.extend(trues)

print(f"\nMean 5-Fold CV Accuracy: {np.mean(fold_accs):.2f}% (+/- {np.std(fold_accs):.2f}%)")

print("\n========== CNN CLASSIFICATION REPORT (pooled across folds) ==========")
print(classification_report(all_trues, all_preds, target_names=['False Positive', 'Planet']))

cm = confusion_matrix(all_trues, all_preds)
print("Confusion Matrix (pooled across folds):")
print(cm)

# ════════════════════════════════════════════════════════════
# FINAL DEPLOYMENT MODEL — trained on ALL data so the dashboard
# benefits from every sample. CV above is purely to honestly
# report how well this approach generalizes.
# ════════════════════════════════════════════════════════════
print("\nTraining final deployment model on full dataset...")
final_ds = LightCurveDataset(fluxes, labels, augment=True)
final_loader = DataLoader(final_ds, batch_size=16, shuffle=True, drop_last=True)

final_model = ExoplanetCNN().to(device)
optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(30):
    final_model.train()
    for flux_batch, label_batch in final_loader:
        flux_batch, label_batch = flux_batch.to(device), label_batch.to(device)
        optimizer.zero_grad()
        output = final_model(flux_batch)
        loss = criterion(output, label_batch)
        loss.backward()
        optimizer.step()

torch.save(final_model.state_dict(), "exoplanet_cnn.pth")
print("Final model saved: exoplanet_cnn.pth")

# ════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar([f"Fold {i+1}" for i in range(5)], fold_accs, color='steelblue')
axes[0].axhline(y=float(np.mean(fold_accs)), color='red', linestyle='--',
                 label=f'Mean: {np.mean(fold_accs):.1f}%')
axes[0].set_title("5-Fold CV Accuracy")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_ylim(0, 100)
axes[0].legend()

im = axes[1].imshow(cm, cmap='Blues')
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['False Positive', 'Planet'])
axes[1].set_yticklabels(['False Positive', 'Planet'])
axes[1].set_title("Pooled Confusion Matrix (5-Fold CV)")
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, cm[i, j], ha='center', va='center',
                      color='white' if cm[i, j] > cm.max()/2 else 'black',
                      fontsize=14, fontweight='bold')
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig("cnn_validation.png", dpi=150)
plt.show()
print("CNN validation plot saved: cnn_validation.png")
print("Next: update dashboard to use ENSEMBLE (RF + CNN combined)!")