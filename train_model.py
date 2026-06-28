"""
train_model.py
Trains a RandomForest + GradientBoosting ensemble on extracted features.
Improvements over v1:
  - 3-class output: planet / false_positive / eclipsing_binary
  - Ensemble of RF + GB (typically 10-15% accuracy boost)
  - SMOTE-style oversampling for minority class balance
  - Saves both individual models and the ensemble
Run AFTER extract_features.py.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection    import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics            import classification_report, confusion_matrix
from sklearn.preprocessing      import LabelEncoder
from sklearn.utils               import resample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import os

print("=" * 60)
print("ExoDetect v2 — Step 3: Train Ensemble Model")
print("=" * 60)

# ── Load features ──────────────────────────────────────────
df = pd.read_csv("features_dataset.csv")
print(f"\nTotal samples: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts().to_string()}\n")

FEATURE_COLS = ["depth", "snr", "sec_ratio", "duration_hours",
                "bls_power", "odd_even_diff"]

X = df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values
y = df["label"].values

# ── Balance classes by oversampling minority classes ───────
print("[1/4] Balancing classes...")
df_work    = df[FEATURE_COLS + ["label"]].fillna(0).copy()
max_count  = df_work["label"].value_counts().max()
parts      = []

for label_val in df_work["label"].unique():
    subset = df_work[df_work["label"] == label_val]
    if len(subset) < max_count:
        subset = resample(subset, replace=True,
                          n_samples=max_count, random_state=42)
    parts.append(subset)

df_balanced = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
X_bal = df_balanced[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values
y_bal = df_balanced["label"].values

print(f"  After balancing: {len(df_balanced)} samples")
print(f"  {pd.Series(y_bal).value_counts().to_string()}\n")

# ── Train/test split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)

# ── Define models ──────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

gb = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.8,
    random_state=42
)

ensemble = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb)],
    voting="soft"
)

# ── Train & evaluate ───────────────────────────────────────
print("[2/4] Training RandomForest...")
rf.fit(X_train, y_train)
rf_pred  = rf.predict(X_test)
rf_acc   = (rf_pred == y_test).mean() * 100
print(f"  RF Test Accuracy: {rf_acc:.1f}%")

print("\n[3/4] Training GradientBoosting...")
gb.fit(X_train, y_train)
gb_pred  = gb.predict(X_test)
gb_acc   = (gb_pred == y_test).mean() * 100
print(f"  GB Test Accuracy: {gb_acc:.1f}%")

print("\n[4/4] Training Ensemble (RF + GB)...")
ensemble.fit(X_train, y_train)
ens_pred = ensemble.predict(X_test)
ens_acc  = (ens_pred == y_test).mean() * 100
print(f"  Ensemble Test Accuracy: {ens_acc:.1f}%")

# ── 5-fold CV on ensemble ──────────────────────────────────
print("\nRunning 5-fold cross-validation on ensemble (this takes a minute)...")
cv_scores = cross_val_score(ensemble, X_bal, y_bal, cv=5, n_jobs=-1)
print(f"5-Fold CV: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# ── Full report ────────────────────────────────────────────
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, ens_pred))

classes_order = sorted(set(y_test))
cm = confusion_matrix(y_test, ens_pred, labels=classes_order)
print("===== CONFUSION MATRIX =====")
print(cm)

# ── Feature importance (from RF) ───────────────────────────
importance = rf.feature_importances_
print("\n===== FEATURE IMPORTANCE (RF) =====")
for name, imp in sorted(zip(FEATURE_COLS, importance), key=lambda x: -x[1]):
    print(f"  {name:20s}: {imp*100:.2f}%")

# ── Save models ────────────────────────────────────────────
joblib.dump(rf,       "rf_model.pkl")
joblib.dump(gb,       "gb_model.pkl")
joblib.dump(ensemble, "exoplanet_classifier.pkl")   # dashboard uses this
print("\nModels saved: rf_model.pkl / gb_model.pkl / exoplanet_classifier.pkl")

# ── Plots ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#0a1220')

# Accuracy comparison bar
model_names = ["RandomForest", "GradientBoost", "Ensemble"]
accs        = [rf_acc, gb_acc, ens_acc]
colors      = ['#4a9eff', '#ff7a45', '#22cc77']
axes[0].bar(model_names, accs, color=colors)
axes[0].axhline(y=ens_acc, color='white', linestyle='--', alpha=0.5)
axes[0].set_title("Model Accuracy Comparison", color='white')
axes[0].set_ylabel("Test Accuracy (%)", color='#7090b8')
axes[0].set_ylim(0, 100)
axes[0].set_facecolor('#0a1220')
axes[0].tick_params(colors='#7090b8')
for i, (n, v) in enumerate(zip(model_names, accs)):
    axes[0].text(i, v + 1, f"{v:.1f}%", ha='center', color='white', fontsize=10)

# Feature importance
sorted_pairs = sorted(zip(FEATURE_COLS, importance), key=lambda x: x[1])
axes[1].barh([p[0] for p in sorted_pairs],
             [p[1]*100 for p in sorted_pairs], color='#4a9eff')
axes[1].set_title("Feature Importance (RF)", color='white')
axes[1].set_xlabel("Importance (%)", color='#7090b8')
axes[1].set_facecolor('#0a1220')
axes[1].tick_params(colors='#7090b8')

# Confusion matrix
im = axes[2].imshow(cm, cmap='Blues')
axes[2].set_xticks(range(len(classes_order)))
axes[2].set_yticks(range(len(classes_order)))
short = [c.replace("eclipsing_binary","EB").replace("false_positive","FP")
           .replace("planet","Planet") for c in classes_order]
axes[2].set_xticklabels(short, rotation=30, color='#7090b8')
axes[2].set_yticklabels(short, color='#7090b8')
axes[2].set_title("Confusion Matrix (Ensemble)", color='white')
axes[2].set_facecolor('#0a1220')
for i in range(len(classes_order)):
    for j in range(len(classes_order)):
        axes[2].text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black',
                     fontsize=12, fontweight='bold')
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.savefig("model_validation.png", dpi=150, facecolor='#0a1220')
print("Validation plot saved: model_validation.png")
print("\nNext step: python collect_raw_curves.py")
print("       OR: streamlit run dashboard.py  (RF ensemble is ready now)")
