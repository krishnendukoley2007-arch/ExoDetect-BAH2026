"""
train_model.py
Trains a RandomForest + GradientBoosting ensemble on extracted features.

v8.1 fix: oversampling is now applied AFTER the train/test split (and
per-fold during CV), never before. Oversampling before splitting let
duplicate copies of the same star appear in both train and test,
inflating the reported accuracy (was ~89%, corrected to ~78-79%).
The deployed model still trains on all available data — only the
*reported metrics* changed, not what ships to the dashboard.

  - 3-class output: planet / false_positive / eclipsing_binary
  - Ensemble of RF + GB
  - Oversampling for minority class balance, applied leak-free
  - Saves both individual models and the ensemble
Run AFTER extract_features.py.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection    import train_test_split, StratifiedKFold
from sklearn.metrics            import classification_report, confusion_matrix
from sklearn.utils               import resample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import os

print("=" * 60)
print("ExoDetect v8.1 — Step 3: Train Ensemble Model (leak-free eval)")
print("=" * 60)

# ── Load features ──────────────────────────────────────────
df = pd.read_csv("features_dataset.csv")
print(f"\nTotal samples: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts().to_string()}\n")

FEATURE_COLS = ["depth", "snr", "sec_ratio", "duration_hours",
                "bls_power", "odd_even_diff"]

# Clean inf/nan on the dataframe itself (a few stars have inf bls_power
# from BLS edge cases) — must happen before any split/balance step.
for col in FEATURE_COLS:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[col] = df[col].fillna(df[col].median())

X = df[FEATURE_COLS].values
y = df["label"].values


def oversample_balance(df_in, feature_cols, label_col="label", seed=42):
    """Oversample minority class(es) up to the size of the majority class.
    IMPORTANT: call this ONLY on a train split, never on data that also
    contains the test split — oversampling before splitting lets exact
    duplicate rows leak across train/test and inflates accuracy."""
    max_count = df_in[label_col].value_counts().max()
    parts = []
    for label_val in df_in[label_col].unique():
        subset = df_in[df_in[label_col] == label_val]
        if len(subset) < max_count:
            subset = resample(subset, replace=True, n_samples=max_count, random_state=seed)
        parts.append(subset)
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out[feature_cols].values, out[label_col].values


def make_models():
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=10, min_samples_leaf=2,
        random_state=42, class_weight="balanced", n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.8, random_state=42
    )
    ensemble = VotingClassifier(estimators=[("rf", rf), ("gb", gb)], voting="soft")
    return rf, gb, ensemble


# ── Held-out test split for the headline report numbers ───────────────
# Split FIRST on raw (unbalanced) data, THEN oversample only the train
# side. The test side stays untouched and real-world-distributed.
print("[1/4] Splitting before balancing (leak-free)...")
df_train_raw, df_test_raw = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
X_train, y_train = oversample_balance(df_train_raw, FEATURE_COLS)
X_test = df_test_raw[FEATURE_COLS].values
y_test = df_test_raw["label"].values
print(f"  Train (balanced): {len(X_train)} | Test (real distribution): {len(X_test)}")
print(f"  Test label distribution:\n{pd.Series(y_test).value_counts().to_string()}\n")

rf, gb, ensemble = make_models()

# ── Honest held-out evaluation ──────────────────────────────
# These models are trained ONLY on df_train_raw (oversampled). They
# have never seen df_test_raw in any form, so scoring them on X_test
# is a fair, leak-free measurement of generalization.
print("\n[2/4] Training RandomForest (train split only, for evaluation)...")
rf.fit(X_train, y_train)
rf_pred  = rf.predict(X_test)
rf_acc   = (rf_pred == y_test).mean() * 100
print(f"  RF Held-out Test Accuracy: {rf_acc:.1f}%")

print("\n[3/4] Training GradientBoosting (train split only, for evaluation)...")
gb.fit(X_train, y_train)
gb_pred  = gb.predict(X_test)
gb_acc   = (gb_pred == y_test).mean() * 100
print(f"  GB Held-out Test Accuracy: {gb_acc:.1f}%")

print("\n[4/4] Training Ensemble (train split only, for evaluation)...")
ensemble.fit(X_train, y_train)
ens_pred = ensemble.predict(X_test)
ens_acc  = (ens_pred == y_test).mean() * 100
print(f"  Ensemble Held-out Test Accuracy: {ens_acc:.1f}%")

# ── 5-fold CV — leak-free: balance INSIDE each fold, not before ───
print("\nRunning leak-free 5-fold cross-validation on ensemble...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accs = []
cv_y_true, cv_y_pred = [], []

for fold_i, (tr_idx, te_idx) in enumerate(skf.split(df[FEATURE_COLS], df["label"])):
    df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
    Xtr, ytr = oversample_balance(df_tr, FEATURE_COLS, seed=42 + fold_i)
    Xte = df_te[FEATURE_COLS].values
    yte = df_te["label"].values

    _, _, fold_ens = make_models()
    fold_ens.fit(Xtr, ytr)
    pred = fold_ens.predict(Xte)

    fold_acc = (pred == yte).mean() * 100
    fold_accs.append(fold_acc)
    cv_y_true.extend(yte)
    cv_y_pred.extend(pred)
    print(f"  Fold {fold_i+1}: {fold_acc:.2f}%")

cv_mean = float(np.mean(fold_accs))
cv_std = float(np.std(fold_accs))
print(f"5-Fold CV (leak-free): {cv_mean:.2f}% (+/- {cv_std:.2f}%)")
cv_scores = np.array(fold_accs) / 100.0  # keep variable name used below

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
# The models saved to disk are trained on ALL real data (oversampled
# once) so the deployed dashboard benefits from every star we have.
# This is a SEPARATE fit from the eval models above — it's fine for the
# deployed artifact to see everything, since we're not scoring it
# against itself. The numbers we report (rf_acc/gb_acc/ens_acc/cv_mean)
# all came from models that never saw their own test data.
print("\nTraining final deployment models on full dataset...")
rf_deploy, gb_deploy, ensemble_deploy = make_models()
X_full_bal, y_full_bal = oversample_balance(df, FEATURE_COLS, seed=42)
rf_deploy.fit(X_full_bal, y_full_bal)
gb_deploy.fit(X_full_bal, y_full_bal)
ensemble_deploy.fit(X_full_bal, y_full_bal)

joblib.dump(rf_deploy,       "rf_model.pkl")
joblib.dump(gb_deploy,       "gb_model.pkl")
joblib.dump(ensemble_deploy, "exoplanet_classifier.pkl")   # dashboard uses this
print("Models saved: rf_model.pkl / gb_model.pkl / exoplanet_classifier.pkl")

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
