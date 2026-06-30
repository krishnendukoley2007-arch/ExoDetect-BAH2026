"""
baseline_comparison.py
========================
Answers the judge's #1 question: "89% sounds good, but compared to what?"

Evaluates FOUR approaches on the EXACT SAME 5-fold stratified CV splits,
on the real 556-star features_dataset.csv, so the comparison is apples
to apples:

  1. BLS-power-only baseline   (no ML — just a threshold on bls_power)
  2. SNR-only baseline          (no ML — just a threshold on snr)
  3. RandomForest alone
  4. GradientBoosting alone
  5. RF+GB Ensemble (your current production model)

For each: Accuracy, Precision/Recall/F1 on the 'planet' class specifically
(not just overall accuracy — recall on real planets is what actually
matters scientifically), and a confusion matrix.

Run from the ExoDetect-v8.0/ directory:
    python baseline_comparison.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
from sklearn.utils import resample
import json

RANDOM_STATE = 42
N_FOLDS = 5
FEATURE_COLS = ["depth", "snr", "sec_ratio", "duration_hours", "bls_power", "odd_even_diff"]

print("=" * 70)
print("ExoDetect — Baseline Comparison (BLS-only vs RF vs GB vs Ensemble)")
print("=" * 70)

# ── Load data exactly as train_model.py does ──────────────────────────
df = pd.read_csv("features_dataset.csv")
print(f"\nTotal samples: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts().to_string()}\n")

# Clean inf/nan in-place on the dataframe itself (a few stars have
# inf bls_power from BLS edge cases) so every downstream user of df
# — balance(), threshold baselines, model fits — sees clean values.
for col in FEATURE_COLS:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[col] = df[col].fillna(df[col].median())

X_raw = df[FEATURE_COLS]
y_raw = df["label"].values

PLANET_LABEL = "planet"


def balance(df_in, feature_cols, label_col="label"):
    """Same oversampling strategy as train_model.py, applied per-fold
    (fit only on train data — avoids leaking test samples via resampling)."""
    max_count = df_in[label_col].value_counts().max()
    parts = []
    for label_val in df_in[label_col].unique():
        subset = df_in[df_in[label_col] == label_val]
        if len(subset) < max_count:
            subset = resample(subset, replace=True, n_samples=max_count,
                               random_state=RANDOM_STATE)
        parts.append(subset)
    out = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    X_out = out[feature_cols].replace([np.inf, -np.inf], 0).fillna(0).values
    return X_out, out[label_col].values


def eval_predictions(y_true, y_pred, planet_label=PLANET_LABEL):
    acc = accuracy_score(y_true, y_pred)
    labels_sorted = sorted(set(y_true) | set(y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_sorted, zero_division=0
    )
    planet_idx = labels_sorted.index(planet_label) if planet_label in labels_sorted else None
    planet_recall = rec[planet_idx] if planet_idx is not None else float("nan")
    planet_prec = prec[planet_idx] if planet_idx is not None else float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    return {
        "accuracy": acc,
        "planet_recall": planet_recall,
        "planet_precision": planet_prec,
        "labels_order": labels_sorted,
        "confusion_matrix": cm.tolist(),
    }


def threshold_baseline_predict(values, threshold):
    """Single-feature threshold classifier: above threshold -> planet,
    else -> false_positive. Threshold chosen on TRAIN fold only."""
    return np.where(values >= threshold, "planet", "false_positive")


def best_threshold(train_values, train_labels, planet_label=PLANET_LABEL):
    """Scan candidate thresholds (sorted unique train values) and pick
    the one maximizing TRAIN accuracy. This is the best a 'no-ML, just
    eyeball a cutoff' approach could realistically do."""
    candidates = np.unique(train_values)
    best_acc, best_t = -1, candidates[0]
    y_bin = (train_labels == planet_label)
    for t in candidates:
        pred_bin = train_values >= t
        acc = (pred_bin == y_bin).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t


# ── Set up identical CV folds for every method ─────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
X_idx = np.arange(len(df))

results = {
    "bls_power_only": {"y_true": [], "y_pred": []},
    "snr_only":       {"y_true": [], "y_pred": []},
    "random_forest":  {"y_true": [], "y_pred": []},
    "grad_boosting":  {"y_true": [], "y_pred": []},
    "ensemble":       {"y_true": [], "y_pred": []},
}

print(f"Running identical {N_FOLDS}-fold stratified CV across all methods...\n")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_idx, y_raw)):
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    y_test = df_test["label"].values

    # Balance ONLY the training fold (test fold stays untouched/real-world)
    X_train_bal, y_train_bal = balance(df_train, FEATURE_COLS)
    X_test = df_test[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values

    # --- Baseline 1: BLS power threshold ---
    t_bls = best_threshold(df_train["bls_power"].fillna(0).values, df_train["label"].values)
    pred_bls = threshold_baseline_predict(df_test["bls_power"].fillna(0).values, t_bls)
    results["bls_power_only"]["y_true"].extend(y_test)
    results["bls_power_only"]["y_pred"].extend(pred_bls)

    # --- Baseline 2: SNR threshold ---
    t_snr = best_threshold(df_train["snr"].fillna(0).values, df_train["label"].values)
    pred_snr = threshold_baseline_predict(df_test["snr"].fillna(0).values, t_snr)
    results["snr_only"]["y_true"].extend(y_test)
    results["snr_only"]["y_pred"].extend(pred_snr)

    # --- RandomForest alone ---
    rf = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=2,
                                 random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
    rf.fit(X_train_bal, y_train_bal)
    pred_rf = rf.predict(X_test)
    results["random_forest"]["y_true"].extend(y_test)
    results["random_forest"]["y_pred"].extend(pred_rf)

    # --- GradientBoosting alone ---
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.08,
                                     subsample=0.8, random_state=RANDOM_STATE)
    gb.fit(X_train_bal, y_train_bal)
    pred_gb = gb.predict(X_test)
    results["grad_boosting"]["y_true"].extend(y_test)
    results["grad_boosting"]["y_pred"].extend(pred_gb)

    # --- RF+GB Ensemble (production model) ---
    ens = VotingClassifier(estimators=[("rf", rf), ("gb", gb)], voting="soft")
    ens.fit(X_train_bal, y_train_bal)
    pred_ens = ens.predict(X_test)
    results["ensemble"]["y_true"].extend(y_test)
    results["ensemble"]["y_pred"].extend(pred_ens)

    print(f"  Fold {fold+1}/{N_FOLDS} done "
          f"(train={len(df_train)}, test={len(df_test)})")

# ── Summarize ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS — pooled across all 5 folds (every star used as test exactly once)")
print("=" * 70)

summary_rows = []
pretty_names = {
    "bls_power_only": "BLS-power threshold only (no ML)",
    "snr_only":       "SNR threshold only (no ML)",
    "random_forest":  "RandomForest alone",
    "grad_boosting":  "GradientBoosting alone",
    "ensemble":       "RF+GB Ensemble (production model)",
}

final_json = {}

for key, name in pretty_names.items():
    r = results[key]
    metrics = eval_predictions(np.array(r["y_true"]), np.array(r["y_pred"]))
    summary_rows.append({
        "method": name,
        "accuracy_pct": round(metrics["accuracy"] * 100, 2),
        "planet_recall_pct": round(metrics["planet_recall"] * 100, 2),
        "planet_precision_pct": round(metrics["planet_precision"] * 100, 2),
    })
    final_json[key] = metrics
    print(f"\n{name}")
    print(f"  Overall accuracy:        {metrics['accuracy']*100:.2f}%")
    print(f"  Planet recall:           {metrics['planet_recall']*100:.2f}%   "
          f"(of real planets, % correctly caught)")
    print(f"  Planet precision:        {metrics['planet_precision']*100:.2f}%   "
          f"(of predicted planets, % actually real)")
    print(f"  Confusion matrix {metrics['labels_order']}:")
    for row in metrics["confusion_matrix"]:
        print(f"    {row}")

# ── Clean summary table ─────────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
print("\n" + "=" * 70)
print("SUMMARY TABLE (copy this into slides)")
print("=" * 70)
print(summary_df.to_string(index=False))

summary_df.to_csv("baseline_comparison_results.csv", index=False)
with open("baseline_comparison_results.json", "w") as f:
    json.dump(final_json, f, indent=2)

print("\nSaved: baseline_comparison_results.csv")
print("Saved: baseline_comparison_results.json")
print("\nUplift of ensemble over best no-ML baseline:")
best_baseline_acc = max(summary_df.loc[summary_df['method'].str.contains('no ML'), 'accuracy_pct'])
ens_acc = float(summary_df.loc[summary_df['method'].str.contains('Ensemble'), 'accuracy_pct'].iloc[0])
print(f"  {ens_acc:.2f}% (ensemble) vs {best_baseline_acc:.2f}% (best threshold-only baseline) "
      f"= +{ens_acc - best_baseline_acc:.2f} points from ML")
