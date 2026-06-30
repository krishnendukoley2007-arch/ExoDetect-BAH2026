<div align="center">

# 🪐 ExoDetect v8.0

### AI-Enabled Detection of Exoplanets from Noisy Astronomical Light Curves

[![BAH2026](https://img.shields.io/badge/BAH2026-Problem%20Statement%207-blue?style=for-the-badge)](https://www.isro.gov.in)
[![Accuracy](https://img.shields.io/badge/CV%20Accuracy-78.4%25%20(leak--free)-brightgreen?style=for-the-badge)]()
[![Python](https://img.shields.io/badge/Python-3.11+-yellow?style=for-the-badge&logo=python)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)]()

**Bharatiya Antariksh Hackathon 2026 (BAH2026) | ISRO | Problem Statement 7**

**Author:** Krishnendu Koley | **Institution:** Jadavpur University, Kolkata

---

![ExoDetect Dashboard](https://img.shields.io/badge/Status-Submission%20Ready-success?style=flat-square)

</div>

---

## 🌟 What Is This?

ExoDetect is a complete end-to-end AI pipeline that **automatically detects and classifies exoplanet transit signals** from real NASA TESS light curves.

When a planet passes in front of its host star, the star dims slightly and periodically. ExoDetect finds these tiny dips — sometimes as small as 0.001% — in the brightness of thousands of stars, and determines whether each dip is a real planet, an eclipsing binary star, or a false alarm.

**Flagship Detection: Pi Mensae c (TIC 261136679)**
- Orbital period detected: **6.270 days** ✓ (matches NASA catalog exactly)
- Classification: **Exoplanet Candidate** — Sub-Neptune, ~2.1 Earth radii
- SNR: **19.4** (high confidence detection)

---

## 📊 Results

**Headline result (leak-free, methodologically validated):**

| Model | 5-Fold CV Accuracy | Samples |
|---|---|---|
| **RF+GB Ensemble** | **78.41% ± 2.32%** | 556 real NASA TOI stars |
| **1D-CNN** | **88.65% ± 2.92%** | 493 phase-folded curves |

> **Methodology note:** An earlier internal version reported 88.92% CV accuracy for the ensemble. We identified that this number came from oversampling the minority class *before* the train/test split — which let duplicate copies of the same star leak across folds and inflated the score. We corrected this by oversampling only inside each training fold, so no test sample's duplicate is ever seen during training. The corrected, leak-free number above (78.41%) is what we report and stand behind. We're flagging this ourselves rather than letting a reviewer catch it, because we think methodological honesty matters more than a higher headline number.

**How this compares to not using ML at all:**

| Method | Accuracy | Planet Recall | Planet Precision |
|---|---|---|---|
| BLS-power threshold only (no ML) | 58.1% | 97.5% | 58.5% |
| SNR threshold only (no ML) | 62.4% | 90.2% | 62.3% |
| RandomForest alone | 79.0% | 84.6% | 80.4% |
| GradientBoosting alone | 77.5% | 86.5% | 77.6% |
| **RF+GB Ensemble (ours)** | **78.4%** | **85.2%** | **78.9%** |

Naive thresholding on a single BLS feature catches almost every real planet (97.5% recall) but floods the result with false alarms — only 58.5% of its "planet" predictions are actually planets. Our ensemble trades a small amount of recall for a substantial precision gain: **+15-20 accuracy points over the best single-feature baseline**, while still catching the large majority of true planets. This is the realistic trade-off in transit classification — perfect recall is easy (flag everything), the hard part is not drowning astronomers in false candidates.

Both the RF+GB ensemble and the independently-trained 1D-CNN converge on similar accuracy on different representations of the same underlying signal (engineered features vs. raw phase-folded flux) — consistent with the models learning real transit structure rather than each overfitting to noise in its own way.

**Classification Report (Ensemble, held-out test set):**
```
                precision  recall  f1-score
false_positive     0.84     0.66    0.74
planet             0.79     0.91    0.84
accuracy                            0.80
```

---
## ⚡ Quick Start — Run Dashboard in 10 Minutes

Pre-trained models are included. No retraining needed. (If you don't have Git installed, get it from https://git-scm.com/downloads first.)

```bash
# 1. Clone the repo
git clone https://github.com/krishnendukoley2007-arch/ExoDetect-BAH2026.git
cd ExoDetect-BAH2026

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch dashboard
python -m streamlit run dashboard.py
```

Open your browser at `http://localhost:8501` — done.

---

## 🔬 Full Pipeline — Train From Scratch

Only needed if you want to retrain on fresh NASA data.
> ⚠️ Steps 2 and 4 take 3–5 hours each due to NASA MAST server download time.

```bash
# Step 1 — Pull 800 labeled stars from NASA TOI catalog (~2 min)
python build_targets.py

# Step 2 — Download TESS light curves + extract BLS features (~4 hours)
# Auto-resumes from checkpoint if interrupted — just run again
python extract_features.py

# Step 3 — Train RF+GB ensemble (~10 min)
python train_model.py

# Step 4 — Download phase-folded curves for CNN (~3 hours)
python collect_raw_curves.py

# Step 5 — Train 1D-CNN (~30 min)
python train_cnn.py

# Step 6 — Launch dashboard
python -m streamlit run dashboard.py
```

---

## 🏗️ Pipeline Architecture

```
NASA MAST / TESS Satellite
           │
           ▼
┌──────────────────────────────────────────┐
│  Stage 1: Data Acquisition               │
│  lightkurve — SPOC 2-min cadence         │
│  Fallback: TESS-SPOC FFI → QLP FFI       │
├──────────────────────────────────────────┤
│  Stage 2: De-trending                    │
│  Savitzky-Golay flatten (window=401)     │
│  Outlier removal (4-sigma clip)          │
├──────────────────────────────────────────┤
│  Stage 3: BLS Period Search              │
│  Bounded grid (≤2000 periods)            │
│  Auto-retry on grid overflow             │
│  Narrow search ±3% around known period   │
├──────────────────────────────────────────┤
│  Stage 4: Feature Extraction             │
│  6 physical features per star            │
├──────────────────────────────────────────┤
│  Stage 5: ML Classification              │
│  RF+GB Ensemble (78.4% leak-free CV)     │
│  1D-CNN on phase-folded curves           │
├──────────────────────────────────────────┤
│  Stage 6: AI Interpretation              │
│  Plain-English signal analysis           │
│  Confidence score + rule-based crosscheck│
└──────────────────────────────────────────┘
           │
           ▼
    Streamlit Dashboard
```

---

## 🔢 The 6 Features

| Feature | What It Measures | Why It Matters |
|---|---|---|
| `depth` | Fractional dip in stellar brightness | Planets: <1% dip. Binaries: >1% |
| `snr` | Signal-to-noise ratio of transit | Higher SNR = more confident detection |
| `sec_ratio` | Secondary eclipse / primary eclipse depth | Binaries show large secondary eclipses at phase 0.5 |
| `duration_hours` | How long the transit lasts | Linked to planet size and orbital distance |
| `bls_power` | Box Least Squares periodogram peak power | Measures strength of periodicity |
| `odd_even_diff` | Depth alternation between consecutive transits | Binaries show alternating depths |

**Feature Importance (from RandomForest, leak-free retrain):**
```
duration_hours  : 25.56%  ← most discriminating
depth           : 18.24%
snr             : 16.30%
bls_power       : 14.12%
odd_even_diff   : 13.63%
sec_ratio       : 12.14%
```

---

## 🧠 1D-CNN Architecture

The CNN operates on raw phase-folded flux curves (493 samples) — a completely independent data representation from the 6 engineered BLS features used by the RF+GB ensemble. Architecture:

```
Input: 1 × N (phase-folded flux, normalized)
  │
  ├─ Conv1d(1→8,  kernel=7) → BatchNorm → ReLU → MaxPool(2)
  ├─ Conv1d(8→16, kernel=5) → BatchNorm → ReLU → MaxPool(2)
  ├─ Conv1d(16→32,kernel=3) → BatchNorm → ReLU → AdaptiveAvgPool(4)
  │
  ├─ Flatten → Linear(128→32) → ReLU → Dropout(0.5)
  └─ Linear(32→2)  [false_positive / planet]
```

**Design choices and why:**
- **Deliberately small (≈10K params).** An earlier, larger CNN was prone to memorizing the ~144-sample dataset rather than learning transit shape. This version trades capacity for generalization.
- **Train-time augmentation:** random circular phase-shift (±8 bins) to simulate epoch timing jitter, plus Gaussian flux noise (σ=0.01) to simulate photometric scatter — applied only during training, not validation.
- **Early stopping** (patience=6 epochs) on validation loss, with the best-epoch weights restored before evaluation.
- **5-fold stratified cross-validation**, identical methodology to the ensemble, so both models' reported accuracies are directly comparable.

---

## 🇮🇳 Relevance to ISRO / Indian Space Missions

India does not currently operate a dedicated continuous-photometry transit-survey mission analogous to TESS or Kepler — AstroSat's instruments are optimized for UV/X-ray imaging and spectroscopy rather than long-baseline stellar photometry, so it isn't a direct fit for transit detection in its current form.

The relevant point for ISRO is architectural, not mission-specific: **ExoDetect is designed as a data-format-agnostic triage layer.** The pipeline ingests any time-series flux curve, runs BLS period search, extracts 6 physics-informed features, and classifies — none of which assumes TESS-specific calibration. A future ISRO-led photometric survey instrument would need only a data-ingestion adapter (matching its file format to our `lightkurve`-based loader), not a redesign of the detection or classification logic. As India's space program scales toward higher-cadence, higher-volume observational data, automated triage of this kind becomes necessary simply because manual vetting of thousands of candidate signals doesn't scale — this is the same problem NASA's pipelines (Astronet, ExoMiner, Robovetter) were built to solve for Kepler/TESS.

---

## 🖥️ Dashboard Features

| Page | What It Does |
|---|---|
| **🔭 Individual Analysis** | Enter any TIC ID → full analysis in real time |
| **⚖️ Compare Stars** | Side-by-side comparison of multiple stars with overlaid transit shapes |
| **📄 Project Report** | Full methodology, dataset stats, model validation plots |
| **📜 History** | Every analysis saved in session, exportable as CSV |

**Each analysis produces:**
- Raw light curve plot
- BLS periodogram with detected period marked
- Phase-folded transit with binned model
- Secondary eclipse check
- ML classification + confidence %
- Planet radius estimate (Earth radii)
- AI plain-English interpretation
- Downloadable PDF report

---

## 📦 Dataset

| Property | Value |
|---|---|
| Source | NASA Exoplanet Archive — TOI catalog |
| Total stars | 556 labeled |
| Planets | 325 confirmed (KP/CP/PC disposition) |
| False positives | 231 (FP/FA disposition) |
| Data type | Real TESS SPOC 2-min + FFI fallback |
| Features | 6 BLS-derived physical features |

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| `lightkurve` | ≥2.4.0 | TESS data download and processing |
| `scikit-learn` | ≥1.3.0 | RF+GB ensemble classifier |
| `PyTorch` | ≥2.0.0 | 1D-CNN on phase-folded curves |
| `astropy` | ≥5.3.0 | Astronomical unit and time handling |
| `streamlit` | ≥1.28.0 | Interactive web dashboard |
| `pandas/numpy` | latest | Data processing |
| NASA MAST | — | TESS light curve archive |

---

## 📁 File Structure

```
ExoDetect-BAH2026/
│
├── 🚀 dashboard.py               ← Main Streamlit app — run this first
│
├── 📊 Pipeline Scripts
│   ├── build_targets.py          ← Downloads 800 labeled stars from NASA
│   ├── extract_features.py       ← Extracts BLS features (resumes on crash)
│   ├── train_model.py            ← Trains RF+GB ensemble (leak-free CV)
│   ├── baseline_comparison.py    ← BLS/SNR-only baselines vs RF/GB/Ensemble
│   ├── collect_raw_curves.py     ← Downloads phase-folded curves
│   ├── train_cnn.py              ← Trains 1D-CNN
│   └── add_eb_targets.py         ← Adds eclipsing binary samples
│
├── 🤖 Pre-trained Models (included — no retraining needed)
│   ├── exoplanet_classifier.pkl  ← RF+GB Ensemble (78.4% leak-free CV)
│   ├── rf_model.pkl              ← RandomForest component
│   ├── gb_model.pkl              ← GradientBoosting component
│   └── exoplanet_cnn.pth         ← 1D-CNN weights (88.65% CV accuracy)
│
├── 📈 Dataset (included)
│   └── features_dataset.csv      ← 556-star labeled feature dataset
│
├── raw_curves/                   ← Phase-folded .npy curves (493 files)
├── requirements.txt              ← Python dependencies
├── SETUP.md                      ← Detailed setup guide for beginners
└── README.md                     ← This file
```

---

## 🌍 Try These Stars

| TIC ID | Star | Type | Expected Result |
|---|---|---|---|
| `261136679` | Pi Mensae c | Super-Earth | Exoplanet Candidate, ~6.27d |
| `25155310` | WASP-126 b | Hot Jupiter | Exoplanet Candidate, short period |
| `441075486` | Known FP | Eclipsing Binary | False Positive / Binary |

---

## ❓ Common Issues

**"No TESS data found"**
That star has no TESS 2-min cadence data. Try a different TIC ID or widen the search.

**Dashboard is slow on first run**
It downloads real-time data from NASA's servers. Normal — takes 30–60 seconds per star.

**Graph shows no clear transit**
Narrow your period range closer to the expected value, or increase sectors to stack to 3–5.

**Model gives wrong classification**
Check the SNR — if below 5, the signal is too weak to classify reliably. Try more sectors.

---

## 📋 Problem Statement

**BAH2026 PS7:** Develop an AI-based pipeline capable of automatically detecting exoplanet transit signals from noisy TESS light curve data, including:
- ✅ Periodic dip identification
- ✅ Classification (transit vs eclipse vs false positive)
- ✅ SNR / confidence estimation
- ✅ Parameter extraction (period, depth, duration)
- ✅ Visualization of detected signals

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

<div align="center">

**ExoDetect v8.0 | Bharatiya Antariksh Hackathon 2026 | Jadavpur University**

*Built with real NASA TESS data | Trained on 556 confirmed stars | 78.4% leak-free CV accuracy*

</div>
