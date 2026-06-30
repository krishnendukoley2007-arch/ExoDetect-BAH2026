# 🪐 ExoDetect v8.0

### AI-Enabled Detection of Exoplanets from Noisy Astronomical Light Curves

![BAH2026](https://img.shields.io/badge/BAH2026-Problem%20Statement%207-blue?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/CV%20Accuracy-87.35%25-brightgreen?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-yellow?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)

**Bharatiya Antariksh Hackathon 2026 (BAH2026) | ISRO | Problem Statement 7**

**Team:** OrbitX2026 | Jadavpur University, Kolkata

| Role | Name | Email |
|---|---|---|
| Team Leader | Krishnendu Koley | krishnendukoley2007@gmail.com |
| Team Member | Abhradeep Bera | abhradeepmsk23@gmail.com |
| Team Member | Asmit Dey | deyasmit07@gmail.com |

---

## 🌟 What Is This?

ExoDetect is a complete end-to-end AI pipeline that **automatically detects and classifies exoplanet transit signals** from real NASA TESS light curves.

When a planet passes in front of its host star, the star dims slightly and periodically. ExoDetect finds these tiny dips — sometimes as small as 0.001% — in the brightness of thousands of stars, and determines whether each dip is a real planet, an eclipsing binary star, or a false alarm.


## 📊 Results

| Model | 5-Fold CV Accuracy | Samples |
|---|---|---|
| **RF+GB Ensemble** | **87.35%** | 641 real NASA TOI stars |
| **1D-CNN** | 88.65% ± 2.92% | 493 phase-folded curves |

**Classification Report (Ensemble, post-retrain):**
```
                precision  recall  f1-score
false_positive     0.88-0.92 range across both classes
planet
accuracy           ~0.87 (5-fold CV)
```

---

## ⚡ Quick Start — Run Dashboard in 10 Minutes

Pre-trained models are included. No retraining needed.

```bash
# 1. Clone the repo
git clone https://github.com/krishnendukoley2007-arch/ExoDetect-BAH2026.git
cd ExoDetect-BAH2026

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch dashboard
streamlit run dashboard.py
```

Open your browser at `http://localhost:8501` — done.

---

## 🔬 Full Pipeline — Train From Scratch

Only needed if you want to retrain on fresh NASA data.
> ⚠️ Steps 2 and 4 take 3–5 hours each due to NASA MAST server download time.

```bash
# Step 1 — Pull labeled stars from NASA TOI catalog
python build_targets.py

# Step 2 — Download TESS light curves + extract BLS features
# Auto-resumes from checkpoint if interrupted — just run again
python extract_features.py

# Step 3 — Train RF+GB ensemble
python train_model.py

# Step 4 — Download phase-folded curves for CNN
python collect_raw_curves.py

# Step 5 — Train 1D-CNN
python train_cnn.py

# Step 6 — Launch dashboard
streamlit run dashboard.py
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
│  RF+GB Ensemble (87.35% CV accuracy)     │
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

---

## 🖥️ Dashboard Features

| Page | What It Does |
|---|---|
| **🔭 Individual Analysis** | Enter any TIC ID → full analysis in real time |
| **🔬 Deep Diagnostics** | 5 additional diagnostic plots per star — odd/even transit comparison, secondary eclipse zoom, river plot, residuals, periodogram zoom — all downloadable as a single PDF |
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
- Downloadable PDF report (standard + deep diagnostics)

---

## 📦 Dataset

| Property | Value |
|---|---|
| Source | NASA Exoplanet Archive — TOI catalog |
| Total stars | 641 labeled |
| Planets | 399 confirmed (KP/CP/PC disposition) |
| False positives | 242 (FP/FA disposition) |
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
│   ├── build_targets.py          ← Downloads labeled stars from NASA
│   ├── extract_features.py       ← Extracts BLS features (resumes on crash)
│   ├── train_model.py            ← Trains RF+GB ensemble
│   ├── collect_raw_curves.py     ← Downloads phase-folded curves
│   ├── train_cnn.py              ← Trains 1D-CNN
│   └── add_eb_targets.py         ← Adds eclipsing binary samples
│
├── 🤖 Pre-trained Models (included — no retraining needed)
│   ├── exoplanet_classifier.pkl  ← RF+GB Ensemble (87.35% CV accuracy)
│   ├── rf_model.pkl              ← RandomForest component
│   ├── gb_model.pkl              ← GradientBoosting component
│   └── exoplanet_cnn.pth         ← 1D-CNN weights (88.65% CV accuracy)
│
├── 📈 Dataset (included)
│   └── features_dataset.csv      ← 641-star labeled feature dataset
│
├── raw_curves/                   ← Phase-folded .npy curves
├── requirements.txt              ← Python dependencies
└── README.md                     ← This file

```

---

## 🌍 Try These Stars

| TIC ID | Star | Type | Expected Result |
|---|---|---|---|
| `261136679` | Pi Mensae c | Sub-Neptune (~2.1 R⊕) | Exoplanet Candidate, ~6.27d, shallow transit (~320 ppm) |
| `25155310` | WASP-126 b | Hot Jupiter | Exoplanet Candidate, short period |
| `441075486` | Known FP | Eclipsing Binary | False Positive / Binary |

---

## ❓ Common Issues

**"No TESS data found"** — That star has no TESS 2-min cadence data. Try a different TIC ID or widen the search.

**Dashboard is slow on first run** — It downloads real-time data from NASA's servers. Normal — takes 30–60 seconds per star.

**Graph shows no clear transit** — Narrow your period range closer to the expected value, or increase sectors to stack to 3–5.

**BLS locks onto the wrong period** — On noisy/shallow signals, BLS can occasionally lock onto a harmonic (an integer multiple) of the true period instead of the true period itself. If your result period looks like a multiple of the expected value, narrow the Min/Max period sliders close to the expected period.

**Model gives wrong classification** — Check the SNR — if below 5, the signal is too weak to classify reliably. Try more sectors.

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

**ExoDetect v8.0 | Bharatiya Antariksh Hackathon 2026 | Jadavpur University**

*Built with real NASA TESS data | Trained on 641 confirmed/false-positive stars | 87.35% accuracy*
