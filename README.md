<div align="center">

# 🪐 ExoDetect v8.0

### AI-Enabled Detection of Exoplanets from Noisy Astronomical Light Curves

[![BAH2026](https://img.shields.io/badge/BAH2026-Problem%20Statement%207-blue?style=for-the-badge)](https://www.isro.gov.in)
[![Accuracy](https://img.shields.io/badge/CV%20Accuracy-88.92%25-brightgreen?style=for-the-badge)](https://github.com/krishnendukoley2007-arch/ExoDetect-BAH2026)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Download ZIP](https://img.shields.io/badge/⬇️%20Download-Ready%20to%20Run%20ZIP-orange?style=for-the-badge)](https://github.com/krishnendukoley2007-arch/ExoDetect-BAH2026/releases/latest)

**Bharatiya Antariksh Hackathon 2026 (BAH2026) | ISRO | Problem Statement 7**

**Team:** Krishnendu Koley (Leader) · Asmit Dey · Abhradeep Bera | Jadavpur University, Kolkata

</div>

---

## 🌟 What Is This?

ExoDetect is a complete end-to-end AI pipeline that **automatically detects and classifies exoplanet transit signals** from real NASA TESS light curves.

When a planet passes in front of its host star, the star dims slightly and periodically. ExoDetect finds these tiny dips — sometimes as small as 0.001% — in the brightness of thousands of stars, and determines whether each dip is a real planet, an eclipsing binary star, or a false alarm.

**Flagship Detection: Pi Mensae c (TIC 261136679)**
- Orbital period detected: **6.270 days** ✓ (matches NASA catalog, <0.01% error)
- Classification: **Exoplanet Candidate** — Sub-Neptune, ~2.1 Earth radii
- SNR: **19.4** · BLS Power: **2741** · ML Confidence: **94.2%**

---

## 📊 Results

| Model | 5-Fold CV Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **RF+GB Ensemble** | **88.92% ± 1.04%** | 0.92 | 0.92 | 0.92 |
| **1D-CNN (PyTorch)** | **88.65% ± 2.92%** | 0.89 | 0.89 | 0.88 |
| Baseline RF (v7) | 67.2% ± 9.2% | 0.68 | 0.67 | 0.67 |

Both models trained independently on 556 real NASA TOI stars reach ~88.9% accuracy — confirming genuine astrophysical signal learning, not overfitting. **21-point improvement** over the v7 single-model baseline.

---

## ⚡ How to Run — Choose Your Method

### ✅ Option 1 — Clone from GitHub (recommended if you have Git)

**Step 1: Install Git** (skip if already installed)
→ Download from: https://git-scm.com/downloads and install with default settings.

**Step 2: Clone and run**
```bash
git clone https://github.com/krishnendukoley2007-arch/ExoDetect-BAH2026.git
cd ExoDetect-BAH2026
pip install -r requirements.txt
python -m streamlit run dashboard.py
```

Open your browser at `http://localhost:8501` — done. Pre-trained models are included, no retraining needed.

---

### 📦 Option 2 — Download ZIP (easiest, no Git required)

**If you have trouble with Git, or just want the simplest possible setup — use this.**

**Step 1:** Download the ready-to-run ZIP from the [**Releases page →**](https://github.com/krishnendukoley2007-arch/ExoDetect-BAH2026/releases/latest)
> File name: `ExoDetect_BAH2026_RunPackage.zip`

**Step 2:** Extract the ZIP anywhere on your computer.

**Step 3:** Run the launcher for your OS:

| Your OS | What to do |
|---|---|
| **Windows** | Double-click **`▶ RUN ME - Windows.bat`** |
| **Mac / Linux** | Open Terminal in that folder → `bash "▶ RUN ME - Mac_Linux.sh"` |

**That's it.** The launcher automatically:
- Checks if Python is installed (opens python.org if not)
- Installs all required packages (first run only, ~2–3 minutes)
- Starts the dashboard and opens your browser at `http://localhost:8501`

> **Python not installed?** → https://www.python.org/downloads/
> Windows users: during install, check ✅ **"Add Python to PATH"**

---

## 🔬 Full Pipeline — Train From Scratch

Only needed if you want to retrain on fresh NASA data. Pre-trained models included are sufficient to run the dashboard.

> ⚠️ Steps 2 and 4 take 3–5 hours each due to NASA MAST server download times.

```bash
python build_targets.py       # Step 1 — Pull labeled stars from NASA (~2 min)
python extract_features.py    # Step 2 — Download light curves + BLS features (~4 hrs, auto-resumes)
python train_model.py         # Step 3 — Train RF+GB ensemble (~10 min)
python collect_raw_curves.py  # Step 4 — Download phase-folded curves (~3 hrs)
python train_cnn.py           # Step 5 — Train 1D-CNN (~30 min GPU / 2 hrs CPU)
python -m streamlit run dashboard.py  # Step 6 — Launch
```

---

## 🏗️ Pipeline Architecture
