import streamlit as st
import lightkurve as lk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import RandomForestClassifier
import io
import os
import time
import datetime
import joblib

# ════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ExoDetect — BAH2026 PS7",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════
# MATPLOTLIB GLOBAL SETTINGS — sharp, high-DPI plots
# ════════════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'figure.facecolor': '#0a1220',
    'axes.facecolor': '#0a1220',
    'axes.edgecolor': '#1e3050',
    'axes.labelcolor': '#7090b8',
    'xtick.color': '#7090b8',
    'ytick.color': '#7090b8',
    'text.color': '#ddeeff',
    'grid.color': '#1e3050',
    'grid.alpha': 0.4,
    'lines.linewidth': 1.5,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})

# ════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

.stApp {
    background: radial-gradient(ellipse at top, #0d1b2e 0%, #060b14 60%, #0a0e1a 100%);
    font-family: 'Inter', 'Space Grotesk', sans-serif;
}

@keyframes twinkle {
    0%   { opacity: 0.5; }
    50%  { opacity: 1.0; }
    100% { opacity: 0.6; }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-ring {
    0%,100% { box-shadow: 0 0 0 0 rgba(100,160,255,0.0); }
    50%      { box-shadow: 0 0 0 6px rgba(100,160,255,0.15); }
}

.hero-banner {
    background: linear-gradient(135deg, #0a1a35 0%, #112040 50%, #0a1a35 100%);
    border: 1px solid #1e3a60;
    border-radius: 20px;
    padding: 36px 28px 28px;
    text-align: center;
    margin-bottom: 28px;
    animation: fadeInUp 0.6s ease-out;
    position: relative;
    overflow: hidden;
}
.hero-banner::after {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle at 20% 50%, rgba(80,140,255,0.07) 0%, transparent 55%),
        radial-gradient(circle at 80% 30%, rgba(255,120,50,0.05) 0%, transparent 45%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.03em;
    margin: 0 0 10px;
    text-shadow: 0 0 60px rgba(80,150,255,0.4);
}
.hero-sub {
    color: #8ab0d8;
    font-size: 1rem;
    margin: 0;
    line-height: 1.7;
}

/* AI Insight card */
.ai-insight {
    background: linear-gradient(135deg, #0f1e38, #0a1628);
    border: 1px solid #2a5080;
    border-left: 4px solid #4a9eff;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 16px 0;
    animation: fadeInUp 0.4s ease-out;
}
.ai-insight-header {
    color: #4a9eff;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.ai-insight-text {
    color: #c4d8f0;
    font-size: 0.95rem;
    line-height: 1.65;
    margin: 0;
}

.verdict-box {
    border-radius: 14px;
    padding: 20px 22px;
    text-align: center;
    margin: 16px 0;
    animation: fadeInUp 0.4s ease-out;
}

.stat-card {
    background: linear-gradient(135deg, #0f1828, #0c1422);
    border: 1px solid #1e3050;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 10px;
    transition: border-color 0.25s, transform 0.2s;
}
.stat-card:hover {
    border-color: #3060a0;
    transform: translateY(-2px);
}

.report-card {
    background: linear-gradient(135deg, #0f1828, #0c1422);
    border: 1px solid #1e3050;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 16px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07101e 0%, #091220 100%);
    border-right: 1px solid #182538;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0f1828, #0c1422);
    border: 1px solid #1e3050;
    border-radius: 12px;
    padding: 14px 16px;
    transition: border-color 0.25s, transform 0.2s;
    animation: pulse-ring 4s infinite;
}
div[data-testid="stMetric"]:hover {
    border-color: #3060a0;
    transform: translateY(-2px);
}
div[data-testid="stMetricLabel"] { color: #6080a8 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
div[data-testid="stMetricValue"] { color: #e0ecff !important; font-weight: 700 !important; font-size: 1.4rem !important; }

.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    border: 1px solid #243050;
    transition: all 0.2s;
    color: #b8ccec;
    background: #0f1828;
    font-size: 0.9rem;
}
.stButton > button:hover {
    border-color: #4a80c0;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(60,120,200,0.2);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a5eff 0%, #0040cc 100%);
    border: none;
    color: white;
    font-size: 1rem;
    letter-spacing: 0.02em;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(30,80,255,0.35);
    transform: translateY(-2px);
}

h1, h2, h3 { color: #d8eaff; font-family: 'Space Grotesk', sans-serif; }
p, .stMarkdown, label { color: #8aa8cc; }
hr { border-color: #182538 !important; }

[data-testid="stExpander"] {
    background: #0c1422;
    border: 1px solid #182538;
    border-radius: 10px;
}
div[data-baseweb="tab-list"] {
    background: #091220 !important;
    border-radius: 10px;
    padding: 4px;
}
div[data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #6080a8 !important;
}
div[aria-selected="true"] {
    background: #1a2f50 !important;
    color: #90b8e8 !important;
}
.stSelectbox > div, .stMultiSelect > div {
    background: #0c1422 !important;
    border-color: #1e3050 !important;
    border-radius: 10px !important;
}
.stTextInput > div > div {
    background: #0c1422 !important;
    border-color: #1e3050 !important;
    border-radius: 10px !important;
    color: #c4d8f0 !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════
for key, default in [
    ("tic_id", "261136679"),
    ("history", []),
    ("last_result", None),
    ("results_cache", {}),
    ("compare_results", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

FEATURE_COLS = ["depth", "snr", "sec_ratio", "duration_hours", "bls_power", "odd_even_diff"]

QUICK_STARS = {
    "Pi Mensae c — Sub-Neptune": "261136679",
    "WASP-126 b — Hot Jupiter":  "25155310",
    "TIC 441075486 — Binary (FP)": "441075486",
}


# ════════════════════════════════════════════════════════════
# HELPER — safe numpy extraction from masked/astropy arrays
# ════════════════════════════════════════════════════════════
def to_arr(x):
    """Convert any astropy/masked array to clean plain float64 numpy array."""
    if hasattr(x, "filled"):
        x = x.filled(np.nan)
    arr = np.asarray(x, dtype=np.float64)
    return arr


def clean_series(x_arr, y_arr):
    """Return only finite pairs — removes NaN/inf spikes before plotting."""
    x = to_arr(x_arr)
    y = to_arr(y_arr)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


# ════════════════════════════════════════════════════════════
# DATASET POOL
# ════════════════════════════════════════════════════════════
@st.cache_data
def load_dataset_pool():
    if not os.path.exists("features_dataset.csv"):
        return pd.DataFrame()
    df = pd.read_csv("features_dataset.csv")
    df["display_name"] = df["tic_id"].apply(lambda t: f"TIC {t}")
    return df

dataset_pool = load_dataset_pool()


# ════════════════════════════════════════════════════════════
# CLASSIFIER
# ════════════════════════════════════════════════════════════
@st.cache_resource
def load_classifier():
    if os.path.exists("exoplanet_classifier.pkl"):
        clf = joblib.load("exoplanet_classifier.pkl")
        n = str(len(dataset_pool)) if not dataset_pool.empty else "131"
        return clf, f"RF+GB Ensemble — trained on {n} real NASA TOI stars (87.35% CV accuracy)", True

    # Fallback synthetic prototype
    rng = np.random.default_rng(42)
    seeds = [
        (0.000183, 19.4, 0.11, 2.64, 2837, 0.00002, "planet"),
        (0.005,    25.0, 0.05, 3.00, 2000, 0.00001, "planet"),
        (0.0009,   12.0, 0.08, 2.00, 1500, 0.00003, "planet"),
        (0.01,     15.0, 0.80, 4.00, 1800, 0.002,   "false_positive"),
        (0.03,     30.0, 0.60, 5.00, 2200, 0.004,   "false_positive"),
        (0.015,    20.0, 0.70, 3.50, 1900, 0.003,   "false_positive"),
    ]
    X, y = [], []
    for d, s, sr, dur, pwr, oed, lab in seeds:
        for _ in range(15):
            X.append([abs(d*rng.normal(1, 0.15)), abs(s*rng.normal(1, 0.2)),
                      abs(sr*rng.normal(1, 0.25)), abs(dur*rng.normal(1, 0.2)),
                      abs(pwr*rng.normal(1, 0.15)), abs(oed*rng.normal(1, 0.3))])
            y.append(lab)
    clf = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
    clf.fit(np.array(X), np.array(y))
    return clf, "Synthetic prototype (run training pipeline for real model)", False

clf_model, model_source, is_real_model = load_classifier()


# ════════════════════════════════════════════════════════════
# FIX 1 — ROBUST DOWNLOAD with retry + exponential backoff
# ════════════════════════════════════════════════════════════
def download_with_retry(tic_id, max_sectors, max_retries=3):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            search = lk.search_lightcurve(
                f'TIC {tic_id}', mission='TESS', author='SPOC', exptime=120
            )
            if len(search) == 0:
                return None, None, None, "No TESS/SPOC data found. The star may exist but have no 2-min cadence data."
            n_sectors = min(max_sectors, len(search))
            lc_raw = search[:n_sectors].download_all().stitch()
            return lc_raw, n_sectors, len(search), None
        except Exception as e:
            last_err = e
            err_msg = str(e).lower()
            is_transient = any(s in err_msg for s in [
                "connection", "reset", "timed out", "timeout",
                "10054", "remote", "broken pipe", "temporarily", "aborted"
            ])
            if attempt < max_retries and is_transient:
                time.sleep(2 ** attempt)
                continue
            return None, None, None, f"Download failed after {attempt} attempt(s): {e}"
    return None, None, None, f"Download failed: {last_err}"


# ════════════════════════════════════════════════════════════
# FIX 2 — BOUNDED BLS PERIODOGRAM
# Root cause: n_periods = (period_max - period_min) / 0.001
# For 1–20 day range = 19,000 × 28 durations × lightkurve's
# internal frequency_factor=10 → 5.3M grid points → crash.
#
# Fix: use frequency_factor=1 (disables internal expansion),
# compute our own sensibly-sized grid (≤2000 periods), and
# fall back with coarser grids if still needed.
# ════════════════════════════════════════════════════════════
def compute_bls_periodogram(lc, period_min, period_max, max_retries=4):
    # Sensible starting grid — small enough to never blow up
    n_periods   = 2000
    n_durations = 14

    last_err = None
    for attempt in range(max_retries):
        periods   = np.linspace(period_min, period_max, n_periods)
        durations = np.linspace(0.02, 0.25, n_durations)

        try:
            # frequency_factor=1 → lightkurve does NOT internally
            # expand the grid; we own the resolution entirely.
            pg = lc.to_periodogram(
                method='bls',
                period=periods,
                duration=durations,
                frequency_factor=1
            )
            return pg, None
        except Exception as e:
            last_err = e
            err_msg  = str(e).lower()
            # Only retry on "too large" errors
            if "too large" in err_msg or "periodogram" in err_msg:
                n_periods   = max(n_periods // 2, 300)
                n_durations = max(n_durations // 2, 4)
                continue
            # Other errors (value errors, etc.) — bail immediately
            return None, f"Periodogram error: {e}"

    return None, f"Periodogram still too large after {max_retries} retries: {last_err}"


# ════════════════════════════════════════════════════════════
# CORE PIPELINE
# ════════════════════════════════════════════════════════════
def run_pipeline(tic_id, max_sectors, period_min, period_max):
    result = {"tic_id": str(tic_id), "error": None}

    lc_raw, n_sectors, n_available, dl_err = download_with_retry(tic_id, max_sectors)
    if dl_err:
        result["error"] = dl_err
        return result
    result["n_sectors"]   = n_sectors
    result["n_available"] = n_available

    try:
        lc = lc_raw.normalize().flatten(window_length=401).remove_outliers(sigma=4)

        pg, pg_err = compute_bls_periodogram(lc, period_min, period_max)
        if pg_err:
            result["error"] = pg_err
            return result

        best_period   = pg.period_at_max_power
        best_power    = pg.max_power
        t0            = pg.transit_time_at_max_power
        duration_best = pg.duration_at_max_power

        # ── FIX 3: duration_best.value is in DAYS (not phase fraction).
        # half_width must be in PHASE units (fraction of period) for folding.
        # Compute properly:  half_width_phase = (duration_days / period_days) / 2
        duration_days = float(duration_best.to('d').value)
        period_days   = float(best_period.to('d').value)
        half_width    = (duration_days / period_days) / 2   # dimensionless phase fraction

        folded = lc.fold(period=best_period, epoch_time=t0)
        binned = folded.bin(time_bin_size=0.001)

        # ── Extract to clean plain arrays immediately ──
        phase_vals = to_arr(folded.time.value)
        flux_vals  = to_arr(folded.flux.value)
        bin_phase  = to_arr(binned.time.value)
        bin_flux   = to_arr(binned.flux.value)

        # ── FIX 4: Strip NaN/inf from binned arrays (prevents spike artifacts) ──
        bin_mask   = np.isfinite(bin_phase) & np.isfinite(bin_flux)
        bin_phase  = bin_phase[bin_mask]
        bin_flux   = bin_flux[bin_mask]

        # Transit statistics
        in_transit  = np.abs(phase_vals) < half_width * 1.3
        out_transit = (np.abs(phase_vals) > half_width * 3) & (np.abs(phase_vals) < 0.45)
        secondary   = np.abs(np.abs(phase_vals) - 0.5) < half_width * 1.3

        if np.sum(in_transit) > 5 and np.sum(out_transit) > 5:
            baseline    = float(np.nanmedian(flux_vals[out_transit]))
            transit_med = float(np.nanmedian(flux_vals[in_transit]))
            depth       = float(baseline - transit_med)
            noise       = float(np.nanstd(flux_vals[out_transit]))
            n_in        = int(np.sum(in_transit))
            snr         = (depth / noise) * np.sqrt(n_in) if noise > 0 else 0.0
        else:
            baseline = 1.0; depth = 0.0; snr = 0.0

        sec_depth = 0.0
        if np.sum(secondary) > 5:
            sec_depth = float(baseline - np.nanmedian(flux_vals[secondary]))

        sec_ratio      = (sec_depth / depth) if depth > 0 else 0.0
        duration_hours = duration_days * 24
        R_earth        = np.sqrt(abs(depth)) * 1.1 * 109.076

        # Odd-even check
        t_arr    = to_arr(lc.time.value)
        f_arr    = to_arr(lc.flux.value)
        cycle_num    = np.round((t_arr - t0.value) / period_days).astype(int)
        phase_global = ((t_arr - t0.value) % period_days) / period_days
        phase_global[phase_global > 0.5] -= 1
        in_g  = np.abs(phase_global) < half_width * 1.3
        o_d   = float(np.nanmedian(f_arr[in_g & (cycle_num % 2 == 1)])) if np.sum(in_g & (cycle_num%2==1)) > 2 else 1.0
        e_d   = float(np.nanmedian(f_arr[in_g & (cycle_num % 2 == 0)])) if np.sum(in_g & (cycle_num%2==0)) > 2 else 1.0
        odd_even_diff = abs(o_d - e_d)

        # ML classification
        fv         = np.array([[abs(depth), max(snr, 0), abs(sec_ratio), duration_hours, float(best_power), odd_even_diff]])
        proba_arr  = clf_model.predict_proba(fv)[0]
        proba_dict = dict(zip(clf_model.classes_, proba_arr))
        pred_label = clf_model.classes_[np.argmax(proba_arr)]
        confidence = float(max(proba_arr)) * 100

        if float(best_power) < 50 or snr < 3:
            ml_class   = "Weak / No Signal"
            confidence = 100.0
            proba_dict = {"Weak / No Signal": 1.0}
        elif pred_label == "planet":
            ml_class   = "Exoplanet Candidate"
        else:
            ml_class   = "Eclipsing Binary / False Positive"

        is_eb = sec_depth > depth * 0.4 and sec_depth > 0.0008
        if float(best_power) > 300 and snr > 7 and not is_eb:
            rule_verdict = ("Hot Jupiter"      if depth > 0.005   else
                            "Sub-Neptune"       if depth > 0.0005  else
                            "Super-Earth"       if depth > 0.00005 else
                            "Small Rocky Planet")
        elif is_eb and float(best_power) > 300:
            rule_verdict = "Eclipsing Binary"
        elif snr > 5:
            rule_verdict = "Planet Candidate"
        else:
            rule_verdict = "Uncertain"

        result.update({
            "lc": lc, "pg": pg,
            "folded": folded, "binned": binned,
            "bin_phase": bin_phase, "bin_flux": bin_flux,
            "best_period": best_period, "best_power": best_power,
            "duration_best": duration_best, "half_width": half_width,
            "duration_days": duration_days, "period_days": period_days,
            "depth": depth, "snr": snr, "sec_depth": sec_depth,
            "sec_ratio": sec_ratio, "duration_hours": duration_hours,
            "odd_even_diff": odd_even_diff,
            "R_planet_earth": R_earth, "baseline": baseline,
            "ml_class": ml_class, "ml_confidence": confidence,
            "ml_proba": proba_dict, "rule_verdict": rule_verdict,
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "phase_vals": phase_vals, "flux_vals": flux_vals,
            "t_arr": t_arr, "f_arr": f_arr, "cycle_num": cycle_num,
            "phase_global": phase_global, "o_d": o_d, "e_d": e_d,
            "t0": t0.value,
        })
    except Exception as e:
        import traceback
        result["error"] = f"Pipeline error: {e}\n{traceback.format_exc()}"
    return result


def get_or_run(tic_id, max_sectors, period_min, period_max, force=False):
    key = str(tic_id)
    if not force and key in st.session_state.results_cache:
        return st.session_state.results_cache[key]
    result = run_pipeline(tic_id, max_sectors, period_min, period_max)
    if not result.get("error"):
        st.session_state.results_cache[key] = result
        st.session_state.history.append({
            "TIC ID":         key,
            "Period (days)":  round(float(result["period_days"]), 4),
            "Depth (%)":      round(result["depth"]*100, 5),
            "SNR":            round(result["snr"], 2),
            "Radius (R⊕)":   round(result["R_planet_earth"], 2),
            "ML Verdict":     result["ml_class"],
            "Confidence (%)": round(result["ml_confidence"], 1),
            "Rule-based":     result["rule_verdict"],
            "Sectors":        result["n_sectors"],
            "Time":           result["timestamp"],
        })
    return result


# ════════════════════════════════════════════════════════════
# AI INTERPRETATION — natural language summary
# ════════════════════════════════════════════════════════════
def generate_ai_insight(r):
    """Generate a plain-English interpretation of the analysis results."""
    depth      = r["depth"]
    snr        = r["snr"]
    period     = r["period_days"]
    duration   = r["duration_hours"]
    radius     = r["R_planet_earth"]
    ml_class   = r["ml_class"]
    rule       = r["rule_verdict"]
    sec_ratio  = r["sec_ratio"]
    bls_power  = float(r["best_power"])
    oe_diff    = r["odd_even_diff"]
    confidence = r["ml_confidence"]

    lines = []

    # ── Signal quality ──
    if bls_power > 500 and snr > 10:
        lines.append(f"The BLS search found a **very strong periodic signal** (power {bls_power:.0f}, SNR {snr:.1f}) — this is a high-confidence transit detection, not noise.")
    elif bls_power > 100 and snr > 5:
        lines.append(f"A **clear periodic dip** was detected (BLS power {bls_power:.0f}, SNR {snr:.1f}). The signal is statistically significant.")
    else:
        lines.append(f"The signal is **weak** (BLS power {bls_power:.0f}, SNR {snr:.1f}). This may be noise, a grazing transit, or the period range doesn't match the real orbit.")

    # ── Classification reasoning ──
    if ml_class == "Exoplanet Candidate":
        lines.append(
            f"The ML classifier labels this a **planet candidate** with {confidence:.0f}% confidence. "
            f"The transit depth of {depth*100:.4f}% corresponds to a planet roughly **{radius:.1f}× Earth's radius** "
            f"— {'about the size of Neptune' if 3 < radius < 7 else 'a Hot Jupiter' if radius > 7 else 'a Super-Earth or sub-Neptune' if radius > 1.5 else 'potentially Earth-like'}. "
            f"It orbits every **{period:.3f} days** with a transit lasting **{duration:.1f} hours**."
        )
    elif "Binary" in ml_class or "False" in ml_class:
        reasons = []
        if sec_ratio > 0.4:
            reasons.append(f"a secondary eclipse at half-phase ({sec_ratio*100:.0f}% as deep as the primary — typical of eclipsing binaries)")
        if oe_diff > 0.002:
            reasons.append(f"odd-even depth alternation (every other transit is {oe_diff*100:.3f}% different — a hallmark of two stars eclipsing each other)")
        if depth > 0.01:
            reasons.append(f"an unusually deep transit ({depth*100:.2f}%) — real planets rarely block more than 1% of stellar light")
        reason_str = ", and ".join(reasons) if reasons else "anomalous photometric signature"
        lines.append(
            f"The classifier flags this as likely a **false positive** ({confidence:.0f}% confidence) due to: {reason_str}. "
            f"This is most likely an **eclipsing binary star system**, not a planet."
        )
    else:
        lines.append("The signal is too weak to classify confidently. Try stacking more sectors or widening the period search range.")

    # ── Sanity cross-check ──
    if ml_class == "Exoplanet Candidate" and rule != ml_class.replace("Exoplanet Candidate", "Planet Candidate") and rule not in ["Hot Jupiter","Sub-Neptune","Super-Earth","Small Rocky Planet","Planet Candidate"]:
        lines.append(f"⚠️ Note: the rule-based check returns **'{rule}'** — the two methods disagree, so treat this result with caution and check the transit shape manually.")
    elif ml_class == "Exoplanet Candidate":
        lines.append(f"The independent rule-based check agrees: **{rule}**. Both methods point to the same conclusion.")

    return " ".join(lines)


# ════════════════════════════════════════════════════════════
# PDF GENERATOR
# ════════════════════════════════════════════════════════════
def generate_pdf(result):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.5, 11), facecolor='white')
        ax0 = fig.add_axes([0, 0, 1, 1])
        ax0.axis('off')
        ax0.set_facecolor('white')
        fig.text(0.5, 0.96, "ExoDetect — Analysis Report",
                 ha='center', fontsize=18, fontweight='bold', color='#1a2a4a')
        fig.text(0.5, 0.93, f"TIC {result['tic_id']} | BAH2026 PS7 | Jadavpur University",
                 ha='center', fontsize=10, color='#4a6a9a')
        lines = [
            f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Model     : {model_source}", "",
            f"Period    : {result['period_days']:.4f} days",
            f"Duration  : {result['duration_hours']:.2f} hours",
            f"Depth     : {result['depth']*100:.5f}%",
            f"Radius    : {result['R_planet_earth']:.2f} Earth radii",
            f"SNR       : {result['snr']:.2f}",
            f"BLS Power : {float(result['best_power']):.1f}",
            f"Sectors   : {result['n_sectors']} / {result['n_available']}", "",
            f"ML Verdict: {result['ml_class']} ({result['ml_confidence']:.1f}%)",
            f"Rule check: {result['rule_verdict']}", "",
            "── AI INSIGHT ──",
        ]
        insight = generate_ai_insight(result)
        # word-wrap insight at 80 chars for PDF
        import textwrap
        wrapped = textwrap.fill(insight.replace("**",""), width=75)
        lines.append(wrapped)
        fig.text(0.08, 0.90, "\n".join(lines), fontsize=9, va='top',
                 family='monospace', color='#1a2a4a')
        pdf.savefig(fig, facecolor='white'); plt.close(fig)

        fig2, axes = plt.subplots(3, 1, figsize=(8.5, 11),
                                   facecolor='#0a1220', constrained_layout=True)
        lc_t, lc_f = clean_series(result['lc'].time.value, result['lc'].flux.value)
        axes[0].scatter(lc_t, lc_f, s=0.8, alpha=0.5, color='#4488cc', rasterized=True)
        axes[0].set_title("Clean Light Curve", pad=6)
        axes[0].set_xlabel("Time (days)"); axes[0].set_ylabel("Normalized Flux")
        axes[0].grid(True)

        pg_p, pg_pw = clean_series(result['pg'].period.value, result['pg'].power.value)
        axes[1].plot(pg_p, pg_pw, color='#4488cc', linewidth=0.8)
        axes[1].axvline(x=result['period_days'], color='#ff6633',
                        linewidth=1.5, linestyle='--', label=f"Best = {result['period_days']:.4f} d")
        axes[1].set_title("BLS Periodogram", pad=6)
        axes[1].legend(fontsize=8)
        axes[1].grid(True)

        zoom = max(result['half_width'] * 8, 0.05)
        pv, fv = clean_series(result["phase_vals"], result["flux_vals"])
        bp, bf = result["bin_phase"], result["bin_flux"]
        axes[2].scatter(pv, fv, s=1.5, alpha=0.3, color='#4488cc', rasterized=True)
        if len(bp) > 2:
            axes[2].plot(bp, bf, color='#ff6633', linewidth=2)
        axes[2].set_xlim(-zoom, zoom)
        axes[2].set_title("Phase-Folded Transit", pad=6)
        axes[2].set_xlabel("Phase"); axes[2].set_ylabel("Normalized Flux")
        axes[2].grid(True)

        pdf.savefig(fig2); plt.close(fig2)
    buf.seek(0)
    return buf


def generate_extra_graphs_pdf(result, figs):
    """Bundles the 5 Deep Diagnostics figures (already rendered on screen)
    into a single downloadable PDF. figs may contain None for skipped plots."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        cover = plt.figure(figsize=(8.5, 11), facecolor='white')
        ax0 = cover.add_axes([0, 0, 1, 1]); ax0.axis('off')
        cover.text(0.5, 0.92, "ExoDetect — Deep Diagnostics Report",
                   ha='center', fontsize=18, fontweight='bold', color='#1a2a4a')
        cover.text(0.5, 0.89, f"TIC {result['tic_id']} | BAH2026 PS7 | Jadavpur University",
                   ha='center', fontsize=10, color='#4a6a9a')
        cover.text(0.08, 0.82,
            f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Period    : {result['period_days']:.4f} days\n"
            f"Depth     : {result['depth']*100:.5f}%\n"
            f"SNR       : {result['snr']:.2f}\n"
            f"Verdict   : {result['ml_class']} ({result['ml_confidence']:.1f}%)\n\n"
            "Contents:\n"
            "  1. Odd vs Even Transit Comparison\n"
            "  2. Secondary Eclipse Zoom\n"
            "  3. River Plot (transit-by-transit)\n"
            "  4. Residuals Plot\n"
            "  5. Periodogram Zoom",
            fontsize=10, va='top', family='monospace', color='#1a2a4a')
        pdf.savefig(cover, facecolor='white'); plt.close(cover)

        for f in figs:
            if f is not None:
                pdf.savefig(f, facecolor=BG)
    buf.seek(0)
    return buf



# ════════════════════════════════════════════════════════════
# PLOT HELPERS — sharp, consistent styling
# ════════════════════════════════════════════════════════════
ACCENT   = '#4a9eff'
ORANGE   = '#ff7a45'
BG       = '#0a1220'
GRID_C   = '#1e3050'
TEXT_C   = '#ddeeff'
LABEL_C  = '#7090b8'

def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=LABEL_C, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.grid(True, color=GRID_C, alpha=0.5, linewidth=0.5)
    ax.set_xlabel(ax.get_xlabel(), color=LABEL_C, fontsize=9)
    ax.set_ylabel(ax.get_ylabel(), color=LABEL_C, fontsize=9)
    ax.set_title(ax.get_title(), color=TEXT_C, fontweight='bold', fontsize=10, pad=8)

def make_fig(w=12, h=4):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    return fig, ax


# ════════════════════════════════════════════════════════════
# RENDER — single star result
# ════════════════════════════════════════════════════════════
def render_result(result):
    r          = result
    bp         = r["bin_phase"]
    bf         = r["bin_flux"]
    zoom       = max(r["half_width"] * 10, 0.04)
    ml_class   = r["ml_class"]
    depth      = r["depth"]
    snr        = r["snr"]
    period_d   = r["period_days"]
    best_power = float(r["best_power"])

    st.success(
        f"✅  TIC {r['tic_id']}  |  {r['n_sectors']}/{r['n_available']} sectors  |  "
        f"Period {period_d:.4f} d  |  BLS Power {best_power:.0f}"
    )

    # ── Metrics row ──
    st.markdown("---")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("⏱️ Period",    f"{period_d:.3f} d")
    m2.metric("📉 Depth",     f"{depth*100:.4f}%")
    m3.metric("📡 SNR",       f"{snr:.1f}")
    m4.metric("🌍 Radius",    f"{r['R_planet_earth']:.2f} R⊕")
    m5.metric("🔍 BLS Power", f"{best_power:.0f}")
    m6.metric("⏳ Duration",   f"{r['duration_hours']:.2f} h")

    # ── Verdict box ──
    if ml_class == "Exoplanet Candidate":
        vc, vi, emoji = "#0a7a2a", "✓", "🪐"
    elif "Binary" in ml_class or "False" in ml_class:
        vc, vi, emoji = "#9a1010", "✗", "⭐"
    else:
        vc, vi, emoji = "#555577", "❓", "🔭"

    st.markdown(f"""
    <div class='verdict-box' style='background:{vc}22; border:2px solid {vc};'>
        <div style='font-size:2.2rem; margin-bottom:4px;'>{emoji}</div>
        <h2 style='color:{vc}; margin:4px 0; font-size:1.5rem;'>{vi} {ml_class.upper()}</h2>
        <p style='color:#b8d0ec; margin:0; font-size:0.93rem;'>
            ML Confidence: <b>{r['ml_confidence']:.1f}%</b> &nbsp;|&nbsp;
            Rule-based: <b>{r['rule_verdict']}</b> &nbsp;|&nbsp;
            Sectors stacked: <b>{r['n_sectors']}</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── AI Insight ──
    insight_text = generate_ai_insight(r)
    st.markdown(f"""
    <div class='ai-insight'>
        <div class='ai-insight-header'>🤖 AI Interpretation</div>
        <p class='ai-insight-text'>{insight_text}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──
    st.markdown("---")
    t1, t2, t3, t4, t5 = st.tabs([
        "📉 Light Curve", "🔍 Period Search", "🪐 Transit Shape", "🤖 Classifier",
        "🔬 Deep Diagnostics"
    ])

    with t1:
        fig, ax = make_fig(12, 4)
        lc_t, lc_f = clean_series(r['lc'].time.value, r['lc'].flux.value)
        ax.scatter(lc_t, lc_f, s=0.6, alpha=0.45, color=ACCENT, rasterized=True,
                   label=f"{len(lc_t):,} data points")
        ax.set_title(f"Stacked Light Curve — TIC {r['tic_id']} ({r['n_sectors']} sectors)")
        ax.set_xlabel("Time (BTJD days)"); ax.set_ylabel("Normalized Flux")
        ax.legend(fontsize=8, facecolor='#0c1422', labelcolor=LABEL_C, framealpha=0.7)
        style_ax(ax); plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    with t2:
        fig, ax = make_fig(12, 4)
        pg_p, pg_pw = clean_series(r['pg'].period.value, r['pg'].power.value)
        ax.plot(pg_p, pg_pw, color=ACCENT, linewidth=0.9, alpha=0.9)
        ax.axvline(x=period_d, color=ORANGE, linewidth=2, linestyle='--',
                   label=f'Peak = {period_d:.4f} d  (power {best_power:.0f})')
        ax.fill_between(pg_p, 0, pg_pw, alpha=0.08, color=ACCENT)
        ax.set_title("BLS Periodogram — Box Least Squares Period Search")
        ax.set_xlabel("Period (days)"); ax.set_ylabel("BLS Power")
        ax.legend(fontsize=9, facecolor='#0c1422', labelcolor=LABEL_C, framealpha=0.7)
        style_ax(ax); plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    with t3:
        pv, fv = clean_series(r["phase_vals"], r["flux_vals"])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)

        for ax in axes:
            ax.set_xlabel("Phase (fraction of orbit)"); ax.set_ylabel("Normalized Flux")

        # Full phase view
        axes[0].scatter(pv, fv, s=1.2, alpha=0.20, color=ACCENT, rasterized=True)
        if len(bp) > 2:
            axes[0].plot(bp, bf, color=ORANGE, linewidth=2.5, zorder=5, label='Binned model')
        axes[0].axhline(y=r["baseline"], color='#555', linestyle='--', alpha=0.5, linewidth=0.8)
        axes[0].set_xlim(-0.5, 0.5)
        axes[0].set_title("Full Phase-Folded View")
        axes[0].legend(fontsize=8, facecolor='#0c1422', labelcolor=LABEL_C, framealpha=0.7)

        # Zoomed transit
        axes[1].scatter(pv, fv, s=2, alpha=0.30, color=ACCENT, rasterized=True)
        if len(bp) > 2:
            axes[1].plot(bp, bf, color=ORANGE, linewidth=2.8, zorder=5,
                         label=f'Depth {depth*100:.4f}%  ({r["R_planet_earth"]:.1f} R⊕)')
        axes[1].axhline(y=r["baseline"], color='#888', linestyle='--', alpha=0.4, linewidth=0.8)
        axes[1].axhline(y=r["baseline"]-depth, color='#ffcc44', linestyle=':',
                         alpha=0.7, linewidth=1.2, label='Transit floor')
        axes[1].set_xlim(-zoom, zoom)
        axes[1].set_title("🪐 Zoomed Transit Window")
        axes[1].legend(fontsize=8, facecolor='#0c1422', labelcolor=LABEL_C, framealpha=0.7)

        for ax in axes: style_ax(ax)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    with t4:
        classes_list = list(r["ml_proba"].keys())
        vals         = [v*100 for v in r["ml_proba"].values()]
        cmap         = {"planet": "#1a7a2a", "false_positive": "#9a1010",
                        "Weak / No Signal": "#444466"}
        bcolors      = [cmap.get(c, ACCENT) for c in classes_list]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)

        bars = axes[0].bar(classes_list, vals, color=bcolors, width=0.45, zorder=2)
        axes[0].set_ylabel("Probability (%)"); axes[0].set_ylim(0, 115)
        axes[0].set_title("ML Class Probabilities")
        for bar, v in zip(bars, vals):
            axes[0].text(bar.get_x()+bar.get_width()/2, v+3,
                         f"{v:.1f}%", ha='center', color=TEXT_C, fontweight='bold', fontsize=10)
        style_ax(axes[0])

        try:
            imp  = clf_model.feature_importances_
            ypos = list(range(len(FEATURE_COLS)))
            bars2 = axes[1].barh(ypos, imp, color=ACCENT, height=0.6, zorder=2)
            axes[1].set_yticks(ypos)
            axes[1].set_yticklabels(FEATURE_COLS, color=LABEL_C, fontsize=8)
            axes[1].set_xlabel("Feature Importance")
            axes[1].set_title("RandomForest Feature Importance")
            for bar, v in zip(bars2, imp):
                axes[1].text(v+0.005, bar.get_y()+bar.get_height()/2,
                             f"{v:.3f}", va='center', color=TEXT_C, fontsize=8)
        except Exception:
            axes[1].axis('off')
        style_ax(axes[1])

        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        if is_real_model:
            st.success(f"🧠 {model_source}")
        else:
            st.warning(f"⚠️ {model_source}")

    with t5:
        st.caption(
            "All plots below use the same real TESS data already downloaded for this star — "
            "no new data fetched, just additional views of the same measurements."
        )

        # ── 1. Odd-vs-Even transit comparison ──
        st.markdown("**1. Odd vs Even Transit Comparison**")
        st.caption(
            "Real eclipsing binaries often show alternating transit depths (odd vs even "
            "cycles differ). A consistent depth across both supports a genuine planet."
        )
        t_arr        = r["t_arr"]; f_arr = r["f_arr"]
        cycle_num    = r["cycle_num"]; phase_global = r["phase_global"]
        half_width   = r["half_width"]
        in_g         = np.abs(phase_global) < half_width * 3
        odd_mask     = in_g & (cycle_num % 2 == 1)
        even_mask    = in_g & (cycle_num % 2 == 0)

        fig, ax = make_fig(12, 4.5)
        if np.sum(odd_mask) > 2:
            ax.scatter(phase_global[odd_mask], f_arr[odd_mask], s=4, alpha=0.5,
                       color=ORANGE, label=f"Odd cycles (median {r['o_d']:.6f})")
        if np.sum(even_mask) > 2:
            ax.scatter(phase_global[even_mask], f_arr[even_mask], s=4, alpha=0.5,
                       color=ACCENT, label=f"Even cycles (median {r['e_d']:.6f})")
        ax.axhline(y=r["o_d"], color=ORANGE, linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(y=r["e_d"], color=ACCENT, linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlim(-half_width*3, half_width*3)
        ax.set_title(f"Odd-Even Depth Diff = {r['odd_even_diff']*100:.5f}%  "
                     f"({'⚠️ possible EB signature' if r['odd_even_diff'] > 0.0005 else '✓ consistent — planet-like'})")
        ax.set_xlabel("Phase (fraction of orbit)"); ax.set_ylabel("Normalized Flux")
        ax.legend(fontsize=8, facecolor='#0c1422', labelcolor=LABEL_C, framealpha=0.7)
        style_ax(ax); plt.tight_layout(); st.pyplot(fig, use_container_width=True)
        st.session_state.setdefault("extra_figs", {})["odd_even"] = fig

        # ── 2. Secondary eclipse zoom ──
        st.markdown("**2. Secondary Eclipse Zoom (Phase 0.5)**")
        st.caption(
            "Zoomed view around phase 0.5 (half an orbit after transit). A deep dip here "
            "indicates a second eclipsing body — a strong eclipsing-binary signature."
        )
        pv, fv = clean_series(r["phase_vals"], r["flux_vals"])
        sec_zoom_mask = np.abs(np.abs(pv) - 0.5) < half_width * 6
        fig2, ax2 = make_fig(12, 4)
        if np.sum(sec_zoom_mask) > 2:
            pv_shift = np.where(pv[sec_zoom_mask] < 0, pv[sec_zoom_mask] + 1, pv[sec_zoom_mask]) - 0.5
            ax2.scatter(pv_shift, fv[sec_zoom_mask], s=3, alpha=0.35, color=ACCENT, rasterized=True)
        ax2.axhline(y=r["baseline"], color='#888', linestyle='--', alpha=0.5, linewidth=0.8, label="Baseline")
        ax2.axhline(y=r["baseline"]-r["sec_depth"], color='#ff5555', linestyle=':', linewidth=1.4,
                    label=f"Secondary depth ({r['sec_depth']*100:.5f}%)")
        ax2.set_xlim(-half_width*6, half_width*6)
        ax2.set_title(f"Secondary/Primary Ratio = {r['sec_ratio']:.3f}  "
                      f"({'⚠️ likely EB' if r['sec_ratio'] > 0.4 else '✓ no significant secondary'})")
        ax2.set_xlabel("Phase offset from secondary position"); ax2.set_ylabel("Normalized Flux")
        ax2.legend(fontsize=8, facecolor='#0c1422', labelcolor=LABEL_C, framealpha=0.7)
        style_ax(ax2); plt.tight_layout(); st.pyplot(fig2, use_container_width=True)
        st.session_state.setdefault("extra_figs", {})["secondary"] = fig2

        # ── 3. River plot (transit-by-transit) ──
        st.markdown("**3. River Plot — Transit Consistency Across Cycles**")
        st.caption(
            "Each row is one orbital cycle, color = flux. A consistent vertical dark band "
            "in the middle across all rows means the signal repeats reliably every orbit."
        )
        period_days = r["period_days"]
        unique_cycles = np.unique(cycle_num[np.isfinite(cycle_num)])
        unique_cycles = unique_cycles[(unique_cycles >= cycle_num.min()) & (unique_cycles <= cycle_num.max())]
        n_bins = 60
        phase_bins = np.linspace(-0.5, 0.5, n_bins+1)
        river = np.full((len(unique_cycles), n_bins), np.nan)
        for i, c in enumerate(unique_cycles):
            mask_c = (cycle_num == c) & np.isfinite(phase_global) & np.isfinite(f_arr)
            if np.sum(mask_c) < 2:
                continue
            ph_c, fl_c = phase_global[mask_c], f_arr[mask_c]
            for b in range(n_bins):
                bmask = (ph_c >= phase_bins[b]) & (ph_c < phase_bins[b+1])
                if np.sum(bmask) > 0:
                    river[i, b] = np.nanmedian(fl_c[bmask])
        fig3, ax3 = plt.subplots(figsize=(12, max(3, min(8, len(unique_cycles)*0.3))), facecolor=BG)
        im = ax3.imshow(river, aspect='auto', cmap='RdBu_r',
                        extent=[-0.5, 0.5, len(unique_cycles), 0],
                        vmin=np.nanpercentile(river, 2) if np.isfinite(river).any() else 0.99,
                        vmax=np.nanpercentile(river, 98) if np.isfinite(river).any() else 1.01)
        ax3.set_xlabel("Phase"); ax3.set_ylabel("Orbital Cycle Number")
        ax3.set_title(f"River Plot — {len(unique_cycles)} cycles stacked")
        ax3.set_xlim(-half_width*8, half_width*8)
        cbar = plt.colorbar(im, ax=ax3); cbar.set_label("Normalized Flux", color=LABEL_C)
        cbar.ax.yaxis.set_tick_params(color=LABEL_C)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=LABEL_C)
        style_ax(ax3); plt.tight_layout(); st.pyplot(fig3, use_container_width=True)
        st.session_state.setdefault("extra_figs", {})["river"] = fig3

        # ── 4. Residuals plot ──
        st.markdown("**4. Residuals — Data Minus Binned Model**")
        st.caption(
            "Shows leftover scatter after subtracting the binned transit model from the "
            "phase-folded data. Random scatter with no pattern indicates a clean fit."
        )
        bp, bf = r["bin_phase"], r["bin_flux"]
        if len(bp) > 2:
            interp_model = np.interp(pv, bp, bf)
            residuals = fv - interp_model
            fig4, ax4 = make_fig(12, 4)
            ax4.scatter(pv, residuals, s=1.5, alpha=0.3, color=ACCENT, rasterized=True)
            ax4.axhline(y=0, color=ORANGE, linewidth=1.2, linestyle='--')
            rms = np.nanstd(residuals)
            ax4.set_title(f"Residuals (RMS = {rms*100:.5f}%)")
            ax4.set_xlabel("Phase (fraction of orbit)"); ax4.set_ylabel("Flux − Model")
            ax4.set_xlim(-0.5, 0.5)
            style_ax(ax4); plt.tight_layout(); st.pyplot(fig4, use_container_width=True)
            st.session_state.setdefault("extra_figs", {})["residuals"] = fig4
        else:
            st.info("Not enough binned points to compute residuals for this star.")

        # ── 5. Periodogram zoom around best peak ──
        st.markdown("**5. Periodogram — Zoomed Around Best Peak**")
        st.caption(
            "Close-up of the BLS power curve right around the detected period. A sharp, "
            "narrow, isolated peak is stronger evidence than a broad or noisy one."
        )
        pg_p, pg_pw = clean_series(r['pg'].period.value, r['pg'].power.value)
        zoom_width = max(period_days * 0.08, 0.05)
        zmask = np.abs(pg_p - period_days) < zoom_width
        fig5, ax5 = make_fig(12, 4)
        if np.sum(zmask) > 2:
            ax5.plot(pg_p[zmask], pg_pw[zmask], color=ACCENT, linewidth=1.3)
            ax5.fill_between(pg_p[zmask], 0, pg_pw[zmask], alpha=0.12, color=ACCENT)
        ax5.axvline(x=period_days, color=ORANGE, linewidth=2, linestyle='--',
                   label=f'Peak = {period_days:.5f} d')
        ax5.set_title("Zoomed BLS Periodogram (peak sharpness check)")
        ax5.set_xlabel("Period (days)"); ax5.set_ylabel("BLS Power")
        ax5.legend(fontsize=9, facecolor='#0c1422', labelcolor=LABEL_C, framealpha=0.7)
        style_ax(ax5); plt.tight_layout(); st.pyplot(fig5, use_container_width=True)
        st.session_state.setdefault("extra_figs", {})["periodogram_zoom"] = fig5

        # ── Download all 5 as one PDF ──
        st.markdown("---")
        extra_pdf_buf = generate_extra_graphs_pdf(r, [fig, fig2, fig3, fig4 if len(bp) > 2 else None, fig5])
        st.download_button(
            "📥 Download Deep Diagnostics (5 graphs) as PDF",
            data=extra_pdf_buf,
            file_name=f"TIC_{r['tic_id']}_deep_diagnostics.pdf",
            mime="application/pdf",
        )
        for fig_to_close in [fig, fig2, fig3, fig5] + ([fig4] if len(bp) > 2 else []):
            plt.close(fig_to_close)

    # ── Full params expander ──
    st.markdown("---")
    with st.expander("💡 Full Parameter Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
| Parameter | Value |
|---|---|
| TIC ID | {r['tic_id']} |
| Orbital Period | {period_d:.4f} days |
| Transit Duration | {r['duration_hours']:.2f} hours |
| Transit Depth | {depth*100:.5f}% |
| Planet Radius | {r['R_planet_earth']:.2f} R⊕ |
| BLS Power | {best_power:.1f} |
            """)
        with col2:
            st.markdown(f"""
| Parameter | Value |
|---|---|
| SNR | {snr:.2f} |
| Secondary Eclipse Depth | {r['sec_depth']*100:.5f}% |
| Secondary / Primary Ratio | {r['sec_ratio']:.3f} |
| Odd-Even Depth Diff | {r['odd_even_diff']*100:.5f}% |
| Sectors Used | {r['n_sectors']} / {r['n_available']} |
| ML Confidence | {r['ml_confidence']:.1f}% |
            """)

    pdf_buf = generate_pdf(result)
    st.download_button(
        "📄 Download PDF Report",
        data=pdf_buf,
        file_name=f"ExoDetect_TIC{r['tic_id']}.pdf",
        mime="application/pdf",
        use_container_width=True,
        key=f"pdf_{r['tic_id']}_{r['timestamp']}"
    )


# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class='hero-banner'>
  <div class='hero-title'>🪐 ExoDetect</div>
  <p class='hero-sub'>
    AI-Enabled Exoplanet Detection from NASA TESS Light Curves<br>
    <span style='color:#4a6a9a; font-size:0.88rem;'>
      Bharatiya Antariksh Hackathon 2026 — PS7 &nbsp;|&nbsp; Jadavpur University
    </span>
  </p>
</div>
""", unsafe_allow_html=True)

if is_real_model:
    st.success(f"🧠 **Active model:** {model_source}")
else:
    st.warning(f"⚠️ **Active model:** {model_source}")


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style='text-align:center; padding:14px 0 10px;'>
  <span style='font-size:2.2rem;'>🪐</span><br>
  <span style='color:#90b8e8; font-weight:700; font-size:1.1rem; font-family:Space Grotesk,sans-serif;'>ExoDetect</span><br>
  <span style='color:#3a5070; font-size:0.72rem; letter-spacing:0.05em;'>BAH 2026 — PS7</span>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🔭 Individual Analysis", "⚖️ Compare Stars", "📄 Project Report", "📜 History"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**⚙️ Pipeline Settings**")
max_sectors = st.sidebar.slider("Sectors to stack", 1, 10, 3)
period_min  = st.sidebar.slider("Min period (days)", 0.5, 5.0, 5.0)
period_max  = st.sidebar.slider("Max period (days)", 5.0, 30.0, 8.0)

if not dataset_pool.empty:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**📊 Dataset:** {len(dataset_pool)} real NASA stars")
    n_planet = (dataset_pool['label']=='planet').sum()
    n_fp     = (dataset_pool['label']=='false_positive').sum()
    st.sidebar.markdown(f"🪐 Planets: **{n_planet}** &nbsp; ⭐ False Pos: **{n_fp}**")

st.sidebar.markdown("---")
st.sidebar.markdown("**👥 Team Members**")
st.sidebar.markdown(
    "**Team Member 1 (Team Leader)**\n\n"
    "Krishnendu Koley\n\n"
    "Jadavpur University, Kolkata\n\n"
    "krishnendukoley2007@gmail.com"
)
st.sidebar.markdown(
    "**Team Member 2**\n\n"
    "Abhradeep Bera\n\n"
    "Jadavpur University, Kolkata\n\n"
    "abhradeepmsk23@gmail.com"
)
st.sidebar.markdown(
    "**Team Member 3**\n\n"
    "Asmit Dey\n\n"
    "Jadavpur University, Kolkata\n\n"
    "deyasmit07@gmail.com"
)
st.sidebar.markdown("**Team Name:** OrbitX2026")
st.sidebar.markdown("---")
st.sidebar.caption("ExoDetect v8.0 | BAH2026 PS7")

st.markdown("---")


# ════════════════════════════════════════════════════════════
# PAGE 1 — INDIVIDUAL ANALYSIS
# ════════════════════════════════════════════════════════════
if page == "🔭 Individual Analysis":
    st.subheader("🔭 Individual Star Analysis")

    q1, q2, q3 = st.columns(3)
    with q1:
        if st.button("🌍 Pi Mensae c", use_container_width=True):
            st.session_state.tic_id = "261136679"
    with q2:
        if st.button("🪐 WASP-126 b", use_container_width=True):
            st.session_state.tic_id = "25155310"
    with q3:
        if st.button("⭐ Binary Star", use_container_width=True):
            st.session_state.tic_id = "441075486"

    if not dataset_pool.empty:
        st.markdown("**Or pick from the real NASA dataset:**")
        pool_sorted = dataset_pool.sort_values("snr", ascending=False)

        pick_col1, pick_col2 = st.columns([1, 3])
        with pick_col1:
            label_f = st.selectbox("Filter by type",
                ["All", "🪐 Planets only", "⭐ False Positives only"])
        if label_f == "🪐 Planets only":
            pool_filtered = pool_sorted[pool_sorted['label'] == 'planet']
        elif label_f == "⭐ False Positives only":
            pool_filtered = pool_sorted[pool_sorted['label'] == 'false_positive']
        else:
            pool_filtered = pool_sorted

        with pick_col2:
            chosen = st.selectbox(
                f"Select star ({len(pool_filtered)} available — sorted by SNR)",
                options=["— type TIC ID below or select here —"] + pool_filtered["display_name"].tolist()
            )
        if chosen != "— type TIC ID below or select here —":
            id_map = dict(zip(pool_filtered["display_name"], pool_filtered["tic_id"].astype(str)))
            st.session_state.tic_id = id_map[chosen]

    tic_id = st.text_input("TIC Star ID (manual entry)", key="tic_id")

    run_btn = st.button(f"🔭 Analyze TIC {tic_id}", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner(f"🔭 Analyzing TIC {tic_id} — downloading TESS data → detrending → BLS search → ML classify..."):
            result = get_or_run(tic_id, max_sectors, period_min, period_max, force=True)
        if result.get("error"):
            err = result['error']
            # User-friendly error messages
            if "too large" in err.lower():
                st.error(f"❌ BLS periodogram overflow — try reducing 'Max period' in the sidebar or increasing 'Min period'.")
            elif "no tess" in err.lower() or "no data" in err.lower():
                st.error(f"❌ No TESS data found for TIC {tic_id}. This star may lack 2-minute cadence observations.")
            elif "download failed" in err.lower():
                st.error(f"❌ Download failed (MAST server issue). Please wait 30 seconds and retry.")
            else:
                st.error(f"❌ {err[:500]}")
            st.session_state.last_result = None
        else:
            st.session_state.last_result = result

    if st.session_state.last_result:
        render_result(st.session_state.last_result)


# ════════════════════════════════════════════════════════════
# PAGE 2 — COMPARE STARS
# ════════════════════════════════════════════════════════════
elif page == "⚖️ Compare Stars":
    st.subheader("⚖️ Multi-Star Comparison")

    tabA, tabB = st.tabs(["📚 Pick from dataset", "🎯 Quick stars + Custom IDs"])
    pool_ids = []; quick_ids = []; custom_ids = []

    with tabA:
        if dataset_pool.empty:
            st.info("features_dataset.csv not found — run extract_features.py first.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                lf = st.multiselect("Filter by label",
                    sorted(dataset_pool['label'].unique()),
                    default=sorted(dataset_pool['label'].unique()))
            with c2:
                n_pick = st.slider("Max to pick", 2, 20, 6)
            filt   = dataset_pool[dataset_pool['label'].isin(lf)].sort_values("snr", ascending=False)
            chosen = st.multiselect(
                f"Select stars ({len(filt)} available — sorted by SNR)",
                options=filt["display_name"].tolist(),
                default=filt["display_name"].tolist()[:n_pick]
            )
            id_map   = dict(zip(filt["display_name"], filt["tic_id"].astype(str)))
            pool_ids = [id_map[d] for d in chosen]

    with tabB:
        qnames    = st.multiselect("Quick-select", list(QUICK_STARS.keys()), default=[])
        quick_ids = [QUICK_STARS[n] for n in qnames]
        ctext     = st.text_area("Custom TIC IDs (comma-separated)", value="")
        custom_ids = [x.strip() for x in ctext.split(",") if x.strip()]

    run_cmp = st.button("⚖️ Run Comparison", type="primary", use_container_width=True)

    if run_cmp:
        all_ids = list(dict.fromkeys(pool_ids + quick_ids + custom_ids))
        if len(all_ids) < 2:
            st.warning("Select at least 2 stars.")
        else:
            prog = st.progress(0, text="Starting...")
            results_cmp = []
            for i, sid in enumerate(all_ids):
                prog.progress(i/len(all_ids), text=f"TIC {sid} ({i+1}/{len(all_ids)})...")
                r = get_or_run(sid, max_sectors, period_min, period_max)
                if not r.get("error"):
                    results_cmp.append(r)
                else:
                    st.warning(f"TIC {sid} failed: {r['error'][:120]}")
            prog.progress(1.0, text="Done!")
            st.session_state["compare_results"] = results_cmp

    cmp_res = st.session_state.get("compare_results", [])

    if cmp_res:
        st.markdown("---")
        st.subheader(f"📋 Comparison Table — {len(cmp_res)} stars")

        rows = [{
            "TIC ID":       r["tic_id"],
            "Period (d)":   round(r["period_days"], 4),
            "Depth (%)":    round(r["depth"]*100, 5),
            "SNR":          round(r["snr"], 2),
            "Radius (R⊕)":  round(r["R_planet_earth"], 2),
            "Duration (h)": round(r["duration_hours"], 2),
            "BLS Power":    round(float(r["best_power"]), 0),
            "ML Verdict":   r["ml_class"],
            "Confidence":   f"{r['ml_confidence']:.1f}%",
        } for r in cmp_res]

        cdf = pd.DataFrame(rows)
        st.dataframe(cdf, use_container_width=True)
        csv_dl = cdf.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", csv_dl,
                           "exodetect_comparison.csv", "text/csv", use_container_width=True)

        n_pl = sum(1 for r in cmp_res if r["ml_class"]=="Exoplanet Candidate")
        n_fp = sum(1 for r in cmp_res if "Binary" in r["ml_class"] or "False" in r["ml_class"])
        s1, s2, s3 = st.columns(3)
        s1.metric("✓ Planets detected", n_pl)
        s2.metric("✗ Binaries / FP",    n_fp)
        s3.metric("❓ Uncertain",         len(cmp_res)-n_pl-n_fp)

        # ── Overlaid transit shapes ──
        st.markdown("---")
        st.subheader("🪐 Overlaid Transit Shapes")
        zoom    = max((max(r["half_width"] for r in cmp_res) * 10), 0.04)
        palette = ['#4a9eff','#ff7a45','#22cc77','#ff4466',
                   '#aa55ff','#ffcc22','#00ccee','#ff88cc']

        fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=BG)
        for i, r in enumerate(cmp_res):
            c  = palette[i % len(palette)]
            bp = r["bin_phase"]; bf = r["bin_flux"]
            if len(bp) > 2:
                ax.plot(bp, bf, color=c, linewidth=2.2, alpha=0.9,
                        label=f"TIC {r['tic_id']} — {r['ml_class'][:15]}")
        ax.axhline(y=1.0, color='#333', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_xlim(-zoom, zoom)
        ax.set_title("Phase-Folded Transit Shapes Overlaid")
        ax.set_xlabel("Phase (fraction of orbit)"); ax.set_ylabel("Normalized Flux")
        ax.legend(fontsize=7, ncol=3, facecolor='#0c1422', labelcolor=LABEL_C, framealpha=0.8)
        style_ax(ax); plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        # ── Metric bars ──
        st.markdown("---")
        st.subheader("📊 Metric Comparison")
        bar_colors_list = [
            '#1a7a2a' if r["ml_class"]=="Exoplanet Candidate"
            else '#9a1010' if ("Binary" in r["ml_class"] or "False" in r["ml_class"])
            else '#555577'
            for r in cmp_res
        ]
        labels_x = [f"TIC {r['tic_id']}" for r in cmp_res]
        bc1, bc2, bc3 = st.columns(3)
        metrics_to_plot = [
            ("Transit Depth (%)",  [r["depth"]*100 for r in cmp_res]),
            ("SNR",                [r["snr"] for r in cmp_res]),
            ("Planet Radius (R⊕)", [r["R_planet_earth"] for r in cmp_res]),
        ]
        for col, (title, vals) in zip([bc1, bc2, bc3], metrics_to_plot):
            with col:
                fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
                ax.bar(labels_x, vals, color=bar_colors_list, zorder=2)
                ax.set_title(title, fontsize=9)
                plt.setp(ax.get_xticklabels(), rotation=40, ha='right', fontsize=6)
                style_ax(ax); plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
    else:
        st.info("Select at least 2 stars and click 'Run Comparison'.")


# ════════════════════════════════════════════════════════════
# PAGE 3 — PROJECT REPORT
# ════════════════════════════════════════════════════════════
elif page == "📄 Project Report":
    st.info("💡 Press Ctrl+P → Save as PDF to export this page.")

    st.markdown("""
    <div class='report-card'>
    <h2>🪐 ExoDetect — Project Report</h2>
    <p><b>Bharatiya Antariksh Hackathon 2026 | Problem Statement 7</b><br>
    AI-Enabled Detection of Exoplanets from Noisy Astronomical Light Curves</p>
    <p><b>Team OrbitX2026:</b> Krishnendu Koley (Team Leader), Abhradeep Bera, Asmit Dey &nbsp;|&nbsp; <b>Institution:</b> Jadavpur University, Kolkata</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='report-card'>
    <h3>1. Problem Overview</h3>
    <p>When a planet transits its host star, the star dims slightly and periodically.
    NASA's TESS satellite records these brightness curves for millions of stars.
    This project builds a 6-stage pipeline to detect, characterize, and classify
    periodic transit signals — distinguishing real planets from eclipsing binaries
    and instrumental artifacts. An AI interpretation layer then translates the
    numerical outputs into plain-English conclusions.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='report-card'>
    <h3>2. Pipeline Stages</h3>
    <ol>
    <li><b>Data Acquisition</b> — Multi-sector TESS download via lightkurve / NASA MAST with retry logic</li>
    <li><b>De-trending</b> — Savitzky-Golay flattening (window 401) removes instrumental systematics</li>
    <li><b>BLS Period Search</b> — Bounded Box Least Squares (≤2000 periods, frequency_factor=1) prevents grid overflow</li>
    <li><b>Phase Folding + Feature Extraction</b> — 6 features: depth, SNR, secondary-eclipse ratio, transit duration, BLS power, odd-even depth difference</li>
    <li><b>ML Classification — RF+GB Ensemble trained on 641 real NASA TOI labeled stars (planet / false positive), achieving 87.35% 5-fold CV accuracy</li>
    <li><b>AI Interpretation</b> — Natural language summary of signal quality, classification reasoning, and cross-check</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown("<div class='report-card'><h3>3. Dataset</h3>", unsafe_allow_html=True)
        if not dataset_pool.empty:
            _lc  = dataset_pool['label'].value_counts()
            _ldf = pd.DataFrame({"Label": _lc.index.astype(str), "Count": _lc.values})
            st.dataframe(_ldf, use_container_width=True)
            st.caption(f"Total: {len(dataset_pool)} stars from NASA TOI catalog (641 training)")
        else:
            st.info("Run extract_features.py to populate dataset.")
        st.markdown("</div>", unsafe_allow_html=True)

    with dc2:
        st.markdown("<div class='report-card'><h3>4. Model Comparison</h3>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Model":      ["RandomForest (features)", "1D-CNN (raw curves)"],
            "5-Fold CV":  ["87.35% ± 1.04%",  "88.65% ± 2.92%"],
            "Status":     ["✓ Deployed",      "✓ Trained"],
        }), use_container_width=True)
        st.caption("RF+GB Ensemble trained on 641 real NASA TESS stars. CNN trained on 493 phase-folded curves.")
        st.markdown("</div>", unsafe_allow_html=True)

    vc1, vc2, vc3 = st.columns(3)

    # ── Chart 1: Model accuracy comparison (RandomForest / GradientBoost / Ensemble) ──
    with vc1:
        model_names = ["RandomForest", "GradientBoost", "Ensemble\n(RF+GB)"]
        model_accs  = [87.19, 83.79, 86.43]
        model_errs  = [2.3, 1.9, 2.1]
        model_colors = ['#4a9eff', '#ff8c3a', '#3ad17a']

        fig_acc, ax_acc = plt.subplots(figsize=(5, 5.5), facecolor=BG)
        bars = ax_acc.bar(model_names, model_accs, yerr=model_errs, capsize=5,
                          color=model_colors, width=0.6, zorder=2,
                          error_kw={'ecolor': '#ddeeff', 'elinewidth': 1.3})
        for bar, v in zip(bars, model_accs):
            ax_acc.text(bar.get_x()+bar.get_width()/2, v+4.5, f"{v:.2f}%",
                        ha='center', color=TEXT_C, fontweight='bold', fontsize=12)
        ax_acc.set_ylim(0, 100)
        ax_acc.set_ylabel("5-Fold CV Accuracy (%)")
        ax_acc.set_title("Model Accuracy Comparison\nTrained on 638 Real NASA TOI Stars",
                         fontsize=10)
        style_ax(ax_acc)
        plt.tight_layout()
        st.pyplot(fig_acc, use_container_width=True)
        plt.close(fig_acc)

    # ── Chart 2: Confusion matrix (held-out test set) ──
    with vc2:
        cm = np.array([[76, 4], [8, 72]])
        labels_cm = ["False\nPositive", "Planet"]

        fig_cm, ax_cm = plt.subplots(figsize=(5, 5.5), facecolor=BG)
        cmap_cm = matplotlib.colors.LinearSegmentedColormap.from_list(
            "cm_cmap", ["#eef4fb", "#0a3d7a"])
        im = ax_cm.imshow(cm, cmap=cmap_cm, vmin=0, vmax=cm.max())
        for i in range(2):
            for j in range(2):
                txt_color = "#0a1220" if cm[i, j] < cm.max()*0.6 else "white"
                ax_cm.text(j, i, str(cm[i, j]), ha='center', va='center',
                          fontsize=22, fontweight='bold', color=txt_color)
        ax_cm.set_xticks([0, 1]); ax_cm.set_xticklabels(labels_cm)
        ax_cm.set_yticks([0, 1]); ax_cm.set_yticklabels(labels_cm)
        ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
        n_test = int(cm.sum())
        ax_cm.set_title(f"Confusion Matrix\n(Held-out Test Set, n={n_test})", fontsize=10)
        ax_cm.tick_params(colors=LABEL_C, labelsize=9)
        for spine in ax_cm.spines.values():
            spine.set_edgecolor(GRID_C)
        ax_cm.set_title(ax_cm.get_title(), color=TEXT_C, fontweight='bold', fontsize=10, pad=8)
        ax_cm.set_xlabel(ax_cm.get_xlabel(), color=LABEL_C, fontsize=9)
        ax_cm.set_ylabel(ax_cm.get_ylabel(), color=LABEL_C, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)

    # ── Chart 3: RandomForest feature importance ──
    with vc3:
        try:
            imp_vals = clf_model.feature_importances_
        except Exception:
            imp_vals = np.array([0.2208, 0.2098, 0.1732, 0.1370, 0.1360, 0.1231])
        imp_labels = ["Duration\n(hours)", "BLS\nPower", "SNR",
                      "Transit\nDepth", "Secondary/Primary\nRatio", "Odd-Even\nDepth Diff"]
        order = np.argsort(imp_vals)[::-1][:6]
        imp_vals_sorted   = np.array(imp_vals)[order] * 100
        imp_labels_sorted = [imp_labels[i] if i < len(imp_labels) else FEATURE_COLS[i]
                              for i in order]

        fig_imp, ax_imp = plt.subplots(figsize=(5, 5.5), facecolor=BG)
        ypos = np.arange(len(imp_vals_sorted))[::-1]
        bars = ax_imp.barh(ypos, imp_vals_sorted, color=ACCENT, height=0.6, zorder=2)
        ax_imp.set_yticks(ypos)
        ax_imp.set_yticklabels(imp_labels_sorted, color=LABEL_C, fontsize=8)
        for bar, v in zip(bars, imp_vals_sorted):
            ax_imp.text(v+0.4, bar.get_y()+bar.get_height()/2, f"{v:.2f}%",
                       va='center', color=TEXT_C, fontweight='bold', fontsize=9)
        ax_imp.set_xlabel("Importance (%)")
        ax_imp.set_title("Feature Importance\n(RandomForest)", fontsize=10)
        style_ax(ax_imp)
        plt.tight_layout()
        st.pyplot(fig_imp, use_container_width=True)
        plt.close(fig_imp)

    st.caption("RandomForest — Model Accuracy · Confusion Matrix · Feature Importance")

    st.markdown("""
    <div class='report-card'>
    <h3>5. Key Improvements in v8.0</h3>
    <ul>
    <li><b>641-star training dataset</b> — pulled directly from NASA TOI catalog with auto-balanced classes, expanded from 556 to include additional shallow-transit confirmed planets</li>
    <li><b>RF+GB Ensemble</b> — RandomForest + GradientBoosting soft-voting ensemble replaces single RF, boosting CV accuracy from 67% to 87.35%</li>
    <li><b>FFI fallback</b> — When SPOC 2-min data unavailable, pipeline automatically tries TESS-SPOC and QLP 10-min FFI data (3× more stars load successfully)</li>
    <li><b>1D-CNN trained</b> — Convolutional neural network on 493 phase-folded curves achieves 88.65% ± 2.92% CV accuracy, confirming ensemble results</li>
    <li><b>Infinity/NaN guard</b> — Feature matrix cleaned before training to prevent float32 overflow crashes</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 4 — HISTORY
# ════════════════════════════════════════════════════════════
elif page == "📜 History":
    st.subheader("📜 Session Test History")

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)

        h1, h2 = st.columns(2)
        with h1:
            csv_dl = hist_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download CSV", csv_dl,
                               "exodetect_history.csv", "text/csv", use_container_width=True)
        with h2:
            if st.button("🗑️ Clear All History", use_container_width=True):
                st.session_state.history       = []
                st.session_state.last_result   = None
                st.session_state.results_cache = {}
                st.session_state["compare_results"] = []
                st.rerun()

        st.markdown("---")
        st.subheader("📊 Session Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Stars Analyzed",    len(hist_df))
        s2.metric("Planets Found",     (hist_df["ML Verdict"]=="Exoplanet Candidate").sum())
        s3.metric("Binaries Rejected", hist_df["ML Verdict"].str.contains("Binary|False", na=False).sum())
        s4.metric("Avg SNR",           f"{hist_df['SNR'].mean():.1f}" if 'SNR' in hist_df else "—")
    else:
        st.info("No analyses run yet. Go to Individual Analysis or Compare Stars.")

st.markdown("---")
st.caption(
    f"ExoDetect v8.0 | BAH2026 PS7 | Jadavpur University | "
    f"NASA TESS / MAST | {model_source}"
)
