"""
build_targets.py
Pulls confirmed planets, false positives, and eclipsing binaries
from NASA Exoplanet Archive TOI table and saves training_targets.csv.
Run this FIRST before anything else.
"""

import requests
import pandas as pd
import numpy as np
import os

print("=" * 60)
print("ExoDetect v2 — Step 1: Build Training Targets")
print("=" * 60)

# ── 1. Download TOI table from NASA Exoplanet Archive ──────
print("\n[1/4] Downloading TOI catalog from NASA Exoplanet Archive...")

TOI_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+toi,tid,tfopwg_disp,pl_orbper,pl_trandep,pl_tranmid"
    "+from+toi"
    "+where+tfopwg_disp+is+not+null"
    "+and+pl_orbper+is+not+null"
    "+and+pl_orbper+>+0.5"
    "+and+pl_orbper+<+30"
    "&format=csv"
)

try:
    resp = requests.get(TOI_URL, timeout=120)
    resp.raise_for_status()
    with open("toi_raw.csv", "wb") as f:
        f.write(resp.content)
    print("  Downloaded successfully.")
except Exception as e:
    print(f"  ERROR: {e}")
    print("  Check your internet connection and try again.")
    exit(1)

# ── 2. Load and inspect ────────────────────────────────────
print("\n[2/4] Loading and inspecting data...")
df = pd.read_csv("toi_raw.csv", comment="#")
print(f"  Total TOI entries: {len(df)}")
print(f"  Columns: {list(df.columns)}")
print(f"  Disposition counts:\n{df['tfopwg_disp'].value_counts().to_string()}")

# ── 3. Map dispositions to 3 classes ──────────────────────
print("\n[3/4] Mapping to 3 classes: planet / false_positive / eclipsing_binary...")

# NASA TFOPWG dispositions:
# KP  = Known Planet (confirmed)
# CP  = Confirmed Planet
# PC  = Planet Candidate
# APC = Ambiguous Planet Candidate (skip — too uncertain)
# FP  = False Positive (generic)
# FA  = False Alarm
# EB  = Eclipsing Binary (explicit)
# NEB = Nearby Eclipsing Binary

def map_label(disp):
    d = str(disp).strip().upper()
    if d in ["KP", "CP"]:
        return "planet"
    elif d == "PC":
        return "planet"          # planet candidate — treat as planet
    elif d in ["EB", "NEB"]:
        return "eclipsing_binary"
    elif d in ["FP", "FA"]:
        return "false_positive"
    else:
        return None              # APC and unknown — skip

df["label"] = df["tfopwg_disp"].apply(map_label)
df = df[df["label"].notna()].copy()
print(f"  After filtering: {len(df)} usable entries")
print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")

# ── 4. Balance classes and cap size ───────────────────────
print("\n[4/4] Balancing classes and building final target list...")

planets   = df[df["label"] == "planet"]
fps       = df[df["label"] == "false_positive"]
ebs       = df[df["label"] == "eclipsing_binary"]

print(f"  Raw planets:           {len(planets)}")
print(f"  Raw false positives:   {len(fps)}")
print(f"  Raw eclipsing binary:  {len(ebs)}")

# Take up to 350 planets, 250 FP, 200 EB — total ~800
# Sort by period so we get diversity (short + long period planets)
planets_sel = planets.sort_values("pl_orbper").head(350)
fps_sel     = fps.head(250)
ebs_sel     = ebs.head(200)

final = pd.concat([planets_sel, fps_sel, ebs_sel], ignore_index=True)
final = final[["tid", "pl_orbper", "label"]].copy()
final.columns = ["tid", "pl_orbper", "label"]
final["tid"] = final["tid"].astype(int)
final = final.dropna(subset=["tid", "pl_orbper"])
final = final.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

final.to_csv("training_targets.csv", index=False)

print(f"\n  DONE! Final dataset:")
print(f"  {final['label'].value_counts().to_string()}")
print(f"  Total: {len(final)} stars")
print(f"\n  Saved: training_targets.csv")
print("\n  Next step: python extract_features.py")
