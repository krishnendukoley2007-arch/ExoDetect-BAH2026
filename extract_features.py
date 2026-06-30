"""
extract_features.py
Downloads TESS light curves and extracts 6 BLS features per star.
Improvements over v1:
  - FFI (10-min) fallback when SPOC 2-min data is unavailable
  - Handles 3 classes: planet / false_positive / eclipsing_binary
  - Checkpoint saves every 10 stars so you don't lose progress
  - Better error messages
Run AFTER build_targets.py.
"""

import lightkurve as lk
import numpy as np
import pandas as pd
import os

print("=" * 60)
print("ExoDetect v2 — Step 2: Extract Features")
print("=" * 60)

targets = pd.read_csv("training_targets.csv")
print(f"\nTotal targets: {len(targets)}")
print(f"Label distribution:\n{targets['label'].value_counts().to_string()}\n")

# Load checkpoint if it exists (so you can resume interrupted runs)
checkpoint_file = "features_dataset_partial.csv"
done_tics = set()
features_list = []

if os.path.exists(checkpoint_file):
    existing = pd.read_csv(checkpoint_file)
    features_list = existing.to_dict("records")
    done_tics = set(existing["tic_id"].astype(str).tolist())
    print(f"  Resuming from checkpoint — {len(done_tics)} already done.\n")

success_count = 0
fail_count    = 0


def download_lc(tic_id):
    """
    Try SPOC 2-min first. If nothing found, fall back to TESS-SPOC
    or QLP 10-min FFI data. Returns (lc, source_label) or (None, reason).
    """
    # Attempt 1: SPOC 2-min (best quality)
    try:
        search = lk.search_lightcurve(f'TIC {tic_id}', mission='TESS',
                                       author='SPOC', exptime=120)
        if len(search) > 0:
            n = min(3, len(search))
            lc = search[:n].download_all().stitch()
            return lc, f"SPOC-2min({n}sectors)"
    except Exception:
        pass

    # Attempt 2: TESS-SPOC FFI (10-min)
    try:
        search = lk.search_lightcurve(f'TIC {tic_id}', mission='TESS',
                                       author='TESS-SPOC')
        if len(search) > 0:
            n = min(2, len(search))
            lc = search[:n].download_all().stitch()
            return lc, f"TESS-SPOC-FFI({n}sectors)"
    except Exception:
        pass

    # Attempt 3: QLP (MIT Quick Look Pipeline) FFI
    try:
        search = lk.search_lightcurve(f'TIC {tic_id}', mission='TESS',
                                       author='QLP')
        if len(search) > 0:
            n = min(2, len(search))
            lc = search[:n].download_all().stitch()
            return lc, f"QLP-FFI({n}sectors)"
    except Exception:
        pass

    return None, "No TESS data found in SPOC/TESS-SPOC/QLP"


for idx, row in targets.iterrows():
    tic_id       = str(int(row['tid']))
    label        = row['label']
    known_period = float(row['pl_orbper'])

    # Skip if already done
    if tic_id in done_tics:
        success_count += 1
        continue

    print(f"[{idx+1}/{len(targets)}] TIC {tic_id} ({label}) | period={known_period:.3f}d")

    try:
        lc_raw, source = download_lc(tic_id)

        if lc_raw is None:
            print(f"  SKIP: {source}")
            fail_count += 1
            continue

        lc = lc_raw.normalize().flatten(window_length=401).remove_outliers(sigma=4)

        if len(lc) < 100:
            print(f"  SKIP: too few points ({len(lc)})")
            fail_count += 1
            continue

        # BLS search narrowly around known period (±3%)
        p_lo = max(known_period * 0.97, 0.3)
        p_hi = known_period * 1.03
        if p_hi <= p_lo:
            fail_count += 1
            continue

        periods   = np.linspace(p_lo, p_hi, 300)
        durations = np.arange(0.02, 0.3, 0.02)
        durations = durations[durations < p_lo]

        if len(durations) == 0:
            fail_count += 1
            continue

        try:
            pg = lc.to_periodogram(method='bls', period=periods,
                                    duration=durations, frequency_factor=10)
        except Exception as e:
            print(f"  SKIP: BLS failed ({e})")
            fail_count += 1
            continue

        best_period   = pg.period_at_max_power
        best_power    = pg.max_power
        t0            = pg.transit_time_at_max_power
        duration_best = pg.duration_at_max_power
        half_width    = (float(duration_best.to('d').value) /
                         float(best_period.to('d').value)) / 2

        folded     = lc.fold(period=best_period, epoch_time=t0)
        phase_vals = np.asarray(folded.time.value,  dtype=float)
        flux_vals  = np.asarray(folded.flux.value,  dtype=float)
        if hasattr(flux_vals, 'filled'):
            flux_vals = flux_vals.filled(np.nan)

        in_transit  = np.abs(phase_vals) < half_width * 1.3
        out_transit = ((np.abs(phase_vals) > half_width * 3) &
                       (np.abs(phase_vals) < 0.45))
        secondary   = np.abs(np.abs(phase_vals) - 0.5) < half_width * 1.3

        if np.sum(in_transit) < 3 or np.sum(out_transit) < 10:
            print(f"  SKIP: not enough transit points")
            fail_count += 1
            continue

        baseline    = float(np.nanmedian(flux_vals[out_transit]))
        transit_med = float(np.nanmedian(flux_vals[in_transit]))
        depth       = float(baseline - transit_med)
        noise       = float(np.nanstd(flux_vals[out_transit]))
        n_in        = int(np.sum(in_transit))
        snr         = (depth / noise) * np.sqrt(n_in) if noise > 0 else 0.0

        sec_depth = 0.0
        if np.sum(secondary) > 3:
            sec_depth = float(baseline - np.nanmedian(flux_vals[secondary]))
        sec_ratio = (sec_depth / depth) if depth > 0 else 0.0

        t_arr    = np.asarray(lc.time.value,  dtype=float)
        f_arr    = np.asarray(lc.flux.value,  dtype=float)
        if hasattr(f_arr, 'filled'):
            f_arr = f_arr.filled(np.nan)

        cycle_num    = np.round((t_arr - t0.value) /
                                 float(best_period.value)).astype(int)
        phase_global = ((t_arr - t0.value) % float(best_period.value)) / float(best_period.value)
        phase_global[phase_global > 0.5] -= 1
        in_g = np.abs(phase_global) < half_width * 1.3

        o_d = (float(np.nanmedian(f_arr[in_g & (cycle_num % 2 == 1)]))
               if np.sum(in_g & (cycle_num % 2 == 1)) > 2 else 1.0)
        e_d = (float(np.nanmedian(f_arr[in_g & (cycle_num % 2 == 0)]))
               if np.sum(in_g & (cycle_num % 2 == 0)) > 2 else 1.0)
        odd_even_diff = abs(o_d - e_d)

        duration_hours = float(duration_best.to('d').value) * 24

        features_list.append({
            "tic_id":         tic_id,
            "label":          label,
            "catalog_period": known_period,
            "period_days":    float(best_period.value),
            "depth":          depth,
            "snr":            snr,
            "sec_ratio":      sec_ratio,
            "duration_hours": duration_hours,
            "bls_power":      float(best_power),
            "odd_even_diff":  odd_even_diff,
            "source":         source,
        })
        done_tics.add(tic_id)
        success_count += 1
        print(f"  OK [{source}] period={best_period.value:.4f}d "
              f"depth={depth*100:.4f}% snr={snr:.1f}")

    except Exception as e:
        print(f"  ERROR: {e}")
        fail_count += 1

    # Save checkpoint every 10 stars
    if (idx + 1) % 10 == 0:
        pd.DataFrame(features_list).to_csv(checkpoint_file, index=False)
        print(f"  [Checkpoint saved — {len(features_list)} done so far]")

# Final save
features_df = pd.DataFrame(features_list)
features_df.to_csv("features_dataset.csv", index=False)

print("\n" + "=" * 60)
print(f"DONE!  Success: {success_count}  |  Failed: {fail_count}")
print(f"Total features saved: {len(features_df)}")
print(f"\nLabel distribution:")
print(features_df['label'].value_counts().to_string())
print("\nNext step: python train_model.py")
