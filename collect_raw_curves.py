import lightkurve as lk
import numpy as np
import pandas as pd
import os

print("Loading training targets...")
targets = pd.read_csv("training_targets.csv")
os.makedirs("raw_curves", exist_ok=True)

N_PHASE_POINTS = 200
success = 0
fail = 0

def to_plain_array(arr):
    """
    Converts astropy MaskedNDArray (or any masked array) into a
    plain numpy float array, with masked/invalid points as NaN.
    This is needed because newer astropy versions return a special
    masked array type from .flux.value that numpy.save/numpy.pad
    cannot handle directly.
    """
    if hasattr(arr, "filled"):
        arr = arr.filled(np.nan)
    return np.asarray(arr, dtype=float)

for idx, row in targets.iterrows():
    tic_id = str(int(row['tid']))
    label = row['label']
    known_period = float(row['pl_orbper'])
    save_path = f"raw_curves/TIC{tic_id}_{label}.npy"

    if os.path.exists(save_path):
        print(f"[{idx+1}/{len(targets)}] TIC {tic_id} already done, skipping.")
        success += 1
        continue

    print(f"[{idx+1}/{len(targets)}] TIC {tic_id} ({label})...")

    try:
        search = lk.search_lightcurve(
            f'TIC {tic_id}', mission='TESS',
            author='SPOC', exptime=120
        )
        if len(search) == 0:
            fail += 1
            continue

        n_use = min(2, len(search))
        lc = search[:n_use].download_all().stitch()
        lc = lc.normalize().flatten(window_length=401).remove_outliers(sigma=4)

        p_lo = max(known_period * 0.97, 0.3)
        p_hi = known_period * 1.03

        if p_hi <= p_lo:
            fail += 1
            continue

        periods = np.linspace(p_lo, p_hi, 300)
        durations = np.arange(0.02, 0.3, 0.02)
        durations = durations[durations < p_lo]

        if len(durations) == 0:
            fail += 1
            continue

        try:
            pg = lc.to_periodogram(
                method='bls', period=periods, duration=durations,
                frequency_factor=10
            )
        except Exception:
            fail += 1
            continue

        best_period = pg.period_at_max_power
        t0 = pg.transit_time_at_max_power

        folded = lc.fold(period=best_period, epoch_time=t0)
        binned = folded.bin(time_bin_size=1.0 / N_PHASE_POINTS)

        # ── KEY FIX: convert masked astropy array to plain numpy array ──
        flux = to_plain_array(binned.flux.value)

        if len(flux) >= N_PHASE_POINTS:
            flux = flux[:N_PHASE_POINTS]
        else:
            flux = np.pad(flux, (0, N_PHASE_POINTS - len(flux)), mode='edge')

        fmin = np.nanmin(flux)
        fmax = np.nanmax(flux)
        flux = (flux - fmin) / (fmax - fmin + 1e-10)
        flux = np.nan_to_num(flux, nan=0.5)

        np.save(save_path, flux)
        success += 1
        print(f"  Saved!")

    except Exception as e:
        print(f"  Error: {e}")
        fail += 1

print(f"\nDONE! Success: {success} | Failed: {fail}")
print("Next: run train_cnn.py")