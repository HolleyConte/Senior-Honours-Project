
"""
Reweighting S8 via parallel lines method!!

- "along the line" = T = "S8-perpendicular" direction = find by fitting a straight line to the index samples in (Omega_m, S8) space
- "perpendicular to the line" = S = "S8" direction


We load the big schmear+temp chain ONCE. For each index file:
  1) read the index chain, compute (Omega_m, S8), fit its line;
  2) build a thin band (two parallel lines) around that fit (padding in t and s);
  3) cut the schmear samples to that band;
  4) evaluate those schmear points with the per-index pipeline;
  5) write a cosmosis-friendly .txt: [schmear row ...] prior log_weight post

This script reweights the big schmear+temp chain *through each index pipeline*
by only re-evaluating schmear samples that lie inside a thin band that hugs the
index posterior in the (Omega_m, sigma_8) plane.

We read each index .txt file to get the index cloud, fit its line, build a band
(two parallel lines) around it, cut the schmear chain to that band, and then
evaluate those schmear samples using the index pipeline.

Outputs: one .txt per index.
"""


import os
import re
import glob
import numpy as np
import cosmosis


# =========================
# ======  CONFIG  =========
# =========================

# Paths
PARAMS_INI = "/Users/holleyconte/Desktop/Senior-Honours-Project/baseline/params.ini"
SCHMEAR_FILE = "/Users/holleyconte/Desktop/Senior-Honours-Project/baseline/schmear_0.2_AND_temp_20.txt"
INDEX_GLOB  = "/Users/holleyconte/Desktop/Senior-Honours-Project/baseline/all_n_z_realizations/*.txt"
OUT_DIR     = "/Users/holleyconte/Desktop/Senior-Honours-Project/baseline/importance_sampling_results"

# Column mappings (0-based)
OMEGA_M_COL   = 0         # Ω_m = column 1
SIGMA8_COL    = 4         # σ_8 = column 5
OLD_LOGW_COL  = 16        # schmear file has log-weights in column 17

# Index chain rows end with [log_weight, prior, post] so add 3 "bookkeeping columns" at the end
N_TRAILING_TO_DROP = 3

# Band padding:
# - PAD_T grows the span *along the index line* slightly beyond [min, max] of the index points.
# - PAD_S sets the half-width *perpendicular* to the line, i.e., the "thickness" of the band.
PAD_T = 0.15
PAD_S = 0.15


# ========================
# Testing!! cap the number of evaluations per index (None = use all of them, 10 = use first 10, etc)
FIRST_N_EVAL = 500
# Testing!! cap how many index files to run (None = use all of them)
MAX_INDEX_FILES = None
# ========================



# EXACT parameter columns to pass to the pipeline and to save, in the order
# the pipeline expects. Adjust if your params.ini uses a different order.
PARAM_COLS = [0, 1, 2, 3, 4, 5]
# Names must exactly match what cosmosis-postprocess expects for the originals:
PARAM_NAMES = [
    "cosmological_parameters--omega_m",
    "cosmological_parameters--h0",
    "cosmological_parameters--ombh2",
    "cosmological_parameters--n_s",
    "cosmological_parameters--sigma_8",
    "halo_model_parameters--logt_agn",
]



# =========================
# ======  HELPERS  ========
# =========================

def compute_S8(omega_m, sigma8):
    return sigma8 * np.sqrt(omega_m / 0.3)

def fit_line_S8_vs_Om(omega_m, S8):
    """
    Fit S8 ≈ a*Omega_m + b.
    Returns:
      a, b         : slope, intercept
      t_hat (unit): along-line direction (S8-perpendicular)
      s_hat (unit): perpendicular direction (S8 direction)
    """
    A = np.vstack([omega_m, np.ones_like(omega_m)]).T
    a, b = np.linalg.lstsq(A, S8, rcond=None)[0]

    t = np.array([1.0, a], dtype=float)   # along the line
    t_hat = t / np.linalg.norm(t)

    s = np.array([-a, 1.0], dtype=float)  # perpendicular to the line
    s_hat = s / np.linalg.norm(s)
    return a, b, t_hat, s_hat


def project_ts(omega_m, S8, origin, t_hat, s_hat):
    """
    Project points to (t, s), where:
      t = along the index line (S8-perp)
      s = across the line (S8)
    """
    P  = np.column_stack([omega_m, S8])
    P0 = P - origin[None, :]
    t = P0 @ t_hat
    s = P0 @ s_hat
    return t, s


def parse_nz_index_from_filename(path):
    """
    Extract first integer from filename to use as the n(z) index override
    """
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1




# =========================
# ======  MAIN RUN  =======
# =========================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Load schmear once
    schmear = np.loadtxt(SCHMEAR_FILE)
    n_cols = schmear.shape[1]
    if N_TRAILING_TO_DROP < 0 or N_TRAILING_TO_DROP > n_cols - 1:
        raise SystemExit(f"N_TRAILING_TO_DROP={N_TRAILING_TO_DROP} is inconsistent with schmear columns={n_cols}")

    Om_s   = schmear[:, OMEGA_M_COL]
    sig8_s = schmear[:, SIGMA8_COL]
    S8_s   = compute_S8(Om_s, sig8_s)

    old_logw = schmear[:, OLD_LOGW_COL] if (OLD_LOGW_COL is not None) else None

    # 2) Index files to process
    index_files = sorted(glob.glob(INDEX_GLOB))
    if not index_files:
        raise SystemExit(f"No index files found for glob: {INDEX_GLOB}")
    if MAX_INDEX_FILES is not None:
        index_files = index_files[:MAX_INDEX_FILES]
        print(f"Limiting to first {len(index_files)} index files.")

    for k, idx_path in enumerate(index_files, start=1):
        print(f"\n[{k}/{len(index_files)}] Processing index: {idx_path}")

        # 2.1) Index cloud -> (Ωm, S8)
        idx_arr = np.loadtxt(idx_path)
        Om_i    = idx_arr[:, OMEGA_M_COL]
        sig8_i  = idx_arr[:, SIGMA8_COL]
        S8_i    = compute_S8(Om_i, sig8_i)

        # 2.2) Fit line (t_hat along, s_hat across)
        a, b, t_hat, s_hat = fit_line_S8_vs_Om(Om_i, S8_i)
        origin = np.array([np.mean(Om_i), np.mean(S8_i)], dtype=float)

        # t-range from index points (+ padding)
        t_i, s_i = project_ts(Om_i, S8_i, origin, t_hat, s_hat)
        t_min, t_max = t_i.min() - PAD_T, t_i.max() + PAD_T

        # s half-width from index spread (+ padding)
        s_half = np.max(np.abs(s_i)) + PAD_S

        # 2.3) Cut schmear to the band
        t_s, s_s = project_ts(Om_s, S8_s, origin, t_hat, s_hat)
        band_mask = (t_s >= t_min) & (t_s <= t_max) & (np.abs(s_s) <= s_half)

        cut = schmear[band_mask]
        if cut.shape[0] == 0:
            print(f"  -> Band kept 0 points for {os.path.basename(idx_path)}. Increase PAD_T and/or PAD_S.")
            continue

        if FIRST_N_EVAL is not None:
            cut = cut[:FIRST_N_EVAL]

        # 2.4) Pipeline for this index
        nz_idx = parse_nz_index_from_filename(idx_path)
        overrides = {("load_nz", "index"): str(nz_idx)}
        index_pipeline = cosmosis.LikelihoodPipeline(PARAMS_INI, override=overrides)

        # 2.5) Evaluate log-posterior for each cut schmear row (ONLY the 6 model params)
        theta_mat = cut[:, PARAM_COLS]


        new_logpost = np.empty(theta_mat.shape[0], dtype=float)
        for i, theta in enumerate(theta_mat):
            lp, _ = index_pipeline.posterior(theta)
            new_logpost[i] = float(lp)

        # 2.6) Save with the SAME 9 columns as original: 6 params + log_weight + prior + post
        # old log-weight sliced to match this cut:
        old_logw_cut = old_logw[band_mask] if old_logw is not None else np.zeros(len(t_s))
        old_logw_cut = old_logw_cut[:len(new_logpost)]  # align with FIRST_N_EVAL if used
        prior_col = np.zeros_like(new_logpost)

        # Assemble output: 6 params (only!) + log_weight + prior + post
        to_save = np.column_stack([cut[:, PARAM_COLS], old_logw_cut, prior_col, new_logpost])

        # Header: EXACTLY the same names & order as the originals
        column_header = " ".join(PARAM_NAMES + ["log_weight", "prior", "post"])

        meta_lines = [
            "## reweighted schmear subset evaluated through index pipeline",
            f"## index_file = {idx_path}",
            f"## nz_index_override = {nz_idx}",
            "## geometry = band around line fitted to index cloud",
            f"## t_hat = [{t_hat[0]:.6f}, {t_hat[1]:.6f}]",
            f"## s_hat = [{s_hat[0]:.6f}, {s_hat[1]:.6f}]",
            f"## t_window = [{t_min:.6f}, {t_max:.6f}]",
            f"## s_half_width = {s_half:.6f}",
            f"## PAD_T = {PAD_T:.3f}, PAD_S = {PAD_S:.3f}",
            f"## Number of evaluations (samples) used = {FIRST_N_EVAL}",]
        header = "# " + column_header + "\n" + "\n".join(meta_lines)

        # >>> define output filename
        base = os.path.splitext(os.path.basename(idx_path))[0]
        out_txt = os.path.join(OUT_DIR, f"reweight_{base}.txt")

        np.savetxt(out_txt, to_save, fmt="%.8f", header=header, comments="")
        print(f"  -> saved {to_save.shape[0]} rows to: {out_txt}")


    print("\nDone! Now you can cosmosis-postprocess the outputs and compare to the original index chains!\n")


if __name__ == "__main__":
    main()



