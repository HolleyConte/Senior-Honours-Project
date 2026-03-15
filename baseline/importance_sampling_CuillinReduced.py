"""
Reduced-parameter importance-sampling reweighting script

Designed for:

1) Reduced original index chains with columns: omega_m  sigma_8  log_weight  prior  post

2) Reduced schmear+temp chain with columns like: omega_m  sigma_8  
    nz_source_errors--bias_1 ... bias_5  
    nz_source_errors--width_1 ... width_5  
    log_weight
    prior
    post

This script:
  1) Loads the reduced schmear+temp chain once
  2) For each reduced index file:
       - computes the weighted mean in (Omega_m, S8)
       - optionally cuts the schmear chain to a box in (Omega_m, S8)
       - evaluates the NEW log-posterior using a per-index CosmoSIS pipeline
       - computes reweighted log-weights:
            log_w_new = log_w_old + (log_post_new - log_post_old)
       - rescales log_w_new by subtracting max(log_w_new)
       - saves:
            a reduced cosmosis-style chain
            a diagnostics file
            a simple Ωm–σ8 diagnostic plot


"""

import os
import re
import glob
import numpy as np
import cosmosis
from mpi4py.MPI import COMM_WORLD as COMM

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# ======  CONFIG  =========
# =========================

PARAMS_INI   = "params.ini"
SCHMEAR_FILE = "S2_reducedParams/xS2-reduced_params_posterior_schmear_temp.txt"
INDEX_GLOB   = "S2_reducedParams/xS2-reduced_n_z_real_index_*.txt"
OUT_DIR      = "reduced_params_importance_sampling_NoCut"

# Toggle: True = apply rectangular cut in (Omega_m, S8), False = use full schmear chain
USE_BAND_MASK = False

# Box half-widths in the reduced space
BOX_HALF_OM = 0.036
BOX_HALF_S8 = 0.030

# Optional testing truncation
FIRST_N_EVAL = None         # None = evaluate all kept samples
MAX_INDEX_FILES = None      # None = process all index files

# Optional manual filter for plots / files
PLOT_INCLUDE_TOKENS = None
# Example:
# PLOT_INCLUDE_TOKENS = ["index_1.txt", "index_7.txt"]

# Output parameter names for reduced chains
OUTPUT_PARAM_NAMES = [
    "cosmological_parameters--omega_m",
    "cosmological_parameters--sigma_8",
]


# =========================
# ======  HELPERS  ========
# =========================

def load_txt_2d(path):
    """Load txt and ensure a 2D array even if only one row survives."""
    arr = np.loadtxt(path)
    return np.atleast_2d(arr)


def compute_S8(omega_m, sigma8):
    """S8 = sigma8 * sqrt(omega_m / 0.3)"""
    omega_m = np.asarray(omega_m, dtype=float)
    sigma8  = np.asarray(sigma8, dtype=float)
    return sigma8 * np.sqrt(omega_m / 0.3)


def parse_nz_index_from_filename(path):
    """
    Extract integer n(z) index from filename like:
      reduced_n_z_real_index_1.txt  -> 1
      xS2-reduced_n_z_real_index_37.txt -> 37
    """
    name = os.path.basename(path)
    m = re.search(r"n_z[_-]real[_-]index[_-](\d+)\.txt$", name)
    if m:
        return int(m.group(1))

    # fallback: last integer before .txt
    m = re.search(r"(\d+)\.txt$", name)
    if m:
        return int(m.group(1))

    raise ValueError(f"Could not extract n(z) index from filename: {path}")


def read_header_columns(path):
    """
    Read the first header line that looks like a column header.
    Returns a list of column names.

    We look for the first line starting with '#' that contains either:
      - 'cosmological_parameters--'
      - 'log_weight'
      - 'post'

    This is much safer than hardcoding column positions.
    """
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("#"):
                continue

            s = s[1:].strip()  # remove leading '#'

            if (
                "cosmological_parameters--" in s
                or "nz_source_errors--" in s
                or "log_weight" in s
                or "post" in s
            ):
                cols = s.split()
                return cols

    raise RuntimeError(f"Could not find a usable header row in {path}")


def build_colmap(path):
    """
    Return dict: column_name -> column_index
    """
    cols = read_header_columns(path)
    return {name: i for i, name in enumerate(cols)}


def find_required_col(colmap, candidates, what, path):
    """
    Return the first matching column index from candidate names.
    Raises a clear error if not found.
    """
    for c in candidates:
        if c in colmap:
            return colmap[c]
    raise KeyError(
        f"Could not find column for '{what}' in file:\n"
        f"  {path}\n"
        f"Available columns are:\n"
        f"  {list(colmap.keys())}"
    )


def maybe_filter_files(files, include_tokens):
    """
    Optionally keep only files whose basename contains any token
    """
    if include_tokens is None:
        return files
    keep = []
    for f in files:
        b = os.path.basename(f)
        if any(tok in b for tok in include_tokens):
            keep.append(f)
    return keep


def weighted_mean(x, w):
    """
    Simple weighted mean with a safety check
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    ws = np.sum(w)
    if ws <= 0 or not np.isfinite(ws):
        return np.nan
    return np.sum(w * x) / ws


def make_diagnostic_plot(
    outdir,
    idx_number,
    Om_s,
    sig8_s,
    Om_cut,
    sig8_cut,
    Om_center,
    sig8_center,
    use_band_mask,
    box_half_om,
    box_half_sig8,
):
    """Save a simple Ωm vs σ8 diagnostic plot."""
    plt.figure(figsize=(8, 6))

    # Background proposal cloud
    plt.scatter(
        sig8_s, Om_s,
        s=2, alpha=0.08,
        color="deepskyblue",
        label="full reduced schmear+temp"
    )

    # Kept / cut cloud
    plt.scatter(
        sig8_cut, Om_cut,
        s=8, alpha=0.7,
        color="deeppink",
        label=f"kept proposal samples (index {idx_number})"
    )

    if use_band_mask:
        plt.axvline(sig8_center - box_half_sig8, color="deeppink", linestyle="-", linewidth=1.5)
        plt.axvline(sig8_center + box_half_sig8, color="deeppink", linestyle="-", linewidth=1.5)
        plt.axhline(Om_center   - box_half_om,   color="deeppink", linestyle="--", linewidth=1.5)
        plt.axhline(Om_center   + box_half_om,   color="deeppink", linestyle="--", linewidth=1.5)

    plt.xlabel(r"$\sigma_8$", fontsize=18)
    plt.ylabel(r"$\Omega_m$", fontsize=18)
    plt.title(rf"Reduced-space diagnostic (index {idx_number})", fontsize=20)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12, markerscale=2)
    plt.tight_layout()

    outpath = os.path.join(outdir, f"diagnostic_{idx_number}_omegam_vs_sigma8.png")
    plt.savefig(outpath)
    plt.close()


# =========================
# ======  MAIN  ===========
# =========================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # -----------------------------------
    # 1) Build column maps from headers
    # -----------------------------------
    schmear_colmap = build_colmap(SCHMEAR_FILE)

    # Required schmear columns
    SCHMEAR_OMEGA_M_COL = find_required_col(
        schmear_colmap,
        ["cosmological_parameters--omega_m"],
        "schmear omega_m",
        SCHMEAR_FILE,
    )
    SCHMEAR_SIGMA8_COL = find_required_col(
        schmear_colmap,
        ["cosmological_parameters--sigma_8"],
        "schmear sigma_8",
        SCHMEAR_FILE,
    )
    SCHMEAR_LOGW_COL = find_required_col(
        schmear_colmap,
        ["log_weight"],
        "schmear log_weight",
        SCHMEAR_FILE,
    )
    SCHMEAR_PRIOR_COL = find_required_col(
        schmear_colmap,
        ["prior"],
        "schmear prior",
        SCHMEAR_FILE,
    )
    SCHMEAR_POST_COL = find_required_col(
        schmear_colmap,
        ["post"],
        "schmear post",
        SCHMEAR_FILE,
    )

    print("Loaded schmear header columns:")
    for k, v in schmear_colmap.items():
        print(f"  {v:2d}  {k}")

    # -----------------------------------
    # 2) Load schmear chain once
    # -----------------------------------
    schmear = load_txt_2d(SCHMEAR_FILE)

    Om_s   = schmear[:, SCHMEAR_OMEGA_M_COL]
    sig8_s = schmear[:, SCHMEAR_SIGMA8_COL]
    S8_s   = compute_S8(Om_s, sig8_s)

    logw_old_full    = schmear[:, SCHMEAR_LOGW_COL]
    prior_old_full   = schmear[:, SCHMEAR_PRIOR_COL]
    logpost_old_full = schmear[:, SCHMEAR_POST_COL]

    # theta passed to reduced pipeline: only omega_m, sigma_8
    PARAM_COLS = [SCHMEAR_OMEGA_M_COL, SCHMEAR_SIGMA8_COL]

    # -----------------------------------
    # 3) Index files
    # -----------------------------------
    index_files = sorted(glob.glob(INDEX_GLOB))
    index_files = maybe_filter_files(index_files, PLOT_INCLUDE_TOKENS)

    if not index_files:
        raise SystemExit(f"No index files found for glob: {INDEX_GLOB}")

    if MAX_INDEX_FILES is not None:
        index_files = index_files[:MAX_INDEX_FILES]
        print(f"Limiting to first {len(index_files)} index files.")

    # Build index colmap from the first file
    first_index_colmap = build_colmap(index_files[0])

    INDEX_OMEGA_M_COL = find_required_col(
        first_index_colmap,
        ["cosmological_parameters--omega_m"],
        "index omega_m",
        index_files[0],
    )
    INDEX_SIGMA8_COL = find_required_col(
        first_index_colmap,
        ["cosmological_parameters--sigma_8"],
        "index sigma_8",
        index_files[0],
    )
    INDEX_LOGW_COL = find_required_col(
        first_index_colmap,
        ["log_weight"],
        "index log_weight",
        index_files[0],
    )
    INDEX_PRIOR_COL = find_required_col(
        first_index_colmap,
        ["prior"],
        "index prior",
        index_files[0],
    )
    INDEX_POST_COL = find_required_col(
        first_index_colmap,
        ["post"],
        "index post",
        index_files[0],
    )

    print("\nLoaded index header columns:")
    for k, v in first_index_colmap.items():
        print(f"  {v:2d}  {k}")

    # -----------------------------------
    # 4) Loop over index files
    # -----------------------------------
    for k, idx_path in enumerate(index_files, start=1):
        # MPI sharding
        if k % COMM.Get_size() != COMM.Get_rank():
            continue

        print(f"\n[{k}/{len(index_files)}] Rank {COMM.Get_rank()} processing: {idx_path}")

        idx_arr = load_txt_2d(idx_path)

        Om_i   = idx_arr[:, INDEX_OMEGA_M_COL]
        sig8_i = idx_arr[:, INDEX_SIGMA8_COL]
        S8_i   = compute_S8(Om_i, sig8_i)

        # old chain weights for weighted means
        idx_logw = idx_arr[:, INDEX_LOGW_COL]
        idx_w = np.exp(idx_logw - np.max(idx_logw))

        Om_center   = weighted_mean(Om_i, idx_w)
        S8_center   = weighted_mean(S8_i, idx_w)
        sig8_center = weighted_mean(sig8_i, idx_w)

        idx_number = parse_nz_index_from_filename(idx_path)

        # -------------------------
        # 4.1) Optional cut in (Om, S8)
        # -------------------------
        if USE_BAND_MASK:
            band_mask = (
                (np.abs(Om_s - Om_center) <= BOX_HALF_OM) &
                (np.abs(S8_s - S8_center) <= BOX_HALF_S8)
            )
            cut = schmear[band_mask]
            if cut.shape[0] == 0:
                print(f"  -> band cut kept 0 points for index {idx_number}; skipping.")
                continue
        else:
            band_mask = np.ones_like(Om_s, dtype=bool)
            cut = schmear

        if FIRST_N_EVAL is not None:
            cut = cut[:FIRST_N_EVAL]

        # Match the exact same kept rows for old logw / old post / old prior
        logw_old    = logw_old_full[band_mask]
        prior_old   = prior_old_full[band_mask]
        logpost_old = logpost_old_full[band_mask]

        if FIRST_N_EVAL is not None:
            ncut = cut.shape[0]
            logw_old    = logw_old[:ncut]
            prior_old   = prior_old[:ncut]
            logpost_old = logpost_old[:ncut]

        Om_cut   = cut[:, SCHMEAR_OMEGA_M_COL]
        sig8_cut = cut[:, SCHMEAR_SIGMA8_COL]

        # -------------------------
        # 4.2) Diagnostic plot
        # -------------------------
        make_diagnostic_plot(
            OUT_DIR,
            idx_number,
            Om_s,
            sig8_s,
            Om_cut,
            sig8_cut,
            Om_center,
            sig8_center,
            USE_BAND_MASK,
            BOX_HALF_OM,
            BOX_HALF_S8,
        )

        # -------------------------
        # 4.3) Build per-index pipeline
        # -------------------------
        from cosmosis.runtime.config import Inifile

        nz_idx = parse_nz_index_from_filename(idx_path)
        ini = Inifile(PARAMS_INI)
        ini.set("load_nz", "index", str(nz_idx))
        index_pipeline = cosmosis.LikelihoodPipeline(ini)

        try:
            chosen = index_pipeline.options["load_nz"].getint("index")
        except Exception:
            chosen = ini.getint("load_nz", "index")

        if chosen != nz_idx:
            raise RuntimeError(f"n(z) override failed: wanted {nz_idx}, got {chosen}")
        else:
            print(f"  -> using n(z) index = {chosen}")

        # -------------------------
        # 4.4) Evaluate NEW log-posterior
        # -------------------------
        theta_mat = cut[:, PARAM_COLS]   # shape (N, 2)
        logpost_new = np.empty(theta_mat.shape[0], dtype=float)

        for i, theta in enumerate(theta_mat):
            lp, _ = index_pipeline.posterior(theta)
            logpost_new[i] = float(lp)

        # -------------------------
        # 4.5) Drop unusable samples
        # -------------------------
        good = (
            np.isfinite(logpost_new) &
            np.isfinite(logpost_old) &
            np.isfinite(logw_old)
        )

        n_bad = good.size - np.count_nonzero(good)
        if n_bad > 0:
            print(f"  -> dropping {n_bad} / {good.size} samples with non-finite logpost/logw")

        cut         = cut[good]
        theta_mat   = theta_mat[good]
        logpost_new = logpost_new[good]
        logpost_old = logpost_old[good]
        logw_old    = logw_old[good]
        prior_old   = prior_old[good]

        if cut.shape[0] == 0:
            print(f"  -> 0 finite samples remain for index {idx_number}; skipping.")
            continue

        # -------------------------
        # 4.6) Importance sampling
        # -------------------------
        logw_new = logw_old + (logpost_new - logpost_old)

        # rescale to avoid catastrophic underflow
        logw_new = logw_new - np.max(logw_new)

        # For output prior column:
        # zero is usually fine for postprocessing, and matches your earlier workflow
        prior_col = np.zeros_like(logpost_new)

        # -------------------------
        # 4.7) Save outputs
        # -------------------------
        # Main reduced cosmosis-style chain:
        #   omega_m, sigma_8, log_weight, prior, post
        main_table = np.column_stack([
            theta_mat[:, 0],
            theta_mat[:, 1],
            logw_new,
            prior_col,
            logpost_new,
        ])

        # Diagnostics sidecar:
        #   omega_m, sigma_8, log_weight_old, log_weight_new, prior_old, post_old, post_new
        diag_table = np.column_stack([
            theta_mat[:, 0],
            theta_mat[:, 1],
            logw_old,
            logw_new,
            prior_old,
            logpost_old,
            logpost_new,
        ])

        header_main = "# " + " ".join(OUTPUT_PARAM_NAMES + ["log_weight", "prior", "post"])

        n_used = cut.shape[0]
        meta_lines = [
            "## reduced-parameter reweighted schmear subset evaluated through per-index pipeline",
            f"## index_file = {idx_path}",
            f"## nz_index_override = {nz_idx}",
            f"## box_center_Om = {Om_center:.8f}, half_width = {BOX_HALF_OM}",
            f"## box_center_S8 = {S8_center:.8f}, half_width = {BOX_HALF_S8}",
            f"## box_center_sigma8 = {sig8_center:.8f}",
            f"## USE_BAND_MASK = {USE_BAND_MASK}",
            f"## FIRST_N_EVAL = {FIRST_N_EVAL}",
            f"## number_of_samples_used = {n_used}",
        ]
        header_main = header_main + "\n" + "\n".join(meta_lines)

        header_diag = "# " + " ".join(
            OUTPUT_PARAM_NAMES +
            ["log_weight_old", "log_weight_new", "prior_old", "post_old", "post_new"]
        )

        base = os.path.splitext(os.path.basename(idx_path))[0]
        out_txt_main = os.path.join(OUT_DIR, f"reweight_{base}.txt")
        out_txt_diag = os.path.join(OUT_DIR, f"reweight_{base}_diagnostics.txt")

        np.savetxt(out_txt_main, main_table, fmt="%.10f", header=header_main, comments="")
        np.savetxt(out_txt_diag, diag_table, fmt="%.10f", header=header_diag, comments="")

        print(f"  -> saved {main_table.shape[0]} rows to: {out_txt_main}")
        print(f"  -> saved diagnostics to: {out_txt_diag}")

    print("\nDone! Reduced importance-sampling outputs are in:", OUT_DIR)


if __name__ == "__main__":
    main()