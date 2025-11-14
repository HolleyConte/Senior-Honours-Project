
"""
Reweighting S8 via parallel lines method!!

- "along the line" = "S8-perpendicular" direction
- "perpendicular to the line" = "S8" direction

Reweighting S8 via simple weighted-mean box around each index cloud.
We cut the schmear+temp chain to a rectangle centered on the index's
weighted mean in (Omega_m, S8), with half-widths 0.05 in each direction.

We:
  1) Load the big schmear+temp chain once.
  2) For each index file, compute the weighted means of (Omega_m, S8) from the index chain.
  3) Cut the schmear+temp samples to a rectangle centered on those means.
  4) Build a per-index CosmoSIS pipeline and evaluate the NEW log-posterior at the cut points.
  5) Do importance sampling in log-space:
         log_w_new = log_w_old + (log_post_new - log_post_old)
     where log_post_old is the schmear file's last column (proposal posterior),
           log_post_new is from the per-index pipeline (target posterior).
  6) Save a cosmosis-friendly 9-col txt (6 params + log_weight + prior + post) with meta header.
     Also save a diagnostics txt keeping both old/new weights & old/new posteriors.

Outputs: one .txt per index (+ diagnostics .txt)
"""


import os
import re
import glob
import numpy as np
import cosmosis
from mpi4py.MPI import COMM_WORLD as COMM



# =========================
# ======  CONFIG  =========
# =========================

# Paths
PARAMS_INI = "params.ini"
SCHMEAR_FILE = "schmear_0.2_AND_temp_20.txt"
INDEX_GLOB  = "n_z_real_index_*.txt"
OUT_DIR     = "importance_sampling_resultsCuillinALL"

# Column mappings (0-based)
OMEGA_M_COL       = 0         # Ω_m = column 1
SIGMA8_COL        = 4         # σ_8 = column 5
SCHMEAR_LOGW_COL  = 16        # schmear file has log-weights in column 17
SCHMEAR_POST_COL  = -1     # Schmear file has log-posterior in the last column (col 19)


# selection box half-widths
BOX_HALF_OM = 0.025
BOX_HALF_S8 = 0.025
BOX_HALF_SIG8 = BOX_HALF_S8



# Testing!! cap the number of samples per index (None = use all of them, 10 = use first 10, etc)
FIRST_N_EVAL = None
# Testing!! cap how many index files to run (None = use all of them)
MAX_INDEX_FILES = None


# To manually choose which indices go into the combined plots.
# If not None, only indices whose basename contains any of these tokens will be visualized
PLOT_INCLUDE_TOKENS = None
# E.g. PLOT_INCLUDE_TOKENS = ["index_2.txt", "index_7.txt"]


# EXACT parameter columns to pass to the pipeline and to save, in correct order
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


def parse_nz_index_from_filename(path):
    name = os.path.basename(path)
    m = re.search(r"n_z[_-]real[_-]index[_-](\d+)\.txt$", name)
    if not m:
        # Fallback to first integer in the basename
        m = re.search(r"(\d+)", name)
    if not m:
        raise ValueError(f"Error: Could not extract n(z) index from filename: {path}")
    return int(m.group(1))



# =========================
# ======  MAIN RUN  =======
# =========================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Load schmear once
    schmear = np.loadtxt(SCHMEAR_FILE)
    Om_s    = schmear[:, OMEGA_M_COL]
    sig8_s  = schmear[:, SIGMA8_COL]
    S8_s    = compute_S8(Om_s, sig8_s)

    # proposal (old) log weights and log posterior (from schmear file)
    logw_old_full  = schmear[:, SCHMEAR_LOGW_COL]
    logpost_old_full = schmear[:, SCHMEAR_POST_COL]


    # 2) Index files to process
    index_files = sorted(glob.glob(INDEX_GLOB))
    if not index_files:
        raise SystemExit(f"No index files found for glob: {INDEX_GLOB}")
    if MAX_INDEX_FILES is not None:
        index_files = index_files[:MAX_INDEX_FILES]
        print(f"Limiting to first {len(index_files)} index files.")




    for k, idx_path in enumerate(index_files, start=1):
        # Simple MPI sharding: only process files where (k mod size) == rank
        if k% COMM.Get_size() != COMM.Get_rank():
            continue
        print(f"\n[{k}/{len(index_files)}] Rank: {COMM.Get_rank()} Processing index: {idx_path}")


        # 2.1) read index cloud and compute weighted means in (Ωm, S8)
        idx_arr = np.loadtxt(idx_path)
        Om_i    = idx_arr[:, OMEGA_M_COL]
        sig8_i  = idx_arr[:, SIGMA8_COL]
        S8_i    = compute_S8(Om_i, sig8_i)

        # weights from the index file (log_weight column)
        weights = np.exp(idx_arr[:, -3])   # columns: [6 params | log_weight | prior | post]


        # 2.2) cut schmear+temp chain to a simple rectangle centered at means
        # Centers in BOTH spaces:
        Om_center    = np.average(Om_i,   weights=weights)
        S8_center    = np.average(S8_i,   weights=weights)   # used for S8-space selection
        sig8_center  = np.average(sig8_i, weights=weights)   # used for σ8-space plotting edges

        band_mask = (np.abs(Om_s - Om_center) <= BOX_HALF_OM) & \
                    (np.abs(S8_s - S8_center) <= BOX_HALF_S8)

        cut = schmear[band_mask]
        if cut.shape[0] == 0:
            print(f"  -> Box kept 0 points for {os.path.basename(idx_path)}. Increase BOX_HALF_OM/S8.")
            continue
        if FIRST_N_EVAL is not None:
            cut = cut[:FIRST_N_EVAL]


        # proposal (old) log posterior and log weight for these exact rows
        logpost_old = logpost_old_full[band_mask]
        logw_old    = logw_old_full[band_mask]
        if FIRST_N_EVAL is not None:
            logpost_old = logpost_old[:len(cut)]
            logw_old    = logw_old[:len(cut)]



        # 2.3) Build per-index pipeline with explicit n(z) override
        from cosmosis.runtime.config import Inifile
        nz_idx = parse_nz_index_from_filename(idx_path)
        # Build a new ini, set the index explicitly, and pass the ini object
        ini = Inifile(PARAMS_INI)
        ini.set("load_nz", "index", str(nz_idx))
        index_pipeline = cosmosis.LikelihoodPipeline(ini)

        # Assert we really changed it (fails fast if not picked up)
        try:
            chosen = index_pipeline.options["load_nz"].getint("index")
        except Exception:
            # Fallback way to read it if the above accessor differs in your version
            chosen = ini.getint("load_nz", "index")
        if chosen != nz_idx:
            raise RuntimeError(f"n(z) override failed: wanted {nz_idx}, got {chosen}")
        else:
            print(f"    -> using n(z) index = {chosen}")


        # 2.4) Evaluate NEW log-posterior for each cut schmear row
        theta_mat = cut[:, PARAM_COLS]
        logpost_new = np.empty(theta_mat.shape[0], dtype=float)
        for i, theta in enumerate(theta_mat):
            lp, _ = index_pipeline.posterior(theta)
            logpost_new[i] = float(lp)


        # 2.5) Importance sampling (log-space)
        # log_w_new = log_w_old + (log_post_new - log_post_old)
        logw_new = logw_old + (logpost_new - logpost_old)

        # prior column: if you don't have one, zero is fine for postprocessing
        prior_col = np.zeros_like(logpost_new)


        # 2.6) Primary output: EXACT same 9 columns as the originals
        # Replace 'log_weight' with the reweighted one
        main_table = np.column_stack([cut[:, PARAM_COLS], logw_new, prior_col, logpost_new])

        # Diagnostics sidecar: keep both old/new weights and old/new posteriors
        diag_table = np.column_stack([
            cut[:, PARAM_COLS],
            logw_old,          # log_weight_old
            logw_new,          # log_weight_new
            logpost_old,       # post_old (proposal)
            logpost_new        # post_new (target)
        ])


        # 2.7) Save with a cosmosis-friendly header + meta lines
        # First line MUST be column names only:
        header_main = "# " + " ".join(PARAM_NAMES + ["log_weight", "prior", "post"])

        # Actual number of samples used (after any truncation)
        n_used = cut.shape[0]

        meta_lines = [
            "## reweighted schmear subset evaluated through index pipeline (weighted-mean box)",
            f"## index_file = {idx_path}",
            f"## nz_index_override = {nz_idx}",
            f"## box_center_Om = {Om_center:.6f}, half_width = {BOX_HALF_OM:.3f}",
            f"## box_center_S8 = {S8_center:.6f}, half_width = {BOX_HALF_S8:.3f}",
            f"## box_center_sigma8 = {sig8_center:.6f}, half_width = {BOX_HALF_SIG8:.3f}",
            f"## number_of_samples_used = {n_used}",
        ]
        header_main = header_main + "\n" + "\n".join(meta_lines)

        header_diag = "# " + " ".join(PARAM_NAMES + ["log_weight_old", "log_weight_new", "post_old", "post_new"])

        base = os.path.splitext(os.path.basename(idx_path))[0]
        out_txt_main = os.path.join(OUT_DIR, f"reweight_{base}.txt")
        out_txt_diag = os.path.join(OUT_DIR, f"reweight_{base}__diagnostics.txt")

        np.savetxt(out_txt_main, main_table, fmt="%.8f", header=header_main, comments="")
        np.savetxt(out_txt_diag, diag_table, fmt="%.8f", header=header_diag, comments="")
        print(f"  -> saved {main_table.shape[0]} rows to: {out_txt_main}")
        print(f"  -> saved diagnostics to: {out_txt_diag}")


    print("\nDone! Now you can cosmosis-postprocess the outputs and compare to the original index chains!\n")

if __name__ == "__main__":
    main()



