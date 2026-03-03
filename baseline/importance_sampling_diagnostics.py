"""
Diagnostic script for S8 reweighting / importance sampling

This assumes I've have already run the importance–sampling code to get:
    reweight_n_z_real_index_k.txt
    reweight_n_z_real_index_k__diagnostics.txt

    
Visual part:
    - Compare COMBINED ORIGINAL index chains (green)
      vs COMBINED IMPORTANCE–REWEIGHTED chains (pink)
      in (Omega_m, sigma_8) and (logT_agn, sigma_8).

Numerical part:
    - Per–index importance–weight stats (ESS, var(w), etc.)
    - Global importance–weight stats (all indices combined)
    - ESS of baseline chain and combined reweighted chain
"""

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =========================
# ======  CONFIG  =========
# =========================

# ---- Files / paths ----
BASELINE_FILE       = "/Users/holleyconte/Desktop/Senior-Honours-Project/baseline-cuillin.txt"  
ORIGINAL_INDEX_GLOB = "/Users/holleyconte/Desktop/Senior-Honours-Project/baseline/all_n_z_realizations/n_z_real_index_*.txt"
REWEIGHT_GLOB       = "importance_sampling_resultsCuillinALLNew_Wider/reweight_n_z_real_index_*.txt"
DIAG_GLOB           = "importance_sampling_resultsCuillinALLNew_Wider/reweight_n_z_real_index_*_diagnostics.txt"
OUT_DIR             = "importance_sampling_diagnosticsresultsCuillinALLNew_Wider"

# ---- Column indices (0-based) ----
OMEGA_M_COL     = 0
SIGMA8_COL      = 4
LOGT_COL        = 5
BASE_LOGW_COL   = 6   # baseline log_weight column
BASE_POST_COL   = 8   # baseline post column

# diagnostics file layout:
DIAG_LOGW_OLD_COL = 6
DIAG_LOGW_NEW_COL = 7
DIAG_POST_OLD_COL = 8
DIAG_POST_NEW_COL = 9

# ---- Plot ranges ----
OM_RANGE    = (0.22, 0.36)
SIG8_RANGE  = (0.70, 0.90)
LOGT_RANGE  = (7.4, 8.0)

NBINS_OM_SIG8   = 80
NBINS_LOGT_SIG8 = 80

# contour probability levels (68% and 95%)
CONTOUR_LEVELS = [0.68, 0.95]


# =========================
# ======  HELPERS  ========
# =========================

def load_txt(path):
    return np.loadtxt(path)

def weighted_2d_hist(x, y, w, x_range, y_range, nbins):
    """
    Return (H, xedges, yedges) for weighted 2D histogram
    """
    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=nbins,
        range=[x_range, y_range],
        weights=w
    )
    return H.T, xedges, yedges   # transpose so that H[y, x]

def get_contour_levels_from_pdf(pdf, levels):
    """
    Given a 2D pdf (array summing to 1), find density thresholds
    corresponding to given *cumulative* probability levels (e.g. 0.68, 0.95)
    """
    flat = pdf.ravel()
    idx = np.argsort(flat)[::-1]     # sort descending
    flat_sorted = flat[idx]
    cumsum = np.cumsum(flat_sorted)
    thr = []
    for p in levels:
        j = np.searchsorted(cumsum, p)
        thr.append(flat_sorted[j])
    return thr



def plot_contours(ax, H, xedges, yedges, levels, color, label):
    """
    Draw contour lines of H at thresholds corresponding to 'levels' (cumulative probs)
    and add a legend entry using a dummy Line2D.
    """
    pdf = H / np.sum(H)

    # Get density thresholds for the desired cumulative probabilities
    thr = get_contour_levels_from_pdf(pdf, levels)

    # Matplotlib requires strictly increasing, positive contour levels
    thr = np.sort(thr)          # ensure increasing
    thr = np.unique(thr)        # drop duplicates
    thr = thr[thr > 0]          # keep only positive levels

    if len(thr) == 0:
        print("Warning: no positive contour levels; skipping plot.")
        return
    if len(thr) == 1:
        # If we only got one level, just use that single level
        thr = [thr[0]]

    # Bin centres
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xc, yc)

    # Draw contours
    ax.contour(X, Y, pdf, levels=thr, colors=[color], linewidths=1.8)

    


def compute_importance_stats(weights):
    """
    Given importance weights w (ratio new/old posterior), compute:

      ESS = (sum w)^2 / sum(w^2)
      ESS_norm = ESS / N
      mean, variance, max, min

    All numerically stable if weights already rescaled.
    """
    w = np.asarray(weights, dtype=float)
    N = w.size
    if N == 0:
        return dict(N=0, ESS=np.nan, ESS_norm=np.nan, mean=np.nan,
                    var=np.nan, w_min=np.nan, w_max=np.nan)

    s1 = np.sum(w)
    s2 = np.sum(w**2)
    ESS = (s1**2) / s2
    ESS_norm = ESS / N
    return dict(
        N=N,
        ESS=ESS,
        ESS_norm=ESS_norm,
        mean=np.mean(w),
        var=np.var(w),
        w_min=np.min(w),
        w_max=np.max(w),
    )

def parse_index_from_diag_name(path):
    """
    Extract integer index from a diagnostics filename like
    'reweight_n_z_real_index_37_diagnostics.txt' -> 37
    """
    name = os.path.basename(path)
    try:
        core = name.split("index_")[1]
        idx_str = core.split("_diagnostics")[0]
        return int(idx_str)
    except Exception:
        return -1




def compute_S8(omega_m, sigma8):
    """S8 = sigma8 * sqrt(omega_m/0.3)"""
    omega_m = np.asarray(omega_m, float)
    sigma8  = np.asarray(sigma8, float)
    return sigma8 * np.sqrt(omega_m / 0.3)


def weighted_mean_and_var(x, w):
    """
    Return weighted mean and variance of x with weights w.
    (variance is E[(x-mean)^2] using the weighted expectation)
    """
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    w_sum = np.sum(w)
    if w_sum <= 0:
        return np.nan, np.nan
    mean = np.sum(w * x) / w_sum
    var  = np.sum(w * (x - mean)**2) / w_sum
    return mean, var


# =========================
# ======  MAIN  ===========
# =========================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- 1. Load baseline chain (for ESS only) ----------
    baseline_exists = os.path.exists(BASELINE_FILE)
    if baseline_exists:
        base = load_txt(BASELINE_FILE)
        base_om   = base[:, OMEGA_M_COL]
        base_sig8 = base[:, SIGMA8_COL]
        base_logt = base[:, LOGT_COL]
        base_logw = base[:, BASE_LOGW_COL]
        base_w_for_ess = np.exp(base_logw - np.max(base_logw))  # for ESS only
    else:
        print(f"WARNING: baseline file not found: {BASELINE_FILE}")
        base_w_for_ess = None

    # ---------- 2. Load all ORIGINAL index chains  ----------
    orig_files = sorted(glob.glob(ORIGINAL_INDEX_GLOB))
    if not orig_files:
        raise SystemExit(f"No original index chain files found for glob: {ORIGINAL_INDEX_GLOB}")

    all_orig_om   = []
    all_orig_sig8 = []
    all_orig_logt = []
    all_orig_logw = []

    for path in orig_files:
        arr = load_txt(path)
        all_orig_om.append(arr[:, OMEGA_M_COL])
        all_orig_sig8.append(arr[:, SIGMA8_COL])
        all_orig_logt.append(arr[:, LOGT_COL])
        # index files: [6]=log_weight, [7]=prior, [8]=post
        all_orig_logw.append(arr[:, 6])

    orig_om   = np.concatenate(all_orig_om)
    orig_sig8 = np.concatenate(all_orig_sig8)
    orig_logt = np.concatenate(all_orig_logt)
    orig_logw = np.concatenate(all_orig_logw)

    orig_w_for_contours = np.exp(orig_logw - orig_logw.max())


    # ---------- 3. Load all reweighted chains ----------
    reweight_files = sorted(glob.glob(REWEIGHT_GLOB))
    if not reweight_files:
        raise SystemExit(f"No reweighted chain files found for glob: {REWEIGHT_GLOB}")

    all_rw_om    = []
    all_rw_sig8  = []
    all_rw_logt  = []
    all_rw_logw  = []   # store log_weight_new 

    for path in reweight_files:
        arr = load_txt(path)
        all_rw_om.append(arr[:, OMEGA_M_COL])
        all_rw_sig8.append(arr[:, SIGMA8_COL])
        all_rw_logt.append(arr[:, LOGT_COL])
        all_rw_logw.append(arr[:, 6])   # col 6 = reweighted log_weight_new

    rw_om   = np.concatenate(all_rw_om)
    rw_sig8 = np.concatenate(all_rw_sig8)
    rw_logt = np.concatenate(all_rw_logt)
    rw_logw = np.concatenate(all_rw_logw)

    rw_w_for_contours = np.exp(rw_logw - rw_logw.max())

    print(f"Loaded {len(orig_files)} original index chains with total {orig_om.size} samples.")
    print(f"Loaded {len(reweight_files)} reweighted chains with total {rw_om.size} samples.")


    # ---------- 4. S8 accuracy diagnostic (combined original vs reweighted) ----------
    S8_orig = compute_S8(orig_om, orig_sig8)
    S8_rw   = compute_S8(rw_om,  rw_sig8)

    # use the same (rescaled) weights we already use for contours
    mean_S8_orig, var_S8_orig = weighted_mean_and_var(S8_orig, orig_w_for_contours)
    mean_S8_rw,   var_S8_rw   = weighted_mean_and_var(S8_rw,   rw_w_for_contours)

    std_S8_orig = np.sqrt(var_S8_orig)
    std_S8_rw   = np.sqrt(var_S8_rw)

    print("\nWeighted S8 statistics for combined index chains:")
    print(f"  Original chains:   <S8> = {mean_S8_orig:.4f},  std(S8) = {std_S8_orig:.4f}")
    print(f"  Reweighted chains: <S8> = {mean_S8_rw:.4f},  std(S8) = {std_S8_rw:.4f}")

    d_mean  = mean_S8_rw - mean_S8_orig
    d_sigma = std_S8_rw  - std_S8_orig

    print(f"  Δ<S8> (reweighted - original) = {d_mean:.4e}")
    print(f"  Δ std(S8)                     = {d_sigma:.4e}")

    if np.isfinite(mean_S8_orig) and mean_S8_orig != 0:
        print(f"  Relative shift in <S8>        = {d_mean/mean_S8_orig:.3e}")
    if np.isfinite(std_S8_orig) and std_S8_orig != 0:
        print(f"  Relative shift in std(S8)     = {d_sigma/std_S8_orig:.3e}")







    # ---------- 4. Visual diagnostic: Ωm vs σ8 ----------
    H_orig, xedges_o, yedges_o = weighted_2d_hist(
        orig_sig8, orig_om, orig_w_for_contours,
        SIG8_RANGE, OM_RANGE, NBINS_OM_SIG8
    )
    H_rw, xedges_rw, yedges_rw = weighted_2d_hist(
        rw_sig8, rw_om, rw_w_for_contours,
        SIG8_RANGE, OM_RANGE, NBINS_OM_SIG8
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_contours(ax, H_orig, xedges_o, yedges_o,
                  CONTOUR_LEVELS, color="blue", label="combined original indices")
    plot_contours(ax, H_rw, xedges_rw, yedges_rw,
                  CONTOUR_LEVELS, color="deeppink", label="combined reweighted indices")

    ax.set_xlim(SIG8_RANGE)
    ax.set_ylim(OM_RANGE)
    ax.set_xlabel(r"$\sigma_8$", fontsize=16)
    ax.set_ylabel(r"$\Omega_m$", fontsize=16)
    ax.set_title(r"Original vs importance–reweighted: $\Omega_m$–$\sigma_8$", fontsize=16)
    ax.legend(
        handles=[
            Line2D([], [], color="blue", linewidth=1.8, label="combined original indices"),
            Line2D([], [], color="deeppink", linewidth=1.8, label="combined reweighted indices"),
        ],
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "original_vs_reweighted_omegam_sigma8.png"))
    plt.close(fig)





    # ---------- 5. Visual diagnostic: logT_AGN vs σ8 ----------
    H_orig2, xedges_o2, yedges_o2 = weighted_2d_hist(
        orig_logt, orig_sig8, orig_w_for_contours,
        LOGT_RANGE, SIG8_RANGE, NBINS_LOGT_SIG8
    )
    H_rw2, xedges_rw2, yedges_rw2 = weighted_2d_hist(
        rw_logt, rw_sig8, rw_w_for_contours,
        LOGT_RANGE, SIG8_RANGE, NBINS_LOGT_SIG8
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_contours(ax, H_orig2, xedges_o2, yedges_o2,
                  CONTOUR_LEVELS, color="blue", label="combined original indices")
    plot_contours(ax, H_rw2, xedges_rw2, yedges_rw2,
                  CONTOUR_LEVELS, color="deeppink", label="combined reweighted indices")

    ax.set_xlim(LOGT_RANGE)
    ax.set_ylim(SIG8_RANGE)
    ax.set_xlabel(r"$\log_{10}(T_{\mathrm{AGN}})$", fontsize=16)
    ax.set_ylabel(r"$\sigma_8$", fontsize=16)
    ax.set_title(r"Original vs importance–reweighted: $\log T_{\rm AGN}$–$\sigma_8$", fontsize=16)
    ax.legend(
        handles=[
            Line2D([], [], color="blue", linewidth=1.8, label="combined original indices"),
            Line2D([], [], color="deeppink", linewidth=1.8, label="combined reweighted indices"),
        ],
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "original_vs_reweighted_logTagn_sigma8.png"))
    plt.close(fig)





    # ---------- 6. Numerical diagnostics: importance weights ----------
    diag_files = sorted(glob.glob(DIAG_GLOB))
    if not diag_files:
        print(f"\nWARNING: no diagnostics files found for glob {DIAG_GLOB}")
    else:
        print(f"\nFound {len(diag_files)} diagnostics files. Computing importance-weight stats...\n")

    # We'll store per–index stats + all raw log-ratios for a *global* diagnostic
    per_index_summaries = []
    all_log_ratios = []   # post_new - post_old for every sample across all indices

    for diag_path in diag_files:
        arr = load_txt(diag_path)
        post_old = arr[:, DIAG_POST_OLD_COL]
        post_new = arr[:, DIAG_POST_NEW_COL]

        # raw log importance ratios for this index
        log_ratio = post_new - post_old
        all_log_ratios.append(log_ratio)

        # per–index stats with rescaling
        log_ratio_shift = log_ratio - np.max(log_ratio)
        w_imp = np.exp(log_ratio_shift)
        stats = compute_importance_stats(w_imp)
        idx = parse_index_from_diag_name(diag_path)
        per_index_summaries.append((idx, stats))

    # Sort summaries by index number for a pretty table
    per_index_summaries.sort(key=lambda t: t[0])

    if per_index_summaries:
        print("{:<18s} {:>8s} {:>10s} {:>13s} {:>12s} {:>10s} {:>10s}".format(
            "reweighted index", "N", "ESS", "ESS/N", "Var(w)", "w_min", "w_max"
        ))
        print("-" * 90)
        for idx, stats in per_index_summaries:
            print("{:<18s} {:>8d} {:>10.1f} {:>13.3e} {:>12.3e} {:>10.2e} {:>10.2e}".format(
                f"index {idx}",
                stats["N"],
                stats["ESS"],
                stats["ESS_norm"],
                stats["var"],
                stats["w_min"],
                stats["w_max"],
            ))


    # ---------- 7. Global importance-weight diagnostics ----------
    if per_index_summaries:
        # Concatenate *raw* log-ratios and rescale ONCE globally.
        log_ratio_all = np.concatenate(all_log_ratios)
        log_ratio_all -= np.max(log_ratio_all)
        w_all = np.exp(log_ratio_all)

        global_imp_stats = compute_importance_stats(w_all)
        print("\nGlobal importance weights (all indices combined):")
        print(f"  N_total   = {global_imp_stats['N']}")
        print(f"  ESS       = {global_imp_stats['ESS']:.1f}")
        print(f"  ESS/N     = {global_imp_stats['ESS_norm']:.3e}")
        print(f"  mean(w)   = {global_imp_stats['mean']:.3e}")
        print(f"  var(w)    = {global_imp_stats['var']:.3e}")
        print(f"  w_min     = {global_imp_stats['w_min']:.3e}")
        print(f"  w_max     = {global_imp_stats['w_max']:.3e}")

        # One *global* histogram of importance weights
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.hist(w_all, bins=60, histtype="stepfilled", alpha=0.7)
        ax.set_xlabel(r"importance weight $w = e^{\Delta \ln p}$", fontsize=13)
        ax.set_ylabel("count", fontsize=13)
        ax.set_title("Importance weights across all reweighted indices", fontsize=14)
        ax.set_yscale("log")   
        # x-axis on *linear* scale so we actually see spread near w≈1
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "importance_weights_all_indices.png"))
        plt.close(fig)

    # ---------- 8. ESS for baseline + chains using log_weight ----------
    if baseline_exists:
        base_chain_stats = compute_importance_stats(base_w_for_ess)
        rw_chain_w = np.exp(rw_logw - np.max(rw_logw))
        rw_chain_stats = compute_importance_stats(rw_chain_w)

        print("\nEffective sample size of baseline vs combined reweighted chains:")
        print("  (using CosmoSIS log_weight columns)")
        print(f"  Baseline chain:   N = {base_chain_stats['N']}, "
              f"ESS = {base_chain_stats['ESS']:.1f}, "
              f"ESS/N = {base_chain_stats['ESS_norm']:.3f}")
        print(f"  Reweighted chain: N = {rw_chain_stats['N']}, "
              f"ESS = {rw_chain_stats['ESS']:.1f}, "
              f"ESS/N = {rw_chain_stats['ESS_norm']:.7f}")

    print("\nDone! Visual & numerical diagnostics saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
