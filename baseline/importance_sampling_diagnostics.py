"""
Diagnostic script for S8 reweighting / importance sampling

This script assumes you have already run the importance–sampling code, which
produced, for each index k:

    reweight_n_z_real_index_k.txt
    reweight_n_z_real_index_k__diagnostics.txt

with the diagnostics file having columns:

    [0:5]  six model parameters
    [6]    log_weight_old
    [7]    log_weight_new
    [8]    post_old   (log posterior from schmear+temp proposal)
    [9]    post_new   (log posterior from per-index pipeline)

It also assumes you have a *baseline* chain for index = -1, with columns:

    [0:5]  six model parameters
    [6]    log_weight
    [7]    prior
    [8]    post

We produce:

  - baseline_vs_combined_omegam_sigma8.png
  - baseline_vs_combined_logTagn_sigma8.png
  - weights_hist_index_<k>.png    (importance-weight histograms)
  - a printed table of ESS statistics per index.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# =========================
# ======  CONFIG  =========
# =========================

# ---- Files / paths ----
BASELINE_FILE   = "/Users/holleyconte/Desktop/Senior-Honours-Project/baseline-cuillin.txt"   # baseline chain at average n(z)
REWEIGHT_GLOB   = "importance_sampling_resultsCuillinALLNew/reweight_n_z_real_index_*.txt"
DIAG_GLOB       = "importance_sampling_resultsCuillinALLNew/reweight_n_z_real_index_*_diagnostics.txt"
OUT_DIR         = "importance_sampling_diagnosticsResults"

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
LOGT_RANGE  = (7.0, 8.2)

NBINS_OM_SIG8   = 80
NBINS_LOGT_SIG8 = 80

# contour probability levels (e.g. 68% and 95%)
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
    """
    pdf = H / np.sum(H)
    thr = get_contour_levels_from_pdf(pdf, levels)

    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xc, yc)

    cs = ax.contour(X, Y, pdf, levels=thr, colors=[color], linewidths=1.8)
    cs.collections[0].set_label(label)  # only label the innermost for legend

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


# =========================
# ======  MAIN  ===========
# =========================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- 1. Load baseline chain ----------
    if not os.path.exists(BASELINE_FILE):
        raise SystemExit(f"Baseline file not found: {BASELINE_FILE}")

    base = load_txt(BASELINE_FILE)
    base_om   = base[:, OMEGA_M_COL]
    base_sig8 = base[:, SIGMA8_COL]
    base_logt = base[:, LOGT_COL]
    base_logw = base[:, BASE_LOGW_COL]

    # weight rescaling for numerical stability (doesn't change contours)
    base_w = np.exp(base_logw - np.max(base_logw))

    # ---------- 2. Load all reweighted chains ----------
    reweight_files = sorted(glob.glob(REWEIGHT_GLOB))
    if not reweight_files:
        raise SystemExit(f"No reweighted chain files found for glob: {REWEIGHT_GLOB}")

    all_rw_om   = []
    all_rw_sig8 = []
    all_rw_logt = []
    all_rw_w    = []

    for path in reweight_files:
        arr = load_txt(path)
        om   = arr[:, OMEGA_M_COL]
        sig8 = arr[:, SIGMA8_COL]
        logt = arr[:, LOGT_COL]
        logw = arr[:, 6]    # in reweight files: col6 = reweighted log_weight

        w = np.exp(logw - np.max(logw))

        all_rw_om.append(om)
        all_rw_sig8.append(sig8)
        all_rw_logt.append(logt)
        all_rw_w.append(w)

    # concatenate
    rw_om   = np.concatenate(all_rw_om)
    rw_sig8 = np.concatenate(all_rw_sig8)
    rw_logt = np.concatenate(all_rw_logt)
    rw_w    = np.concatenate(all_rw_w)

    print(f"Loaded baseline chain with {base_om.size} samples.")
    print(f"Loaded {len(reweight_files)} reweighted chains with total {rw_om.size} samples.")

    # ---------- 3. Visual diagnostic: Ωm vs σ8 ----------
    H_base, xedges_bs, yedges_bs = weighted_2d_hist(
        base_sig8, base_om, base_w,
        SIG8_RANGE, OM_RANGE, NBINS_OM_SIG8
    )
    H_rw, xedges_rw, yedges_rw = weighted_2d_hist(
        rw_sig8, rw_om, rw_w,
        SIG8_RANGE, OM_RANGE, NBINS_OM_SIG8
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_contours(ax, H_base, xedges_bs, yedges_bs,
                  CONTOUR_LEVELS, color="limegreen", label="baseline (index = -1)")
    plot_contours(ax, H_rw, xedges_rw, yedges_rw,
                  CONTOUR_LEVELS, color="deeppink", label="combined reweighted")

    ax.set_xlim(SIG8_RANGE)
    ax.set_ylim(OM_RANGE)
    ax.set_xlabel(r"$\sigma_8$", fontsize=16)
    ax.set_ylabel(r"$\Omega_m$", fontsize=16)
    ax.set_title(r"Baseline vs combined reweighted: $\Omega_m$–$\sigma_8$", fontsize=18)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "baseline_vs_combined_omegam_sigma8.png"))
    plt.close(fig)

    # ---------- 4. Visual diagnostic: logT_AGN vs σ8 ----------
    H_base2, xedges_bs2, yedges_bs2 = weighted_2d_hist(
        base_logt, base_sig8, base_w,
        LOGT_RANGE, SIG8_RANGE, NBINS_LOGT_SIG8
    )
    H_rw2, xedges_rw2, yedges_rw2 = weighted_2d_hist(
        rw_logt, rw_sig8, rw_w,
        LOGT_RANGE, SIG8_RANGE, NBINS_LOGT_SIG8
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_contours(ax, H_base2, xedges_bs2, yedges_bs2,
                  CONTOUR_LEVELS, color="limegreen", label="baseline (index = -1)")
    plot_contours(ax, H_rw2, xedges_rw2, yedges_rw2,
                  CONTOUR_LEVELS, color="deeppink", label="combined reweighted")

    ax.set_xlim(LOGT_RANGE)
    ax.set_ylim(SIG8_RANGE)
    ax.set_xlabel(r"$\log_{10}(T_{\mathrm{AGN}})$", fontsize=16)
    ax.set_ylabel(r"$\sigma_8$", fontsize=16)
    ax.set_title(r"Baseline vs combined reweighted: $\log T_{\rm AGN}$–$\sigma_8$", fontsize=18)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "baseline_vs_combined_logTagn_sigma8.png"))
    plt.close(fig)





    # ---------- 5. Numerical diagnostics: importance weights ----------
    diag_files = sorted(glob.glob(DIAG_GLOB))
    if not diag_files:
        print(f"\nWARNING: no diagnostics files found for glob {DIAG_GLOB}")
    else:
        print(f"\nFound {len(diag_files)} diagnostics files. Computing importance-weight stats...\n")
        print("{:<30s} {:>8s} {:>10s} {:>9s} {:>10s} {:>10s}".format(
            "index_file", "N", "ESS", "ESS/N", "w_min", "w_max"
        ))
        print("-"*90)

    for diag_path in diag_files:
        arr = load_txt(diag_path)
        post_old = arr[:, DIAG_POST_OLD_COL]
        post_new = arr[:, DIAG_POST_NEW_COL]

        # importance weights = exp(post_new - post_old), rescaled for stability
        log_ratio = post_new - post_old
        log_ratio -= np.max(log_ratio)
        w_imp = np.exp(log_ratio)

        stats = compute_importance_stats(w_imp)

        # print short summary line
        print("{:<30s} {:>8d} {:>10.1f} {:>9.3f} {:>10.2e} {:>10.2e}".format(
            os.path.basename(diag_path),
            stats["N"],
            stats["ESS"],
            stats["ESS_norm"],
            stats["w_min"],
            stats["w_max"],
        ))

        # Histogram of importance weights
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(w_imp, bins=50, histtype="stepfilled", alpha=0.7)
        ax.set_xlabel("importance weight  $w = e^{\\Delta \\ln \\, p}$", fontsize=12)
        ax.set_ylabel("count", fontsize=12)
        ax.set_title("Importance weights: " + os.path.basename(diag_path), fontsize=13)
        ax.set_yscale("log")  # log y helps show tails
        fig.tight_layout()

        base_name = os.path.basename(diag_path).replace("__diagnostics.txt", "")
        out_hist = os.path.join(OUT_DIR, f"weights_hist_{base_name}.png")
        fig.savefig(out_hist)
        plt.close(fig)

    print("\nDone! Visual & numerical diagnostics saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
