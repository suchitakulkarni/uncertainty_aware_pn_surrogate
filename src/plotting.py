import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRAIN_COLOR = "#2a78d6"
HELDOUT_COLOR = "#eb6834"


def _device_param_str(source):
    """One-line device-parameter caption, so a saved PNG is self-describing
    without cross-referencing the run's stdout/out_dir name.

    Args:
        source: FiniteVoltageDataSource.
    Returns:
        A single-line string of the device's key parameters.
    """
    return (f"L={source.L:.3e} cm | nD={source.nD:.3e} cm^-3 | V={source.v_target:.3g} V | "
            f"mu_n={source.mu_n:.4g} cm^2/Vs | mu_p={source.mu_p:.4g} cm^2/Vs | "
            f"taun={source.taun:.3g} s | taup={source.taup:.3g} s | T={source.T:.4g} K")


def plot_loss_curves(history, source=None):
    """Plots training loss curves.

    Args:
        history: dict returned by src/train_qfl.py:train_qfl (total_loss/bc_loss/data_loss plus poisson_loss/
            electron_continuity_loss/hole_continuity_loss).
        source: optional FiniteVoltageDataSource; if given, stamps the
            device parameters at the bottom of the figure.
    Returns:
        The matplotlib Figure.
    """
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    ax.plot(history["poisson_loss"], label = "Poisson (pointwise MSE, diagnostic)")
    ax.plot(history["electron_continuity_loss"], label = "Electron continuity")
    ax.plot(history["hole_continuity_loss"], label = "Hole continuity")
    ax.plot(history["bc_loss"], label = "BC")
    if any(v > 0 for v in history.get("data_loss", [])):
        ax.plot(history["data_loss"], label = "Data")
    ax.plot(history["total_loss"], label = "Total", lw=2, color="k")
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 1e5)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    if source is not None:
        fig.text(0.5, 0.01, _device_param_str(source), ha="center", va="bottom", fontsize=7)
        plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig

def plot_eval_summary(x, psi_true, psi_pred, n_true, n_pred, p_true, p_pred,
                             J_total, I_devsim, E_pred, E_true, source=None):
    """2x2 diagnostic panel used by src/evaluate.py: state-space
    reconstruction (potential, carriers) plus engineering quantities of
    interest (current conservation, E-field) against DEVSIM ground truth.

    Args:
        x: physical coordinates (cm); converted to um for display.
        psi_true, psi_pred: potential (mV), DEVSIM and predicted.
        n_true, n_pred, p_true, p_pred: carrier densities (cm^-3).
        J_total: predicted Jn+Jp (A/cm^2).
        I_devsim: DEVSIM's terminal current (A/cm^2).
        E_pred, E_true: electric field (V/cm), predicted and DEVSIM.
        source: optional FiniteVoltageDataSource; if given, stamps the
            device parameters at the bottom of the figure, so the saved
            PNG is self-describing without cross-referencing the run's
            out_dir name.
    Returns:
        The matplotlib Figure.
    """
    x = x * 1e4  # cm -> um
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(x, psi_true, label="DEVSIM", lw=2)
    axes[0, 0].plot(x, psi_pred, label="PINN", ls="--")
    axes[0, 0].set_xlabel("x (um)"); axes[0, 0].set_ylabel("Potential (mV)"); axes[0, 0].legend()
    axes[0, 0].set_title("State-space reconstruction: potential")

    # Residual inset: aggregate L1 can hide error that's spatially
    # concentrated at high-curvature regions (the junction bend) - a
    # flat-looking main panel can still have a residual spike the eye
    # won't catch at that scale.
    inset_psi = axes[0, 0].inset_axes([0.55, 0.08, 0.4, 0.32])
    inset_psi.plot(x, psi_pred - psi_true, color="tab:red", lw=1)
    inset_psi.axhline(0, color="k", lw=0.5)
    inset_psi.set_title("residual (mV)", fontsize=8)
    inset_psi.tick_params(labelsize=6)

    axes[0, 1].semilogy(x, n_true, label="n DEVSIM", lw=2)
    axes[0, 1].semilogy(x, n_pred, label="n PINN", ls="--")
    axes[0, 1].semilogy(x, p_true, label="p DEVSIM", lw=2)
    axes[0, 1].semilogy(x, p_pred, label="p PINN", ls="--")
    axes[0, 1].set_xlabel("x (um)"); axes[0, 1].set_ylabel("Density (cm^-3)"); axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_title("State-space reconstruction: carriers")

    inset_np = axes[0, 1].inset_axes([0.55, 0.08, 0.4, 0.32])
    inset_np.plot(x, 100 * (n_pred - n_true) / n_true, color="tab:orange", lw=1, label="n")
    inset_np.plot(x, 100 * (p_pred - p_true) / p_true, color="tab:red", lw=1, label="p")
    inset_np.axhline(0, color="k", lw=0.5)
    inset_np.set_title("rel. residual (%)", fontsize=8)
    inset_np.legend(fontsize=6)
    inset_np.tick_params(labelsize=6)

    axes[1, 0].plot(x, J_total * 1e3)
    axes[1, 0].axhline(I_devsim * 1e3, color="k", ls=":", label="DEVSIM terminal I")
    axes[1, 0].set_xlabel("x (um)"); axes[1, 0].set_ylabel("Jn+Jp (mA/cm^2)"); axes[1, 0].legend()
    axes[1, 0].set_title("EQI: current conservation (should be flat)")

    axes[1, 1].semilogy(x, abs(E_pred), label="PINN |E| (from network's phi_hat)")
    axes[1, 1].semilogy(x, abs(E_true), label="Our ground truth |E| (d/dx of DEVSIM's phi)", ls="--")
    axes[1, 1].set_xlabel("x (um)"); axes[1, 1].set_ylabel("|E| (V/cm)"); axes[1, 1].legend()
    axes[1, 1].set_title("EQI: electric field")

    if source is not None:
        fig.text(0.5, 0.01, _device_param_str(source), ha="center", va="bottom", fontsize=8)
        plt.tight_layout(rect=[0, 0.03, 1, 1])
    else:
        plt.tight_layout()
    return fig


def region_stats(name: str, J_net: np.ndarray, J_true: np.ndarray) -> dict:
    """Current-conservation stats for one spatial region.

    Args:
        name: region label (e.g. "bulk", "junction", "total").
        J_net, J_true: same-shape arrays of network and DEVSIM total
            current density (A/cm^2) at the points in this region.
    Returns:
        dict of region/n_points/I_network_mean/I_true_mean/
        I_rel_error_mean/J_std_over_mean_true/L1_abs_error.
    """
    return {
        "region": name,
        "n_points": int(J_net.size),
        "I_network_mean": float(np.mean(J_net)),
        "I_true_mean": float(np.mean(J_true)),
        "I_rel_error_mean": float(abs(np.mean(J_net) - np.mean(J_true)) / (abs(np.mean(J_true)) + 1e-30)),
        "J_std_over_mean_true": float(np.std(J_net) / (abs(np.mean(J_true)) + 1e-30)),
        "L1_abs_error": float(np.sum(np.abs(J_net - J_true))),
    }


def bulk_vs_junction(arrays: dict, tag: str, out_dir: str) -> dict:
    """Bulk-vs-junction current-conservation decomposition, pipeline-
    agnostic (works on the recovered J_total for any parametrization, since it only
    looks at the already-recovered J_total): never call a current-
    conservation verdict from an aggregate metric alone - a good aggregate
    can hide a bad junction-region fit. Saves a plot to
    out_dir/bulk_vs_junction_{tag}.png.

    Args:
        arrays: evaluate(..., return_arrays=True)["arrays"] - needs
            x_dense, J_total, J_true_dense, x_dep_lo_true, x_dep_hi_true.
        tag: label for the plot title/filename/printed summary.
        out_dir: directory to save the plot to.
    Returns:
        dict with "bulk"/"junction"/"total" region_stats() sub-dicts.
    """
    x, J_net, J_true = arrays["x_dense"], arrays["J_total"], arrays["J_true_dense"]
    x_lo, x_hi = arrays["x_dep_lo_true"], arrays["x_dep_hi_true"]
    junction_mask = (x >= x_lo) & (x <= x_hi)
    bulk_mask = ~junction_mask
    stats = {
        "bulk": region_stats("bulk", J_net[bulk_mask], J_true[bulk_mask]),
        "junction": region_stats("junction", J_net[junction_mask], J_true[junction_mask]),
        "total": region_stats("total", J_net, J_true),
    }
    print(f"  [{tag}] bulk I_rel_error={stats['bulk']['I_rel_error_mean']:.4f}  "
          f"junction I_rel_error={stats['junction']['I_rel_error_mean']:.4f}  "
          f"total I_rel_error={stats['total']['I_rel_error_mean']:.4f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, J_net, label=f"J_total ({tag})", lw=1)
    ax.plot(x, J_true, label="J_total (DEVSIM)", lw=1, ls="--")
    ax.axvspan(x_lo, x_hi, color="orange", alpha=0.15, label="junction band")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("J_total [A/cm^2]")
    ax.set_title(f"Bulk-vs-junction decomposition ({tag})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/bulk_vs_junction_{tag}.png", dpi=130)
    plt.close(fig)
    return stats


# (column, display title, log-scale) - the error-quantifying subset of
# evaluate.py's EQI_KEYS (excludes raw network/devsim value pairs and
# current itself - current near a zero-denominator anchor needs a floored
# metric, a generic box plot doesn't apply one).
EQI_METRICS = [
    ("recon.psi_L1_rel", "psi_L1_rel", True),
    ("recon.n_L1_rel", "n_L1_rel", True),
    ("recon.p_L1_rel", "p_L1_rel", True),
    ("eqi.depletion_width_rel_error", "depletion_width_rel_error", True),
    ("eqi.dEF_rmse_mV", "dEF_rmse_mV (mV)", True),
    ("eqi.Q_n_stored_rel_error", "Q_n_stored_rel_error", True),
    ("eqi.Q_p_stored_rel_error", "Q_p_stored_rel_error", True),
    ("eqi.E_peak_rel_error", "E_peak_rel_error", True),
    ("eqi.R_L1_rel", "R_L1_rel", True),
    ("eqi.gauss_law_rel_residual", "gauss_law_rel_residual", True),
]


def plot_eqi_suite_boxplots(df: pd.DataFrame, label: str, out_path: str) -> None:
    """Full EQI suite (10 metrics), train vs held-out, as box plots rather
    than a table of numbers. Box plots, not violins: EQI metrics are
    heavily right-skewed across several orders of magnitude with small-to-
    moderate sample sizes - quartiles/whiskers stay legible on a log axis
    at this scale in a way a KDE-based violin wouldn't. Scales to any
    anchor-pool size for free (a box plot compresses arbitrary N into a
    5-number summary), unlike overlaying one line/point per anchor.

    Args:
        df: flatten_eval_result() rows, one per anchor, with a "held_out"
            column and the columns EQI_METRICS names (rename
            "reconstruction."->"recon." first if built from
            flatten_eval_result's own default section naming - see
            evaluate.py:flatten_eval_result's own docstring).
        label: title label (n_train/n_heldout counts are added
            automatically).
        out_path: PNG output path.
    """
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    for ax, (col, title, logscale) in zip(axes.flat, EQI_METRICS):
        train = df.loc[~df["held_out"], col].dropna()
        held = df.loc[df["held_out"], col].dropna()
        bp = ax.boxplot([train, held], tick_labels=["train", "held-out"], patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], [TRAIN_COLOR, HELDOUT_COLOR]):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for median in bp["medians"]:
            median.set_color("black")
        if logscale:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    n_train, n_held = (~df["held_out"]).sum(), df["held_out"].sum()
    fig.suptitle(f"EQI suite: {label} (n_train={n_train}, n_heldout={n_held})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved {out_path}")


