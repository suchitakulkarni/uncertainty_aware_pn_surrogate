"""Plots of the DEVSIM reference solutions of the planar 2D diode (data/planar2d/planar2d_V*.npz, written by
devsim_data_generator/generate_planar_diode_2d.py): state maps at three biases, line cuts through the junction,
the current flow, the I-V curve and a consistency check of the reconstructed current against the anode current.

    PYTHONPATH=. python -m scripts.plot_planar2d_devsim --data-dir data/planar2d --out-dir <dropbox>/devsim_planar2d
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import CubicTriInterpolator, LinearTriInterpolator, Triangulation

UM = 1e-4
Q_E = 1.602176634e-19
VOLTAGES_MAPS = (0.16, 0.48, 0.64)
CUT_VOLTAGES = (0.04, 0.16, 0.32, 0.48, 0.64)


def load(data_dir, v):
    d = dict(np.load(os.path.join(data_dir, f"planar2d_V{v:.3f}.npz")))
    d["xu"], d["yu"] = d["x"] / UM, d["y"] / UM
    d["tri"] = Triangulation(d["xu"], d["yu"])
    return d


def field(d):
    """|E| in V/cm from the nodal potential (cubic interpolation on the node triangulation)."""
    ex, ey = CubicTriInterpolator(d["tri"], d["potential"]).gradient(d["xu"], d["yu"])
    return ex * 1e4, ey * 1e4  # d/d(um) -> d/d(cm); E = -grad(psi) handled by the sign below


def currents(d):
    """Electron and hole current-density vectors (A/cm^2) at the nodes from the quasi-Fermi potentials,
    J_n = -q mu_n n grad(phi_n), J_p = -q mu_p p grad(phi_p), phi_n = psi - Ut ln(n/ni), phi_p = psi + Ut ln(p/ni)."""
    Ut = d["k_const"] * d["T_const"] / d["q_const"]
    ni = float(d["n_i"])
    tri = d["tri"]
    phin = d["potential"] - Ut * np.log(d["electrons"] / ni)
    phip = d["potential"] + Ut * np.log(d["holes"] / ni)
    out = {}
    for name, phi, dens, mu in (("n", phin, d["electrons"], float(d["mu_n"])), ("p", phip, d["holes"], float(d["mu_p"]))):
        gx, gy = CubicTriInterpolator(tri, phi).gradient(d["xu"], d["yu"])
        out[name] = (-Q_E * mu * dens * gx * 1e4, -Q_E * mu * dens * gy * 1e4)
    return out


def _decorate(ax, d, title, cbar=None):
    ax.set_ylim(2, 0); ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
    ax.set_xlabel(r"$x$ ($\mu$m)"); ax.set_ylabel(r"depth $y$ ($\mu$m)")
    ax.plot([float(d["win_x0"]) / UM, float(d["win_x1"]) / UM], [0, 0], "k-", lw=4, clip_on=False)


def maps(data_dir, out_dir):
    rows = [("potential", r"$\psi$ (mV)", lambda d: d["potential"] * 1e3, None, None),
            ("electrons", r"$\log_{10}\,n$ (cm$^{-3}$)", lambda d: np.log10(d["electrons"]), None, None),
            ("holes", r"$\log_{10}\,p$ (cm$^{-3}$)", lambda d: np.log10(d["holes"]), None, None),
            ("field", r"$\log_{10}|E|$ (V/cm)", None, None, None)]
    fig, axes = plt.subplots(len(rows), len(VOLTAGES_MAPS), figsize=(11, 3.1 * len(rows)), layout="constrained")
    ds = {v: load(data_dir, v) for v in VOLTAGES_MAPS}
    for i, (_, label, fn, _, _) in enumerate(rows):
        vals = {}
        for v, d in ds.items():
            if fn is None:
                gx, gy = CubicTriInterpolator(d["tri"], d["potential"]).gradient(d["xu"], d["yu"])
                vals[v] = np.log10(np.hypot(gx, gy) * 1e4 + 1)
            else:
                vals[v] = fn(d)
        vmin, vmax = min(a.min() for a in vals.values()), max(a.max() for a in vals.values())
        for j, v in enumerate(VOLTAGES_MAPS):
            ax = axes[i, j]
            c = ax.tricontourf(ds[v]["tri"], vals[v], levels=np.linspace(vmin, vmax, 50), cmap="viridis")
            ax.tricontour(ds[v]["tri"], ds[v]["net_doping"], levels=[0], colors="w", linewidths=0.8)
            _decorate(ax, ds[v], f"V = {v:.2f} V" if i == 0 else "")
            if j == len(VOLTAGES_MAPS) - 1:
                plt.colorbar(c, ax=axes[i, :], shrink=0.85, pad=0.02, label=label)
    fig.savefig(os.path.join(out_dir, "devsim_planar2d_maps.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def cuts(data_dir, out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.8))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(CUT_VOLTAGES)))
    for v, col in zip(CUT_VOLTAGES, cols):
        d = load(data_dir, v)
        gx, gy = CubicTriInterpolator(d["tri"], d["potential"]).gradient(d["xu"], d["yu"])
        for r, (cut, label, fixed, along) in enumerate([("vertical cut at x = 1.5 um (implant centre)", "y", 1.5, "yu"),
                                                         ("horizontal cut at y = 0.25 um (through the side walls)", "x", 0.25, "xu")]):
            s = np.linspace(0.0, 2.0 if along == "yu" else 3.0, 600)
            px, py = (np.full_like(s, fixed), s) if along == "yu" else (s, np.full_like(s, fixed))
            ip = lambda arr: CubicTriInterpolator(d["tri"], arr)(px, py)
            ip_log = lambda arr: 10 ** LinearTriInterpolator(d["tri"], np.log10(arr))(px, py)  # densities: log space
            axes[r, 0].plot(s, ip(d["potential"]) * 1e3, color=col, label=f"{v:.2f} V")
            axes[r, 1].semilogy(s, ip_log(d["electrons"]), color=col)
            axes[r, 2].semilogy(s, ip_log(d["holes"]), color=col)
            E = np.hypot(ip(gx), ip(gy)) * 1e4
            axes[r, 3].semilogy(s, E + 1, color=col)
        for c_, t in enumerate([r"$\psi$ (mV)", r"$n$ (cm$^{-3}$)", r"$p$ (cm$^{-3}$)", r"$|E|$ (V/cm)"]):
            axes[r, c_].set_ylabel(t); axes[r, c_].set_xlabel(f"{label} ($\\mu$m)")
        axes[r, 1].set_title(cut, fontsize=10, loc="left", pad=8)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "devsim_planar2d_cuts.png"), dpi=140); plt.close(fig)


def band_flux(d):
    """Exact current flux through every band between consecutive node rows, from DEVSIM's own edge currents:
    flux of an edge = (Jn + Jp) * EdgeCouple (A/cm, per unit depth), signed by its direction n0 -> n1.
    Returns row coordinates y_k (cm) and, per band, the x-midpoints and signed fluxes of the edges crossing it."""
    yk = np.unique(np.round(d["y"], 12))
    ylo, yhi = np.minimum(d["e_y0"], d["e_y1"]), np.maximum(d["e_y0"], d["e_y1"])
    flux = (d["e_Jn"] + d["e_Jp"]) * d["e_couple"] * np.sign(d["e_y1"] - d["e_y0"])
    xm = 0.5 * (d["e_x0"] + d["e_x1"])
    bands = []
    for k in range(len(yk) - 1):
        m = np.isclose(ylo, yk[k], atol=1e-12) & np.isclose(yhi, yk[k + 1], atol=1e-12)
        bands.append((xm[m], flux[m]))
    return yk, bands


def current_flow(data_dir, out_dir, v=0.48):
    d = load(data_dir, v)
    yk, bands = band_flux(d)
    I = float(d["iv_trace"][-1, 3])
    sign = np.sign(sum(b[1].sum() for b in bands) * I)
    xs = np.linspace(0.0, float(d["width"]), 400)
    # stream function: cumulative current from the left wall, as a fraction of the terminal current
    phi = np.array([np.interp(xs, np.sort(xm), np.cumsum(f[np.argsort(xm)]) * sign / I, left=0.0) for xm, f in bands])
    ym = 0.5 * (yk[:-1] + yk[1:]) / UM
    ratio = np.array([sign * f.sum() / I for _, f in bands])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw=dict(width_ratios=[1.35, 1, 0.55]), layout="constrained")
    c = ax[0].contourf(xs / UM, ym, phi, levels=np.linspace(0, 1, 21), cmap="Blues", alpha=0.35)
    ax[0].contour(xs / UM, ym, phi, levels=np.linspace(0.05, 0.95, 19), colors="k", linewidths=0.7)
    ax[0].tricontour(d["tri"], d["net_doping"], levels=[0], colors="r", linewidths=1.2)
    _decorate(ax[0], d, f"current streamlines, V = {v:.2f} V (each line = 5% of the anode current); red: junction")
    # cumulative current fraction left of x at a few depths (smooth; the slope is the vertical current density)
    for depth, col in zip((0.8, 1.2, 1.8), ("C0", "C1", "C2")):
        k = int(np.argmin(abs(ym - depth)))
        ax[1].plot(xs / UM, phi[k], color=col, label=f"y = {ym[k]:.2f} $\\mu$m")
    ax[1].plot([0, float(d["width"]) / UM], [0, 1], "k:", lw=1, label="uniform over the width")
    ax[1].set_xlabel(r"$x$ ($\mu$m)"); ax[1].set_ylabel(r"current left of $x$ / $I_{\rm anode}$"); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3); ax[1].set_title("cumulative current across the device", fontsize=9)
    ax[2].plot(ratio, ym, "o-", ms=3); ax[2].set_ylim(2, 0); ax[2].axvline(1, color="k", ls=":")
    ax[2].set_xlabel(r"flux through band / $I_{\rm anode}$"); ax[2].set_ylabel(r"depth ($\mu$m)")
    ax[2].set_title("current conservation", fontsize=9); ax[2].set_xlim(0.99, 1.01)
    fig.savefig(os.path.join(out_dir, "devsim_planar2d_current.png"), dpi=140); plt.close(fig)
    return float(ratio.min()), float(ratio.max())


def iv(data_dir, out_dir):
    fs = sorted(glob.glob(os.path.join(data_dir, "planar2d_V*.npz")))
    arr = np.array([np.load(f)["iv_trace"][-1] for f in fs])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].semilogy(arr[:, 0], np.abs(arr[:, 3]), "o-", label="total")
    ax[0].semilogy(arr[:, 0], np.abs(arr[:, 1]), "s--", ms=3, label="electrons")
    ax[0].semilogy(arr[:, 0], np.abs(arr[:, 2]), "^--", ms=3, label="holes")
    ax[0].set_xlabel("V (V)"); ax[0].set_ylabel("anode current (A/cm)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ut = 0.025852
    slope = np.diff(np.log(np.abs(arr[:, 3]))) / np.diff(arr[:, 0]) * ut
    ax[1].plot(0.5 * (arr[1:, 0] + arr[:-1, 0]), 1 / slope, "o-")
    ax[1].axhline(1, color="k", ls=":"); ax[1].set_ylim(0.8, 2.2)
    ax[1].set_xlabel("V (V)"); ax[1].set_ylabel("ideality factor"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "devsim_planar2d_iv.png"), dpi=140); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/planar2d")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    maps(a.data_dir, a.out_dir); cuts(a.data_dir, a.out_dir); iv(a.data_dir, a.out_dir)
    lo, hi = current_flow(a.data_dir, a.out_dir)
    print(f"flux through every horizontal band / anode current: min {lo:.5f}, max {hi:.5f}")
