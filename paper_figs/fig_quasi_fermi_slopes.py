"""Fig. quasi-Fermi-slopes: quasi-Fermi levels v_n, v_p and their slopes along the device, DEVSIM against the derivative-control network (eta column + SRH divisor, seed 0),
for three unseen anchors of increasing L/lambda_D (about 10, 100, 1000) at V = 0.52 V. Network slope by autograd of the network output (exported), DEVSIM slope by finite differences.
Reads paper_data/figure_data/quasi_fermi_slopes_V0.52.npz (written once by export/export_slopes_data.py). Output: fig_quasi_fermi_slopes_by_stiffness.{png,pdf}"""
from common import *  # noqa: F401,F403
import matplotlib.pyplot as plt

plt.style.use(STYLE)
Z = np.load(f"{DATA}/figure_data/quasi_fermi_slopes_V0.52.npz")
fig, axes = plt.subplots(3, 3, figsize=(12, 9), layout="constrained")
for k in range(3):
    x = Z[f"{k}_x_cm"]
    xn = (x - x.min()) / (x.max() - x.min())
    vn_t, vn_p, vp_t, vp_p = (Z[f"{k}_{q}"] for q in ("vn_true", "vn_pred", "vp_true", "vp_pred"))
    ax = axes[0, k]  # the quasi-Fermi levels themselves (whole device)
    for t, p, c in ((vn_t, vn_p, "#1f4e8c"), (vp_t, vp_p, "#2f855a")):
        ax.plot(xn, t, color="k", lw=1.6)
        ax.plot(xn, p, color=c, lw=1.4, ls="--")
    ax.text(0.04, 0.5, f"$L/\\lambda_D$ = {float(Z[f'{k}_L_over_lambda_D']):.0f}\n$J^{{\\rm grad}}_{{\\rm pred}}/J_{{\\rm truth}}$ = {float(Z[f'{k}_I_ratio']):.2f}", transform=ax.transAxes, fontsize=TEXT_FS, va="center", ha="left",
            bbox=dict(facecolor="w", edgecolor="none", alpha=.85, pad=3))
    for row, (name, t, p) in enumerate((("$v_n$", vn_t, Z[f"{k}_dvn_net"]), ("$v_p$", vp_t, Z[f"{k}_dvp_net"])), start=1):
        ax = axes[row, k]
        ax.plot(xn, np.gradient(t, x), color="k", lw=1.8, label="DEVSIM")
        ax.plot(xn, p, color="#d9531e", lw=1.5, ls="--", label="network")
        ax.set_yscale("symlog", linthresh=100.0)
        if k == 0:
            ax.set_ylabel(f"d{name}/dx (V/cm)")
        if row == 2:
            ax.set_xlabel("position $x/L$")
axes[0, 0].set_ylabel("$v_n$, $v_p$ (V)")
axes[0, 0].plot([], [], "k-", label="DEVSIM"); axes[0, 0].plot([], [], "--", color="#1f4e8c", label="network $v_n$"); axes[0, 0].plot([], [], "--", color="#2f855a", label="network $v_p$")
axes[0, 0].legend(fontsize=8)
axes[1, 0].legend(fontsize=8)
fig.savefig(f"{OUT}/fig_quasi_fermi_slopes_by_stiffness.png", dpi=140, bbox_inches="tight")
