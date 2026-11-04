from paths import *  # noqa: F401  (also writes a PDF next to every PNG)
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from src.doping import Uniform, ErfBox, SumDoping

# proposed planar diode (micron units; y = depth below the top surface)
W, H = 3.0, 2.0                      # device width, depth
imp_x0, imp_x1, imp_depth = 1.0, 2.0, 0.5   # p+ implant: lateral window and junction depth
edge = 0.08                          # erf smoothing width of the implant edge (um)
nD, nA = 1e16, 1e18                  # substrate donors, implant acceptors (cm^-3)
win_x0, win_x1 = 1.1, 1.9            # anode contact on the top surface (inside the implant window)

net = SumDoping([Uniform(nD, 2), ErfBox([imp_x0, -1.0], [imp_x1, imp_depth], edge, -(nA + nD))])
xs = np.linspace(0, W, 600); ys = np.linspace(0, H, 400)
X, Y = np.meshgrid(xs, ys); N = net(X, Y)
# metallurgical junction (N = 0) from the profile itself
col = N[:, np.argmin(abs(xs - W / 2))]; jdepth = ys[np.argmax(col > 0)]
row = N[np.argmin(abs(ys - 0.05)), :]; neg = xs[row < 0]; jx0, jx1 = neg.min(), neg.max()
S_ = np.sign(N) * np.log10(np.abs(N) + 1)

fig, a = plt.subplots(figsize=(9, 6.4))
# ---- left: schematic ----
a.add_patch(Rectangle((0, 0), W, H, fc="#dbe7f7", ec="k", lw=1.5))
a.contourf(X, Y, (N < 0).astype(float), levels=[0.5, 1.5], colors=["#f4c7c3"])
a.contour(X, Y, N, levels=[0], colors="k", linestyles="--", linewidths=1.2)
a.add_patch(Rectangle((win_x0, -0.07), win_x1 - win_x0, 0.07, fc="#7a7a7a", ec="k"))
a.add_patch(Rectangle((0, H), W, 0.07, fc="#7a7a7a", ec="k"))
a.text(W / 2, -0.17, "anode (top contact)", ha="center", va="bottom", fontsize=14)
a.text(W / 2, H + 0.17, "cathode (bottom contact, full width)", ha="center", va="top", fontsize=14)
a.text(W / 2, 0.22, r"p$^+$ implant" "\n" r"$n_A=10^{18}$ cm$^{-3}$", ha="center", va="center", fontsize=14)
a.text(W / 2, 1.35, r"n$^-$ substrate" "\n" r"$n_D=10^{16}$ cm$^{-3}$", ha="center", va="center", fontsize=14)
a.text(0.04, 0.95, "insulating\n(zero normal\nfield and flux)", fontsize=12, va="center")
a.text(W - 0.04, 0.95, "insulating", fontsize=12, va="center", ha="right", rotation=90)
a.text((imp_x0 + win_x0) / 2 - 0.45, -0.05, "insulating", fontsize=12, ha="center", va="bottom")
dim = lambda p, q, t, off=(0, 0), **k: (a.add_patch(FancyArrowPatch(p, q, arrowstyle="<->", mutation_scale=9, lw=0.8, color="0.25")),
                                          a.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], t, fontsize=12, ha="center", va="center", **k))
dim((0, H + 0.45), (W, H + 0.45), r"$W=3\,\mu$m", (0, 0.1))
dim((-0.3, 0), (-0.3, H), r"$H=2\,\mu$m", (-0.2, 0), rotation=90)
dim((jx0, jdepth + 0.22), (jx1, jdepth + 0.22), rf"{jx1-jx0:.2f} $\mu$m (junction width)", (0, 0.1))
dim((jx1 + 0.25, 0), (jx1 + 0.25, jdepth), rf"{jdepth:.2f} $\mu$m", (0.25, 0), rotation=90)
a.set_xlim(-0.65, W + 0.35); a.set_ylim(H + 0.75, -0.55); a.set_aspect("equal"); a.axis("off")
fig.tight_layout(); import os; os.makedirs(f"{ROOT}/figures", exist_ok=True); fig.savefig(f"{ROOT}/figures/fig_planar2d_geometry.png", dpi=170, bbox_inches="tight")
print("ok")
