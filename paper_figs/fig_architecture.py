"""Fig. architecture: FiLM-conditioned network (x -> Fourier features -> psi/n/p trunks; (L, nD, V, ...) -> one FiLM hypernet per trunk). Output: architecture_film.{png,pdf}"""
from common import *  # noqa: F401,F403
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrow, FancyArrowPatch, Rectangle, Circle

W, H = 1260, 620
fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=200)
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
fig.patch.set_facecolor("#f8f8f8")
GREY = "#555555"


def arrow(p, q, color=GREY, lw=1.8, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, color=color, lw=lw, connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=0, shrinkB=0, zorder=3))


def box(x0, y0, x1, y1, fc, label="", ec=GREY, fs=11, lw=1.0, ls="-"):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, ha="center", va="center", fontsize=fs, zorder=4)


def mini_net(x0, y0, x1, y1, fc):
    box(x0, y0, x1, y1, fc)
    xs = np.linspace(x0 + 25, x1 - 25, 3)
    ys = np.linspace(y0 + 30, y1 - 12, 3)
    for a, b in ((0, 1), (1, 2)):
        for ya in ys:
            for yb in ys:
                ax.plot([xs[a], xs[b]], [ya, yb], color="#777777", lw=.5, zorder=3)
    for xx in xs:
        for yy in ys:
            ax.add_patch(Circle((xx, yy), 6.5, fc="white", ec="#555555", lw=.8, zorder=4))


ax.add_patch(Rectangle((8, 8), 960, 600, fc="#eeeeee", ec="k", lw=1.6, ls=(0, (1, 3)), zorder=0))
ax.add_patch(Rectangle((8, 8), 1236, 600, fc="none", ec="#999999", lw=1, zorder=0))

rows = (22, 212, 402)
names = ("Phi-net", "n-net", "p-net")
cols = ("#fbe5d0", "#f8cb9c", "#ffe599")
NET = (305, 450)
FILM = (322, 433)
net_mid, film_mid = [], []
for r, nm, c in zip(rows, names, cols):
    box(*FILM[:1], r, FILM[1], r + 52, "#cfe2f3", "FiLM\nhypernet", ec="#2a6099", fs=9.5, lw=1.3)
    mini_net(NET[0], r + 84, NET[1], r + 178, c)
    ax.text(NET[0] + 8, r + 98, nm, fontsize=9.5, va="center", zorder=5)
    arrow((377.5, r + 52), (377.5, r + 84), color="#2a6099", lw=2)
    ax.text(386, r + 69, r"$\gamma,\ \beta$", fontsize=10.5, color="#2a6099", va="center", zorder=5)
    film_mid.append(r + 26)
    net_mid.append(r + 131)

# inputs
ax.add_patch(Ellipse((80, 160), 104, 160, fc="#f4cccc", ec=GREY, lw=1, zorder=2))
ax.text(80, 160, "cond:\n$L$\n$n_D$\n$V$\n$\\eta$", ha="center", va="center", fontsize=11, zorder=4)
ax.add_patch(Ellipse((80, 345), 76, 76, fc="#f4cccc", ec=GREY, lw=1, zorder=2))
ax.text(80, 345, "$x$", ha="center", va="center", fontsize=15, zorder=4)
ax.text(80, 392, "(+Fourier\nfeatures)", ha="center", va="top", fontsize=8.5, color=GREY, zorder=4)
ax.add_patch(Circle((80, 535), 42, fc="#dd7e6b", ec=GREY, lw=1, zorder=2))
ax.text(80, 535, "Data", ha="center", va="center", fontsize=13, zorder=4)
for fm in film_mid:
    arrow((119, 160 + (fm - 160) * 0.12), (FILM[0], fm), color="#2a6099")
for nm_ in net_mid:
    arrow((117, 345 + (nm_ - 345) * 0.1), (NET[0], nm_))
for nm_ in net_mid:
    arrow((118, 530 + (nm_ - 530) * 0.08), (NET[0], nm_ + 12), color="#999999", lw=1.2)

# losses
LX = (620, 740)
box(LX[0], 40, LX[1], 160, "#d9ead3", "")
ax.text(LX[0] + 12, 72, r"$L_{\mathrm{phys}}$", fontsize=16, va="center", zorder=5); ax.text(LX[0] + 12, 128, "(PDEs)", fontsize=11, va="center", zorder=5)
box(LX[0], 225, LX[1], 345, "#b6d7a8", "")
ax.text(LX[0] + 12, 257, r"$L_{\mathrm{BC}}$", fontsize=16, va="center", zorder=5); ax.text(LX[0] + 12, 315, "(Dirichlet\nBC)", fontsize=11, va="center", zorder=5)
box(LX[0], 415, LX[1], 535, "#d9ead3", "")
ax.text(LX[0] + 12, 447, r"$L_{\mathrm{MSE}}$", fontsize=16, va="center", zorder=5); ax.text(LX[0] + 12, 505, "(DevSim −\nNN o/p)", fontsize=11, va="center", zorder=5)
for nm_ in net_mid:
    for ly in (100, 285, 475):
        arrow((NET[1], nm_), (LX[0], ly), lw=1.5)
box(830, 250, 940, 370, "#d0e0e3", "")
ax.text(845, 305, r"$L_{\mathrm{tot}}$", fontsize=16, va="center", zorder=5)
for ly in (100, 285, 475):
    arrow((LX[1], ly), (830, 310), lw=1.8)
arrow((885, 370), (885, 590), style="-", lw=1.8); arrow((885, 590), (377.5, 590), style="-", lw=1.8); arrow((377.5, 590), (377.5, rows[2] + 178 + 1), lw=1.8)

# derived quantities
ax.add_patch(FancyArrow(980, 305, 40, 0, width=26, head_width=56, head_length=30, length_includes_head=True, fc="#eeeeee", ec=GREY, lw=1, zorder=3))
box(1040, 210, 1235, 400, "#ead1dc", "Derived\nquantities\n(readout)", fs=12)

# FiLM legend
box(990, 430, 1238, 575, "#ffffff", "", ec="#2a6099", lw=1)
ax.text(1000, 450, "FiLM (per hidden layer $\\ell$):", fontsize=9.5, va="center", color="#2a6099", zorder=5)
ax.text(1000, 488, r"$h_\ell=\tanh(\gamma_\ell\odot W_\ell h_{\ell-1}+\beta_\ell)$", fontsize=10, va="center", zorder=5)
ax.text(1000, 538, "$(\\gamma,\\beta)$ = hypernet(cond)\ninit: $\\gamma{=}1,\\ \\beta{=}0$ (identity)", fontsize=8.5, va="center", color=GREY, zorder=5)
fig.savefig(os.path.join(OUT, "architecture_film.png"), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
