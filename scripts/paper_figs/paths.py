"""Single place for every location the paper-figure scripts read from or write to (import with `from paths import *`).

ROOT     paper root of the 1D symmetric study (full-domain truth): data/ (ensembles, cal, test sets, ...), figures/ (outputs)
DERIV    derivative-control run (eta column + SRH divisor, seed 0): profiles, rows, bulk-slope decomposition
EXTRAP   forward-extrapolation pilot (per-anchor folders above 0.64 V)
STYLE    matplotlib style used by every figure
All inputs live in <repo>/paper_data; outputs go to ROOT/figures.
"""
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROOT = os.environ.get("PAPER_ROOT") or os.path.join(REPO, "paper_data", "sixseed_v10", "mirror")
DERIV = os.environ.get("PAPER_DERIV") or os.path.join(REPO, "paper_data", "sixseed_v10", "derivative_control")
EXTRAP = os.path.join(REPO, "paper_data", "extrap_pilot")
STYLE = os.path.join(os.path.dirname(__file__), "plot.mplstyle")
TEST_SET = os.environ.get("TEST_SET", "test_fresh250")  # the paper's test set (246 devices)
U_T_DATA = 0.025887193125  # k T/q of the DEVSIM data (k = 1.3806503e-23, q = 1.6e-19, T = 300 K): the constant of the V/V_bi < 0.8 training filter
TEXT_FS = 12  # minimum size of every in-plot text()/annotate(); at least the largest legend font used in the figures


# Every figure saved as PNG is also written as a vector PDF next to it (pcolormesh maps stay rasterised inside the PDF to keep the files small).
import matplotlib.axes as _mpl_axes
import matplotlib.figure as _mpl_fig
_orig_savefig, _orig_pcolormesh = _mpl_fig.Figure.savefig, _mpl_axes.Axes.pcolormesh


def _save_pdf_too(self, fname, *args, **kwargs):
    _orig_savefig(self, fname, *args, **kwargs)
    if isinstance(fname, (str, os.PathLike)) and str(fname).endswith(".png"):
        _orig_savefig(self, str(fname)[:-4] + ".pdf", *args, **kwargs)


def _pcolormesh_raster(self, *args, **kwargs):
    kwargs.setdefault("rasterized", True)
    return _orig_pcolormesh(self, *args, **kwargs)


_mpl_fig.Figure.savefig = _save_pdf_too
_mpl_axes.Axes.pcolormesh = _pcolormesh_raster
