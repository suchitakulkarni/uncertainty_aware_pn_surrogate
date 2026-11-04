"""Analytic net-doping profiles C(x) / C(x, y) [cm^-3], signed (+ donors, - acceptors).

A profile is a callable taking one coordinate array per spatial dimension (all in cm, same
shape) and returning C of that shape. Profiles are sums of simple components, so one
vocabulary covers asymmetric 1D junctions, planar 2D junctions and laterally-doped 2D regions:

    Uniform(N)                          constant background
    ErfStep(axis, x0, width, amp)       amp * erf((c_axis - x0)/width)   (width 0 -> abrupt amp * sign)
    ErfBox(lo, hi, width, amp)          amp inside the box [lo, hi] (per-axis), smooth erf edges

make_junction_1d() builds the standard asymmetric junction (acceptors N_A on the left of x0, donors
N_D on the right, or the flipped-polarity "left" family). Profiles round-trip through to_spec() /
from_spec() (plain dicts, JSON-serialisable) so a data file can carry its own description.
"""
import json

import numpy as np
from scipy.special import erf


class DopingProfile:
    n_dims: int = 1

    def __call__(self, *coords: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def to_spec(self) -> dict:
        raise NotImplementedError


class Uniform(DopingProfile):
    def __init__(self, value: float, n_dims: int = 1):
        self.value, self.n_dims = float(value), n_dims

    def __call__(self, *coords):
        return np.full(np.shape(coords[0]), self.value)

    def to_spec(self):
        return dict(kind="uniform", value=self.value, n_dims=self.n_dims)


class ErfStep(DopingProfile):
    """amp * erf((c_axis - x0)/width); width == 0 is the abrupt amp * (+1 if c >= x0 else -1)."""

    def __init__(self, axis: int, x0: float, width: float, amp: float, n_dims: int = 1):
        self.axis, self.x0, self.width, self.amp, self.n_dims = int(axis), float(x0), float(width), float(amp), n_dims

    def __call__(self, *coords):
        c = coords[self.axis]
        if self.width > 0:
            return self.amp * erf((c - self.x0) / self.width)
        return np.where(c < self.x0, -self.amp, self.amp)

    def to_spec(self):
        return dict(kind="erf_step", axis=self.axis, x0=self.x0, width=self.width, amp=self.amp, n_dims=self.n_dims)


class ErfBox(DopingProfile):
    """amp inside the axis-aligned box lo..hi, with erf-smoothed edges of the given width."""

    def __init__(self, lo, hi, width: float, amp: float):
        self.lo, self.hi, self.width, self.amp = list(map(float, lo)), list(map(float, hi)), float(width), float(amp)
        self.n_dims = len(self.lo)

    def __call__(self, *coords):
        out = np.full(np.shape(coords[0]), self.amp)
        for c, a, b in zip(coords, self.lo, self.hi):
            if self.width > 0:
                out = out * 0.5 * (erf((c - a) / self.width) - erf((c - b) / self.width))
            else:
                out = out * ((c >= a) & (c <= b))
        return out

    def to_spec(self):
        return dict(kind="erf_box", lo=self.lo, hi=self.hi, width=self.width, amp=self.amp)


class SumDoping(DopingProfile):
    def __init__(self, parts):
        self.parts = list(parts)
        self.n_dims = self.parts[0].n_dims

    def __call__(self, *coords):
        total = self.parts[0](*coords)
        for p in self.parts[1:]:
            total = total + p(*coords)
        return total

    def to_spec(self):
        return dict(kind="sum", parts=[p.to_spec() for p in self.parts])


def from_spec(spec: dict | str) -> DopingProfile:
    if isinstance(spec, str):
        spec = json.loads(spec)
    kind = spec["kind"]
    if kind == "uniform":
        return Uniform(spec["value"], spec.get("n_dims", 1))
    if kind == "erf_step":
        return ErfStep(spec["axis"], spec["x0"], spec["width"], spec["amp"], spec.get("n_dims", 1))
    if kind == "erf_box":
        return ErfBox(spec["lo"], spec["hi"], spec["width"], spec["amp"])
    if kind == "sum":
        return SumDoping([from_spec(p) for p in spec["parts"]])
    raise ValueError(f"unknown doping kind {kind!r}")


def make_junction(nA: float, nD: float, x0: float, width: float, donor_side: str = "right", axis: int = 0,
                  n_dims: int = 1) -> DopingProfile:
    """Planar junction normal to `axis` at x0 (uniform along the other axes).

    donor_side="right": acceptors nA below x0, donors nD above. "left": donors nD below, acceptors nA above.
    erf form: N = (N_hi + N_lo)/2 + (N_hi - N_lo)/2 * erf((c - x0)/width), which for nA == nD is exactly
    +-nD * erf(.) (the symmetric family).
    """
    n_lo, n_hi = (-nA, nD) if donor_side == "right" else (nD, -nA)
    return SumDoping([Uniform(0.5 * (n_hi + n_lo), n_dims), ErfStep(axis, x0, width, 0.5 * (n_hi - n_lo), n_dims)])
