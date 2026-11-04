# QFL physics-informed surrogate for p-n diodes

Code accompanying the paper (`main_v10.tex`): a FiLM-conditioned PINN in quasi-Fermi-level (QFL) space for a p-n diode
(1D symmetric, 1D asymmetric doping, and a 2D planar benchmark), trained against DEVSIM data. The networks output the
electrostatic potential and the two quasi-Fermi levels; every quantity of interest (current, depletion width, peak field,
stored charge, ...) is read from that state. Background: `THEORY.pdf` (physics, QFL formalism) and `EQI_definitions.pdf`.

This release contains only the recipe used for the paper's production results (the six-seed 1D runs and the asymmetric and
2D benchmarks). It has no option switches; the recipe is fixed in `src/config.py`:

| Setting | Value |
|---|---|
| Contact trim | 1 raw DEVSIM node (1 nm) per contact; 1 nm contact inset for the 2D devices |
| Epochs / optimiser | 20000, Adam 1e-3, StepLR (step 7000, gamma 0.1), minibatch of 64 anchors |
| Voltage centring | fixed `V_c = 12.361324787139893` (V/Ut) |
| Conditioning | 6 FiLM columns (no eta column); 2D adds 4 geometry columns |
| Continuity loss | pointwise residual times `J_ref * ell` (no SRH-peak divisor) |
| Boundary pins | DEVSIM's own node values at the window edge (no symmetric pinning) |
| Reference doping | `n_max = max(nA, nD)` |
| Network | TripleNet, 4 x 128 Tanh, 128-wide phi-net, unbounded FiLM gamma, v_p recentred |

The seed is the only environment option: `QFL_SEED` (default 0).

## Layout

- `src/` - model, losses, training, evaluation (`run_train_eval_qfl_film.py` for 1D, `run_train_eval_qfl_2d.py` for 2D,
  `evaluate.py` / `evaluate2d.py` / `qoi2d.py` / `readout2d.py` for the quantities of interest).
- `devsim_data_generator/` - DEVSIM scripts that produce the 1D grids, the asymmetric devices and the planar 2D diode.
- `scripts/` - 2D data conversion (`planar2d_to_anchors.py`), 2D QoI table (`eval_planar2d_qoi.py`), re-evaluation of a
  trained run (`reeval_film.py`), asymmetric state figure (`make_paper_asym_figs.py`), asymmetric data check.
- `data/anchor_lists/` - anchor lists (plain text, one `.npz` path per line) for the production split
  (`qfl_pe_dh3_*`; `qfl_pe_dh3_train_vbi08.txt` is the 404-anchor training list with V/V_bi >= 0.8 removed, as in the paper) and the asymmetric benchmark (`showcase_asym_*`). The `.npz` files themselves are not in this repository.
- `handoff_data/` - three anchors used by the tests.
- `tests/` - `python -m pytest tests -q`.

## Running

```
python -m src.run_train_eval_qfl_film --train-list data/anchor_lists/qfl_pe_dh3_train_vbi08.txt \
    --holdout-list data/anchor_lists/qfl_pe_dh3_holdout.txt --out-dir results/run0
python -m src.run_train_eval_qfl_2d --train-list <train.txt> [--holdout-list <holdout.txt>] --out-dir results/run2d
```

A checkpoint loads only into `build_model(v_centre)` from `run_train_eval_qfl_film.py`; the runs save `v_centre.json`
next to it. Each run writes `loss_curves.png`, `eval_summary.csv` (full per-anchor quantity-of-interest suite) and
per-anchor plots.

## Reproducing the paper figures

`paper_data/` holds the tables the figures are plotted from (six-seed ensemble outputs, calibration and test sets, bar,
derivative control, extrapolation pilot, data-only ablation; 76 MB). `scripts/paper_figs/make_all.sh` regenerates the
table-based figures into `paper_data/sixseed_v10/mirror/figures/`. Not yet included: the figures that re-evaluate trained
networks (1D and 2D state figures, loss curves and state-space best/worst, quasi-Fermi slopes, asymmetric state).

## Deployment check

`src/uq.py: query_status(L, nD, V, bar_json, eqi)` returns the 90% bar of a query only if it is inside the trained box
(L in [1e-5, 7e-4] cm, nD in [1e15, 1e19] cm^-3, V in [0.04, 0.64] V), has V/V_bi < 0.8, and `log10(W_dep/L)` is inside the
calibrated support of the quantity; otherwise it returns the reason and no bar.

## Not included

Data (DEVSIM grids, 2D planar solutions), trained checkpoints, and the network-based figure scripts. The development repository also contains the older non-FiLM
pipeline and the switches for the ablations (eta column, SRH divisor, symmetric pins, trim width), which are not part of
this release.
