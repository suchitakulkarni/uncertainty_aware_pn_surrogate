"""v4 seed-independence table (2026-10-14): for every EQI, the median test-set error of each of the six single networks (own informative/measurable masks), reported as mean +- std
(ddof=1) and min-max over the six networks, next to the ensemble error and its 95% device-bootstrap interval (data/table2_v4.csv). Writes data/seed_table_v4.csv and
data/seed_table_v4_body.tex (LaTeX rows)."""
import numpy as np
import pandas as pd

import pv3
import score_table2_v4 as t2
from paths import TEST_SET

P = t2.P
SC = [("I", r"Readout current $I_{\rm ro}$"), ("Jnet", r"Network $J$ (differentiated)"), ("W", r"Depletion width $W$"), ("Ep", r"Peak field $\mathcal{E}_{\rm peak}$"),
      ("Qn", r"Stored charge $Q_n$"), ("Qp", r"Stored charge $Q_p$")]
PR = [("psi", r"$\psi$"), ("n", "$n$"), ("p", "$p$"), ("R", "$R$"), ("dEF", r"$\Delta E_F$")]
PC = {"psi": "reconstruction.psi_L1_rel", "n": "reconstruction.n_L1_rel", "p": "reconstruction.p_L1_rel", "R": "eqi.R_L1_rel", "dEF": "eqi.dEF_L1_rel"}


def f(x: float) -> str:
    return f"{100 * x:.2f}" if x < 0.01 else f"{100 * x:.1f}"


def main() -> None:
    ps = pd.read_csv(f"{P}/data/{TEST_SET}/per_seed_rows.csv")
    ps = ps[ps.provenance.isin(["test60", "test_extra100", "test_fresh250"])]
    s = pv3.add_errors(pv3.canonicalise(ps))
    s["e_Jnet"] = (s.ratio_net - 1).abs()
    ek = {"I": "e_I", "Jnet": "e_Jnet", "W": "e_W", "Ep": "e_Ep", "Qn": "e_Qn", "Qp": "e_Qp"}
    mk = {"I": s.measurable, "Jnet": s.measurable, "W": s.inf_W, "Ep": s.measurable | True, "Qn": s.inf_Qn, "Qp": s.inf_Qp}
    t = pd.read_csv(f"{P}/data/table2_v4.csv").set_index(["group", "eqi"])
    seeds = sorted(s.seed.unique())
    rows, tex = [], []
    for block, items in (("Scalar EQIs (relative error)", SC), ("Profiles (relative $L_1$ error)", PR)):
        if tex:
            tex.append(r"\midrule")
        tex.append(r"\multicolumn{4}{@{}l}{\emph{" + block + r"}}\\")
        for k, name in items:
            if k in ek:
                med = [s[(s.seed == sd) & mk[k]][ek[k]].median() for sd in seeds]
            else:
                med = [ps[ps.seed == sd][PC[k]].median() for sd in seeds]
            r = t.loc[("test", k)]
            rows.append(dict(eqi=k, ens=r.err, ens_lo=r.err_lo, ens_hi=r.err_hi, single_mean=np.mean(med), single_std=np.std(med, ddof=1), single_min=min(med), single_max=max(med),
                             std_over_ci_halfwidth=np.std(med, ddof=1) / ((r.err_hi - r.err_lo) / 2)))
            tex.append(f"{name} & {f(r.err)} [{f(r.err_lo)}, {f(r.err_hi)}] & {f(np.mean(med))} $\\pm$ {f(np.std(med, ddof=1))} & {f(min(med))}--{f(max(med))} \\\\")
    out = pd.DataFrame(rows)
    out.to_csv(f"{P}/data/seed_table_v4.csv", index=False)
    open(f"{P}/data/seed_table_v4_body.tex", "w").write("\n".join(tex) + "\n")
    print(out.to_string(float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
