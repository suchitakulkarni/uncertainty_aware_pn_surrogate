import os

import numpy as np
import pytest

from src.uq import load_bar, query_status

BAR = os.path.join(os.path.dirname(__file__), "..", "paper_data", "sixseed_v10", "mirror", "calibration_bar.json")
pytestmark = pytest.mark.skipif(not os.path.exists(BAR), reason="paper_data not present")


def test_inside_gets_a_bar():
    q = query_status(4e-4, 6e16, 0.4, load_bar(BAR), "I")
    assert q.status == "ok" and q.bar is not None and q.bar > 0


@pytest.mark.parametrize("L,nD,V", [(9e-6, 1e17, 0.4), (8e-4, 1e17, 0.4), (4e-4, 5e14, 0.4), (4e-4, 2e19, 0.4),
                                    (4e-4, 1e17, 0.02), (4e-4, 1e17, 0.8)])
def test_outside_box_is_refused(L, nD, V):
    q = query_status(L, nD, V, load_bar(BAR), "I")
    assert q.status == "out_of_box" and q.bar is None


def test_vbi_limit_and_support():
    bar = load_bar(BAR)
    assert query_status(4e-4, 1e15, 0.64, bar, "I").status == "above_vbi_limit"  # V_bi = 0.58 V
    q = query_status(1e-5, 1e15, 0.04, bar, "I")  # W/L >> 1: x above the calibrated support
    assert q.status == "out_of_support"
