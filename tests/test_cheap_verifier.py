import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import numpy as np
import run_ragtruth_cheap_verifier as C


def test_unsupported_fraction():
    assert C.unsupported_fraction("Paris is the capital of France", "France has Paris as capital") == 0.0
    assert C.unsupported_fraction("the 1999 award", "nothing here") == 1.0
    assert C.unsupported_fraction("the of and", "x") == 0.0  # only stopwords


def test_auc_ties_and_order():
    assert C.auc(np.array([2., 3.]), np.array([0., 1.])) == 1.0
    assert C.auc(np.array([1.]), np.array([1.])) == 0.5


def test_calibrate_picks_separating_tau():
    rows = [{"gold": "HALLUCINATED", "u": 0.9}] * 5 + [{"gold": "UNFLAGGED", "u": 0.1}] * 5
    assert 0.1 < C.calibrate(rows)["tau"] <= 0.9
