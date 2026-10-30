import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import make_ieee_figures as F  # noqa: E402


def _synthetic(tmp_path, drop=None):
    """Tiny replay-schema fixture (2 policies, 3 questions): NOT a result, used only to test aggregation and validation."""
    def run(em0, em1, n_in):
        return [{"sample_id": f"q{i}", "n_claims": 4, "strategies": {
            "S0_current": {"str_em": em0, "n_in_answer": n_in[0]}, "S1_conservative": {"str_em": em1, "n_in_answer": n_in[1]},
            "S2_repair_first": {"str_em": em1, "n_in_answer": n_in[1]}}} for i in range(3)]
    d = {"manifest": {}, "summary": {"draft_str_em": 0.5}, "per_question_runs": {"4|A": [run(.1, .3, (1, 2)), run(.3, .3, (1, 2))], "4|B": [run(.2, .4, (2, 4))]}}
    if drop:
        del d[drop]
    p = tmp_path / "replay.json"
    p.write_text(json.dumps(d))
    return p


def test_alce_tables_aggregation(tmp_path):
    t = F.alce_tables(_synthetic(tmp_path), n_boot=200)
    # policy A seeds average to (.2, .3); policy B (.2, .4); equal-weight mean over policies
    assert t["strategies"]["S0_current"]["str_em"] == pytest.approx(0.2)
    assert t["strategies"]["S1_conservative"]["str_em"] == pytest.approx(0.35)
    assert t["strategies"]["S0_current"]["retention"] == pytest.approx((1 + 2) / 2 / 4)
    assert t["paired"]["S2-S1 str_em"]["diff"] == 0 and t["n_questions"] == 3


def test_alce_tables_rejects_missing_fields(tmp_path):
    with pytest.raises(ValueError):
        F.alce_tables(_synthetic(tmp_path, drop="per_question_runs"))


def test_alce_tables_rejects_missing_strategy(tmp_path):
    p = _synthetic(tmp_path)
    d = json.loads(p.read_text())
    del d["per_question_runs"]["4|A"][0][0]["strategies"]["S2_repair_first"]
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        F.alce_tables(p)


def test_real_replay_matches_documented_values():
    t = F.alce_tables(F.ALCE_REPLAY)
    s = t["strategies"]
    assert round(s["S0_current"]["str_em"], 3) == 0.182 and round(s["S1_conservative"]["str_em"], 3) == 0.225
    assert round(s["S0_current"]["retention"], 2) == 0.32 and round(s["S1_conservative"]["retention"], 2) == 0.49
    assert t["n_questions"] == 30


def test_fig2_and_fig3_are_reproducible(tmp_path):
    def digest(name, fn):
        out = tmp_path / name
        fn(None if name.startswith("fig3") else F.ALCE_REPLAY, out)
        return hashlib.sha256(out.read_bytes()).hexdigest(), out.with_suffix(".pdf").exists()
    for name, fn in (("fig2.png", F.fig2), ("fig3.png", F.fig3)):
        a, b = digest(name, fn), digest(name, fn)
        assert a == b and a[1]
