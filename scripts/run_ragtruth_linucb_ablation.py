#!/usr/bin/env python3
"""LinUCB feature ablation with multiple training seeds, evaluated on the HELD-OUT RAGTruth test split (oracle-verifier control, no LLM).

Design (fixed before running):
  * condition: oracle (gold-label) verifier control, repair disabled, budgets {2,4,8}; metric = selection recall (verifier-independent);
  * variants (feature masks applied at feature extraction in BOTH training and evaluation): full | no_CTS (x1 := 0) | no_type_risk (x2 := 0) | no_CTS_no_type;
    The 8 LinUCB features are bias, CTS, claim-type risk, evidence coverage, assessment status, interaction history, budget fraction left,
    action-state affinity. There is NO claim-position feature; under this control only CTS and type risk can distinguish two pending claims;
  * training seeds 0..4: each seed trains on a different seeded stratified sample of 1,000 TRAIN-split responses (alpha 0.5, budget 4, proxy reward),
    then is frozen; all variants of a seed use the same training sample; the test split is never used for training (asserted);
  * position / CTS-direction are probed with reference policies that use only that information: ReverseFIFO (last claim first),
    LowCTSFirst (ascending CTS), plus the benchmark's CTS-ranked (descending CTS), FIFO and Random;
  * uncertainty: percentile bootstrap over test responses (pooled over training seeds), plus the per-seed spread (training variance).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_ragtruth_comparison as R  # noqa: E402
import run_ragtruth_noisy_simulation as N  # noqa: E402
import src.controller.mab_policy as MP  # noqa: E402
from src.controller.baselines import _EligibilityBaseline  # noqa: E402
from src.controller.models import ControllerAction  # noqa: E402
from src.evaluation import ragtruth_metrics as M  # noqa: E402
from src.evaluation.ragtruth_adapter import RAGTruthDatasetAdapter  # noqa: E402

FEATURES = ["bias", "CTS", "type_risk", "evid_cov", "assess_status", "interact_hist", "budget_left", "affinity"]
VARIANTS = {"full": [], "no_CTS": [1], "no_type_risk": [2], "no_CTS_no_type": [1, 2]}
REFERENCES = ["StaticCTSRanked", "StaticFIFO", "Random", "ReverseFIFO", "LowCTSFirst"]


class OrderedPolicy(_EligibilityBaseline):
    """Verify-only reference policy that orders eligible claims by a fixed rule (used to probe position and CTS direction)."""

    def __init__(self, order: str, config=None) -> None:
        super().__init__(config)
        self.order = order

    def select_next_action(self, claims, evidence_store, assessments, repairs, budget_tracker):
        if budget_tracker.is_exhausted():
            return ControllerAction.STOP, None, "Global verification budget is exhausted."
        to_verify, _ = self._eligible(claims, evidence_store, assessments, repairs, budget_tracker)
        if not to_verify:
            return ControllerAction.STOP, None, f"{self.order}: nothing eligible."
        pos = {c.claim_id: i for i, c in enumerate(claims)}
        if self.order == "ReverseFIFO":
            c = max(to_verify, key=lambda x: pos[x.claim_id])
        else:  # LowCTSFirst
            c = min(to_verify, key=lambda x: ((x.triage_score if x.triage_score is not None else 0.0), pos[x.claim_id]))
        return ControllerAction.VERIFY, c, f"{self.order}: picked '{c.claim_id}'."


_orig_make_policy = R.make_policy


def _make_policy(name, seed, idx, linucb, enable_repair=True):
    if name in ("ReverseFIFO", "LowCTSFirst"):
        return OrderedPolicy(name, R.ControllerPolicyConfig(enable_repair=enable_repair))
    return _orig_make_policy(name, seed, idx, linucb, enable_repair)


R.make_policy = _make_policy  # run_setting looks the factory up at call time


@contextlib.contextmanager
def feature_mask(masked: List[int]):
    """Zero the given feature indices wherever LinUCB builds a context vector (training and evaluation)."""
    orig = MP.extract_context_features

    def wrapped(*a, **k):
        x = list(orig(*a, **k))
        for i in masked:
            x[i] = 0.0
        return x

    MP.extract_context_features = wrapped
    try:
        yield
    finally:
        MP.extract_context_features = orig


def mean_verified_index(rows) -> float:
    pos, counter, idxs = {}, {}, []
    for r in rows:
        counter[r["response_id"]] = counter.get(r["response_id"], -1) + 1
        if r["label"] is not None:
            idxs.append(counter[r["response_id"]])
    return float(np.mean(idxs)) if idxs else float("nan")


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--budgets", default="2,4,8")
    ap.add_argument("--train-responses", type=int, default=1000)
    ap.add_argument("--train-budget", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--fixture", default=None)
    ap.add_argument("--max-responses", type=int, default=0, help="debug: only the first N test responses")
    ap.add_argument("--out-dir", default=str(R.RUN_DIR))
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    budgets = [int(b) for b in args.budgets.split(",")]

    def load(split):
        ad = RAGTruthDatasetAdapter()
        if args.fixture:
            ad.load_from_fixture(args.fixture)
        else:
            ad.load_from_jsonl(R.DATA_DIR / "source_info.jsonl", R.DATA_DIR / "response.jsonl", split=split)
        return ad

    ad = load("test")
    resp = sorted((r for r in ad.responses if r.source_id in ad.sources), key=lambda r: r.response_id)
    if args.max_responses:
        resp = resp[: args.max_responses]
    prepared, _ = R.prepare(ad, resp)
    ids = [p["resp"].response_id for p in prepared]
    tr = load("train")
    repairer = R._CostedMockRepairer()
    oracle_factory = lambda prep: R.OracleAssessor(prep["gold"])  # noqa: E731
    rng = np.random.default_rng(0)
    boots = rng.integers(0, len(ids), size=(args.n_boot, len(ids)))
    t0 = time.monotonic()

    # ---- train every (seed, variant) policy on that seed's TRAIN sample
    policies: Dict[tuple, Any] = {}
    train_ids: Dict[int, List[str]] = {}
    thetas: Dict[tuple, List[float]] = {}
    for s in seeds:
        tr_prep, _ = R.prepare(tr, R.stratified_sample(tr.responses, tr.sources, args.train_responses, s + 1000))
        train_ids[s] = [p["resp"].response_id for p in tr_prep]
        assert not (set(train_ids[s]) & set(ids)), "train/eval overlap"
        for v, mask in VARIANTS.items():
            with feature_mask(mask):
                pol = R.train_linucb(tr_prep, None, repairer, args.train_budget, s, out_dir / f"ragtruth_ablation_{v}_seed{s}_{ts}.json",
                                     assessor_factory=oracle_factory, enable_repair=False)
            pol.freeze()
            policies[(s, v)] = pol
            thetas[(s, v)] = pol.arms[ControllerAction.VERIFY].get_theta()
        print(f"seed {s}: trained {len(VARIANTS)} variants on {len(tr_prep)} train responses ({time.monotonic() - t0:.0f}s)", flush=True)

    # ---- evaluate on the held-out test split
    arr: Dict[tuple, Dict[str, np.ndarray]] = {}
    meta: Dict[tuple, Dict[str, float]] = {}
    for b in budgets:
        for (s, v), pol in policies.items():
            with feature_mask(VARIANTS[v]):
                res, rows = R.run_setting("LinUCB", b, s, prepared, {"LinUCB": pol}, None, repairer, None,
                                          assessor_factory=oracle_factory, enable_repair=False, n_boot=0)
            arr[(b, v, s)] = N.accumulate([M.claim_counts_by_response(rows)], ids)
            meta[(b, v, s)] = {"mean_verified_index": mean_verified_index(rows), "calls": res["recorded_cost"]["model_calls"]}
        for ref in REFERENCES:
            for s in (seeds[:3] if ref == "Random" else [0]):
                res, rows = R.run_setting(ref, b, s, prepared, {}, None, repairer, None,
                                          assessor_factory=oracle_factory, enable_repair=False, n_boot=0)
                arr[(b, ref, s)] = N.accumulate([M.claim_counts_by_response(rows)], ids)
                meta[(b, ref, s)] = {"mean_verified_index": mean_verified_index(rows), "calls": res["recorded_cost"]["model_calls"]}
        print(f"budget {b} evaluated ({time.monotonic() - t0:.0f}s)", flush=True)

    def pooled(b, name):
        keys = [(b, name, s) for (bb, n, s) in arr if bb == b and n == name]
        return {k: sum(arr[kk][k] for kk in keys) for k in N.ARR_KEYS}, keys

    out_b: Dict[str, Any] = {}
    for b in budgets:
        table, boot_vals = {}, {}
        for name in list(VARIANTS) + REFERENCES:
            A, keys = pooled(b, name)
            point = float(N.metric_values(A)["selection_recall"])
            boot = N.metric_values(A, boots)["selection_recall"]
            per_seed = [float(N.metric_values(arr[k])["selection_recall"]) for k in keys]
            boot_vals[name] = boot
            table[name] = {"selection_recall": point, "ci95": N.ci(boot), "per_run": per_seed, "min": min(per_seed), "max": max(per_seed),
                           "n_runs": len(keys), "mean_verified_index": float(np.mean([meta[k]["mean_verified_index"] for k in keys]))}
            if name in VARIANTS:
                th = np.mean([thetas[(s, name)] for s in seeds], axis=0)
                table[name]["mean_theta_verify"] = {n: round(float(t), 4) for n, t in zip(FEATURES, th)}
        paired = {}
        for a in VARIANTS:
            for c in [x for x in VARIANTS if x != a] + REFERENCES:
                if a != "full" and c in VARIANTS:
                    continue
                d = boot_vals[a] - boot_vals[c]
                paired[f"{a} - {c}"] = {"diff": table[a]["selection_recall"] - table[c]["selection_recall"], "ci95": N.ci(d)}
        out_b[str(b)] = {"policies": table, "paired": paired}

    out = {"manifest": {"run_id": f"ragtruth_linucb_ablation_{ts}{('_' + args.tag) if args.tag else ''}", "command": " ".join(sys.argv),
                        "condition": "oracle (gold-label) verifier control; repair disabled; held-out test split; selection recall only",
                        "n_test_responses": len(ids), "training_seeds": seeds, "variants": VARIANTS, "features": FEATURES,
                        "train_responses_per_seed": args.train_responses, "n_boot": args.n_boot,
                        "note": "No claim-position feature exists in LinUCB; position is probed with ReverseFIFO/LowCTSFirst reference policies."},
           "budgets": out_b}
    jp = out_dir / f"{out['manifest']['run_id']}.json"
    jp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    f = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
    L = [f"# {out['manifest']['run_id']}", "", f"Condition: {out['manifest']['condition']}. Test responses: {len(ids)}. Training seeds: {seeds}.", ""]
    for b, bo in out_b.items():
        L += [f"## Budget {b}", "", "| Policy | Selection recall [95% CI over responses] | per-seed min-max | mean index of verified claims | mean learned VERIFY weights (CTS / type) |", "|---|---|---|---|---|"]
        for name, v in bo["policies"].items():
            th = v.get("mean_theta_verify")
            L.append(f"| {name} | {f(v['selection_recall'])} [{f(v['ci95'][0]) if v['ci95'] else 'n/a'}, {f(v['ci95'][1]) if v['ci95'] else 'n/a'}] | "
                     f"{f(v['min'])}-{f(v['max'])} | {v['mean_verified_index']:.2f} | {('%.3f / %.3f' % (th['CTS'], th['type_risk'])) if th else ''} |")
        L += ["", "Paired differences (95% CI over responses; unadjusted):", "", "| Comparison | Difference | CI |", "|---|---|---|"]
        for k, v in bo["paired"].items():
            L.append(f"| {k} | {v['diff']:+.3f} | [{f(v['ci95'][0]) if v['ci95'] else 'n/a'}, {f(v['ci95'][1]) if v['ci95'] else 'n/a'}] |")
        L.append("")
    (out_dir / f"{out['manifest']['run_id']}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {jp}")
    return out


if __name__ == "__main__":
    main()
