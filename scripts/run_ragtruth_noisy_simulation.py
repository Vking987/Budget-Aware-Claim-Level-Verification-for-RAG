#!/usr/bin/env python3
"""SIMULATED noisy-verifier comparison of policies on the held-out RAGTruth test split (no LLM calls).

This is a SIMULATION, not a result for any real model. A simulated verifier flags a gold-positive claim with probability TPR and
a gold-negative claim with probability FPR. The draw is a deterministic function of (noise seed, response id, claim id), never of
the policy, the budget or the call order, so every policy sees the SAME simulated verdict for the same claim (as with cached real
verdicts) and policies are compared on identical examples and budgets.

Assumptions (all unrealistic in ways that matter, see the report):
  * verdict errors are independent across claims and independent of task type, claim features and evidence;
  * rates are fixed per scenario; ambiguous gold labels (UNCERTAIN, IMPLICIT_TRUE_ABSENT) are treated as negatives (and never scored);
  * a flag is CONTRADICTED, a non-flag SUPPORTED (confidence 0.8); repair is disabled for every policy; 1 model call per verification.

Scenarios: S0/S1/S2 use rates MEASURED in the 77-claim verifier diagnostic; S3 is HYPOTHETICAL (a verifier that works), shown only to
see how selection differences would translate into detection if one existed.
"""

from __future__ import annotations

import argparse
import hashlib
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
from src.controller.mab_policy import ContextualMABPolicy  # noqa: E402
from src.evaluation import ragtruth_metrics as M  # noqa: E402
from src.evaluation.ragtruth_adapter import RAGTruthDatasetAdapter  # noqa: E402
from src.evidence.assessor import BaseAssessor  # noqa: E402
from src.evidence.models import AssessmentLabel, EvidenceAssessment, VerificationCost  # noqa: E402

SCENARIOS = {
    "S0_measured_benchmark_evidence": {"tpr": 0.0, "fpr": 0.10, "kind": "measured",
                                      "source": "diagnostic, benchmark evidence: TP 0/17, FP 6/60"},
    "S1_measured_readable_evidence": {"tpr": 2 / 17, "fpr": 8 / 60, "kind": "measured",
                                      "source": "diagnostic, readable evidence: TP 2/17, FP 8/60"},
    "S2_measured_optimistic_tpr": {"tpr": 0.343, "fpr": 8 / 60, "kind": "measured (upper 95% Wilson bound of S1's TPR)",
                                   "source": "diagnostic: TPR upper bound 0.343, FPR 8/60"},
    "S3_hypothetical_working_verifier": {"tpr": 0.70, "fpr": 0.10, "kind": "HYPOTHETICAL (not measured)",
                                         "source": "assumed"},
}
EVAL_POLICIES = ["StaticCTSRanked", "StaticFIFO", "DeterministicCTS", "Random", "VerifyAll"]
ARR_KEYS = ("btp", "bfn", "bfp", "btn", "stp", "sfn", "sfp", "stn")


class SimulatedAssessor(BaseAssessor):
    """Noisy verifier simulation: see module docstring. NOT a model."""

    def __init__(self, gold: Dict[str, str], response_id: str, tpr: float, fpr: float, noise_seed: int) -> None:
        self.gold, self.response_id, self.tpr, self.fpr, self.noise_seed = gold, response_id, tpr, fpr, noise_seed

    @property
    def assessor_id(self) -> str:
        return f"simulated_tpr{self.tpr:.3f}_fpr{self.fpr:.3f}"

    def _u(self, claim_id: str) -> float:
        h = hashlib.sha256(f"{self.noise_seed}|{self.response_id}|{claim_id}".encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big") / 2.0 ** 64

    def assess(self, claim, passages):
        p = self.tpr if self.gold.get(claim.claim_id) == "HALLUCINATED" else self.fpr  # ambiguous gold -> treated as negative
        flagged = self._u(claim.claim_id) < p
        return EvidenceAssessment(
            claim_id=claim.claim_id, evidence_ids=[x.evidence_id for x in passages], supporting_spans={},
            label=AssessmentLabel.CONTRADICTED if flagged else AssessmentLabel.SUPPORTED, confidence=0.8,
            rationale="simulated verdict", cost=VerificationCost(model_calls=1, latency_seconds=0.0, model_id="simulated"),
        )


def sim_factory(tpr: float, fpr: float, noise_seed: int):
    return lambda prep: SimulatedAssessor(prep["gold"], prep["resp"].response_id, tpr, fpr, noise_seed)


# ----------------------------------------------------------------------------- vectorised bootstrap over responses
def accumulate(counts_list: List[Dict[str, Dict[str, int]]], ids: List[str]) -> Dict[str, np.ndarray]:
    A = {k: np.zeros(len(ids)) for k in ARR_KEYS}
    for cs in counts_list:  # sum the repeated noise draws per response
        for i, rid in enumerate(ids):
            c = cs.get(rid)
            if c:
                for k, src in zip(ARR_KEYS, ("broad_tp", "broad_fn", "broad_fp", "broad_tn", "sel_tp", "sel_fn", "sel_fp", "sel_tn")):
                    A[k][i] += c[src]
    return A


def _ratio(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def metric_values(A, idx=None) -> Dict[str, Any]:
    """idx None -> scalars from all responses; idx (n_boot x n) -> one value per resample."""
    def S(k):
        return A[k].sum() if idx is None else A[k][idx].sum(axis=1)
    btp, bfn, bfp, stp, sfn = S("btp"), S("bfn"), S("bfp"), S("stp"), S("sfn")
    sel_recall = _ratio(stp, stp + sfn)
    det_recall = _ratio(btp, btp + bfn)
    det_prec = _ratio(btp, btp + bfp)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where((det_prec + det_recall) > 0, 2 * det_prec * det_recall / (det_prec + det_recall), np.where(np.isnan(det_prec), np.nan, 0.0))
    return {"selection_recall": sel_recall, "detection_recall": det_recall, "detection_precision": det_prec, "detection_f1": f1}


def ci(v) -> Optional[List[float]]:
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) >= 20 else None


def fnum(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--noise-seeds", default="0,1,2")
    ap.add_argument("--budgets", default="2,4,8")
    ap.add_argument("--random-seeds", default="0,1,2")
    ap.add_argument("--linucb-oracle", default=None, help="frozen oracle-trained LinUCB checkpoint (the one from the full oracle run)")
    ap.add_argument("--train-responses", type=int, default=1000, help="TRAIN-split responses for the per-scenario noisy-trained LinUCB (0 = skip)")
    ap.add_argument("--train-budget", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--fixture", default=None)
    ap.add_argument("--max-responses", type=int, default=0, help="debug: only the first N test responses")
    ap.add_argument("--out-dir", default=str(R.RUN_DIR))
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scen_names = [s for s in args.scenarios.split(",") if s]
    noise_seeds = [int(x) for x in args.noise_seeds.split(",")]
    budgets = [int(x) for x in args.budgets.split(",")]
    rseeds = [int(x) for x in args.random_seeds.split(",")]

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
    t_prep = time.monotonic()

    tr_prep, train_ids = [], []
    if args.train_responses > 0:
        tr = load("train")
        tr_prep, _ = R.prepare(tr, R.stratified_sample(tr.responses, tr.sources, args.train_responses, args.seed + 1000))
        train_ids = [p["resp"].response_id for p in tr_prep]
        assert not (set(train_ids) & set(ids)), "train/eval overlap"

    linucb_oracle = ContextualMABPolicy.load_checkpoint(args.linucb_oracle, frozen=True) if args.linucb_oracle else None
    repairer = R._CostedMockRepairer()  # never invoked: repair is disabled
    rng = np.random.default_rng(args.seed)
    boots = rng.integers(0, len(ids), size=(args.n_boot, len(ids)))

    results: Dict[str, Any] = {}
    for si, sname in enumerate(scen_names):
        sc = SCENARIOS[sname]
        linucb: Dict[str, ContextualMABPolicy] = {}
        if linucb_oracle is not None:
            linucb["LinUCB_oracle_trained"] = linucb_oracle
        ckpt_path = None
        if args.train_responses > 0:  # LinUCB trained on TRAIN claims under the same simulated noise (separate noise stream)
            ckpt_path = out_dir / f"ragtruth_linucb_noisytrained_{sname}_{ts}.json"
            pol = R.train_linucb(tr_prep, None, repairer, args.train_budget, args.seed, ckpt_path,
                                 assessor_factory=sim_factory(sc["tpr"], sc["fpr"], 10_000 + si), enable_repair=False)
            pol.freeze()
            linucb["LinUCB_noisy_trained"] = pol
        rows_specs = [(n, args.seed) for n in linucb] + [(p, args.seed) for p in EVAL_POLICIES if p != "Random"] + [("Random", s) for s in rseeds]
        acc: Dict[tuple, List[Dict]] = {}
        calls: Dict[tuple, int] = {}
        for ns in noise_seeds:
            for b in budgets:
                for name, seed in rows_specs:
                    res, rows = R.run_setting(name, b, seed, prepared, linucb, None, repairer, None,
                                              assessor_factory=sim_factory(sc["tpr"], sc["fpr"], ns), enable_repair=False, n_boot=0)
                    key = (b, name, seed)
                    acc.setdefault(key, []).append(M.claim_counts_by_response(rows))
                    calls[key] = calls.get(key, 0) + res["recorded_cost"]["model_calls"]
                    assert res["recorded_cost"]["model_calls"] <= b * len(ids), "budget exceeded"
            print(f"[{sname}] noise seed {ns} done ({time.monotonic() - t_prep:.0f}s elapsed)", flush=True)
        arrays = {k: accumulate(v, ids) for k, v in acc.items()}
        sc_out: Dict[str, Any] = {"params": sc, "linucb_noisy_checkpoint": str(ckpt_path) if ckpt_path else None, "budgets": {}}
        for b in budgets:
            tab, rec_by_policy = {}, {}
            for (bb, name, seed), A in arrays.items():
                if bb != b:
                    continue
                label = f"{name}({seed})" if name == "Random" else name
                point = {k: fnum(v) for k, v in metric_values(A).items()}
                boot = metric_values(A, boots)
                tab[label] = {**point, **{f"{k}_ci95": ci(v) for k, v in boot.items()},
                              "model_calls": calls[(bb, name, seed)], "noise_draws": len(noise_seeds),
                              "expected_detection_recall_if_independent": fnum(point["selection_recall"] * sc["tpr"])}
                rec_by_policy[label] = boot
            rnd = [l for l in tab if l.startswith("Random(")]
            if rnd:
                tab["Random(mean of seeds)"] = {k: float(np.nanmean([tab[l][k] for l in rnd])) for k in ("selection_recall", "detection_recall")}
            paired = {}
            for lin in [l for l in tab if l.startswith("LinUCB")]:
                for base in [l for l in tab if not l.startswith("LinUCB")
                             and (not l.startswith("Random(") or l == "Random(mean of seeds)")]:
                    if base == "Random(mean of seeds)":
                        bvals = {m: np.mean([rec_by_policy[l][m] for l in rnd], axis=0) for m in ("selection_recall", "detection_recall")}
                    else:
                        bvals = rec_by_policy[base]
                    for m in ("selection_recall", "detection_recall"):
                        d = rec_by_policy[lin][m] - bvals[m]
                        point = tab[lin][m] - (tab[base][m] if base in tab and m in tab[base] else np.nan)
                        paired[f"{lin} - {base} [{m}]"] = {"diff": fnum(point), "ci95": ci(d)}
            sc_out["budgets"][str(b)] = {"policies": tab, "paired": paired}
        results[sname] = sc_out

    out = {
        "manifest": {
            "run_id": f"ragtruth_noisy_sim_{ts}{('_' + args.tag) if args.tag else ''}", "command": " ".join(sys.argv),
            "SIMULATION": "no model or LLM was called; verdicts are simulated from fixed TPR/FPR (see assumptions)",
            "split": "test" if not args.fixture else "fixture", "n_responses": len(ids), "noise_seeds": noise_seeds, "budgets": budgets,
            "n_boot": args.n_boot, "linucb_oracle_checkpoint": args.linucb_oracle,
            "linucb_noisy_training": {"n_train_responses": len(train_ids), "budget": args.train_budget, "train_noise_stream": "10000+scenario index (disjoint from test noise seeds)"},
            "repair_enabled": False, "assumptions": __doc__.split("Assumptions", 1)[1].split("Scenarios:")[0].strip(),
        },
        "scenarios": results,
    }
    jp = out_dir / f"{out['manifest']['run_id']}.json"
    jp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    L = [f"# {out['manifest']['run_id']}", "", "**SIMULATION: no model or LLM was called.** Verifier verdicts are simulated from fixed TPR/FPR; see assumptions in the report.", ""]
    for sname, so in results.items():
        L += [f"## {sname}  (TPR {so['params']['tpr']:.3f}, FPR {so['params']['fpr']:.3f}; {so['params']['kind']})", ""]
        for b, bo in so["budgets"].items():
            L += [f"### budget {b}", "", "| Policy | Selection recall [CI] | Detection recall [CI] | Detection precision | F1 | expected det. recall |", "|---|---|---|---|---|---|"]
            for lab, v in bo["policies"].items():
                if "selection_recall_ci95" not in v:
                    L.append(f"| {lab} | {v['selection_recall']:.3f} | {v['detection_recall']:.3f} | | | |")
                    continue
                c1, c2 = v["selection_recall_ci95"], v["detection_recall_ci95"]
                f = lambda x: "n/a" if x is None else f"{x:.3f}"
                L.append(f"| {lab} | {f(v['selection_recall'])} [{f(c1[0]) if c1 else 'n/a'}, {f(c1[1]) if c1 else 'n/a'}] | "
                         f"{f(v['detection_recall'])} [{f(c2[0]) if c2 else 'n/a'}, {f(c2[1]) if c2 else 'n/a'}] | {f(v['detection_precision'])} | "
                         f"{f(v['detection_f1'])} | {f(v['expected_detection_recall_if_independent'])} |")
            L.append("")
    (out_dir / f"{out['manifest']['run_id']}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {jp}")
    return out


if __name__ == "__main__":
    main()
