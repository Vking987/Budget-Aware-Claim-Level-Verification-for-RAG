#!/usr/bin/env python3
"""Budgeted selection with a REAL (cheap, deterministic) verifier on the held-out RAGTruth test split. No LLM calls.

Why: the 1B verifier is near chance and DeBERTa NLI costs ~40 s/claim on this CPU, so policy comparisons with a working real verifier were
never run. This script uses a lexical-unsupported-fraction verifier whose only free parameter (the flag threshold) is fixed on a TRAIN-split
calibration sample before the test split is touched. It is a *testbed for budget allocation under a real, imperfect, structured-error
verifier*, not a proposed detector: it is cheap enough that, in deployment, it would not need a budget.

Verifier V_lex (sees only the claim text and the evidence passages the controller hands it; never gold labels):
    u = fraction of the claim's content tokens (alphanumeric, not a stopword, length > 1, plural-'s' stripped) absent from the evidence text
    flag (CONTRADICTED) if u >= tau, else SUPPORTED. One verification = one model call, so budgets bind exactly as for any verifier.
tau = argmax Youden J (TPR - FPR) over a fixed grid on the calibration sample (gold HALLUCINATED vs UNFLAGGED; ambiguous labels excluded).

Same claims, evidence and budgets for every policy (asserted); repair disabled; LinUCB frozen at evaluation. Two LinUCB checkpoints, both
declared before running: the frozen oracle-trained one from the full oracle run, and one trained on TRAIN responses under V_lex verdicts.
"""

from __future__ import annotations

import argparse
import json
import re
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
from src.controller.mab_policy import ContextualMABPolicy  # noqa: E402
from src.evaluation import ragtruth_metrics as M  # noqa: E402
from src.evaluation.ragtruth_adapter import RAGTruthDatasetAdapter  # noqa: E402
from src.evidence.assessor import BaseAssessor  # noqa: E402
from src.evidence.models import AssessmentLabel, EvidenceAssessment, VerificationCost  # noqa: E402

STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being it its this that these those he she they them his her their
we you i not no do does did has have had will would can could should may might than then so such into about over after before between during there here which who whom
what when where how also more most some any each other one two""".split())
_TOK = re.compile(r"[a-z0-9]+")
GRID = [round(x * 0.05, 2) for x in range(1, 21)]
EVAL_POLICIES = ["StaticCTSRanked", "StaticFIFO", "DeterministicCTS", "Random", "VerifyAll"]


def content_tokens(text: str) -> List[str]:
    out = []
    for t in _TOK.findall(text.lower()):
        if len(t) > 1 and t not in STOP:
            out.append(t[:-1] if len(t) > 3 and t.endswith("s") else t)
    return out


def unsupported_fraction(claim_text: str, evidence_text: str) -> float:
    ctoks = content_tokens(claim_text)
    if not ctoks:
        return 0.0
    ev = set(content_tokens(evidence_text))
    return sum(1 for t in ctoks if t not in ev) / len(ctoks)


class LexicalVerifier(BaseAssessor):
    def __init__(self, tau: float) -> None:
        self.tau = tau

    @property
    def assessor_id(self) -> str:
        return f"lexical_unsupported_fraction_tau{self.tau:.2f}"

    def assess(self, claim, passages):
        u = unsupported_fraction(claim.normalized_text, " ".join(p.source_text for p in passages))
        flagged = u >= self.tau
        return EvidenceAssessment(
            claim_id=claim.claim_id, evidence_ids=[p.evidence_id for p in passages], supporting_spans={},
            label=AssessmentLabel.CONTRADICTED if flagged else AssessmentLabel.SUPPORTED, confidence=0.8,
            rationale=f"unsupported fraction {u:.2f}", cost=VerificationCost(model_calls=1, latency_seconds=0.0, model_id="lexical"),
        )


def claim_scores(prepared) -> List[Dict[str, Any]]:
    rows = []
    for p in prepared:
        for c in p["claims"]:
            ev = " ".join(x.source_text for x in p["evidence"][c.claim_id])
            rows.append({"rid": p["resp"].response_id, "gold": p["gold"][c.claim_id], "task": p["task"],
                         "u": unsupported_fraction(c.normalized_text, ev)})
    return rows


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float)
    # average ranks for ties
    order = np.argsort(allv, kind="mergesort")
    sv = allv[order]
    r = np.empty(len(allv))
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def calibrate(rows) -> Dict[str, Any]:
    pos = np.array([r["u"] for r in rows if r["gold"] == "HALLUCINATED"])
    neg = np.array([r["u"] for r in rows if r["gold"] == "UNFLAGGED"])
    table = []
    for t in GRID:
        tpr, fpr = float((pos >= t).mean()), float((neg >= t).mean())
        table.append({"tau": t, "tpr": tpr, "fpr": fpr, "youden_j": tpr - fpr})
    best = max(table, key=lambda x: (x["youden_j"], -x["tau"]))
    return {"tau": best["tau"], "grid": table, "n_pos": int(len(pos)), "n_neg": int(len(neg)), "auc": auc(pos, neg)}


def verifier_on_test(rows, tau, n_boot, seed) -> Dict[str, Any]:
    pos = [r for r in rows if r["gold"] == "HALLUCINATED"]
    neg = [r for r in rows if r["gold"] == "UNFLAGGED"]
    tp = sum(r["u"] >= tau for r in pos)
    fp = sum(r["u"] >= tau for r in neg)
    out = {"tau": tau, "n_pos": len(pos), "n_neg": len(neg), "tp": tp, "fp": fp, "fn": len(pos) - tp, "tn": len(neg) - fp,
           "tpr": tp / len(pos), "fpr": fp / len(neg), "tpr_ci95": M.wilson_ci(tp, len(pos)), "fpr_ci95": M.wilson_ci(fp, len(neg)),
           "precision_at_natural_prevalence": tp / (tp + fp) if tp + fp else None, "recall": tp / len(pos),
           "auc": auc(np.array([r["u"] for r in pos]), np.array([r["u"] for r in neg]))}
    out["f1"] = 2 * tp / (2 * tp + fp + (len(pos) - tp)) if tp else 0.0
    rids = sorted({r["rid"] for r in rows})
    by = {rid: ([], []) for rid in rids}
    for r in pos:
        by[r["rid"]][0].append(r["u"])
    for r in neg:
        by[r["rid"]][1].append(r["u"])
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(rids), size=len(rids))
        p = np.array([u for i in pick for u in by[rids[i]][0]])
        n = np.array([u for i in pick for u in by[rids[i]][1]])
        aucs.append(auc(p, n))
    out["auc_ci95"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
    # per task (descriptive only)
    out["by_task"] = {}
    for t in sorted({r["task"] for r in rows}):
        p_ = [r["u"] for r in pos if r["task"] == t]
        n_ = [r["u"] for r in neg if r["task"] == t]
        out["by_task"][t] = {"n_pos": len(p_), "n_neg": len(n_), "tpr": (sum(u >= tau for u in p_) / len(p_)) if p_ else None,
                             "fpr": (sum(u >= tau for u in n_) / len(n_)) if n_ else None, "auc": auc(np.array(p_), np.array(n_)) if p_ and n_ else None}
    return out


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budgets", default="2,4,8")
    ap.add_argument("--random-seeds", default="0,1,2")
    ap.add_argument("--linucb-oracle", required=True, help="frozen oracle-trained LinUCB checkpoint from the full oracle run")
    ap.add_argument("--cal-responses", type=int, default=1500, help="TRAIN responses used only to fix tau")
    ap.add_argument("--train-responses", type=int, default=1000, help="TRAIN responses for the lexical-trained LinUCB")
    ap.add_argument("--train-budget", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--tau", type=float, default=None, help="override tau (default: calibrated on TRAIN)")
    ap.add_argument("--max-responses", type=int, default=0, help="debug: first N test responses")
    ap.add_argument("--out-dir", default=str(R.RUN_DIR))
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets = [int(x) for x in args.budgets.split(",")]
    rseeds = [int(x) for x in args.random_seeds.split(",")]
    t0 = time.monotonic()

    def load(split):
        ad = RAGTruthDatasetAdapter()
        ad.load_from_jsonl(R.DATA_DIR / "source_info.jsonl", R.DATA_DIR / "response.jsonl", split=split)
        return ad

    tr = load("train")
    cal_prep, _ = R.prepare(tr, R.stratified_sample(tr.responses, tr.sources, args.cal_responses, args.seed + 2000))
    cal_ids = {p["resp"].response_id for p in cal_prep}
    cal = calibrate(claim_scores(cal_prep))
    tau = args.tau if args.tau is not None else cal["tau"]
    print(f"calibration on {len(cal_ids)} train responses: tau={tau} (calibration AUC {cal['auc']:.3f}, {cal['n_pos']} pos / {cal['n_neg']} neg)", flush=True)

    ad = load("test")
    resp = sorted((r for r in ad.responses if r.source_id in ad.sources), key=lambda r: r.response_id)
    if args.max_responses:
        resp = resp[: args.max_responses]
    prepared, _ = R.prepare(ad, resp)
    ids = [p["resp"].response_id for p in prepared]
    tr_prep, _ = R.prepare(tr, R.stratified_sample(tr.responses, tr.sources, args.train_responses, args.seed + 1000))
    train_ids = {p["resp"].response_id for p in tr_prep}
    assert not (train_ids | cal_ids) & set(ids), "train/eval overlap"
    test_rows = claim_scores(prepared)
    vtest = verifier_on_test(test_rows, tau, 300, args.seed)
    print(f"verifier on test (all claims): TPR {vtest['tpr']:.3f} FPR {vtest['fpr']:.3f} AUC {vtest['auc']:.3f} {vtest['auc_ci95']}", flush=True)

    repairer = R._CostedMockRepairer()  # never invoked: repair disabled
    factory = lambda prep: LexicalVerifier(tau)  # noqa: E731
    linucb = {"LinUCB_oracle_trained": ContextualMABPolicy.load_checkpoint(args.linucb_oracle, frozen=True)}
    ckpt = out_dir / f"ragtruth_linucb_lexicaltrained_{ts}.json"
    pol = R.train_linucb(tr_prep, None, repairer, args.train_budget, args.seed, ckpt, assessor_factory=factory, enable_repair=False)
    pol.freeze()
    linucb["LinUCB_lexical_trained"] = pol

    rng = np.random.default_rng(args.seed)
    boots = rng.integers(0, len(ids), size=(args.n_boot, len(ids)))
    specs = [(n, args.seed) for n in linucb] + [(p, args.seed) for p in EVAL_POLICIES if p != "Random"] + [("Random", s) for s in rseeds]
    arrays, calls, seen = {}, {}, {}
    for b in budgets:
        for name, seed in specs:
            res, rows = R.run_setting(name, b, seed, prepared, linucb, None, repairer, None, assessor_factory=factory, enable_repair=False, n_boot=0)
            assert res["recorded_cost"]["model_calls"] <= b * len(ids), "budget exceeded"
            key = (b, name, seed)
            arrays[key] = N.accumulate([M.claim_counts_by_response(rows)], ids)
            calls[key] = res["recorded_cost"]["model_calls"]
            seen.setdefault(b, set()).add(tuple((x["response_id"], x["claim_id"]) for x in rows))
        assert len(seen[b]) == 1, "policies saw different claims"
        print(f"budget {b} done ({time.monotonic() - t0:.0f}s)", flush=True)

    results: Dict[str, Any] = {}
    for b in budgets:
        tab, boot_by = {}, {}
        for (bb, name, seed), A in arrays.items():
            if bb != b:
                continue
            label = f"{name}({seed})" if name == "Random" else name
            point = {k: N.fnum(v) for k, v in N.metric_values(A).items()}
            boot = N.metric_values(A, boots)
            tp = float(A["btp"].sum())
            tab[label] = {**point, **{f"{k}_ci95": N.ci(v) for k, v in boot.items()}, "model_calls": calls[(bb, name, seed)],
                          "true_positives": int(tp), "gold_positives": int(A["btp"].sum() + A["bfn"].sum()),
                          "true_positives_per_100_calls": 100 * tp / calls[(bb, name, seed)] if calls[(bb, name, seed)] else None}
            boot_by[label] = boot
        rnd = [l for l in tab if l.startswith("Random(")]
        tab["Random(mean of seeds)"] = {k: float(np.nanmean([tab[l][k] for l in rnd])) for k in ("selection_recall", "detection_recall", "true_positives_per_100_calls")}
        paired = {}
        for lin in [l for l in tab if l.startswith("LinUCB")]:
            for base in [l for l in tab if not l.startswith("LinUCB") and (not l.startswith("Random(") or l == "Random(mean of seeds)")]:
                for m in ("selection_recall", "detection_recall"):
                    bv = np.mean([boot_by[l][m] for l in rnd], axis=0) if base == "Random(mean of seeds)" else boot_by[base][m]
                    paired[f"{lin} - {base} [{m}]"] = {"diff": N.fnum(tab[lin][m] - tab[base][m]), "ci95": N.ci(boot_by[lin][m] - bv)}
        results[str(b)] = {"policies": tab, "paired": paired}

    out = {"manifest": {
        "run_id": f"ragtruth_cheap_verifier_{ts}{('_' + args.tag) if args.tag else ''}", "command": " ".join(sys.argv), "split": "test",
        "n_test_responses": len(ids), "n_test_claims": len(test_rows), "verifier": "lexical unsupported-fraction (deterministic, no model)",
        "tau": tau, "tau_source": "override" if args.tau is not None else f"Youden J on {len(cal_ids)} TRAIN responses (seed {args.seed + 2000})",
        "linucb_oracle_checkpoint": args.linucb_oracle, "linucb_lexical_checkpoint": str(ckpt), "lexical_train_responses": len(train_ids),
        "budgets": budgets, "n_boot": args.n_boot, "repair_enabled": False, "evidence": "top-4 lexical chunks of the gold source (oracle context)",
        "runtime_s": round(time.monotonic() - t0, 1)},
        "calibration": cal, "verifier_test": vtest, "budgets": results}
    jp = out_dir / f"{out['manifest']['run_id']}.json"
    jp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {jp}")
    return out


if __name__ == "__main__":
    main()
