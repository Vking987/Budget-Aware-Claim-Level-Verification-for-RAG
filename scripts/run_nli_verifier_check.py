#!/usr/bin/env python3
"""Small, fast verifier check: NLIAssessor on RAGTruth claims with gold labels (separate from the benchmark; no controller, no budget).

Reuses the claim-set construction, evidence conditions (A original benchmark chunks, B readable full-source chunks) and scoring of
scripts/run_ragtruth_verifier_diagnostic.py, so numbers are directly comparable to the 1B-assessor diagnostic. Thresholds are the
assessor's fixed defaults (0.5 / 0.5); nothing is tuned on this data. Reports TPR/FPR with Wilson CIs, never precision on natural prevalence.

  --claims pilot50   the exact 17 gold-hallucinated + 60 sampled unflagged claims of the earlier 1B diagnostic (same seed)
  --claims sample    a larger seeded stratified sample of TEST responses (--n responses)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_ragtruth_comparison as R  # noqa: E402
import run_ragtruth_verifier_diagnostic as D  # noqa: E402
from src.evaluation.ragtruth_adapter import RAGTruthDatasetAdapter  # noqa: E402
from src.evidence.nli_assessor import DEFAULT_MODEL, NLIAssessor  # noqa: E402

PILOT = ROOT / "experiments/runs/ragtruth_comparison_20261028_162847_pilot50.json"


class _NoCache:
    hits = 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claims", choices=["pilot50", "sample"], default="pilot50")
    ap.add_argument("--n", type=int, default=60, help="responses for --claims sample")
    ap.add_argument("--neg-per-task", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--conditions", default="A_original,B_readable")
    ap.add_argument("--out-dir", default=str(ROOT / "experiments/diagnostics"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"ragtruth_nli_check_{ts}{('_' + a.tag) if a.tag else ''}"
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ad = RAGTruthDatasetAdapter()
    if a.claims == "pilot50":
        proto = json.load(open(PILOT, encoding="utf-8"))["manifest"]["protocol"]
        ad.load_from_jsonl(R.DATA_DIR / "source_info.jsonl", R.DATA_DIR / "response.jsonl", split=proto["split"])
        by_id = {r.response_id: r for r in ad.responses}
        chosen = [by_id[i] for i in proto["response_ids"]]
    else:
        ad.load_from_jsonl(R.DATA_DIR / "source_info.jsonl", R.DATA_DIR / "response.jsonl", split="test")
        chosen = R.stratified_sample(ad.responses, ad.sources, a.n, a.seed)
    prepared, mapped = R.prepare(ad, chosen)
    items = D.build_claim_set(prepared, mapped, a.neg_per_task, a.seed)
    if a.limit:
        items = items[: a.limit]
    n_pos = sum(1 for it in items if it["gold"] == "HALLUCINATED")
    print(f"claim set ({a.claims}): {n_pos} gold-hallucinated + {len(items) - n_pos} unflagged", flush=True)

    assessor = NLIAssessor(a.model)
    results = {}
    conds = a.conditions.split(",")
    for cond in conds:
        t0 = time.monotonic()
        results[cond] = D.run_condition(items, cond, assessor, _NoCache(), ad.sources)
        errs = sum(1 for x in results[cond] if x["error"])
        print(f"[{cond}] {len(items)} claims in {time.monotonic() - t0:.0f}s wall, {errs} errors", flush=True)
    D.CONDITIONS = tuple(conds)
    summ = D.summarize(items, results)
    for cond in conds:
        for mode in ("strict", "broad"):
            o = summ[cond]["overall"][mode]
            print(f"  {cond} {mode}: TPR {D.fmt(o['recall'])} ({o['tp']}/{o['tp'] + o['fn']}) FPR {D.fmt(o['fpr'])} ({o['fp']}/{o['fp'] + o['tn']}) F1 {D.fmt(o['f1'])}")
        print(f"  {cond} labels: {json.dumps(summ[cond]['label_counts'])}")
    meta = {"run_id": run_id, "command": " ".join(sys.argv), "model": a.model, "assessor": assessor.assessor_id,
            "thresholds": {"entail": assessor.entail_threshold, "contradict": assessor.contradict_threshold}, "claims": a.claims,
            "n_pos": n_pos, "n_neg": len(items) - n_pos, "seed": a.seed, "n_responses": len(chosen),
            "label_validation": D.validate_label_mapping(items)}
    (out_dir / f"{run_id}.json").write_text(json.dumps({"meta": meta, "summary": summ}, indent=2, default=str), encoding="utf-8")
    print("wrote", out_dir / f"{run_id}.json")
    return summ


if __name__ == "__main__":
    main()
