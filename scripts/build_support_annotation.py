#!/usr/bin/env python3
"""Build a blinded human-annotation sheet for the S0 (drop INSUFFICIENT) vs S1 (retain INSUFFICIENT) question.

NO model inference: every verdict is read from the existing NLI verdict cache (read-only), keyed by (claim text, evidence) exactly as the
runs keyed them. Under the replayed answer rules (docs/alce_answer_handling_report.md), a claim with a verified-claim verdict is placed in:
    SUPPORTED     -> in the S0 and the S1 answer
    INSUFFICIENT  -> in the S1 answer only (the retention gain being tested)
    CONTRADICTED  -> in neither (a check on whether exclusion was right)
The pool is every draft claim of the 30 questions that has a cached verdict, so with 130 claims the sheet is a census of the pool, not a draw.
Rows are shuffled with a fixed seed; the sheet shows only the question, the claim and the retrieved passages (strategy and verdict are in the KEY).

usage: python scripts/build_support_annotation.py [--seed 0] [--max-items 150]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

LABELS = ("supported", "unsupported", "contradicted", "indeterminate")
STRATUM_RULE = {"SUPPORTED": ("S0+S1", True, True), "INSUFFICIENT": ("S1 only", False, True), "CONTRADICTED": ("neither", False, False)}
_SCORES = re.compile(r"max entailment=([\d.]+), max contradiction=([\d.]+)")


def build_records(prep: Dict[str, Any], lookup: Callable[[Dict[str, Any], List[Dict[str, Any]]], Any]) -> List[Dict[str, Any]]:
    """One record per claim that has a verifier verdict. lookup(claim_dict, passage_dicts) -> assessment or None."""
    recs = []
    for sid in sorted(prep):
        p = prep[sid]
        for c in p["claims"]:
            a = lookup(c, p["passages"])
            if a is None or a.label.value not in STRATUM_RULE:
                continue
            m = _SCORES.search(a.rationale or "")
            recs.append({"sample_id": sid, "claim_id": c["claim_id"], "question": p["question"], "claim": c["normalized_text"],
                         "source_sentence": c["original_text"], "passages": p["passages"], "verdict": a.label.value,
                         "nli_max_entailment": float(m.group(1)) if m else None, "nli_max_contradiction": float(m.group(2)) if m else None})
    return recs


def select(recs: List[Dict[str, Any]], max_items: int, seed: int) -> List[Dict[str, Any]]:
    """Census if the pool fits in max_items; otherwise a seeded sample stratified by verdict with equal allocation (remainder to larger strata)."""
    if len(recs) <= max_items:
        return list(recs)
    rng = random.Random(seed)
    by = {}
    for r in recs:
        by.setdefault(r["verdict"], []).append(r)
    quota = {k: 0 for k in by}
    left = max_items
    while left > 0:
        open_ = [k for k in sorted(by) if quota[k] < len(by[k])]
        if not open_:
            break
        for k in open_:
            if left > 0:
                quota[k] += 1
                left -= 1
    out = []
    for k in sorted(by):
        out += rng.sample(sorted(by[k], key=lambda r: (r["sample_id"], r["claim_id"])), quota[k])
    return out


def write_files(items: List[Dict[str, Any]], out_dir: Path, seed: int, meta: Dict[str, Any]):
    rng = random.Random(seed)
    items = sorted(items, key=lambda r: (r["sample_id"], r["claim_id"]))
    rng.shuffle(items)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_p, key_p = out_dir / "alce_support_annotation_sheet.csv", out_dir / "alce_support_annotation_key.json"
    for p in (sheet_p, key_p):
        if p.exists():
            raise FileExistsError(f"{p} exists; refusing to overwrite an annotation artifact (use --out-dir)")
    n_ev = max(len(r["passages"]) for r in items)
    cols = ["item_id", "question", "claim", "claim_source_sentence"] + [f"evidence_{i + 1}" for i in range(n_ev)] + ["label", "notes"]
    key: Dict[str, Any] = {}
    with open(sheet_p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, r in enumerate(items):
            iid = f"C{i + 1:03d}"
            ev = [f"[{k + 1}] {p['doc_title']}: {p['source_text']}" for k, p in enumerate(r["passages"])]
            w.writerow([iid, r["question"], r["claim"], r["source_sentence"] if r["source_sentence"] != r["claim"] else ""] + ev + [""] * (n_ev - len(ev)) + ["", ""])
            strat, s0, s1 = STRATUM_RULE[r["verdict"]]
            key[iid] = {"sample_id": r["sample_id"], "claim_id": r["claim_id"], "nli_verdict": r["verdict"], "answer_membership": strat, "in_S0": s0, "in_S1": s1,
                        "nli_max_entailment": r["nli_max_entailment"], "nli_max_contradiction": r["nli_max_contradiction"]}
    sha = hashlib.sha256(sheet_p.read_bytes()).hexdigest()
    counts = {k: sum(1 for v in key.values() if v["nli_verdict"] == k) for k in STRATUM_RULE}
    key_p.write_text(json.dumps({"meta": {**meta, "seed": seed, "n_items": len(items), "stratum_counts": counts, "sheet_sha256": sha,
                                          "labels": list(LABELS), "created": datetime.now().isoformat(timespec="seconds"),
                                          "note": "KEEP THIS FILE AWAY FROM ANNOTATORS: it contains the strategy membership and the NLI verdict."}, "items": key}, indent=1), encoding="utf-8")
    return sheet_p, key_p, counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", default="alce_asqa_eval_bge-large-en-v1.5")
    ap.add_argument("--start", type=int, default=10)
    ap.add_argument("--n-questions", type=int, default=30)
    ap.add_argument("--cache", default=str(ROOT / "experiments/cache/alce_e2e_nli_cache.pkl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-items", type=int, default=150)
    ap.add_argument("--out-dir", default=str(ROOT / "experiments/annotation"))
    a = ap.parse_args(argv)
    import run_ragtruth_comparison as R
    from src.evidence.models import EvidencePassage
    prep = json.load(open(ROOT / "experiments/runs" / f"alce_e2e_prep_{a.collection}_{a.start}_{a.n_questions}.json", encoding="utf-8"))
    cache = R.LLMCache(Path(a.cache))  # read-only: build_records never calls save()

    def lookup(c, ps):
        passages = [EvidencePassage(evidence_id=x["evidence_id"], doc_title=x["doc_title"], source_text=x["source_text"], retrieval_score=x["retrieval_score"]) for x in ps]
        return cache.data.get(R._key("assess", c["normalized_text"], R._evidence_key(passages)))

    recs = build_records(prep, lookup)
    n_claims = sum(len(p["claims"]) for p in prep.values())
    items = select(recs, a.max_items, a.seed)
    sheet, key, counts = write_files(items, Path(a.out_dir), a.seed, {"source_prep": f"alce_e2e_prep_{a.collection}_{a.start}_{a.n_questions}.json", "verdict_cache": a.cache,
                                                                       "pool_claims": n_claims, "claims_with_cached_verdict": len(recs), "census": len(recs) <= a.max_items,
                                                                       "model_inference": "none (cache lookup only)"})
    print(f"{len(items)} of {len(recs)} claims with cached verdicts ({n_claims} draft claims); strata {counts}")
    print("sheet:", sheet)
    print("key:  ", key)
    return items


if __name__ == "__main__":
    main()
