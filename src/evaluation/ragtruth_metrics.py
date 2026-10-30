"""Verifier-vs-gold metrics for the RAGTruth policy comparison.

These score what the *verifier* concluded about each claim under a policy's budget; they are
separate from the CTS-triage metrics in ragtruth_adapter.compute_claim_evaluation_metrics.

Per-claim row (dict):
    response_id, gold ('HALLUCINATED' | 'UNFLAGGED' | other),
    label  -> first successful assessment label string, or None if the policy never verified it.

Only gold in {HALLUCINATED, UNFLAGGED} is scored; other gold labels (UNCERTAIN,
IMPLICIT_TRUE_ABSENT) are counted in `n_excluded` and never silently dropped.

A claim is *flagged* when the verifier says:
    strict: CONTRADICTED or CONFLICTED
    broad:  anything except SUPPORTED (adds INSUFFICIENT; RAGTruth counts baseless info as hallucination)
Unverified claims are not flagged (a policy that runs out of budget misses them).
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

STRICT = {"CONTRADICTED", "CONFLICTED"}
SCORED_GOLD = {"HALLUCINATED", "UNFLAGGED"}
_KEYS = ("tp", "fp", "fn", "tn")


def prf(tp: int, fp: int, fn: int) -> Dict[str, Optional[float]]:
    """Precision/recall/F1; None only when the denominator is zero, 0.0 F1 when both are 0."""
    p = tp / (tp + fp) if (tp + fp) else None
    r = tp / (tp + fn) if (tp + fn) else None
    if p is None or r is None:
        f1 = None
    else:
        f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return {"precision": p, "recall": r, "f1": f1}


def _flag(label: Optional[str], mode: str) -> bool:
    if label is None:
        return False
    if mode == "sel":  # selection: was the claim verified at all (verifier verdict irrelevant)?
        return True
    return label in STRICT if mode == "strict" else label != "SUPPORTED"


def claim_counts_by_response(claim_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Per-response confusion counts for both flag definitions (claim level)."""
    out: Dict[str, Dict[str, int]] = defaultdict(lambda: {f"{m}_{k}": 0 for m in ("strict", "broad", "sel") for k in _KEYS})
    for r in claim_rows:
        c = out[r["response_id"]]
        if r["gold"] not in SCORED_GOLD:
            continue
        pos = r["gold"] == "HALLUCINATED"
        for mode in ("strict", "broad", "sel"):
            flagged = _flag(r["label"], mode)
            k = ("tp" if flagged else "fn") if pos else ("fp" if flagged else "tn")
            c[f"{mode}_{k}"] += 1
    return dict(out)


def pooled(counts: Sequence[Dict[str, int]], mode: str) -> Dict[str, Optional[float]]:
    s = {k: sum(c[f"{mode}_{k}"] for c in counts) for k in _KEYS}
    n = sum(s.values())
    # flagged_fraction = (tp + fp) / scored claims. For mode "sel" this is COVERAGE (share of scored claims verified);
    # recall in mode "sel" is SELECTION RECALL (share of gold positives that got verified at all).
    return {**prf(s["tp"], s["fp"], s["fn"]), **s, "flagged_fraction": (s["tp"] + s["fp"]) / n if n else None}


def bootstrap_ci(
    counts_by_response: Dict[str, Dict[str, int]],
    mode: str,
    metric: str,
    n_boot: int = 1000,
    seed: int = 0,
) -> Optional[List[float]]:
    """95% percentile CI over responses (resampling whole responses, not claims)."""
    ids = sorted(counts_by_response)
    if not ids:
        return None
    rng = random.Random(seed)
    vals: List[float] = []
    for _ in range(n_boot):
        sample = [counts_by_response[ids[rng.randrange(len(ids))]] for _ in ids]
        v = pooled(sample, mode)[metric]
        if v is not None:
            vals.append(v)
    if len(vals) < 20:
        return None
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]


def response_level(response_rows: Sequence[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    """Response-level detection. Row: gold (bool, RAGTruth has any hallucination span), labels (list of claim labels)."""
    tp = fp = fn = tn = 0
    for r in response_rows:
        flagged = any(_flag(l, mode) for l in r["labels"])
        if r["gold"]:
            tp, fn = (tp + 1, fn) if flagged else (tp, fn + 1)
        else:
            fp, tn = (fp + 1, tn) if flagged else (fp, tn + 1)
    return {**prf(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def repair_summary(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Repair events: {gold, revision_nonempty, reverified (None|bool)}.

    `reverified_pass` means the same verifier judged the revised text SUPPORTED by the same
    evidence; it is NOT independent proof the repair is factually correct.
    """
    n = len(events)
    nonempty = sum(1 for e in events if e["revision_nonempty"])
    attempted = sum(1 for e in events if e["reverified"] is not None)
    passed = sum(1 for e in events if e["reverified"] is True)
    on_gold = [e for e in events if e["gold"] == "HALLUCINATED"]
    return {
        "repairs_triggered": n,
        "revisions_nonempty": nonempty,
        "reverification_attempted": attempted,
        "reverified_pass": passed,
        "reverified_pass_rate_of_triggered": passed / n if n else None,
        "triggered_on_gold_hallucinated": len(on_gold),
        "reverified_pass_on_gold_hallucinated": sum(1 for e in on_gold if e["reverified"] is True),
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> Optional[List[float]]:
    """Wilson score interval for a binomial proportion k/n (None when n == 0)."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]
