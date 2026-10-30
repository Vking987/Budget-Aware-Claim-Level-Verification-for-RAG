#!/usr/bin/env python3
"""Summarize a run_ragtruth_comparison.py run into the final benchmark tables (no LLM, no replay).

Keeps POLICY SELECTION quality (did the policy verify the gold-positive claims at all?) separate from VERIFIER accuracy
(did the verifier flag the claims it examined?), and reports paired bootstrap differences between LinUCB and each baseline
on selection recall by resampling whole responses. Comparisons are UNADJUSTED for multiple testing.

usage: python scripts/summarize_ragtruth_benchmark.py <run.json> [--boot 1000] [--out summary.md]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def f(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def ci(c):
    return "n/a" if not c else f"[{c[0]:.3f}, {c[1]:.3f}]"


def load_rows(run_path: Path):
    for cand in (run_path.with_name(run_path.stem + "_claim_rows.json"), run_path.with_name(run_path.stem + "_claim_rows.json.gz")):
        if cand.exists():
            op = gzip.open if cand.suffix == ".gz" else open
            with op(cand, "rt", encoding="utf-8") as fh:
                return json.load(fh)
    raise SystemExit(f"claim rows file not found next to {run_path}")


def ensure_selection(res, rows_all, n_boot=500):
    """Older runs (e.g. the pilot) have no `selection` block: derive it from the saved claim rows, same definitions."""
    from src.evaluation import ragtruth_metrics as M
    for r in res:
        if "selection" in r:
            continue
        counts = M.claim_counts_by_response(rows_all[f"{r['policy']}|b{r['budget_model_calls']}|s{r['seed']}"])
        sel = M.pooled(counts.values(), "sel")
        n_pos, n = sel["tp"] + sel["fn"], sum(sel[k] for k in ("tp", "fp", "fn", "tn"))
        base = n_pos / n if n else None
        r["selection"] = {
            "gold_positives": n_pos, "verified_positives": sel["tp"], "verified_negatives": sel["fp"],
            "selection_recall": sel["recall"], "selection_recall_ci95": M.bootstrap_ci(counts, "sel", "recall", n_boot=n_boot, seed=r["seed"]),
            "coverage": sel["flagged_fraction"], "coverage_ci95": M.bootstrap_ci(counts, "sel", "flagged_fraction", n_boot=n_boot, seed=r["seed"]),
            "lift_over_base_rate": (sel["precision"] / base) if (sel["precision"] is not None and base) else None,
        }


def per_response_arrays(rows, ids):
    idx = {rid: i for i, rid in enumerate(ids)}
    pos = np.zeros(len(ids)); vpos = np.zeros(len(ids))
    for r in rows:
        if r["gold"] == "HALLUCINATED":
            i = idx[r["response_id"]]
            pos[i] += 1
            vpos[i] += r["label"] is not None
    return pos, vpos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_json")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_path = Path(args.run_json)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    man, res = run["manifest"], run["results"]
    proto = man["protocol"]
    rows_all = load_rows(run_path)
    ensure_selection(res, rows_all)
    ids = proto["response_ids"]
    budgets = sorted({r["budget_model_calls"] for r in res})
    policies = []
    for r in res:
        if r["policy"] not in policies:
            policies.append(r["policy"])
    L = []
    L.append(f"# {man['run_id']}\n")
    L.append(f"- verifier: **{proto.get('verifier', 'llm')}**"
             f"{' (gold-label CONTROL: removes verifier error, repair disabled; not a verifier result)' if proto.get('verifier') == 'oracle' else ''}")
    L.append(f"- split: {proto['split']} | sampling: {proto['sampling']} | responses: {proto['n_responses']} | tasks: {proto['task_counts']}")
    r0 = res[0]
    L.append(f"- claims: {r0['n_claims']} (scored gold positives: {r0['selection']['gold_positives']}; excluded ambiguous gold: {r0['n_excluded_gold']})")
    L.append(f"- budgets (model calls per response): {budgets} | bootstrap resamples over responses: {proto.get('n_boot', 'n/a')} (tables), {args.boot} (paired)")
    L.append(f"- components: assessor={man['components']['assessor']['class']} real={man['components']['assessor']['real']}; "
             f"repairer={man['components']['repairer']['class']} real={man['components']['repairer']['real']}; retriever=oracle gold-source chunks")
    L.append(f"- command: `{man['command']}`")
    lt = proto.get("linucb_training")
    if lt:
        L.append(f"- LinUCB trained on {lt['n_responses']} TRAIN-split responses at budget {lt['budget']} ({lt['seconds']}s), frozen at evaluation; checkpoint `{lt['checkpoint']}`")
    cts = man["cts_triage_metrics_policy_independent"]
    L.append(f"- CTS triage alone (policy-independent, threshold 0.5): precision {f(cts.get('cts_triage_precision'))}, "
             f"recall {f(cts.get('cts_triage_recall'))}, F1 {f(cts.get('cts_triage_f1'))}\n")

    # ---- A. selection
    L.append("## A. Policy selection quality (verifier-independent)\n")
    L.append("Selection recall = gold-positive claims that the policy verified at all / gold positives. Coverage = scored claims verified / scored claims. "
             "Lift = share of verified claims that are gold positives / base rate (1.0 = no better than random picking).\n")
    L.append("| Budget | Policy (seed) | Selection recall [95% CI] | Coverage [95% CI] | Lift | Verified pos / pos | Calls |")
    L.append("|---|---|---|---|---|---|---|")
    for b in budgets:
        for r in [x for x in res if x["budget_model_calls"] == b]:
            s = r["selection"]
            L.append(f"| {b} | {r['policy']} ({r['seed']}) | {f(s['selection_recall'])} {ci(s['selection_recall_ci95'])} | "
                     f"{f(s['coverage'])} {ci(s['coverage_ci95'])} | {f(s['lift_over_base_rate'], 2)} | "
                     f"{s['verified_positives']}/{s['gold_positives']} | {r['recorded_cost']['model_calls']} |")
        rnd = [x["selection"]["selection_recall"] for x in res if x["budget_model_calls"] == b and x["policy"] == "Random"]
        if len(rnd) > 1:
            L.append(f"| {b} | Random, mean of {len(rnd)} seeds | {f(float(np.mean(rnd)))} (range {f(min(rnd))}-{f(max(rnd))}) | | | | |")
    L.append("")

    # ---- paired bootstrap on selection recall
    L.append("## B. Paired differences in selection recall: LinUCB minus baseline (95% CI over responses; unadjusted for multiple comparisons)\n")
    L.append("| Budget | Baseline | Difference | 95% CI | CI excludes 0 |")
    L.append("|---|---|---|---|---|")
    rng = np.random.default_rng(0)
    n = len(ids)
    boots = rng.integers(0, n, size=(args.boot, n))
    paired = []
    for b in budgets:
        keys = {k: v for k, v in rows_all.items() if f"|b{b}|" in k}
        arr = {k: per_response_arrays(v, ids) for k, v in keys.items()}
        lin = [k for k in arr if k.startswith("LinUCB|")]
        if not lin:
            continue
        lp, lv = arr[lin[0]]

        def rec(p, v, idxs):
            return v[idxs].sum(axis=1) / np.maximum(p[idxs].sum(axis=1), 1)

        lrec = rec(lp, lv, boots)
        for base in [p for p in policies if p != "LinUCB"]:
            bks = [k for k in arr if k.startswith(base + "|")]
            brec = np.mean([rec(*arr[k], boots) for k in bks], axis=0)
            d = lrec - brec
            point = lv.sum() / max(lp.sum(), 1) - np.mean([arr[k][1].sum() / max(arr[k][0].sum(), 1) for k in bks])
            lo, hi = np.percentile(d, [2.5, 97.5])
            excl = bool(lo > 0 or hi < 0)
            L.append(f"| {b} | {base}{' (mean of seeds)' if len(bks) > 1 else ''} | {point:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {'yes' if excl else 'no'} |")
            paired.append({"budget": b, "baseline": base, "diff": float(point), "ci95": [float(lo), float(hi)], "excludes_zero": excl})
    L.append("")

    # ---- C. verifier-dependent claim metrics
    L.append("## C. Claim-level verifier metrics (depend on the verifier; flags only on claims the policy verified)\n")
    for mode in ("broad", "strict"):
        L.append(f"**{mode}** flag ({'any label except SUPPORTED' if mode == 'broad' else 'CONTRADICTED or CONFLICTED'}); unverified claims count as not flagged.\n")
        L.append("| Budget | Policy (seed) | TP | FP | FN | Precision [CI] | Recall [CI] | F1 [CI] |")
        L.append("|---|---|---|---|---|---|---|---|")
        for b in budgets:
            for r in [x for x in res if x["budget_model_calls"] == b]:
                c = r[f"claim_{mode}"]
                L.append(f"| {b} | {r['policy']} ({r['seed']}) | {c['tp']} | {c['fp']} | {c['fn']} | {f(c['precision'])} {ci(c['precision_ci95'])} | "
                         f"{f(c['recall'])} {ci(c['recall_ci95'])} | {f(c['f1'])} {ci(c['f1_ci95'])} |")
        L.append("")

    # ---- D. cost + repair
    L.append("## D. Cost and repair\n")
    L.append("| Budget | Policy (seed) | Model calls | Calls/response | Tokens | Recorded LLM latency (s) | Wall (s) | Hallucinations flagged per call (broad) | Repairs triggered / re-verified pass / on gold positives |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for b in budgets:
        for r in [x for x in res if x["budget_model_calls"] == b]:
            rc, rp = r["recorded_cost"], r["repair"]
            L.append(f"| {b} | {r['policy']} ({r['seed']}) | {rc['model_calls']} | {rc['model_calls'] / r['n_responses']:.2f} | {rc['total_tokens']} | "
                     f"{rc['latency_seconds']:.0f} | {r['wall_clock_seconds']} | {f(r['hallucinations_detected_per_model_call_broad'], 4)} | "
                     f"{rp['repairs_triggered']} / {rp['reverified_pass']} / {rp['triggered_on_gold_hallucinated']} |")
    L.append("\nNotes: recorded latency of cached verdicts replays the original call's latency; wall-clock is not comparable across settings. "
             "'Re-verified pass' means the same verifier called the revised text SUPPORTED; it is not independent evidence of repair correctness.")
    text = "\n".join(L) + "\n"
    out = Path(args.out) if args.out else run_path.with_name(run_path.stem + "_summary.md")
    out.write_text(text, encoding="utf-8")
    run_path.with_name(run_path.stem + "_paired.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
