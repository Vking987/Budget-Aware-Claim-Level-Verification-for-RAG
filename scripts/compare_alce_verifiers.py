#!/usr/bin/env python3
"""Paired comparison of two end-to-end ALCE runs that differ ONLY in the verifier (reads saved JSON; same questions, drafts, policies, budget).

usage: python scripts/compare_alce_verifiers.py <run_1b.json> <run_nli.json> --budget 4 --seeds 0,1,2 [--out file.md]

Per policy: seed-averaged per-question metrics for each run, NLI minus 1B paired over questions (percentile bootstrap, 2,000 resamples,
unadjusted). Proxy columns use the lexical 'context-ungrounded' rule (NOT gold hallucination labels).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = ["str_em_final", "n_flagged", "n_repair", "n_reverified_pass", "n_verified_claims_in_answer", "ungrounded_in_final",
           "ungrounded_checked", "ungrounded_flagged", "grounded_checked", "grounded_flagged", "model_calls", "latency_seconds", "total_tokens"]


def load(path, budget, seeds):
    d = json.load(open(path, encoding="utf-8"))
    out = {}
    for key, seed_rows in d["per_question_runs"].items():
        b, pol = key.split("|", 1)
        if int(b) != budget:
            continue
        use = seed_rows if len(seed_rows) == 1 else [seed_rows[s] for s in seeds if s < len(seed_rows)]
        qids = [r["sample_id"] for r in use[0]]
        out[pol] = {"qids": qids, **{m: np.mean([[float(r.get(m) or 0) for r in rows] for rows in use], axis=0) for m in METRICS}, "n_seeds": len(use)}
    return out, d["manifest"]


def boot(diff, n=2000, seed=0):
    if len(diff) < 10:
        return None
    rng = np.random.default_rng(seed)
    m = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def f(x, nd=3):
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_a"), ap.add_argument("run_b")
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    A, ma = load(a.run_a, a.budget, seeds)
    B, mb = load(a.run_b, a.budget, seeds)
    L = [f"# Verifier comparison at budget {a.budget}: NLI minus 1B (paired over questions)", "",
         f"- 1B run `{ma['run_id']}`, NLI run `{mb['run_id']}`; trained/random policies averaged over seeds {seeds}; deterministic policies have one seed.",
         "- Same 30 questions, same drafts, same policies and frozen checkpoints; only the assessor differs (repair stays the 1B model).",
         "- Proxy columns use a lexical context-ungrounded rule, NOT gold labels. STR-EM measures coverage of gold short answers, not the removal of unsupported claims.", "",
         "| Policy | STR-EM 1B | STR-EM NLI | NLI - 1B [95% CI] | Flagged/q 1B -> NLI | Repairs/q (re-verified pass) 1B -> NLI | Ungrounded claims reaching answer/q 1B -> NLI | Proxy TPR 1B -> NLI | Proxy FPR 1B -> NLI |",
         "|---|---|---|---|---|---|---|---|---|"]
    for pol in A:
        if pol not in B or A[pol]["qids"] != B[pol]["qids"]:
            continue
        x, y = A[pol], B[pol]
        diff = y["str_em_final"] - x["str_em_final"]
        ci = boot(diff)
        tpr = lambda r: r["ungrounded_flagged"].sum() / r["ungrounded_checked"].sum() if r["ungrounded_checked"].sum() else float("nan")  # noqa: E731
        fpr = lambda r: r["grounded_flagged"].sum() / r["grounded_checked"].sum() if r["grounded_checked"].sum() else float("nan")  # noqa: E731
        L.append(f"| {pol} | {f(x['str_em_final'].mean())} | {f(y['str_em_final'].mean())} | {diff.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                 f"{f(x['n_flagged'].mean(), 2)} -> {f(y['n_flagged'].mean(), 2)} | {f(x['n_repair'].mean(), 2)} ({f(x['n_reverified_pass'].mean(), 2)}) -> "
                 f"{f(y['n_repair'].mean(), 2)} ({f(y['n_reverified_pass'].mean(), 2)}) | {f(x['ungrounded_in_final'].mean(), 2)} -> {f(y['ungrounded_in_final'].mean(), 2)} | "
                 f"{f(tpr(x))} -> {f(tpr(y))} | {f(fpr(x))} -> {f(fpr(y))} |")
    L += ["", "Cost per question (mean over policies): " + ", ".join(
        f"{n}: calls {np.mean([r['model_calls'].mean() for r in R.values()]):.2f}, tokens {np.mean([r['total_tokens'].mean() for r in R.values()]):.0f}, "
        f"recorded latency {np.mean([r['latency_seconds'].mean() for r in R.values()]):.0f}s" for n, R in (("1B", A), ("NLI", B)))]
    text = "\n".join(L) + "\n"
    out = Path(a.out) if a.out else Path(a.run_b).with_name(Path(a.run_b).stem + f"_vs_1b_b{a.budget}.md")
    out.write_text(text, encoding="utf-8")
    print(text)
    print("wrote", out)


if __name__ == "__main__":
    main()
