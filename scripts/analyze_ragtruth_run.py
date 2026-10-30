#!/usr/bin/env python3
"""Read-only audit of a finished run_ragtruth_comparison.py run (makes NO LLM calls).

Replays every (policy, budget, seed) against a *copy* of the verdict cache, with an assessor/repairer
that raises on a cache miss, so numbers can be re-derived and per-response calls/traces inspected.

Reports, per setting: gold-positive claims, verified positives, TP/FP/FN/TN, P/R/F1 (strict+broad),
coverage, actual calls per response (min/max), and splits gold-positive misses into
  SELECTION failure  (never verified under the budget)  vs  ASSESSOR error (verified, labelled SUPPORTED).
Also: assessor-only confusion over every claim any setting verified, paired bootstrap of policy
differences, and an equivalence check between DeterministicCTS and StaticCTSRanked traces.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_ragtruth_comparison as R  # noqa: E402
from src.controller.mab_policy import ContextualMABPolicy  # noqa: E402
from src.evaluation import ragtruth_metrics as M  # noqa: E402
from src.evaluation.ragtruth_adapter import RAGTruthDatasetAdapter  # noqa: E402


class _NoLLM:
    assessor_id = "none"

    def assess(self, *a, **k):
        raise RuntimeError("cache miss during replay")

    def repair(self, *a, **k):
        raise RuntimeError("cache miss during replay")


def f(v):
    return "n/a" if v is None else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_json")
    ap.add_argument("--cache", default=str(R.DEFAULT_CACHE))
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    run = json.load(open(args.run_json, encoding="utf-8"))
    man, results = run["manifest"], run["results"]
    proto = man["protocol"]
    rows_all = json.load(open(args.run_json.replace(".json", "_claim_rows.json"), encoding="utf-8"))

    tmp = Path(tempfile.mkdtemp()) / "cache.pkl"
    shutil.copy(args.cache, tmp)
    cache = R.LLMCache(tmp)
    assessor, repairer = R.CachedAssessor(_NoLLM(), cache), R.CachedRepairer(_NoLLM(), cache)

    ad = RAGTruthDatasetAdapter()
    ad.load_from_jsonl(R.DATA_DIR / "source_info.jsonl", R.DATA_DIR / "response.jsonl", split=proto["split"])
    by_id = {r.response_id: r for r in ad.responses}
    prepared, _ = R.prepare(ad, [by_id[i] for i in proto["response_ids"]])
    linucb = {}
    for n, kind in proto["linucb_checkpoints_evaluated"].items():
        ck = (proto["linucb_training"] or {}).get("checkpoint")
        if ck:
            linucb[n] = ContextualMABPolicy.load_checkpoint(ck, frozen=True)

    out = {"settings": [], "notes": []}
    traces = {}
    print(f"run: {man['run_id']}  responses={proto['n_responses']}  split={proto['split']}  dry_run={man['dry_run']}")
    hdr = (f"{'policy':<17}{'B':>2} {'s':>1} | {'pos':>3} {'vPos':>4} {'TPb':>3} {'FPb':>3} {'FNb':>3} | "
           f"{'P_b':>5} {'R_b':>5} {'F1_b':>5} | {'P_s':>5} {'R_s':>5} | {'cov':>5} {'calls/resp min-max':>18} | miss-sel miss-assr")
    print(hdr)
    for res in results:
        name, b, seed = res["policy"], res["budget_model_calls"], res["seed"]
        key = f"{name}|b{b}|s{seed}"
        rows = rows_all[key]
        counts = M.claim_counts_by_response(rows)
        pb, ps = M.pooled(counts.values(), "broad"), M.pooled(counts.values(), "strict")
        pos = [r for r in rows if r["gold"] == "HALLUCINATED"]
        v_pos = [r for r in pos if r["label"] is not None]
        sel_fail = sum(1 for r in pos if r["label"] is None)
        assr_fail = sum(1 for r in v_pos if r["label"] == "SUPPORTED")
        scored = [r for r in rows if r["gold"] in M.SCORED_GOLD]
        cov = sum(1 for r in scored if r["label"] is not None) / max(1, len(scored))
        calls = []
        for idx, prep in enumerate(prepared):
            pol = R.make_policy(name, seed, idx, linucb)
            trace, _ = R.run_controller(prep, pol, b, assessor, repairer, f"audit_{key}_{prep['resp'].response_id}")
            tc = trace.total_cost if isinstance(trace.total_cost, dict) else trace.total_cost.to_dict()
            calls.append(tc.get("model_calls") or 0)
            traces.setdefault(key, {})[prep["resp"].response_id] = [(s.action.value, s.claim_id) for s in trace.steps]
        s = {"policy": name, "budget": b, "seed": seed, "gold_pos": len(pos), "verified_pos": len(v_pos),
             "broad": pb, "strict": ps, "coverage_scored": cov, "calls_per_resp_min": min(calls),
             "calls_per_resp_max": max(calls), "calls_total": sum(calls), "recorded_calls_total": res["recorded_cost"]["model_calls"],
             "pos_missed_by_selection": sel_fail, "pos_missed_by_assessor": assr_fail,
             "budget_violations": sum(1 for c in calls if c > b)}
        out["settings"].append(s)
        print(f"{name:<17}{b:>2} {seed:>1} | {len(pos):>3} {len(v_pos):>4} {pb['tp']:>3} {pb['fp']:>3} {pb['fn']:>3} | "
              f"{f(pb['precision']):>5} {f(pb['recall']):>5} {f(pb['f1']):>5} | {f(ps['precision']):>5} {f(ps['recall']):>5} | "
              f"{cov:>5.2f} {min(calls):>8}-{max(calls):<9} | {sel_fail:>8} {assr_fail:>9}")
        assert sum(calls) == res["recorded_cost"]["model_calls"], (key, sum(calls), res["recorded_cost"]["model_calls"])

    # ---- assessor-only confusion: first verdict any setting produced for each claim
    verdict = {}
    for key, rows in rows_all.items():
        for r in rows:
            if r["label"] is not None:
                verdict.setdefault((r["response_id"], r["claim_id"]), (r["gold"], r["label"]))
    conf = collections.Counter(v for v in verdict.values() if v[0] in M.SCORED_GOLD)
    n_scored_total = sum(1 for r in next(iter(rows_all.values())) if r["gold"] in M.SCORED_GOLD)
    n_pos_total = sum(1 for r in next(iter(rows_all.values())) if r["gold"] == "HALLUCINATED")
    tp_b = sum(c for (g, l), c in conf.items() if g == "HALLUCINATED" and l != "SUPPORTED")
    pos_ver = sum(c for (g, l), c in conf.items() if g == "HALLUCINATED")
    fp_b = sum(c for (g, l), c in conf.items() if g == "UNFLAGGED" and l != "SUPPORTED")
    neg_ver = sum(c for (g, l), c in conf.items() if g == "UNFLAGGED")
    print(f"\nASSESSOR-ONLY (any claim verified by any setting): verified {pos_ver}/{n_pos_total} gold positives, "
          f"{neg_ver}/{n_scored_total - n_pos_total} gold negatives")
    for k in sorted(conf, key=str):
        print("   ", k, conf[k])
    print(f"   assessor recall on verified positives (broad) = {tp_b}/{pos_ver}; "
          f"false-flag rate on verified negatives (broad) = {fp_b}/{neg_ver}")
    out["assessor_only"] = {"verified_pos": pos_ver, "total_pos": n_pos_total, "tp_broad": tp_b,
                            "verified_neg": neg_ver, "fp_broad": fp_b, "confusion": {f"{g}|{l}": c for (g, l), c in conf.items()}}

    # ---- paired bootstrap of recall(broad) differences vs LinUCB, per budget (random: mean of seeds)
    def per_resp_tp_fn(key):
        c = M.claim_counts_by_response(rows_all[key])
        return c

    print("\nPAIRED BOOTSTRAP over responses: recall(broad) difference, LinUCB minus baseline (95% CI)")
    budgets = sorted({s["budget"] for s in out["settings"]})
    ids = proto["response_ids"]
    rng = random.Random(0)
    boots = [[rng.randrange(len(ids)) for _ in ids] for _ in range(args.boot)]
    out["paired"] = []
    for b in budgets:
        lk = [k for k in rows_all if k.startswith("LinUCB|") and f"|b{b}|" in k]
        if not lk:
            continue
        lc = per_resp_tp_fn(lk[0])
        for base in ("StaticFIFO", "StaticCTSRanked", "DeterministicCTS", "VerifyAll", "Random"):
            bks = [k for k in rows_all if k.startswith(base + "|") and f"|b{b}|" in k]
            if not bks:
                continue
            bcs = [per_resp_tp_fn(k) for k in bks]
            diffs = []
            for idxs in boots:
                lr = M.pooled([lc.get(ids[i], {}) or {f"broad_{k}": 0 for k in M._KEYS} for i in idxs], "broad")["recall"]
                br = []
                for bc in bcs:
                    v = M.pooled([bc.get(ids[i], {}) or {f"broad_{k}": 0 for k in M._KEYS} for i in idxs], "broad")["recall"]
                    if v is not None:
                        br.append(v)
                if lr is not None and br:
                    diffs.append(lr - sum(br) / len(br))
            diffs.sort()
            if len(diffs) > 40:
                lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]
                point = M.pooled(lc.values(), "broad")["recall"] - sum(M.pooled(bc.values(), "broad")["recall"] for bc in bcs) / len(bcs)
                print(f"   B={b}  LinUCB - {base:<16} = {point:+.3f}  [{lo:+.3f}, {hi:+.3f}]  ({len(bcs)} run(s))")
                out["paired"].append({"budget": b, "baseline": base, "diff": point, "ci95": [lo, hi]})

    # ---- selection quality (independent of the verifier): did the policy spend budget on gold positives?
    ids_all = proto["response_ids"]
    def sel_counts(key):
        d = {i: [0, 0, 0] for i in ids_all}  # pos, verified_pos, verified_scored
        for r in rows_all[key]:
            if r["gold"] not in M.SCORED_GOLD:
                continue
            c = d[r["response_id"]]
            if r["gold"] == "HALLUCINATED":
                c[0] += 1
                c[1] += r["label"] is not None
            c[2] += r["label"] is not None
        return d
    total_scored = sum(1 for r in next(iter(rows_all.values())) if r["gold"] in M.SCORED_GOLD)
    n_pos_all = sum(1 for r in next(iter(rows_all.values())) if r["gold"] == "HALLUCINATED")
    base_rate = n_pos_all / total_scored
    def cov_of(d, idxs):
        pos = sum(d[ids_all[i]][0] for i in idxs); vp = sum(d[ids_all[i]][1] for i in idxs)
        return vp / pos if pos else None
    print(f"\nSELECTION QUALITY (verifier-independent): positive coverage = gold positives verified / {n_pos_all}; "
          f"lift = (verified positives / verified claims) / base rate {base_rate:.3f}")
    out["selection"] = []
    rng2 = random.Random(1)
    boots2 = [[rng2.randrange(len(ids_all)) for _ in ids_all] for _ in range(args.boot)]
    for b in budgets:
        keys_b = [k for k in rows_all if f"|b{b}|" in k]
        pols = sorted({k.split("|")[0] for k in keys_b})
        d_by = {pol: [sel_counts(k) for k in keys_b if k.startswith(pol + "|")] for pol in pols}
        for pol in pols:
            cov = sum(sum(c[i][1] for i in ids_all) for c in d_by[pol]) / (len(d_by[pol]) * n_pos_all)
            vsc = sum(sum(c[i][2] for i in ids_all) for c in d_by[pol]) / len(d_by[pol])
            vp = sum(sum(c[i][1] for i in ids_all) for c in d_by[pol]) / len(d_by[pol])
            lift = (vp / vsc) / base_rate if vsc else None
            diffs = []
            if pol != "LinUCB" and "LinUCB" in d_by:
                for idxs in boots2:
                    l = cov_of(d_by["LinUCB"][0], idxs)
                    bs = [cov_of(c, idxs) for c in d_by[pol]]
                    bs = [x for x in bs if x is not None]
                    if l is not None and bs:
                        diffs.append(l - sum(bs) / len(bs))
                diffs.sort()
            ci = [diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]] if len(diffs) > 40 else None
            dtxt = (f"  LinUCB-this = {cov_of(d_by['LinUCB'][0], range(len(ids_all))) - cov:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "")
            print(f"   B={b} {pol:<17} coverage={cov:.3f} ({vp:.1f}/{n_pos_all})  lift={f(lift)}  runs={len(d_by[pol])}{dtxt}")
            out["selection"].append({"budget": b, "policy": pol, "positive_coverage": cov, "lift": lift, "runs": len(d_by[pol]),
                                     "linucb_minus_this_ci95": ci})

    # ---- CTS-policy equivalence
    print("\nDeterministicCTS vs StaticCTSRanked trace comparison")
    out["cts_equivalence"] = []
    for b in budgets:
        a, c = traces.get(f"DeterministicCTS|b{b}|s0"), traces.get(f"StaticCTSRanked|b{b}|s0")
        if not a or not c:
            continue
        diff_resp = [rid for rid in a if a[rid] != c[rid]]
        tie_resp = []
        for prep in prepared:
            sc = [round(x.triage_score or 0.0, 9) for x in prep["claims"]]
            if len(sc) != len(set(sc)):
                tie_resp.append(prep["resp"].response_id)
        print(f"   B={b}: responses with different action sequences = {len(diff_resp)}/{len(a)}; "
              f"responses containing tied CTS scores = {len(tie_resp)}/{len(prepared)}")
        out["cts_equivalence"].append({"budget": b, "different": len(diff_resp), "total": len(a), "tied_cts_responses": len(tie_resp),
                                       "example_diff_response": diff_resp[:1]})
    print(f"\nreplay cache: hits={cache.hits} misses={cache.misses}")
    out["replay_cache"] = {"hits": cache.hits, "misses": cache.misses}
    outp = Path(args.run_json.replace(".json", "_audit.json"))
    json.dump(out, open(outp, "w", encoding="utf-8"), indent=2, default=str)
    print("wrote", outp)


if __name__ == "__main__":
    main()
