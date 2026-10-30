#!/usr/bin/env python3
"""Does LinUCB add value beyond simple feature-weighted ranking? Oracle-verifier control on the held-out RAGTruth test split (no LLM).

Question design (fixed before running):
  * metric: selection recall (gold positives verified at all) at budgets {2,4,8}, repair disabled, gold-label oracle verifier;
  * conditions: NORMAL, and CTS-BLIND (every claim gets the same CTS, which removes CTS from LinUCB's features AND its tie-break, the confound
    found in the ablation);
  * 5 training seeds, each a different seeded stratified sample of 1,000 TRAIN-split responses (test never used for fitting; asserted);
  * LinUCB (normal): the 5 frozen checkpoints from the ablation run; LinUCB (blind): retrained per seed on CTS-blind train claims;
  * controls fit on the same TRAIN samples: logistic feature rankers (CTS+type, type only, CTS+type+relative position; they use GOLD labels, so
    they are supervised references, not fair competitors on information), TypeThompson (online on train, then frozen), PositionMatchedRandom
    (position profile from LinUCB's own TRAIN-time choices), LexicalSupport (no training), Random x3, FIFO, CTS-ranked.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import policy_controls as PC  # noqa: E402
import run_ragtruth_comparison as R  # noqa: E402
import run_ragtruth_noisy_simulation as N  # noqa: E402
from src.controller.mab_policy import ContextualMABPolicy  # noqa: E402
from src.controller.policy import ControllerPolicyConfig  # noqa: E402
from src.evaluation import ragtruth_metrics as M  # noqa: E402
from src.evaluation.ragtruth_adapter import RAGTruthDatasetAdapter  # noqa: E402

ABLATION_TS = "20261029_002617"  # checkpoints of the earlier 5-seed ablation (feature-complete LinUCB)
REGISTRY: Dict[str, Callable[[int, int, bool], Any]] = {}
_orig_make_policy = R.make_policy


def _make_policy(name, seed, idx, linucb, enable_repair=True):
    if name in REGISTRY:
        return REGISTRY[name](seed, idx, enable_repair)
    return _orig_make_policy(name, seed, idx, linucb, enable_repair)


R.make_policy = _make_policy  # run_setting looks the factory up at call time


def cfg(enable_repair: bool) -> ControllerPolicyConfig:
    return ControllerPolicyConfig(enable_repair=enable_repair)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--budgets", default="2,4,8")
    ap.add_argument("--train-responses", type=int, default=1000)
    ap.add_argument("--train-budget", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--ablation-ts", default=ABLATION_TS)
    ap.add_argument("--fixture", default=None)
    ap.add_argument("--max-responses", type=int, default=0)
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
    test, _ = R.prepare(ad, resp)
    test_blind = PC.blind(test)
    ids = [p["resp"].response_id for p in test]
    tr = load("train")
    repairer = R._CostedMockRepairer()
    oracle = lambda prep: R.OracleAssessor(prep["gold"])  # noqa: E731
    rng = np.random.default_rng(0)
    boots = rng.integers(0, len(ids), size=(args.n_boot, len(ids)))
    t0 = time.monotonic()

    # ------------------------------------------------------------------ per-seed fitting on TRAIN claims only
    per_seed: Dict[int, Dict[str, Any]] = {}
    for s in seeds:
        tr_prep, _ = R.prepare(tr, R.stratified_sample(tr.responses, tr.sources, args.train_responses, s + 1000))
        assert not ({p["resp"].response_id for p in tr_prep} & set(ids)), "train/eval overlap"
        tr_blind = PC.blind(tr_prep)
        d: Dict[str, Any] = {"n_train": len(tr_prep)}
        ck = ROOT / "experiments" / "runs" / f"ragtruth_ablation_full_seed{s}_{args.ablation_ts}.json"
        if ck.exists():
            d["LinUCB"] = ContextualMABPolicy.load_checkpoint(ck, frozen=True)
            d["LinUCB_source"] = f"checkpoint {ck.name}"
        else:  # fixture / smoke fallback
            d["LinUCB"] = R.train_linucb(tr_prep, None, repairer, args.train_budget, s, out_dir / f"ragtruth_ctl_linucb_seed{s}_{ts}.json",
                                         assessor_factory=oracle, enable_repair=False)
            d["LinUCB"].freeze()
            d["LinUCB_source"] = "retrained (checkpoint missing)"
        d["LinUCB"].config.enable_repair = False
        d["LinUCB_blind"] = R.train_linucb(tr_blind, None, repairer, args.train_budget, s, out_dir / f"ragtruth_ctl_linucb_blind_seed{s}_{ts}.json",
                                           assessor_factory=oracle, enable_repair=False)
        d["LinUCB_blind"].freeze()
        d["FR"] = {"FR_cts_type": PC.fit_feature_ranker(tr_prep, ["cts", "type"]), "FR_type": PC.fit_feature_ranker(tr_prep, ["type"]),
                   "FR_cts_type_pos": PC.fit_feature_ranker(tr_prep, ["cts", "type", "pos"])}
        d["FR_blind"] = {"FR_type": PC.fit_feature_ranker(tr_blind, ["type"]), "FR_type_pos": PC.fit_feature_ranker(tr_blind, ["type", "pos"])}
        ts_pol = PC.TypeThompson(seed=s, config=cfg(False), name="TypeThompson")  # online on TRAIN claims (budget 4), then frozen
        for prep in tr_prep:
            R.run_controller(prep, ts_pol, args.train_budget, R.OracleAssessor(prep["gold"]), repairer, f"ts_train_{s}")
        d["TS"] = ts_pol
        d["profiles"], d["profiles_blind"] = {}, {}
        for b in budgets:  # LinUCB's own TRAIN-time position profile at each budget
            _, rows = R.run_setting("LinUCB", b, s, tr_prep, {"LinUCB": d["LinUCB"]}, None, repairer, None, assessor_factory=oracle, enable_repair=False, n_boot=0)
            d["profiles"][b] = PC.estimate_position_profile(rows)
            _, rows_b = R.run_setting("LinUCB", b, s, tr_blind, {"LinUCB": d["LinUCB_blind"]}, None, repairer, None, assessor_factory=oracle, enable_repair=False, n_boot=0)
            d["profiles_blind"][b] = PC.estimate_position_profile(rows_b)
        per_seed[s] = d
        print(f"seed {s}: fitted LinUCB(blind), 3+2 rankers, TypeThompson, position profiles ({time.monotonic() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------ evaluation on the held-out test split
    arr: Dict[tuple, Dict[str, np.ndarray]] = {}
    meta: Dict[tuple, float] = {}

    def run(cond, kind, b, s, prepared, linucb=None, name=None):
        res, rows = R.run_setting(name or kind, b, s, prepared, linucb or {}, None, repairer, None, assessor_factory=oracle, enable_repair=False, n_boot=0)
        arr[(cond, b, kind, s)] = N.accumulate([M.claim_counts_by_response(rows)], ids)
        idx, seen = [], {}
        for r in rows:
            seen[r["response_id"]] = seen.get(r["response_id"], -1) + 1
            if r["label"] is not None:
                idx.append(seen[r["response_id"]])
        meta[(cond, b, kind, s)] = float(np.mean(idx)) if idx else float("nan")

    for b in budgets:
        for s in seeds:
            d = per_seed[s]
            run("normal", "LinUCB", b, s, test, {"LinUCB": d["LinUCB"]}, "LinUCB")
            for fr_name, model in d["FR"].items():
                REGISTRY[f"{fr_name}|{s}"] = (lambda m, n: (lambda seed, idx, er: PC.FeatureRanker(m, n, cfg(er))))(model, fr_name)
                run("normal", fr_name, b, s, test, name=f"{fr_name}|{s}")
            REGISTRY[f"PMR|{s}|{b}"] = (lambda prof: (lambda seed, idx, er: PC.PositionMatchedRandom(prof, seed * 1_000_003 + idx, cfg(er))))(d["profiles"][b])
            run("normal", "PositionMatchedRandom", b, s, test, name=f"PMR|{s}|{b}")
            REGISTRY[f"TS|{s}"] = (lambda tsp: (lambda seed, idx, er: tsp.frozen_copy(seed * 1_000_003 + idx)))(d["TS"])
            run("normal", "TypeThompson", b, s, test, name=f"TS|{s}")
            # CTS-blind condition
            run("blind", "LinUCB_blind", b, s, test_blind, {"LinUCB": d["LinUCB_blind"]}, "LinUCB")
            for fr_name, model in d["FR_blind"].items():
                REGISTRY[f"B{fr_name}|{s}"] = (lambda m, n: (lambda seed, idx, er: PC.FeatureRanker(m, n, cfg(er))))(model, fr_name)
                run("blind", fr_name, b, s, test_blind, name=f"B{fr_name}|{s}")
            REGISTRY[f"BPMR|{s}|{b}"] = (lambda prof: (lambda seed, idx, er: PC.PositionMatchedRandom(prof, seed * 1_000_003 + idx, cfg(er))))(d["profiles_blind"][b])
            run("blind", "PositionMatchedRandom", b, s, test_blind, name=f"BPMR|{s}|{b}")
        REGISTRY["LexicalSupport"] = lambda seed, idx, er: PC.LexicalSupportPolicy(cfg(er))
        for cond, data in (("normal", test), ("blind", test_blind)):
            run(cond, "StaticFIFO", b, 0, data)
            run(cond, "StaticCTSRanked", b, 0, data)
            for rs in (0, 1, 2):
                run(cond, "Random", b, rs, data)
        run("normal", "LexicalSupport", b, 0, test, name="LexicalSupport")
        print(f"budget {b} evaluated ({time.monotonic() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------ summarise
    def pooled(cond, b, kind):
        keys = [k for k in arr if k[0] == cond and k[1] == b and k[2] == kind]
        return {k: sum(arr[kk][k] for kk in keys) for k in N.ARR_KEYS}, keys

    kinds = {"normal": ["LinUCB", "FR_cts_type", "FR_type", "FR_cts_type_pos", "PositionMatchedRandom", "TypeThompson", "LexicalSupport",
                        "Random", "StaticFIFO", "StaticCTSRanked"],
             "blind": ["LinUCB_blind", "FR_type", "FR_type_pos", "PositionMatchedRandom", "Random", "StaticFIFO", "StaticCTSRanked"]}
    out_c: Dict[str, Any] = {}
    for cond, klist in kinds.items():
        out_b = {}
        for b in budgets:
            table, bv = {}, {}
            for kind in klist:
                A, keys = pooled(cond, b, kind)
                point = float(N.metric_values(A)["selection_recall"])
                boot = N.metric_values(A, boots)["selection_recall"]
                per_run = [float(N.metric_values(arr[k])["selection_recall"]) for k in keys]
                bv[kind] = boot
                table[kind] = {"selection_recall": point, "ci95": N.ci(boot), "min": min(per_run), "max": max(per_run), "n_runs": len(keys),
                               "mean_verified_index": float(np.nanmean([meta[k] for k in keys]))}
            ref = "LinUCB" if cond == "normal" else "LinUCB_blind"
            paired = {}
            for a in ([ref, "FR_cts_type"] if cond == "normal" else [ref, "FR_type"]):
                for c in klist:
                    if c != a:
                        paired[f"{a} - {c}"] = {"diff": table[a]["selection_recall"] - table[c]["selection_recall"], "ci95": N.ci(bv[a] - bv[c])}
            out_b[str(b)] = {"policies": table, "paired": paired}
        out_c[cond] = out_b
    out = {"manifest": {"run_id": f"ragtruth_policy_controls_{ts}{('_' + args.tag) if args.tag else ''}", "command": " ".join(sys.argv),
                        "condition": "oracle (gold-label) verifier control; repair disabled; held-out test split; selection recall only",
                        "n_test_responses": len(ids), "training_seeds": seeds, "train_responses_per_seed": args.train_responses,
                        "linucb_sources": {s: per_seed[s]["LinUCB_source"] for s in seeds}, "n_boot": args.n_boot,
                        "ranker_weights": {str(s): {k: v["weights"] for k, v in {**per_seed[s]["FR"], **{"blind_" + k: v for k, v in per_seed[s]["FR_blind"].items()}}.items()} for s in seeds},
                        "type_thompson_posterior_seed0": per_seed[seeds[0]]["TS"].posterior,
                        "note": "Feature rankers use gold labels on train claims (supervised references). TypeThompson is an adaptation of D2R-RAG's context-free TS to claim-type arms."},
           "conditions": out_c}
    jp = out_dir / f"{out['manifest']['run_id']}.json"
    jp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    f = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
    L = [f"# {out['manifest']['run_id']}", "", f"Oracle control, held-out test ({len(ids)} responses), seeds {seeds}. Selection recall.", ""]
    for cond, ob in out_c.items():
        L += [f"# Condition: {cond}", ""]
        for b, bo in ob.items():
            L += [f"## Budget {b}", "", "| Policy | Selection recall [95% CI] | min-max over runs | mean index of verified |", "|---|---|---|---|"]
            for k, v in bo["policies"].items():
                ci = v["ci95"]
                L.append(f"| {k} | {f(v['selection_recall'])} [{f(ci[0]) if ci else 'n/a'}, {f(ci[1]) if ci else 'n/a'}] | {f(v['min'])}-{f(v['max'])} (n={v['n_runs']}) | {v['mean_verified_index']:.2f} |")
            L += ["", "| Paired difference | Value | 95% CI |", "|---|---|---|"]
            for k, v in bo["paired"].items():
                ci = v["ci95"]
                L.append(f"| {k} | {v['diff']:+.3f} | [{f(ci[0]) if ci else 'n/a'}, {f(ci[1]) if ci else 'n/a'}] |")
            L.append("")
    (out_dir / f"{out['manifest']['run_id']}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {jp}")
    return out


if __name__ == "__main__":
    main()
