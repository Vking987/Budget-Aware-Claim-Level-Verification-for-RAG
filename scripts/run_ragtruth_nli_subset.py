#!/usr/bin/env python3
"""Budget-2 policy comparison with the DeBERTa NLI verifier on a POSITIVE-ENRICHED RAGTruth test subset.

Sample: --n responses drawn with a seeded RNG from the TEST responses that contain >= 1 gold-HALLUCINATED claim (gold labels define the sample only;
policies and verifier never see them). Enriched: precision/prevalence here are NOT representative of the test split; recall is unaffected by dropping
positive-free responses because the budget is per response.
Policies (budget 2 calls/response, repair disabled): LinUCB = the existing frozen oracle-trained checkpoint (no online update), StaticCTSRanked,
Random (several seeds). Every NLI verdict is memoized on (claim text, evidence) in a NEW cache, so the same verdict is reused across policies.
--estimate does not load the model: it counts the unique claims the policies would verify under the lexical verifier (cost planning only).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_ragtruth_cheap_verifier as C  # noqa: E402
import run_ragtruth_comparison as R  # noqa: E402
from src.controller.mab_policy import ContextualMABPolicy  # noqa: E402
from src.evaluation import ragtruth_metrics as M  # noqa: E402
from src.evaluation.ragtruth_adapter import RAGTruthDatasetAdapter  # noqa: E402

CACHE_PATH = ROOT / "experiments" / "cache" / "ragtruth_nli_subset_cache.pkl"
KEYS = ("tp", "fn", "fp", "tn")


class SavingCachedAssessor(R.CachedAssessor):
    """CachedAssessor that checkpoints the cache every `every` misses so a long run can resume."""

    def __init__(self, inner, cache, every=20):
        super().__init__(inner, cache)
        self.every, self._last = every, 0

    def assess(self, claim, passages):
        res = super().assess(claim, passages)
        if self.cache.misses - self._last >= self.every:
            self._last = self.cache.misses
            self.cache.save()
        return res


def select_subset(adapter, n, seed):
    resp = sorted((r for r in adapter.responses if r.source_id in adapter.sources), key=lambda r: r.response_id)
    prepared, _ = R.prepare(adapter, resp)
    pos = [p for p in prepared if "HALLUCINATED" in p["gold"].values()]
    chosen = random.Random(seed).sample(pos, min(n, len(pos)))
    return sorted(chosen, key=lambda p: p["resp"].response_id), len(prepared), len(pos)


def arrays(counts, ids, mode):
    A = {k: np.zeros(len(ids)) for k in KEYS}
    for i, rid in enumerate(ids):
        c = counts.get(rid)
        if c:
            for k in KEYS:
                A[k][i] = c[f"{mode}_{k}"]
    return A


def metrics(A, idx=None):
    S = (lambda k: A[k].sum()) if idx is None else (lambda k: A[k][idx].sum(axis=1))
    tp, fn, fp = S("tp"), S("fn"), S("fp")
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        prec = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    return {"recall": rec, "precision": prec, "f1": f1}


def ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) >= 20 else None


def fl(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=2)
    ap.add_argument("--random-seeds", default="0,1,2")
    ap.add_argument("--linucb", default=str(ROOT / "experiments/runs/ragtruth_linucb_train1000_20261028_235615.json"))
    ap.add_argument("--estimate", action="store_true", help="count unique claims to verify with the lexical verifier; no NLI model is loaded")
    ap.add_argument("--cache", default=str(CACHE_PATH))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", default=str(R.RUN_DIR))
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    t0 = time.monotonic()
    ad = RAGTruthDatasetAdapter()
    ad.load_from_jsonl(R.DATA_DIR / "source_info.jsonl", R.DATA_DIR / "response.jsonl", split="test")
    prepared, n_test_prepared, n_pos_resp = select_subset(ad, a.n, a.seed)
    ids = [p["resp"].response_id for p in prepared]
    rseeds = [int(x) for x in a.random_seeds.split(",")]
    linucb = {"LinUCB": ContextualMABPolicy.load_checkpoint(a.linucb, frozen=True)}
    names = [("LinUCB", a.seed), ("StaticCTSRanked", a.seed)] + [("Random", s) for s in rseeds]
    repairer = R._CostedMockRepairer()  # never called: repair disabled
    claims_total = sum(len(p["claims"]) for p in prepared)
    task_mix = {t: sum(1 for p in prepared if p["task"] == t) for t in sorted({p["task"] for p in prepared})}
    print(f"subset: {len(ids)} of {n_pos_resp} positive-bearing test responses ({n_test_prepared} test responses with claims); {claims_total} claims; tasks {task_mix}", flush=True)

    if a.estimate:
        uniq, per = set(), {}
        asr = lambda prep: C.LexicalVerifier(0.4)  # noqa: E731
        for name, seed in names:
            res, rows = R.run_setting(name, a.budget, seed, prepared, linucb, None, repairer, None, assessor_factory=asr, enable_repair=False, n_boot=0)
            v = {(r["response_id"], r["claim_id"]) for r in rows if r["label"] is not None}
            per[f"{name}({seed})"] = len(v)
            uniq |= v
        print(f"ESTIMATE (lexical-verifier trajectories): per policy {per}; unique claims {len(uniq)} (NLI trajectories may differ slightly)")
        return {"unique_claims": len(uniq), "per_policy": per}

    from src.evidence.nli_assessor import NLIAssessor
    nli = NLIAssessor()
    cache = R.LLMCache(Path(a.cache))
    pre_cached = len(cache.data)
    shared = SavingCachedAssessor(nli, cache)
    factory = lambda prep: shared  # noqa: E731
    boots = np.random.default_rng(a.seed).integers(0, len(ids), size=(a.n_boot, len(ids)))
    runs, rows_by = {}, {}
    for name, seed in names:
        label = f"{name}({seed})" if name == "Random" else name
        h0, m0, t1 = cache.hits, cache.misses, time.monotonic()
        res, rows = R.run_setting(name, a.budget, seed, prepared, linucb, None, repairer, None, assessor_factory=factory, enable_repair=False, n_boot=0)
        cache.save()
        assert res["recorded_cost"]["model_calls"] <= a.budget * len(ids), "budget exceeded"
        assert res["assessor_errors"] == 0, "NLI errors"
        runs[label] = {"calls_consumed": res["recorded_cost"]["model_calls"], "cache_hits": cache.hits - h0, "cache_misses": cache.misses - m0,
                       "wall_s": round(time.monotonic() - t1, 1), "claims_verified": res["n_claims_verified"], "action_counts": res["action_counts"]}
        rows_by[label] = rows
        print(f"{label}: calls {runs[label]['calls_consumed']} hits {runs[label]['cache_hits']} misses {runs[label]['cache_misses']} {runs[label]['wall_s']}s", flush=True)
    seen = {tuple((r["response_id"], r["claim_id"]) for r in rows) for rows in rows_by.values()}
    assert len(seen) == 1, "policies saw different claims"

    results: Dict[str, Any] = {}
    for mode in ("broad", "strict"):  # broad (flag = not SUPPORTED) is the primary definition, declared before the run
        arr, tab = {}, {}
        for lab, rows in rows_by.items():
            counts = M.claim_counts_by_response(rows)
            arr[lab] = arrays(counts, ids, mode)
            pt = {k: fl(v) for k, v in metrics(arr[lab]).items()}
            bt = metrics(arr[lab], boots)
            A = arr[lab]
            tab[lab] = {**pt, **{f"{k}_ci95": ci(v) for k, v in bt.items()}, "tp": int(A["tp"].sum()), "fn": int(A["fn"].sum()), "fp": int(A["fp"].sum()),
                        "tn": int(A["tn"].sum()), "calls": runs[lab]["calls_consumed"],
                        "selection_recall": None}
        rnd = [l for l in tab if l.startswith("Random(")]
        mean_rec = np.mean([metrics(arr[l], boots)["recall"] for l in rnd], axis=0)
        mean_prec = np.nanmean([metrics(arr[l], boots)["precision"] for l in rnd], axis=0)
        tab["Random(mean)"] = {"recall": float(np.mean([tab[l]["recall"] for l in rnd])), "recall_ci95": ci(mean_rec),
                               "precision": float(np.nanmean([tab[l]["precision"] for l in rnd]))}
        lin = metrics(arr["LinUCB"], boots)
        paired = {}
        for base in ["StaticCTSRanked", "Random(mean)"] + rnd:
            bm = mean_rec if base == "Random(mean)" else metrics(arr[base], boots)["recall"]
            bp = tab[base]["recall"]
            paired[f"LinUCB - {base} [recall]"] = {"diff_pp": 100 * (tab["LinUCB"]["recall"] - bp), "ci95_pp": [100 * x for x in ci(lin["recall"] - bm)]}
        results[mode] = {"policies": tab, "paired": paired}

    # selection recall (was a gold positive verified at all) per policy
    sel = {}
    for lab, rows in rows_by.items():
        c = M.claim_counts_by_response(rows)
        tp = sum(v["sel_tp"] for v in c.values())
        fn = sum(v["sel_fn"] for v in c.values())
        sel[lab] = {"verified_positives": tp, "gold_positives": tp + fn, "selection_recall": tp / (tp + fn)}

    # NLI verifier on the union of verified claims (policy-selected, NOT a random claim sample)
    claim_meta = {(p["resp"].response_id, c.claim_id): (c, p) for p in prepared for c in p["claims"]}
    union = {}
    for rows in rows_by.values():
        for r in rows:
            if r["label"] is not None:
                union[(r["response_id"], r["claim_id"])] = r
    conf = {"broad": dict.fromkeys(KEYS, 0), "strict": dict.fromkeys(KEYS, 0)}
    labels: Dict[str, Dict[str, int]] = {"HALLUCINATED": {}, "UNFLAGGED": {}, "AMBIGUOUS": {}}
    xs = []
    for (rid, cid), r in union.items():
        g = r["gold"] if r["gold"] in M.SCORED_GOLD else "AMBIGUOUS"
        labels[g][r["label"]] = labels[g].get(r["label"], 0) + 1
        if g == "AMBIGUOUS":
            continue
        for mode in ("broad", "strict"):
            fl_ = M._flag(r["label"], mode)
            conf[mode][("tp" if fl_ else "fn") if g == "HALLUCINATED" else ("fp" if fl_ else "tn")] += 1
        c, p = claim_meta[(rid, cid)]
        ev = " ".join(x.source_text for x in p["evidence"][cid])
        xs.append({"gold": g, "flag": r["label"] != "SUPPORTED", "cts": c.triage_score, "u": C.unsupported_fraction(c.normalized_text, ev)})
    ver = {"n_claims": len(union), "n_scored": len(xs), "label_counts_by_gold": labels, "verifier_positive_rate_broad": float(np.mean([x["flag"] for x in xs])), "note": "claims the policies chose to verify; policy-selected and positive-enriched, so precision/prevalence are not test-split values"}
    for mode in ("broad", "strict"):
        c_ = conf[mode]
        tp, fn, fp, tn = (c_[k] for k in KEYS)
        ver[mode] = {**c_, "tpr": tp / (tp + fn) if tp + fn else None, "tpr_ci95": M.wilson_ci(tp, tp + fn), "fpr": fp / (fp + tn) if fp + tn else None,
                     "fpr_ci95": M.wilson_ci(fp, fp + tn), "precision_enriched": tp / (tp + fp) if tp + fp else None, "f1_enriched": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}
    fl_arr = np.array([x["flag"] for x in xs], float)
    gold_arr = np.array([x["gold"] == "HALLUCINATED" for x in xs], float)
    cts_arr = np.array([x["cts"] if x["cts"] is not None else np.nan for x in xs])
    u_arr = np.array([x["u"] for x in xs])
    ok = ~np.isnan(cts_arr)
    ver["feature_overlap"] = {
        "corr_nli_flag_vs_lexical_unsupported_fraction": float(np.corrcoef(fl_arr, u_arr)[0, 1]),
        "corr_nli_flag_vs_cts": float(np.corrcoef(fl_arr[ok], cts_arr[ok])[0, 1]),
        "corr_gold_vs_cts": float(np.corrcoef(gold_arr[ok], cts_arr[ok])[0, 1]),
        "corr_gold_vs_nli_flag": float(np.corrcoef(gold_arr, fl_arr)[0, 1]),
        "note": "Pearson correlations over verified, scored claims; descriptive only (LinUCB features: CTS, claim-type risk, evidence coverage, ...)"}

    out = {"manifest": {
        "run_id": f"ragtruth_nli_subset_{ts}{('_' + a.tag) if a.tag else ''}", "command": " ".join(sys.argv), "split": "test",
        "sampling": f"seeded random.Random({a.seed}).sample of {a.n} from the {n_pos_resp} test responses (of {n_test_prepared} with claims) that have >=1 gold-HALLUCINATED claim; "
                    "ENRICHED sample: prevalence/precision not representative", "n_responses": len(ids), "n_claims": claims_total, "task_mix": task_mix,
        "response_ids": ids, "verifier": nli.assessor_id, "thresholds": {"entail": nli.entail_threshold, "contradict": nli.contradict_threshold, "tuned": False},
        "budget_calls_per_response": a.budget, "repair": "disabled", "linucb": f"{a.linucb} (frozen, no online update; trained earlier on TRAIN with oracle verdicts)",
        "random_seeds": rseeds, "primary_flag": "broad (label != SUPPORTED); strict (CONTRADICTED) also reported",
        "cache": {"path": str(a.cache), "entries_before": pre_cached, "entries_after": len(cache.data), "hits": cache.hits, "misses": cache.misses},
        "n_boot": a.n_boot, "resampling_unit": "response (paired across policies)", "runtime_s": round(time.monotonic() - t0, 1)},
        "runs": runs, "selection": sel, "detection": results, "nli_verifier_on_verified_claims": ver}
    jp = Path(a.out_dir) / f"{out['manifest']['run_id']}.json"
    jp.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("wrote", jp)
    return out


if __name__ == "__main__":
    main()
