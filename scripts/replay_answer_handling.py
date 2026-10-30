#!/usr/bin/env python3
"""Answer-handling strategies replayed on CACHED NLI verdicts (no new verifier or repair calls; questions, drafts, policies and verdicts identical).

The controller is re-run through scripts/run_alce_end_to_end.py with the NLI verdict cache; any cache miss (NLI or repair model call) aborts the run.
From the SAME trace of each (policy, seed, question) three final answers are assembled:

  S0_current        verified claims + repairs that passed re-verification (the behaviour of the earlier runs; flagged and INSUFFICIENT claims dropped)
  S1_conservative   verified claims + INSUFFICIENT claims (last verdict INSUFFICIENT); CONTRADICTED/CONFLICTED claims excluded; repairs NOT used
  S2_repair_first   S1 + a repair is used (in place of the claim) only if its re-verification passed; otherwise the S1 fallback

Claims that were never verified (budget) are dropped in every strategy, so the budget effect is unchanged. Verifier thresholds, policies and
checkpoints are untouched. Nothing here measures factuality: STR-EM is answer coverage, and the lexical proxy is not ground truth.

usage: python scripts/replay_answer_handling.py --start 10 --n-questions 30 --budgets 4 --seeds 0,1,2 --tag replay_full
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_alce_end_to_end as E  # noqa: E402
import run_ragtruth_comparison as R  # noqa: E402
from src.claims.models import ClaimState  # noqa: E402
from src.controller.models import ControllerAction  # noqa: E402
from src.evidence.models import AssessmentLabel  # noqa: E402
from src.synthesis.grounding import build_items, stitch_answer, support_for_claim  # noqa: E402

STRATEGIES = ("S0_current", "S1_conservative", "S2_repair_first")
EXCLUDE = (AssessmentLabel.CONTRADICTED.value, AssessmentLabel.CONFLICTED.value)
INSUFFICIENT = AssessmentLabel.INSUFFICIENT.value
MISSES: List[str] = []


def claim_outcomes(claims, trace) -> Dict[str, Any]:
    """Per-claim facts needed to assemble every strategy: last verdict, whether verified, repair and its re-verification."""
    last, repairs_all, repairs_pass = {}, {}, {}
    for s in trace.steps:
        if s.action == ControllerAction.VERIFY and s.assessment_label and not s.error:
            last[s.claim_id] = s.assessment_label
        if s.action == ControllerAction.REPAIR and s.repair_data:
            repairs_all[s.claim_id] = s.repair_data
            if (s.repair_data.get("reverification") or {}).get("verified"):
                repairs_pass[s.claim_id] = s.repair_data
    verified = {c.claim_id for c in claims if c.state == ClaimState.VERIFIED}
    return {"last": last, "verified": verified, "repairs_all": repairs_all, "repairs_pass": repairs_pass}


def assemble(strategy: str, claims, out: Dict[str, Any], store, qa_pairs) -> Dict[str, Any]:
    verified, last, rp = out["verified"], out["last"], out["repairs_pass"]
    keep_insufficient = {cid for cid, lab in last.items() if lab == INSUFFICIENT and cid not in verified}
    if strategy == "S0_current":
        ids, repairs = verified | set(rp), rp
    elif strategy == "S1_conservative":
        ids, repairs = verified | keep_insufficient, {}
    elif strategy == "S2_repair_first":
        ids, repairs = verified | keep_insufficient | set(rp), rp
    else:
        raise ValueError(strategy)
    items = build_items(claims, repairs, store, ids)
    answer = stitch_answer(items)
    used = {it["claim_id"] for it in items}
    passages = next(iter(store.values()), [])
    return {"str_em": E.str_em(qa_pairs, E.strip_citations(answer)), "n_in_answer": len(used), "answer": answer,
            "n_insufficient_kept": len(keep_insufficient & used), "n_repairs_used": len(set(rp) & used) if strategy != "S1_conservative" else 0,
            "ungrounded_in_answer": sum(1 for it in items if support_for_claim(it["text"], passages)["title"] is None)}


class _Capture(E.VerificationController):
    last = None

    def run(self, claims, initial_evidence, run_id):
        trace = super().run(claims=claims, initial_evidence=initial_evidence, run_id=run_id)
        _Capture.last = (claims, trace, initial_evidence)
        return trace


_orig_run_one = E.run_one


def run_one_replay(prep, policy, budget, retriever, assessor, repairer, run_id):
    row = _orig_run_one(prep, policy, budget, retriever, assessor, repairer, run_id)
    if MISSES:
        raise RuntimeError(f"new model call during replay: {MISSES[:3]}")
    claims, trace, store = _Capture.last
    out = claim_outcomes(claims, trace)
    row["strategies"] = {s: assemble(s, claims, out, store, prep["qa_pairs"]) for s in STRATEGIES}
    row["strategy_counts"] = {"n_claims": len(claims), "n_verified": len(out["verified"]),
                              "last_labels": {lab: sum(1 for v in out["last"].values() if v == lab) for lab in set(out["last"].values())},
                              "n_repairs": len(out["repairs_all"]), "n_repairs_passed": len(out["repairs_pass"])}
    assert abs(row["strategies"]["S0_current"]["str_em"] - row["str_em_final"]) < 1e-9, "S0 must reproduce the original final answer"
    return row


def install_guards() -> None:
    """Abort on any model call the cache cannot serve; skip the llama-server preflight (no generation happens in a replay)."""
    from src.controller.interfaces import LLMRepairer
    from src.evidence.nli_assessor import NLIAssessor

    def _boom(kind):
        def f(self, *a, **k):
            MISSES.append(kind)
            raise RuntimeError(f"replay attempted a new {kind} call")
        return f
    NLIAssessor.assess = _boom("NLI assess")
    LLMRepairer.repair = _boom("repair")
    R.preflight_llm = lambda: None
    E.VerificationController = _Capture
    E.run_one = run_one_replay


# ----------------------------------------------------------------------------- analysis
def boot(d, n=2000, seed=0):
    if len(d) < 10:
        return None
    rng = np.random.default_rng(seed)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def analyse(run_json: Path, budget: int, saved_nli: Path = None) -> str:
    d = json.load(open(run_json, encoding="utf-8"))
    L = [f"# Answer-handling strategies on cached NLI verdicts (budget {budget})", "",
         f"Source run `{d['manifest']['run_id']}` (llm cache {d['manifest']['llm_cache']}; zero misses required), questions {d['manifest']['questions']}, seeds {d['manifest']['seeds']}.",
         "Strategies: S0 current (drop flagged and INSUFFICIENT; keep passed repairs), S1 conservative (keep INSUFFICIENT, exclude CONTRADICTED, no repairs), "
         "S2 repair-first (S1 + passed repairs). Never-verified claims are dropped in all three. STR-EM is answer coverage, not factuality; proxy columns are lexical, NOT ground truth.", "",
         f"Draft STR-EM (retain every claim, no verification): {d['summary']['draft_str_em']:.3f}.", "",
         "| Policy | Strategy | STR-EM | Delta vs S0 [95% CI, paired over questions] | Claims in answer / draft claims | INSUFFICIENT kept | Repairs used | Lexically ungrounded claims in answer |",
         "|---|---|---|---|---|---|---|---|"]
    pooled = {s: {"str": [], "kept": 0, "n": 0} for s in STRATEGIES}
    for key, seed_rows in d["per_question_runs"].items():
        b, pol = key.split("|", 1)
        if int(b) != budget:
            continue
        def per_q(fn):
            return np.mean([[fn(r) for r in rows] for rows in seed_rows], axis=0)
        base = per_q(lambda r: r["strategies"]["S0_current"]["str_em"])
        for s in STRATEGIES:
            em = per_q(lambda r: r["strategies"][s]["str_em"])
            diff = em - base
            ci = boot(diff) if s != "S0_current" else None
            cl = per_q(lambda r: r["strategies"][s]["n_in_answer"] / max(1, r["n_claims"]))
            L.append(f"| {pol} | {s} | {em.mean():.3f} | " + ("-" if ci is None else f"{diff.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]") +
                     f" | {cl.mean():.2f} | {per_q(lambda r: r['strategies'][s]['n_insufficient_kept']).mean():.2f} | "
                     f"{per_q(lambda r: r['strategies'][s]['n_repairs_used']).mean():.2f} | {per_q(lambda r: r['strategies'][s]['ungrounded_in_answer']).mean():.2f} |")
            pooled[s]["str"].append(em)
    allc = {"n_claims": 0, "n_repairs": 0, "n_repairs_passed": 0}
    labs: Dict[str, int] = {}
    lat = []
    for key, seed_rows in d["per_question_runs"].items():
        if int(key.split("|")[0]) != budget:
            continue
        for rows in seed_rows:
            for r in rows:
                c = r["strategy_counts"]
                for k in allc:
                    allc[k] += c[k]
                for lab, v in c["last_labels"].items():
                    labs[lab] = labs.get(lab, 0) + v
                lat.append(r["latency_seconds"])
    chk = sum(labs.values())
    L += ["", "## Verdicts and repair outcomes (pooled over policies, seeds and questions)", "",
          f"- last verdict per checked claim: " + ", ".join(f"{k} {v} ({v / chk:.0%})" for k, v in sorted(labs.items())) + f" of {chk} checked / {allc['n_claims']} draft claims",
          f"- repair attempts {allc['n_repairs']}, passed re-verification {allc['n_repairs_passed']} "
          f"({allc['n_repairs_passed'] / max(1, allc['n_repairs']):.1%} repair success)",
          f"- recorded verification+repair latency per question run is identical across strategies (mean {np.mean(lat):.0f}s, from the original calls); assembling an answer is deterministic string work with no model call."]
    return "\n".join(L) + "\n"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--verifier" not in argv:
        argv += ["--verifier", "nli"]
    src_cache = ROOT / "experiments/cache/alce_e2e_nli_cache.pkl"
    if "--cache" not in argv:
        replay_cache = ROOT / "experiments/cache/alce_e2e_nli_replay_cache.pkl"
        shutil.copy(src_cache, replay_cache)  # the original NLI cache is never written
        argv += ["--cache", str(replay_cache)]
    install_guards()
    out = E.main(argv)
    budget = int(out["manifest"]["budgets"][0])
    rid = out["manifest"]["run_id"]
    rj = Path(argv[argv.index("--out-dir") + 1] if "--out-dir" in argv else ROOT / "experiments/runs") / f"{rid}.json"
    text = analyse(rj, budget)
    mdp = rj.with_name(rj.stem + "_answer_handling.md")
    mdp.write_text(text, encoding="utf-8")
    print(text)
    print("wrote", mdp)
    return out


if __name__ == "__main__":
    main()
