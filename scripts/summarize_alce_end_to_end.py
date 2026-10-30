#!/usr/bin/env python3
"""Render the end-to-end ALCE-ASQA run (scripts/run_alce_end_to_end.py) into report tables. Reads the saved JSON only.

usage: python scripts/summarize_alce_end_to_end.py <alce_e2e_*.json> [--out summary.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def f(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def ci(c):
    return "n/a" if not c else f"[{c[0]:.3f}, {c[1]:.3f}]"


def render(d) -> str:
    man, s = d["manifest"], d["summary"]
    L = [f"# {man['run_id']}", "",
         f"- questions: {man['questions']} ({s['n_questions']}), collection `{man['collection']}`, seeds {man['seeds']}, budgets {man['budgets']}",
         f"- components: {json.dumps(man['components'])}",
         f"- policies fitted on: {man['policies_fit_on']}; LLM verdict cache: {man['llm_cache']}",
         f"- mean claims per draft: {s['mean_claims_per_draft']:.2f} (a budget of 8 calls is therefore NOT binding for most questions)", "",
         "## Where the answer-bearing content is lost (ALCE STR-EM, mean over questions; 1.0 = every disambiguated question answered)", "",
         "| Stage | STR-EM | 95% CI | Loss vs previous stage |", "|---|---|---|---|",
         f"| Retrieved passages (top 5) | {f(s['retrieval_str_em'])} | {ci(s['retrieval_ci95'])} | retrieval loss = {f(1 - s['retrieval_str_em'])} |",
         f"| Generated draft | {f(s['draft_str_em'])} | {ci(s['draft_ci95'])} | generation loss = {f(s['retrieval_str_em'] - s['draft_str_em'])} (negative: answers the draft has that the passages lack) |", ""]
    for b, bo in s["budgets"].items():
        L += [f"## Budget {b} model calls per question", "",
              "| Policy | Final STR-EM [95% CI over questions] | Selection loss (draft - final) | Repair gain | Verified / checked claims | Flagged | Repairs (re-verified pass) | Calls | Tokens | Recorded latency (s) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k, v in bo["policies"].items():
            L.append(f"| {k} (seeds {v['n_seeds']}) | {f(v['str_em_final'])} {ci(v['ci95'])} | {f(v['selection_loss'])} | {f(v['repair_gain'])} | "
                     f"{v['claims_verified']:.2f} / {v['claims_checked']:.2f} | {v['flagged']:.2f} | {v['repairs']:.2f} ({v['reverified_pass']:.2f}) | "
                     f"{v['model_calls']:.2f} | {v['tokens']:.0f} | {v['latency_seconds']:.0f} |")
        L += ["", "Paired differences in final STR-EM (95% CI over questions; unadjusted):", "", "| Comparison | Difference | CI |", "|---|---|---|"]
        for k, v in bo["paired"].items():
            L.append(f"| {k} | {v['diff']:+.3f} | {ci(v['ci95'])} |")
        L.append("")
    # verifier PROXY analysis, pooled over every run (not gold): context-ungrounded claims vs grounded claims
    L += ["## Verifier behaviour against a PROXY label (not gold)", "",
          "A claim is 'context-ungrounded' when no retrieved passage supports it under the lexical rule (overlap >= 0.5 and all numbers present). "
          "Rates are pooled over every policy/budget/seed run, per question-run averages; this proxy is NOT hallucination ground truth and is lexical.", "",
          "| Budget | Ungrounded claims checked | Flagged (proxy TPR) | Grounded claims checked | Flagged (proxy FPR) | Ungrounded claims that reached the answer |", "|---|---|---|---|---|---|"]
    for b, bo in s["budgets"].items():
        uc = sum(v["ungrounded_checked"] for v in bo["policies"].values()); uf = sum(v["ungrounded_flagged"] for v in bo["policies"].values())
        gc = sum(v["grounded_checked"] for v in bo["policies"].values()); gf = sum(v["grounded_flagged"] for v in bo["policies"].values())
        uin = sum(v["ungrounded_in_final"] for v in bo["policies"].values())
        L.append(f"| {b} | {uc:.1f} | {f(uf / uc if uc else None)} | {gc:.1f} | {f(gf / gc if gc else None)} | {uin:.1f} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    p = Path(a.run_json)
    text = render(json.loads(p.read_text(encoding="utf-8")))
    out = Path(a.out) if a.out else p.with_name(p.stem + "_summary.md")
    out.write_text(text, encoding="utf-8")
    print(text)
    print("wrote", out)


if __name__ == "__main__":
    main()
