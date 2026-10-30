#!/usr/bin/env python3
"""Score a COMPLETED human annotation sheet (scripts/build_support_annotation.py) for the S0 vs S1 question.

Labels are the annotator's judgement of whether the RETRIEVED PASSAGES support the claim (evidence support, not real-world truth):
supported / unsupported / contradicted / indeterminate. NLI verdicts are never used as labels; they only define strata, and the table
shows how often the NLI verdict and the human label agree. Refuses to report rates when no labels are filled in.

S0 answer = SUPPORTED-verdict claims; S1 answer = SUPPORTED + INSUFFICIENT claims (CONTRADICTED claims are in neither). Strategy rates pool the
annotated claims of those strata with weight 1 per claim (claim-level, NOT weighted by how often a policy placed the claim in an answer).
Uncertainty: Wilson 95% intervals within a verdict stratum; percentile bootstrap over QUESTIONS (claims are clustered in questions) for strategy rates.

usage: python scripts/score_support_annotation.py --sheet completed.csv --key key.json [--replay-json ...] [--second other.csv] [--out report.md]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LABELS = ("supported", "unsupported", "contradicted", "indeterminate")
VERDICTS = ("SUPPORTED", "INSUFFICIENT", "CONTRADICTED")


def read_labels(path: Path) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            lab = (row.get("label") or "").strip().lower()
            if lab and lab not in LABELS:
                raise ValueError(f"{path.name}: invalid label {lab!r} for {row['item_id']} (allowed: {LABELS})")
            out[row["item_id"]] = lab or None
    return out


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def cohen_kappa(a: List[str], b: List[str]) -> Optional[float]:
    if not a:
        return None
    po = float(np.mean([x == y for x, y in zip(a, b)]))
    pe = sum((a.count(l) / len(a)) * (b.count(l) / len(b)) for l in LABELS)
    return None if pe == 1 else (po - pe) / (1 - pe)


def f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def fmt_ci(c):
    return "n/a" if c is None else f"[{c[0]:.2f}, {c[1]:.2f}]"


def strategy_rates(rows: List[dict], members, label: str):
    sel = [r for r in rows if members(r)]
    return (sum(1 for r in sel if r["label"] == label) / len(sel)) if sel else None


def boot_strategy(rows: List[dict], members, label: str, n: int = 2000, seed: int = 0):
    by_q: Dict[str, List[dict]] = {}
    for r in rows:
        by_q.setdefault(r["sample_id"], []).append(r)
    qs = sorted(by_q)
    if len(qs) < 5:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        draw = [r for q in rng.choice(qs, size=len(qs)) for r in by_q[q]]
        v = strategy_rates(draw, members, label)
        if v is not None:
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else None


def boot_diff(rows, label, n=2000, seed=0):
    """S1 minus S0 rate for one label, bootstrap over questions."""
    by_q: Dict[str, List[dict]] = {}
    for r in rows:
        by_q.setdefault(r["sample_id"], []).append(r)
    qs = sorted(by_q)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        draw = [r for q in rng.choice(qs, size=len(qs)) for r in by_q[q]]
        a, b = strategy_rates(draw, lambda r: r["in_S1"], label), strategy_rates(draw, lambda r: r["in_S0"], label)
        if a is not None and b is not None:
            d.append(a - b)
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if len(d) > 10 else None


def replay_metrics(path: Path, budget: int = 4):
    d = json.load(open(path, encoding="utf-8"))
    acc = {s: {"str": [], "ret": []} for s in ("S0_current", "S1_conservative", "S2_repair_first")}
    for key, seed_rows in d["per_question_runs"].items():
        if int(key.split("|")[0]) != budget:
            continue
        for s in acc:
            acc[s]["str"].append(np.mean([[r["strategies"][s]["str_em"] for r in rows] for rows in seed_rows]))
            acc[s]["ret"].append(np.mean([[r["strategies"][s]["n_in_answer"] / max(1, r["n_claims"]) for r in rows] for rows in seed_rows]))
    return {s: {"str_em": float(np.mean(v["str"])), "retention": float(np.mean(v["ret"]))} for s, v in acc.items()}, d["manifest"]["run_id"]


def render(sheet: Path, key_path: Path, replay: Optional[Path], second: Optional[Path]) -> Optional[str]:
    key = json.load(open(key_path, encoding="utf-8"))
    labels = read_labels(sheet)
    rows = [{"item_id": i, "label": labels.get(i), **v} for i, v in key["items"].items()]
    labelled = [r for r in rows if r["label"]]
    if not labelled:
        return None
    L = ["# Evidence-support annotation results (S0 vs S1)", "",
         f"Sheet `{sheet.name}`: {len(labelled)} of {len(rows)} items labelled. Pool: {key['meta']['claims_with_cached_verdict']} claims with an NLI verdict "
         f"({'census' if key['meta'].get('census') else 'stratified sample'}), 30 questions. **Labels judge support by the retrieved passages, not real-world truth. "
         "NLI verdicts are not used as labels.**", ""]
    if len(labelled) < len(rows):
        L += [f"**WARNING: {len(rows) - len(labelled)} items unlabelled; estimates use labelled items only and may be biased.**", ""]
    L += ["## Human label distribution by NLI verdict stratum", "", "| NLI verdict (answer membership) | n | supported | unsupported | contradicted | indeterminate | Supported rate [Wilson 95%] | Unsupported+contradicted rate [Wilson 95%] |",
          "|---|---|---|---|---|---|---|---|"]
    for v in VERDICTS:
        sel = [r for r in labelled if r["nli_verdict"] == v]
        c = {l: sum(1 for r in sel if r["label"] == l) for l in LABELS}
        bad = c["unsupported"] + c["contradicted"]
        mem = next((r["answer_membership"] for r in sel), "")
        L.append(f"| {v} ({mem}) | {len(sel)} | {c['supported']} | {c['unsupported']} | {c['contradicted']} | {c['indeterminate']} | "
                 f"{f(c['supported'] / len(sel) if sel else None, 2)} {fmt_ci(wilson(c['supported'], len(sel)))} | {f(bad / len(sel) if sel else None, 2)} {fmt_ci(wilson(bad, len(sel)))} |")
    L += ["", "## Claims in the final answer, by strategy (claim-level pool; bootstrap over questions)", "",
          "| Strategy | n claims | Supported | Unsupported | Contradicted | Indeterminate |", "|---|---|---|---|---|---|"]
    for name, mem in (("S0 current (SUPPORTED verdicts)", lambda r: r["in_S0"]), ("S1 conservative (SUPPORTED + INSUFFICIENT)", lambda r: r["in_S1"])):
        n = sum(1 for r in labelled if mem(r))
        cells = [f"{f(strategy_rates(labelled, mem, l), 2)} {fmt_ci(boot_strategy(labelled, mem, l))}" for l in LABELS]
        L.append(f"| {name} | {n} | " + " | ".join(cells) + " |")
    L += ["", "S1 minus S0 (percentage-point change in the share of answer claims with each label; bootstrap over questions):", ""]
    for l in LABELS:
        a, b = strategy_rates(labelled, lambda r: r["in_S1"], l), strategy_rates(labelled, lambda r: r["in_S0"], l)
        if a is not None and b is not None:
            L.append(f"- {l}: {100 * (a - b):+.1f} pp {fmt_ci(tuple(100 * x for x in boot_diff(labelled, l)) if boot_diff(labelled, l) else None)}")
    ins = [r for r in labelled if r["nli_verdict"] == "INSUFFICIENT"]
    if ins:
        L.append(f"- retained INSUFFICIENT claims judged supported by the passages: {sum(r['label'] == 'supported' for r in ins)} of {len(ins)}")
    L += ["", "## NLI verdict vs human label (agreement; NLI is not ground truth)", "", "| NLI verdict | " + " | ".join(LABELS) + " |", "|---|---|---|---|---|"]
    for v in VERDICTS:
        L.append(f"| {v} | " + " | ".join(str(sum(1 for r in labelled if r['nli_verdict'] == v and r['label'] == l)) for l in LABELS) + " |")
    if replay:
        m, rid = replay_metrics(replay)
        L += ["", f"## Coverage and retention (budget 4, mean over the 8 policies; from `{rid}`)", "", "| Strategy | STR-EM | Draft claims retained in the answer |", "|---|---|---|"]
        for s, v in m.items():
            L.append(f"| {s} | {v['str_em']:.3f} | {v['retention']:.2f} |")
    if second:
        lab2 = read_labels(second)
        both = [i for i in labels if labels[i] and lab2.get(i)]
        k = cohen_kappa([labels[i] for i in both], [lab2[i] for i in both])
        L += ["", f"## Inter-annotator agreement (`{second.name}`): {len(both)} items labelled by both, raw agreement "
                  f"{np.mean([labels[i] == lab2[i] for i in both]):.2f}, Cohen's kappa {f(k, 2)}" if both else "", ""]
    L += ["", "## Reading these numbers", "- Rates are over annotated claims; indeterminate is its own category and is NOT dropped.",
          "- Evidence support is not real-world truth; a claim can be true but unsupported by the five retrieved passages.",
          "- Strategy rates weight each claim once; they do not weight by how often a policy placed a claim in an answer.",
          "- A single annotator gives no reliability estimate; use `--second` for agreement."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--key", default=str(ROOT / "experiments/annotation/alce_support_annotation_key.json"))
    ap.add_argument("--replay-json", default=None)
    ap.add_argument("--second", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = render(Path(a.sheet), Path(a.key), Path(a.replay_json) if a.replay_json else None, Path(a.second) if a.second else None)
    if text is None:
        print("No labels filled in: nothing to report. (No rates are computed from NLI verdicts.)")
        return 2
    out = Path(a.out) if a.out else Path(a.sheet).with_name(Path(a.sheet).stem + "_results.md")
    out.write_text(text, encoding="utf-8")
    print(text)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
