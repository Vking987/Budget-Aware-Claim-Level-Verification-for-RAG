#!/usr/bin/env python3
"""Controlled verifier diagnostic on the RAGTruth pilot-50 claims (separate from the benchmark).

Question: do the verifier's misses on gold-hallucinated claims come from how the benchmark builds
evidence, or from the verifier itself?

Held fixed: claims (text + gold label), model, assessor prompt/code (unchanged LlamaAssessor),
scoring (src/evaluation/ragtruth_metrics.py flag definitions). Varied: ONLY the evidence passages.

  A  original benchmark evidence (top-4 lexical 700-char chunks of the JSON/text source), read from the
     verdict cache, i.e. the exact verdicts the benchmark produced;
  B  the same source rendered as readable text (Key: value lines, nulls shown as "not specified"),
     chunks <= 880 chars so the assessor does not truncate them; for Data2txt the business-record
     chunks always come first. `full_source_visible` records whether NOTHING was omitted.

RAGTruth annotates spans in the RESPONSE, not the source, so there is no span-level source evidence
to render; annotator notes (`meta`) restate the gold label and are deliberately NOT given to the verifier.

Claim set: all gold-HALLUCINATED claims of the pilot sample plus a seeded sample of gold-UNFLAGGED claims
(per task type) so false-positive rate can be measured. Diagnostic prevalence is therefore NOT the
natural prevalence; report TPR/FPR, and read precision/F1 as conditional on this mix.

Writes to experiments/diagnostics/ and uses a separate COPY of the verdict cache.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_ragtruth_comparison as R  # noqa: E402
from src.evaluation import ragtruth_metrics as M  # noqa: E402
from src.evaluation.ragtruth_adapter import RAGTruthDatasetAdapter  # noqa: E402
from src.evidence.models import EvidencePassage  # noqa: E402

DIAG_DIR = ROOT / "experiments" / "diagnostics"
READABLE_CHUNK_CHARS = 880  # < LlamaAssessor.max_passage_chars (900): no silent truncation
MAX_PASSAGES = 4  # LlamaAssessor keeps at most 4 passages
CONDITIONS = ("A_original", "B_readable")


# ----------------------------------------------------------------------------- rendering
def _val(v: Any) -> str:
    if v is None:
        return "not specified"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def render_data2txt(si: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Business record -> readable lines (record lines, review lines). No JSON punctuation."""
    rec: List[str] = []
    rec.append(f"Business name: {_val(si.get('name'))}")
    addr = ", ".join(_val(si.get(k)) for k in ("address", "city", "state") if si.get(k))
    rec.append(f"Address: {addr}")
    rec.append(f"Categories: {_val(si.get('categories'))}")
    if si.get("business_stars") is not None:
        rec.append(f"Overall rating: {si['business_stars']} stars")
    hours = si.get("hours") or {}
    rec.append("Opening hours: " + ("; ".join(f"{d} {h}" for d, h in hours.items()) if hours else "not specified"))
    for k, v in (si.get("attributes") or {}).items():
        if isinstance(v, dict):
            rec.append(f"{k}: " + "; ".join(f"{kk} = {_val(vv)}" for kk, vv in v.items()))
        else:
            rec.append(f"{k}: {_val(v)}")
    reviews = []
    for rv in si.get("review_info") or []:
        reviews.append(f"Review ({_val(rv.get('review_stars'))} stars): {str(rv.get('review_text', '')).strip()}")
    return rec, reviews


def pack_lines(lines: List[str], max_chars: int = READABLE_CHUNK_CHARS) -> List[str]:
    """Pack whole lines into chunks <= max_chars joined by newlines; over-long lines are split on sentences."""
    chunks: List[str] = []
    cur = ""
    for line in lines:
        pieces = [line] if len(line) <= max_chars else R.chunk_source(line, max_chars)
        for piece in pieces:
            if cur and len(cur) + 1 + len(piece) > max_chars:
                chunks.append(cur)
                cur = piece
            else:
                cur = f"{cur}\n{piece}" if cur else piece
    if cur:
        chunks.append(cur)
    return chunks


def _overlap(claim_text: str, chunk: str) -> float:
    ct = set(R._TOK.findall(claim_text.lower()))
    return len(ct & set(R._TOK.findall(chunk.lower()))) / max(1, len(ct))


def readable_evidence(source, claim_text: str, k: int = MAX_PASSAGES) -> Tuple[List[EvidencePassage], bool, int]:
    """Condition-B evidence for one claim. Returns (passages, full_source_visible, total_source_chars)."""
    si = source.source_info
    title = source.source or f"Source_{source.source_id}"
    if source.task_type == "Data2txt" and isinstance(si, dict):
        rec_lines, review_lines = render_data2txt(si)
        record_chunks = pack_lines(rec_lines)
        review_chunks = pack_lines(review_lines)
        fixed = record_chunks[:2]  # business record always shown first
        rest = record_chunks[2:] + review_chunks
    else:
        fixed = []
        rest = pack_lines([ln for ln in source.get_source_text().splitlines() if ln.strip()])
    total_chars = sum(len(c) for c in fixed + rest)
    slots = max(0, k - len(fixed))
    if len(rest) <= slots:
        chosen_rest = list(rest)
    else:
        ranked = sorted(range(len(rest)), key=lambda i: (-_overlap(claim_text, rest[i]), i))[:slots]
        chosen_rest = [rest[i] for i in sorted(ranked)]  # keep document order
    chosen = fixed + chosen_rest
    visible = len(chosen) == len(fixed) + len(rest) and all(len(c) <= READABLE_CHUNK_CHARS for c in chosen)
    passages = [
        EvidencePassage(
            evidence_id=f"diagB_{source.source_id}_p{i}", doc_title=title, source_text=ch,
            retrieval_score=round(_overlap(claim_text, ch), 4), metadata={"source_id": source.source_id, "condition": "B"},
        )
        for i, ch in enumerate(chosen)
    ]
    return passages, visible, total_chars


# ----------------------------------------------------------------------------- claim set + label validation
def label_overlaps(resp, start: int, end: int) -> List[Dict[str, Any]]:
    """Gold annotation spans (non-implicit_true) overlapping the claim's character range in the response."""
    out = []
    for lab in resp.labels:
        if getattr(lab, "implicit_true", False):
            continue
        s, e = lab.start, lab.end
        if min(e, end) - max(s, start) > 0:
            out.append({"label_type": getattr(lab, "label_type", None), "text": getattr(lab, "text", "")})
    return out


def build_claim_set(prepared, mapped, n_neg_per_task: int, seed: int) -> List[Dict[str, Any]]:
    mp = {(m.response_id, m.claim_id): m for m in mapped}
    pos, negs = [], {}
    for p in prepared:
        for c in p["claims"]:
            g = p["gold"][c.claim_id]
            item = {"prep": p, "claim": c, "gold": g, "task": p["task"], "mapped": mp[(p["resp"].response_id, c.claim_id)]}
            if g == "HALLUCINATED":
                pos.append(item)
            elif g == "UNFLAGGED":
                negs.setdefault(p["task"], []).append(item)
    rng = random.Random(seed)
    sel_neg = []
    for t in sorted(negs):
        pool = sorted(negs[t], key=lambda x: (x["prep"]["resp"].response_id, x["claim"].claim_id))
        sel_neg += rng.sample(pool, min(n_neg_per_task, len(pool)))
    return pos + sel_neg


def validate_label_mapping(items) -> Dict[str, Any]:
    """Every positive must overlap >=1 real (non-implicit_true) hallucination span; every sampled negative none."""
    bad_pos, bad_neg, types = [], [], {}
    for it in items:
        m, resp = it["mapped"], it["prep"]["resp"]
        ov = label_overlaps(resp, m.char_start, m.char_end)
        it["label_types"] = sorted({o["label_type"] for o in ov if o["label_type"]})
        if it["gold"] == "HALLUCINATED":
            if not ov:
                bad_pos.append((resp.response_id, it["claim"].claim_id))
            for t in it["label_types"]:
                types[t] = types.get(t, 0) + 1
        elif ov:
            bad_neg.append((resp.response_id, it["claim"].claim_id))
    return {"positives_without_overlapping_span": bad_pos, "negatives_with_overlapping_span": bad_neg,
            "positive_label_type_counts": types}


# ----------------------------------------------------------------------------- running + scoring
def run_condition(items, cond: str, assessor, cache, sources) -> List[Dict[str, Any]]:
    out = []
    for it in items:
        c, p = it["claim"], it["prep"]
        if cond == "A_original":
            ev = p["evidence"][c.claim_id]
            visible, src_chars = None, None
        else:
            ev, visible, src_chars = readable_evidence(sources[p["resp"].source_id], c.normalized_text)
        h0 = cache.hits
        t0 = time.monotonic()
        a = assessor.assess(c, ev)
        wall = time.monotonic() - t0
        out.append({
            "label": None if getattr(a, "is_error", False) else a.label.value,
            "confidence": a.confidence, "error": bool(getattr(a, "is_error", False)),
            "recorded_latency_s": (a.cost.latency_seconds or 0.0) if a.cost else 0.0,
            "wall_s": wall, "cache_hit": cache.hits > h0, "n_passages": len(ev),
            "evidence_chars": sum(len(x.source_text) for x in ev), "full_source_visible": visible,
            "source_chars_rendered": src_chars,
            "rationale": (a.rationale or "")[:200],
            "evidence_preview": [x.source_text for x in ev] if it["gold"] == "HALLUCINATED" else None,
        })
    return out


def score(rows: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    """rows: {gold, label}. Counts over scored gold only; errors/None count as not flagged."""
    tp = fp = fn = tn = 0
    for r in rows:
        if r["gold"] not in M.SCORED_GOLD:
            continue
        flagged = M._flag(r["label"], mode)
        if r["gold"] == "HALLUCINATED":
            tp, fn = (tp + 1, fn) if flagged else (tp, fn + 1)
        else:
            fp, tn = (fp + 1, tn) if flagged else (fp, tn + 1)
    out = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, **M.prf(tp, fp, fn)}
    out["tpr_ci95"] = M.wilson_ci(tp, tp + fn)
    out["fpr"] = fp / (fp + tn) if (fp + tn) else None
    out["fpr_ci95"] = M.wilson_ci(fp, fp + tn)
    return out


def summarize(items, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    summ: Dict[str, Any] = {}
    tasks = sorted({it["task"] for it in items})
    for cond, res in results.items():
        rows = [{"gold": it["gold"], "label": r["label"], "task": it["task"], "visible": r["full_source_visible"]}
                for it, r in zip(items, res)]
        block: Dict[str, Any] = {"overall": {}, "by_task": {}, "label_counts": {}, "latency": {}}
        for mode in ("strict", "broad"):
            block["overall"][mode] = score(rows, mode)
            block["by_task"][mode] = {t: score([r for r in rows if r["task"] == t], mode) for t in tasks}
        for g in ("HALLUCINATED", "UNFLAGGED"):
            block["label_counts"][g] = {}
            for r in rows:
                if r["gold"] == g:
                    block["label_counts"][g][str(r["label"])] = block["label_counts"][g].get(str(r["label"]), 0) + 1
        lat = [x["recorded_latency_s"] for x in res if x["recorded_latency_s"]]
        block["latency"] = {"mean_recorded_s": sum(lat) / len(lat) if lat else None, "n": len(lat),
                            "mean_evidence_chars": sum(x["evidence_chars"] for x in res) / max(1, len(res))}
        if cond == "B_readable":
            for vis in (True, False):
                sub = [r for r in rows if r["visible"] is vis]
                block[f"full_source_visible_{vis}"] = {m: score(sub, m) for m in ("strict", "broad")}
                block[f"full_source_visible_{vis}"]["n_claims"] = len(sub)
        summ[cond] = block
    # paired A -> B on gold positives
    if "A_original" in results and "B_readable" in results:
        pa = [(M._flag(a["label"], "broad"), M._flag(b["label"], "broad"))
              for it, a, b in zip(items, results["A_original"], results["B_readable"]) if it["gold"] == "HALLUCINATED"]
        summ["paired_positives_broad"] = {"flagged_A_only": sum(1 for x, y in pa if x and not y),
                                          "flagged_B_only": sum(1 for x, y in pa if y and not x),
                                          "flagged_both": sum(1 for x, y in pa if x and y),
                                          "flagged_neither": sum(1 for x, y in pa if not x and not y)}
    return summ


def fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def md_report(meta, summ, items, results) -> str:
    L = [f"# RAGTruth verifier diagnostic — {meta['run_id']}", "",
         f"Claims: {meta['n_pos']} gold-hallucinated + {meta['n_neg']} gold-unflagged (seeded sample). "
         "Diagnostic prevalence is not natural prevalence: compare TPR/FPR, read precision/F1 as conditional on this mix.", ""]
    for mode in ("broad", "strict"):
        L += [f"## Flag = {mode}", "", "| Condition | TP | FN | FP | TN | TPR [Wilson 95%] | FPR [Wilson 95%] | Precision | F1 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for cond in CONDITIONS:
            o = summ[cond]["overall"][mode]
            tci, fci = o["tpr_ci95"], o["fpr_ci95"]
            L.append(f"| {cond} | {o['tp']} | {o['fn']} | {o['fp']} | {o['tn']} | {fmt(o['recall'])} [{fmt(tci[0]) if tci else 'n/a'}, {fmt(tci[1]) if tci else 'n/a'}] | "
                     f"{fmt(o['fpr'])} [{fmt(fci[0]) if fci else 'n/a'}, {fmt(fci[1]) if fci else 'n/a'}] | {fmt(o['precision'])} | {fmt(o['f1'])} |")
        L.append("")
        L += [f"By task ({mode}):", "", "| Condition | Task | TP/FN | FP/TN | TPR | FPR |", "|---|---|---|---|---|---|"]
        for cond in CONDITIONS:
            for t, o in summ[cond]["by_task"][mode].items():
                L.append(f"| {cond} | {t} | {o['tp']}/{o['fn']} | {o['fp']}/{o['tn']} | {fmt(o['recall'])} | {fmt(o['fpr'])} |")
        L.append("")
    L += ["## Label counts", ""]
    for cond in CONDITIONS:
        L.append(f"- {cond}: " + json.dumps(summ[cond]["label_counts"]))
    L += ["", "## Latency (recorded per-call, seconds) and evidence size", ""]
    for cond in CONDITIONS:
        la = summ[cond]["latency"]
        L.append(f"- {cond}: mean recorded latency {fmt(la['mean_recorded_s'])}s over {la['n']} calls; mean evidence chars {la['mean_evidence_chars']:.0f}")
    if "full_source_visible_True" in summ["B_readable"]:
        L += ["", "## Condition B split by whether the whole source was visible", ""]
        for vis in (True, False):
            b = summ["B_readable"][f"full_source_visible_{vis}"]
            o = b["broad"]
            L.append(f"- full_source_visible={vis}: n={b['n_claims']} TP={o['tp']} FN={o['fn']} FP={o['fp']} TN={o['tn']} TPR={fmt(o['recall'])} FPR={fmt(o['fpr'])}")
    if "paired_positives_broad" in summ:
        L += ["", "Paired A->B on gold positives (broad flag): " + json.dumps(summ["paired_positives_broad"])]
    L += ["", "Positive label types (response-span annotations): " + json.dumps(meta["label_validation"]["positive_label_type_counts"])]
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-json", default=str(ROOT / "experiments/runs/ragtruth_comparison_20261028_162847_pilot50.json"))
    ap.add_argument("--neg-per-task", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N claims of the set (positives first)")
    ap.add_argument("--src-cache", default=str(R.DEFAULT_CACHE))
    ap.add_argument("--out-dir", default=str(DIAG_DIR))
    ap.add_argument("--dry-run", action="store_true", help="mock assessor, plumbing only")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"ragtruth_verifier_diag_{ts}{('_' + args.tag) if args.tag else ''}"

    if not args.dry_run:
        R.preflight_llm()
    # diagnostic works on a COPY of the verdict cache; the benchmark cache is never written
    diag_cache_path = out_dir / f"{run_id}_cache.pkl"
    if Path(args.src_cache).exists() and not args.dry_run:
        shutil.copy(args.src_cache, diag_cache_path)
    cache = R.LLMCache(None if args.dry_run else diag_cache_path)
    inner = R._CostedMockAssessor() if args.dry_run else R.LlamaAssessor()
    assessor = R.CachedAssessor(inner, cache)

    run = json.load(open(args.run_json, encoding="utf-8"))
    proto = run["manifest"]["protocol"]
    ad = RAGTruthDatasetAdapter()
    ad.load_from_jsonl(R.DATA_DIR / "source_info.jsonl", R.DATA_DIR / "response.jsonl", split=proto["split"])
    by_id = {r.response_id: r for r in ad.responses}
    prepared, mapped = R.prepare(ad, [by_id[i] for i in proto["response_ids"]])
    items = build_claim_set(prepared, mapped, args.neg_per_task, args.seed)
    validation = validate_label_mapping(items)
    n_pos = sum(1 for it in items if it["gold"] == "HALLUCINATED")
    print(f"claim set: {n_pos} positives + {len(items) - n_pos} negatives; mapping validation: "
          f"{len(validation['positives_without_overlapping_span'])} positives lack a span, "
          f"{len(validation['negatives_with_overlapping_span'])} negatives overlap a span; types={validation['positive_label_type_counts']}")
    assert not validation["positives_without_overlapping_span"], "positive claim without gold span"
    assert not validation["negatives_with_overlapping_span"], "negative claim overlapping a gold span"
    if args.limit:
        items = items[: args.limit]

    results: Dict[str, List[Dict[str, Any]]] = {}
    for cond in CONDITIONS:
        t0 = time.monotonic()
        results[cond] = run_condition(items, cond, assessor, cache, ad.sources)
        cache.save()
        print(f"[{cond}] {len(items)} claims, cache hits so far {cache.hits}, misses {cache.misses}, {time.monotonic() - t0:.0f}s wall", flush=True)
        errs = sum(1 for x in results[cond] if x["error"])
        if errs:
            print(f"    WARNING: {errs} assessor errors in {cond} (counted as not flagged)")

    # A must reproduce the benchmark verdicts exactly (served from cache)
    if not args.dry_run:
        a_miss = [i for i, r in enumerate(results["A_original"]) if not r["cache_hit"]]
        print(f"condition A served from the benchmark verdict cache for {len(items) - len(a_miss)}/{len(items)} claims")

    summ = summarize(items, results)
    meta = {"run_id": run_id, "command": " ".join(sys.argv), "dry_run": args.dry_run, "source_run": args.run_json,
            "seed": args.seed, "n_pos": n_pos, "n_neg": len(items) - n_pos, "neg_per_task": args.neg_per_task,
            "label_validation": validation, "model": "LlamaAssessor default (unchanged)", "conditions": list(CONDITIONS),
            "evidence_B": {"chunk_chars": READABLE_CHUNK_CHARS, "max_passages": MAX_PASSAGES},
            "cache": {"hits": cache.hits, "misses": cache.misses, "diag_cache_copy": str(diag_cache_path)}}
    per_claim = [{"response_id": it["prep"]["resp"].response_id, "claim_id": it["claim"].claim_id, "task": it["task"],
                  "gold": it["gold"], "label_types": it.get("label_types"), "claim": it["claim"].normalized_text,
                  **{cond: results[cond][i] for cond in CONDITIONS}} for i, it in enumerate(items)]
    out = {"meta": meta, "summary": summ, "claims": per_claim}
    jp = out_dir / f"{run_id}.json"
    json.dump(out, open(jp, "w", encoding="utf-8"), indent=2, default=str)
    (out_dir / f"{run_id}.md").write_text(md_report(meta, summ, items, results), encoding="utf-8")
    print(f"wrote {jp}")
    return out


if __name__ == "__main__":
    main()
