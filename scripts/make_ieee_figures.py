#!/usr/bin/env python3
"""IEEE-style figures from frozen result JSONs (read-only; new output files only).

fig1: detection recall vs verification budget, RAGTruth TEST (2,700 responses), LEXICAL verifier (ragtruth_cheap_verifier_*.json) + LinUCB advantage vs strongest baseline.
fig2: budget-2 NLI-verifier validation on the positive-enriched 150-response subset (ragtruth_nli_subset_*.json): broad (primary) and strict (secondary) recall + paired differences.
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.linewidth": .6, "font.family": "serif", "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .4})
BLUE, GREY, ORANGE, GREEN, LBLUE = "#1f5fbf", "#6b6b6b", "#c2410c", "#2f8f5b", "#7aa7e8"


def fig1(src: Path, out: Path):
    d = json.loads(src.read_text(encoding="utf-8")); B = sorted(d["budgets"], key=int); x = [int(b) for b in B]
    P = lambda b, k, m="detection_recall": d["budgets"][b]["policies"][k][m]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 4.6), gridspec_kw={"height_ratios": [1.5, 1]}, sharex=True)
    ser = [("LinUCB_oracle_trained", "LinUCB (frozen)", BLUE, "o"), ("StaticCTSRanked", "CTS-ranked", GREY, "^"), ("StaticFIFO", "FIFO", ORANGE, "v")]
    for k, lab, c, mk in ser:
        y = [P(b, k) for b in B]; ci = [P(b, k, "detection_recall_ci95") for b in B]
        a1.errorbar([v * (1.03 if k == "StaticFIFO" else 0.97 if k == "StaticCTSRanked" else 1) for v in x], y, yerr=[[a - q[0] for a, q in zip(y, ci)], [q[1] - a for a, q in zip(y, ci)]], color=c, marker=mk, ms=4, lw=1.2, capsize=2, label=lab)
    rm = [P(b, "Random(mean of seeds)") for b in B]
    lo = [min(P(b, f"Random({s})") for s in (0, 1, 2)) for b in B]; hi = [max(P(b, f"Random({s})") for s in (0, 1, 2)) for b in B]
    a1.errorbar(x, rm, yerr=[[m - l for m, l in zip(rm, lo)], [h - m for m, h in zip(rm, hi)]], color=GREEN, marker="D", ms=3.5, lw=1.2, capsize=2, ls="--", label="Random (mean)")
    a1.set_ylabel("Detection recall"); a1.legend(frameon=False, fontsize=6.5, loc="upper left")
    a1.set_title("RAGTruth test, 2,700 responses, 1,270 gold positives\nlexical verifier, repair off", fontsize=7.5)
    base = {"StaticCTSRanked": "CTS-ranked", "StaticFIFO": "FIFO", "Random(mean of seeds)": "Random (mean)"}
    xs, ys, cis, names = [], [], [], []
    for b in B:
        best = max(base, key=lambda k: P(b, k)); pr = d["budgets"][b]["paired"][f"LinUCB_oracle_trained - {best} [detection_recall]"]
        xs.append(int(b)); ys.append(100 * pr["diff"]); cis.append([100 * v for v in pr["ci95"]]); names.append(base[best])
    a2.errorbar(xs, ys, yerr=[[y - c[0] for y, c in zip(ys, cis)], [c[1] - y for y, c in zip(ys, cis)]], color=BLUE, marker="o", ms=4, lw=1.2, capsize=2)
    a2.axhline(0, color="k", lw=.6)
    for xx, yy, n in zip(xs, ys, names):
        a2.annotate(f"{yy:+.1f} pp\nvs {n}", (xx, yy), textcoords="offset points", xytext=(14 if xx < 8 else 0, 5 if yy >= 0 else -17), ha="center" if xx == 8 else "left", fontsize=6)
    a2.set_ylabel("LinUCB − strongest\nbaseline (pp)"); a2.set_xlabel("Verification calls per response (budget)")
    a2.set_xscale("log", base=2); a2.set_xticks(x); a2.set_xticklabels(B); a2.set_ylim(min(c[0] for c in cis) - 6, max(c[1] for c in cis) + 9); a2.set_xlim(1.6, 9.5)
    fig.text(0.01, 0.005, "95% CIs: bootstrap over responses (paired in lower panel).\nStrongest baseline picked per budget by point estimate.", fontsize=5.5)
    fig.tight_layout(rect=(0, .035, 1, 1)); fig.savefig(out, dpi=300); fig.savefig(out.with_suffix(".pdf")); print(out)


def fig4_nli(src: Path, out: Path):
    d = json.loads(src.read_text(encoding="utf-8")); n = d["manifest"]["n_responses"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw={"width_ratios": [1.1, 1]})
    pols = [("LinUCB", "LinUCB (frozen)", BLUE), ("StaticCTSRanked", "CTS-ranked", GREY), ("Random(mean)", "Random (3 seeds)", GREEN)]
    w = .36
    for j, (mode, hatch, lab) in enumerate([("broad", "", "not SUPPORTED (primary)"), ("strict", "///", "CONTRADICTED only (secondary)")]):
        for i, (k, nm, c) in enumerate(pols):
            p = d["detection"][mode]["policies"][k]; ci = p["recall_ci95"]
            a1.bar(i + (j - .5) * w, p["recall"], w * .92, color=c, alpha=1 if j == 0 else .55, hatch=hatch, edgecolor="white", lw=.4,
                   yerr=[[p["recall"] - ci[0]], [ci[1] - p["recall"]]], error_kw={"lw": .8, "capsize": 2}, label=lab if i == 0 else None)
    a1.set_xticks(range(3)); a1.set_xticklabels([p[1] for p in pols]); a1.set_ylabel("Detection recall of gold positives")
    a1.set_title(f"Budget 2, {n} responses (positive-enriched), DeBERTa NLI", fontsize=7.5); a1.legend(frameon=False, fontsize=6.5)
    rows = []
    for mode in ("broad", "strict"):
        for base, nm in (("StaticCTSRanked", "vs CTS-ranked"), ("Random(mean)", "vs Random")):
            pr = d["detection"][mode]["paired"][f"LinUCB - {base} [recall]"]; rows.append((f"{nm}\n({'primary' if mode == 'broad' else 'strict'})", pr["diff_pp"], pr["ci95_pp"], mode))
    for i, (lab, df, ci, mode) in enumerate(rows):
        a2.errorbar(df, i, xerr=[[df - ci[0]], [ci[1] - df]], fmt="o" if mode == "broad" else "s", color=BLUE if mode == "broad" else LBLUE, ms=4, capsize=2, lw=1.2)
    a2.axvline(0, color="k", lw=.6); a2.set_yticks(range(len(rows))); a2.set_yticklabels([r[0] for r in rows], fontsize=6.5); a2.invert_yaxis()
    a2.set_xlabel("LinUCB − baseline recall (pp)\npaired 95% CI, bootstrap over responses")
    a2.set_title("Predeclared rule: both primary CIs must exclude 0", fontsize=7.5)
    fig.tight_layout(); fig.savefig(out, dpi=300); fig.savefig(out.with_suffix(".pdf")); print(out)


# ----------------------------------------------------------------------------- Figure 2: ALCE-ASQA answer reconstruction (replay of cached verdicts; no model)
ROOT = Path(__file__).resolve().parent.parent
ALCE_REPLAY = ROOT / "experiments/runs/alce_e2e_20261029_133740_replay_full.json"
STRATS = [("S0_current", "S0 current", GREY), ("S1_conservative", "S1 conservative", BLUE), ("S2_repair_first", "S2 repair-first", LBLUE)]


def alce_tables(src: Path = ALCE_REPLAY, n_boot: int = 2000, seed: int = 0):
    """Per-question aggregates and bootstrap summaries from the replay JSON. Resampling unit = question; seeds are averaged within a policy and the
    8 policies with equal weight BEFORE resampling (policies share drafts and verdicts, so they are never treated as independent).
    Retention = mean over questions of (claims in final answer / claims in draft); questions with 0 draft claims would be excluded (count reported)."""
    import numpy as np
    d = json.loads(Path(src).read_text(encoding="utf-8"))
    for key in ("manifest", "summary", "per_question_runs"):
        if key not in d:
            raise ValueError(f"replay JSON lacks '{key}'")
    R = d["per_question_runs"]
    pols = sorted({k.split("|", 1)[1] for k in R})
    qs = sorted({r["sample_id"] for runs in R.values() for run in runs for r in run})
    qi = {q: i for i, q in enumerate(qs)}
    em = {s: np.zeros((len(pols), len(qs))) for s, _, _ in STRATS}
    ia = {s: np.zeros((len(pols), len(qs))) for s, _, _ in STRATS}
    nc = np.zeros((len(pols), len(qs)))
    for i, p in enumerate(pols):
        runs = R["4|" + p]
        for run in runs:
            if {r["sample_id"] for r in run} != set(qs):
                raise ValueError(f"a run for {p} does not cover all questions")
            for r in run:
                j = qi[r["sample_id"]]
                nc[i, j] += r["n_claims"] / len(runs)
                for s, _, _ in STRATS:
                    st = r["strategies"].get(s)
                    if st is None or st.get("str_em") is None or st.get("n_in_answer") is None:
                        raise ValueError(f"missing {s} fields for {p} / {r['sample_id']}")
                    em[s][i, j] += st["str_em"] / len(runs)
                    ia[s][i, j] += st["n_in_answer"] / len(runs)
    valid = nc.mean(0) > 0
    nv = int(valid.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(qs), size=(n_boot, len(qs)))
    ridx = rng.integers(0, nv, size=(n_boot, nv))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    ret_q = {s: ia[s].mean(0)[valid] / nc.mean(0)[valid] for s, _, _ in STRATS}
    em_q = {s: em[s].mean(0) for s, _, _ in STRATS}
    out = {"n_questions": len(qs), "questions_with_zero_draft_claims": len(qs) - nv, "policies": pols, "seeds_per_policy": {p: len(R["4|" + p]) for p in pols},
           "draft_str_em": d["summary"]["draft_str_em"], "draft_claims_per_question": float(nc.mean(0).mean()), "strategies": {},
           "bootstrap": {"unit": "question", "n_boot": n_boot, "seed": seed, "method": "percentile; policies/seeds averaged within question before resampling"}}
    for s, _, _ in STRATS:
        out["strategies"][s] = {"str_em": float(em_q[s].mean()), "str_em_ci95": ci(em_q[s][idx].mean(1)), "str_em_by_policy": {p: float(em[s][i].mean()) for i, p in enumerate(pols)},
                                "retention": float(ret_q[s].mean()), "retention_ci95": ci(ret_q[s][ridx].mean(1)),
                                "retention_pooled_ratio": float(ia[s].mean(0).sum() / nc.mean(0).sum()), "claims_in_answer_per_question": float(ia[s].mean(0).mean())}
    s0, s1, s2 = (em_q[k] for k in ("S0_current", "S1_conservative", "S2_repair_first"))
    out["paired"] = {"S1-S0 str_em": {"diff": float((s1 - s0).mean()), "ci95": ci((s1 - s0)[idx].mean(1))},
                     "S2-S1 str_em": {"diff": float((s2 - s1).mean()), "ci95": ci((s2 - s1)[idx].mean(1)),
                                      "max_abs_policy_question_diff": float(np.abs(em["S2_repair_first"] - em["S1_conservative"]).max())},
                     "S1-S0 retention": {"diff": float((ret_q["S1_conservative"] - ret_q["S0_current"]).mean()), "ci95": ci((ret_q["S1_conservative"] - ret_q["S0_current"])[ridx].mean(1))}}
    return out


def fig2(src, out: Path):
    t = alce_tables(Path(src))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 3.0))
    for ax, key, ylab, title in ((a1, "str_em", "STR-EM (answer coverage)", "(a) Answer coverage"), (a2, "retention", "Claims in answer / claims in draft", "(b) Claim retention")):
        for i, (s, nm, c) in enumerate(STRATS):
            v, ci = t["strategies"][s][key], t["strategies"][s][key + "_ci95"]
            ax.bar(i, v, .62, color=c, edgecolor="white", lw=.4, yerr=[[v - ci[0]], [ci[1] - v]], error_kw={"lw": .8, "capsize": 2.5})
            ax.text(i + .33, v, f"{v:.3f}" if key == "str_em" else f"{v:.2f}", ha="left", va="center", fontsize=6.5)
        if key == "str_em":
            for si, (s, _, _) in enumerate(STRATS):  # one dot per policy: spread across policies, NOT independent replicates
                ys = list(t["strategies"][s]["str_em_by_policy"].values())
                ax.scatter([si + (k - 3.5) * .06 for k in range(len(ys))], ys, s=5, color="k", alpha=.6, zorder=3, label="individual policy (n=8)" if si == 0 else None)
            ax.axhline(t["draft_str_em"], color=ORANGE, lw=.9, ls="--", label="unverified draft (all claims kept)")
            ax.legend(frameon=False, fontsize=6, loc="upper left")
        ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n, _ in STRATS], fontsize=7); ax.set_ylabel(ylab); ax.set_title(title, fontsize=8)
        ax.set_ylim(0, .45 if key == "str_em" else .65); ax.set_xlim(-.6, 2.95)
    fig.text(.01, .005, f"ASQA[10:40], {t['n_questions']} questions, budget 4, DeBERTa-NLI verdicts replayed (no new model call). Bars: mean over {len(t['policies'])} policies (seeds averaged);\n"
             "error bars: 95% bootstrap over questions. Retention = claims kept / draft claims (per-question mean). A replay of one verified run, not independent evaluations.\n"
             "STR-EM is coverage of gold short answers, not factual support; human evidence-support labels are incomplete.", fontsize=5.2)
    fig.tight_layout(rect=(0, .1, 1, 1)); fig.savefig(out, dpi=300, metadata={"Software": None}); fig.savefig(out.with_suffix(".pdf"), metadata={"CreationDate": None}); print(out)


# ----------------------------------------------------------------------------- Figure 3: architecture as implemented (src/pipeline.py, src/controller/, src/synthesis/)
def fig3(src, out: Path):
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.1, 4.2)); ax.set_xlim(0, 100); ax.set_ylim(-3, 58); ax.axis("off")

    def box(x, y, w, h, text, fc="#eef2f8", ec=GREY, fs=6.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fc, ec=ec, lw=.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.2)

    def arrow(x1, y1, x2, y2, c="k", ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=.8, color=c, ls=ls, shrinkA=0, shrinkB=0))

    y, h = 45, 9
    box(0, y, 14, h, "Question +\nretrieved evidence\n(dense top-k)", fc="#f7f7f7"); box(17, y, 13, h, "Initial draft\n(1B LLM)")
    box(33, y, 15, h, "Content filter +\nclaim decomposition\n(rule-based)"); box(51, y, 14, h, "CTS triage score\n+ claim type\n(rule-based)")
    box(68, y, 32, h, "Context features x (8): CTS, claim-type risk,\nevidence coverage, assessment status,\ninteraction history, budget left, action affinity", fc="#e8f0fb", ec=BLUE, fs=5.4)
    for a, b in ((14, 17), (30, 33), (48, 51), (65, 68)):
        arrow(a, y + 4.5, b, y + 4.5)
    ax.add_patch(FancyBboxPatch((0, 8), 100, 33, boxstyle="round,pad=0.3,rounding_size=1", fc="none", ec=BLUE, lw=1, ls="--"))
    ax.text(14, 38.3, "Verification controller: one (claim, action) decision per step until the budget is exhausted", fontsize=6.2, color=BLUE, va="center")
    box(2, 18, 20, 13, "Selection policy\nLinUCB: argmax UCB(x)\n(frozen at test time)\nbaselines: CTS-ranked,\nFIFO, Random", fc="#e8f0fb", ec=BLUE)
    arrow(8, 44.4, 8, 31.5, c=BLUE, ls="--"); ax.text(9, 41.6, "x, claim state", fontsize=5.2, color=BLUE)
    box(27, 28, 20, 6.5, "VERIFY: assessor\n(DeBERTa NLI or 1B LLM)"); box(27, 19.5, 20, 6.5, "RETRIEVE more evidence"); box(27, 11, 20, 6.5, "DEFER / STOP")
    arrow(22, 27, 27, 31); arrow(22, 24.5, 27, 23); arrow(22, 21, 27, 14.5)
    box(51, 28, 24, 6.5, "Verdict: SUPPORTED /\nINSUFFICIENT / CONTRADICTED", fs=5.6); arrow(47, 31.2, 51, 31.2)
    box(51, 16, 24, 9.5, "REPAIR (1B LLM) of a\nflagged claim, then\nre-VERIFY the revision\n(one more call, only if affordable)", fs=5.8)
    arrow(63, 28, 63, 25.6); ax.text(64, 26.8, "flagged", fontsize=5.2, color=ORANGE)
    box(79, 20, 20, 15, "Budget tracker\nmax_model_calls,\nmax_steps; every VERIFY /\nRETRIEVE / REPAIR\nconsumes calls; actions\nover budget are refused", fc="#fff4e5", ec=ORANGE, fs=5.4)
    ax.plot([89, 89, 12], [20, 9.8, 9.8], color=ORANGE, lw=.8, ls="--"); arrow(12, 9.8, 12, 17.7, c=ORANGE, ls="--")
    ax.text(34, 9.8 - 1.6, "remaining budget feeds the policy (budget-left feature) and halts the loop", fontsize=5.2, color=ORANGE, ha="center")
    arrow(63, 16, 63, 3.2, c=GREEN)
    box(2, -2.6, 70, 5.6, "Answer reconstruction: verified claims + re-verified repairs; INSUFFICIENT kept (S1) or dropped (S0);\nCONTRADICTED excluded; citations point to supporting evidence", fc="#eef6ee", ec=GREEN, fs=5.6)
    box(76, -2.6, 24, 5.6, "Final answer\n(stitched claims, cited)", fc="#eef6ee", ec=GREEN, fs=6.0)
    arrow(72, 0.2, 76, 0.2, c=GREEN)
    ax.text(0, 57.2, "Implemented pipeline (src/pipeline.py, src/controller/, src/synthesis/). Gold labels are never used at inference.", fontsize=6.2, fontstyle="italic", va="center")
    fig.text(.01, .004, "ALCE runs stitch claims deterministically; the demo can add an LLM rewrite with citation validation. Retrieval/generation are upstream of the claim-level controller.", fontsize=5.0)
    fig.tight_layout(rect=(0, .025, 1, 1), pad=.2); fig.savefig(out, dpi=300, metadata={"Software": None}); fig.savefig(out.with_suffix(".pdf"), metadata={"CreationDate": None}); print(out)


if __name__ == "__main__":
    which = sys.argv[1]
    src = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    defaults = {"fig2": (ALCE_REPLAY, ROOT / "experiments/runs/ieee_fig2_answer_reconstruction.png"), "fig3": (None, ROOT / "experiments/runs/ieee_fig3_system_architecture.png")}
    if which in defaults:
        src = src or defaults[which][0]
        out = out or defaults[which][1]
    {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4_nli": fig4_nli}[which](src, out)
