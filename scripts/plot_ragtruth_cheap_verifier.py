#!/usr/bin/env python3
"""Plot detection recall vs budget (with 95% CIs) from a ragtruth_cheap_verifier_*.json result."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = Path(sys.argv[1]); out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".png")
d = json.loads(src.read_text(encoding="utf-8")); B = sorted(d["budgets"], key=int)
series = [("LinUCB_oracle_trained", "LinUCB (oracle-trained)", "#1f5fbf", "o"), ("LinUCB_lexical_trained", "LinUCB (verifier-trained)", "#7aa7e8", "s"),
          ("StaticCTSRanked", "CTS-ranked", "#6b6b6b", "^"), ("StaticFIFO", "FIFO / VerifyAll", "#c2410c", "v"), ("Random(0)", "Random (seed 0)", "#2f8f5b", "D")]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for k, lab, col, mk in series:
    y = [d["budgets"][b]["policies"][k]["detection_recall"] for b in B]
    ci = [d["budgets"][b]["policies"][k]["detection_recall_ci95"] for b in B]
    ax.errorbar([int(b) for b in B], y, yerr=[[a - c[0] for a, c in zip(y, ci)], [c[1] - a for a, c in zip(y, ci)]], label=lab, color=col, marker=mk, capsize=3, lw=1.6, ms=5)
ax.set_xscale("log", base=2); ax.set_xticks([int(b) for b in B]); ax.set_xticklabels(B)
ax.set_xlabel("verification calls per response"); ax.set_ylabel("detection recall of gold-hallucinated claims")
ax.set_title("RAGTruth test (2,700 responses, 1,270 positives), lexical verifier", fontsize=10)
ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(out, dpi=150); print(out)
