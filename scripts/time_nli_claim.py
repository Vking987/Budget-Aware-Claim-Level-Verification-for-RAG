#!/usr/bin/env python3
"""Single-claim NLI timing check (no policy, no cache): model load, first (warm-up) call, repeat calls; records device/threads/CPU clock."""
import json, os, platform, subprocess, sys, time
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import run_ragtruth_comparison as R
from src.evaluation.ragtruth_adapter import RAGTruthDatasetAdapter
from src.evidence.nli_assessor import NLIAssessor

ad = RAGTruthDatasetAdapter(); ad.load_from_jsonl(R.DATA_DIR / "source_info.jsonl", R.DATA_DIR / "response.jsonl", split="test")
resp = sorted((r for r in ad.responses if r.source_id in ad.sources and r.labels), key=lambda r: r.response_id)[0]
prep, _ = R.prepare(ad, [resp]); p = prep[0]; c = p["claims"][0]; ev = p["evidence"][c.claim_id]
def clock():
    try: return int(subprocess.run(["powershell", "-c", "(Get-CimInstance Win32_Processor).CurrentClockSpeed"], capture_output=True, text=True).stdout.strip())
    except Exception: return None
import torch
a = NLIAssessor(); out = {"response_id": resp.response_id, "claim_id": c.claim_id, "evidence_chars": sum(len(e.source_text) for e in ev), "n_passages": len(ev),
     "device": "cpu (cuda available: %s)" % torch.cuda.is_available(), "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
     "logical_cpus": os.cpu_count(), "cpu": platform.processor(), "cpu_clock_mhz_before": clock(), "cache": "none (NLIAssessor has no cache; run_ragtruth_comparison.LLMCache wraps it)"}
t = time.monotonic(); a._load(); out["model_load_s"] = round(time.monotonic() - t, 2)
runs = []
for _ in range(3):
    t = time.monotonic(); r = a.assess(c, ev); runs.append({"wall_s": round(time.monotonic() - t, 2), "label": r.label.value, "tokens": r.cost.total_tokens, "error": r.is_error})
out["inference_runs"] = runs; out["cpu_clock_mhz_after"] = clock()
f = ROOT / "experiments/runs" / f"nli_timing_{datetime.now():%Y%m%d_%H%M%S}.json"; f.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2)); print(f)
