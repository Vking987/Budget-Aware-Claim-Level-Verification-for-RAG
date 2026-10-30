#!/usr/bin/env python3
"""Tiny SSE client for the Adaptive RAG Lab API: POST a run, print a one-line digest per event, save every event as JSONL.

usage: python scripts/webapp_client.py --question "..." --budget 4 --policy LinUCB --verifier nli --out experiments/webapp_runs/run.jsonl
"""
import argparse
import json
import sys

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--question", required=True)
    ap.add_argument("--source", default="wikipedia")
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--policy", default="LinUCB")
    ap.add_argument("--verifier", default="nli")
    ap.add_argument("--rule", default="S0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    body = {"question": a.question, "source": a.source, "budget": a.budget, "policy": a.policy, "verifier": a.verifier, "answer_rule": a.rule}
    with requests.post(a.url + "/api/run", json=body, stream=True, timeout=(10, 1800)) as r, open(a.out, "w", encoding="utf-8") as f:
        if r.status_code != 200:
            print("HTTP", r.status_code, r.text)
            return 1
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            f.write(json.dumps(ev) + "\n")
            f.flush()
            t = ev["type"]
            extra = {k: ev[k] for k in ("stage", "status", "round", "claim_id", "label", "model_calls_used", "message", "phase") if k in ev}
            if t == "round":
                extra["chosen"] = ev["chosen"]
            print(f"[{ev['t']:7.1f}s] {t} {extra}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
