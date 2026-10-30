"""Adaptive RAG Lab backend: Starlette app serving the static UI and a Server-Sent-Events run endpoint.

    python -m webapp.server            # http://127.0.0.1:8000
Environment: LLAMA_SERVER_URL (default http://127.0.0.1:8080/v1), ADAPTIVE_RAG_HOST, ADAPTIVE_RAG_PORT.
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from webapp.runner import ANSWER_RULES, LINUCB_CHECKPOINT, POLICY_INFO, SOURCES, STAGES, Backend, RunConfig, run_pipeline

STATIC = Path(__file__).parent / "static"
TRANSCRIPTS = Path(__file__).resolve().parent.parent / "experiments" / "webapp_runs"
LLAMA_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1")
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
BUDGET_PRESETS = {"low": 2, "medium": 4, "high": 8}
TERMINAL = {"result", "error", "cancelled"}


def _get(url: str, timeout: float = 3.0) -> Optional[Any]:
    try:
        import requests
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "AdaptiveRAGLab/1.0 (research demo)"})
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def check_health() -> Dict[str, Any]:
    llama = _get(f"{LLAMA_URL.rstrip('/')}/models", 2.5)
    qd = _get("http://localhost:6333/collections", 2.5)
    wiki = _get("https://en.wikipedia.org/w/api.php?action=query&meta=siteinfo&format=json", 8.0)
    cols = [c["name"] for c in (qd or {}).get("result", {}).get("collections", [])]
    try:
        from huggingface_hub import try_to_load_from_cache
        nli_ok = isinstance(try_to_load_from_cache(NLI_MODEL, "config.json"), str)
    except Exception:
        nli_ok = False
    return {"llama_server": {"ok": llama is not None, "url": LLAMA_URL, "needed_for": "draft generation and repair"},
            "qdrant": {"ok": qd is not None, "collections": cols, "needed_for": "HotpotQA / ASQA sources"},
            "wikipedia_api": {"ok": wiki is not None, "needed_for": "Wikipedia source (internet)"},
            "nli_model": {"ok": nli_ok, "model": NLI_MODEL, "needed_for": "NLI verifier (local files cached)"},
            "linucb_checkpoint": {"ok": LINUCB_CHECKPOINT.exists(), "path": str(LINUCB_CHECKPOINT.relative_to(LINUCB_CHECKPOINT.parents[2])), "needed_for": "LinUCB policy"}}


def build_config(health: Dict[str, Any]) -> Dict[str, Any]:
    cols = health["qdrant"]["collections"]
    sources = []
    for k, v in SOURCES.items():
        ok = health["wikipedia_api"]["ok"] if v["kind"] == "wikipedia" else (health["qdrant"]["ok"] and v["collection"] in cols)
        sources.append({"id": k, "label": v["label"], "available": bool(ok)})
    policies = [{"id": k, "label": v["label"], "note": v["note"], "learns": v["learns"], "available": health["linucb_checkpoint"]["ok"] if k == "LinUCB" else True} for k, v in POLICY_INFO.items()]
    verifiers = [{"id": "nli", "label": "NLI cross-encoder (DeBERTa-v3, local; slower on CPU)", "available": health["nli_model"]["ok"]},
                 {"id": "llama", "label": "Llama-3.2-1B assessor (llama-server; near chance in our diagnostics)", "available": health["llama_server"]["ok"]}]
    return {"sources": sources, "policies": policies, "verifiers": verifiers, "budgets": BUDGET_PRESETS, "answer_rules": ANSWER_RULES, "stages": STAGES,
            "unavailable_by_design": [{"id": "TypeThompson", "reason": "research control only (needs a training run); not exposed in the demo"},
                                      {"id": "oracle / simulated feedback", "reason": "the demo uses real verifier verdicts only; oracle and simulated verifiers exist only in the offline experiments"}]}


def make_app(backend: Optional[Backend] = None, health_fn=check_health) -> Starlette:
    backend = backend or Backend()

    async def index(request: Request):
        return FileResponse(STATIC / "index.html")

    async def health(request: Request):
        return JSONResponse(health_fn())

    async def config(request: Request):
        return JSONResponse(build_config(health_fn()))

    async def transcript(request: Request):
        """Serve a saved run (JSONL of real events) from experiments/webapp_runs so the UI can replay it offline: /?replay=<name>.jsonl"""
        name = request.query_params.get("name", "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.jsonl", name):
            return JSONResponse({"error": "bad transcript name"}, status_code=400)
        p = TRANSCRIPTS / name
        if not p.exists():
            return JSONResponse({"error": "transcript not found"}, status_code=404)
        return StreamingResponse(iter([p.read_bytes()]), media_type="text/plain; charset=utf-8")

    async def run(request: Request):
        try:
            body = await request.json()
            cfg = RunConfig(question=str(body.get("question", "")), source=body.get("source", "wikipedia"), budget=int(body.get("budget", 4)),
                            policy=body.get("policy", "LinUCB"), verifier=body.get("verifier", "nli"), answer_rule=body.get("answer_rule", "S0"),
                            seed=int(body.get("seed", 0)), reuse_prep=bool(body.get("reuse_prep", True)))
        except Exception as e:
            return JSONResponse({"error": f"bad request: {e}"}, status_code=400)
        if not backend.lock.acquire(blocking=False):
            return JSONResponse({"error": "another run is in progress; this demo runs one pipeline at a time"}, status_code=409)
        q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        cancel = threading.Event()
        seq = [0]
        t_start = __import__("time").monotonic()

        def emit(ev: Dict[str, Any]) -> None:
            seq[0] += 1
            q.put({**ev, "seq": seq[0], "t": round(__import__("time").monotonic() - t_start, 3)})

        def worker():
            try:
                run_pipeline(backend, cfg, emit, cancel)
            finally:
                backend.lock.release()
                q.put(None)

        threading.Thread(target=worker, daemon=True).start()

        def stream():
            try:
                while True:
                    try:
                        ev = q.get(timeout=15)
                    except queue.Empty:
                        yield ": keep-alive\n\n"
                        continue
                    if ev is None:
                        break
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
            finally:
                cancel.set()  # client went away (or stream finished): stop between controller steps

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return Starlette(routes=[Route("/", index), Route("/api/health", health), Route("/api/config", config), Route("/api/run", run, methods=["POST"]), Route("/api/transcript", transcript),
                             Mount("/static", StaticFiles(directory=str(STATIC)), name="static")])


def main():
    import uvicorn
    uvicorn.run(make_app(), host=os.environ.get("ADAPTIVE_RAG_HOST", "127.0.0.1"), port=int(os.environ.get("ADAPTIVE_RAG_PORT", "8000")), log_level="info")


if __name__ == "__main__":
    main()
