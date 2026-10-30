# Active RAG Verification System

A budgeted, claim-level active verification and synthesis framework for Retrieval-Augmented Generation (RAG).

## Quick Start & Demo

### 1. Start Prerequisites

Make sure the virtual environment is activated:
```powershell
.venv\Scripts\Activate.ps1
```

Start the local background services (if not already running):

**Qdrant Vector Database:**
```powershell
qdrant.exe
```

**llama.cpp Server:**
```powershell
llama-server.exe -m C:\Users\vrajs\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf --port 8080 --host 127.0.0.1 -c 2048 -ngl 99
```

### 2. Run the Demo

**Single Question (Live with default LinUCB policy):**
```powershell
.venv\Scripts\python.exe demo.py --question "Are Wenling and Xinzheng both in China?"
```

**Interactive Mode:**
```powershell
.venv\Scripts\python.exe demo.py --interactive
```

**Select a Specific Policy:**
```powershell
.venv\Scripts\python.exe demo.py --question "What is the capital of France?" --policy StaticCTSRanked
```
Supported `--policy` values: `ContextualMAB_LinUCB` (default), `DeterministicCTS`, `StaticCTSRanked`, `StaticFIFO`, `ExhaustiveReference`, `VanillaRAG`.

**Offline Dry-Run (Mock Backend):**
```powershell
.venv\Scripts\python.exe demo.py --question "What is the capital of France?" --dry-run
```

**Run the Automated 5-Question Live Demo Suite:**
```powershell
.venv\Scripts\python.exe scripts/run_live_demo_suite.py
```

### 3. Run Tests

Execute the full test suite (165 tests):
```powershell
.venv\Scripts\python.exe -m pytest
```

To run offline unit tests only (excluding live server checks):
```powershell
.venv\Scripts\python.exe -m pytest -m "not live"
```

## System Documentation

For detailed architecture, policy math, evaluation protocols, and benchmark results, see:
- Comprehensive Project Report: [docs/final_project_report.md](file:///C:/Users/vrajs/rag-research/docs/final_project_report.md)
- Controller Architecture: [docs/controller_architecture.md](file:///C:/Users/vrajs/rag-research/docs/controller_architecture.md)
- MAB Policy & LinUCB: [docs/mab_policy.md](file:///C:/Users/vrajs/rag-research/docs/mab_policy.md)
- Warm-Start Training Report: [docs/stage_d5_warmstart_training_report.md](file:///C:/Users/vrajs/rag-research/docs/stage_d5_warmstart_training_report.md)

---

## Adaptive RAG Lab (browser demo)

An interactive dashboard over the real pipeline: question -> retrieval (live Wikipedia or local Qdrant) -> 1B-model draft -> rule-based claim decomposition and CTS -> a selection policy chooses which claim to verify under a call budget -> NLI/LLM verification -> repair and re-verification -> final answer rebuilt from the claim decisions. Everything is streamed from the backend as it happens (Server-Sent Events). Code: `webapp/` (backend `server.py`, `runner.py`, `wikipedia_retriever.py`; UI `static/index.html`). Details of what is real/limited: `docs/adaptive_rag_lab.md`.

### Setup (Windows, PowerShell, from the repo root)
```powershell
# 1) Python packages already used by this repo (starlette, uvicorn, requests, torch, transformers, sentence-transformers, qdrant-client, openai).
#    If uvicorn/starlette are missing:  pip install uvicorn starlette requests
# 2) Start the local LLM server (draft generation and repair; same command as the CLI demo):
#      llama-server -m C:\Users\<you>\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf --port 8080 -c 4096
#    (set LLAMA_SERVER_URL if it is elsewhere; default http://127.0.0.1:8080/v1)
# 3) Optional, only for the HotpotQA / ASQA sources: Qdrant on port 6333 (the same instance the CLI uses).
#    Wikipedia (live) needs internet access only. No API key is required for anything.
# 4) First use of the NLI verifier needs the model files (about 700 MB, cached afterwards):
python -c "from huggingface_hub import snapshot_download; snapshot_download('MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli')"
# 5) Launch:
python -m webapp.server
# open http://127.0.0.1:8000   (ADAPTIVE_RAG_HOST / ADAPTIVE_RAG_PORT change the address)
```
The status chips at the top of the page show which services are reachable; unavailable choices are disabled. Only one pipeline run executes at a time. On a slow CPU the NLI verifier takes roughly 20-50 s per claim, so a budget of 4 takes a few minutes; choose the Llama verifier for speed (it was near chance in our diagnostics).

Command-line client for the same API (prints one line per event and saves the JSONL stream): `python scripts/webapp_client.py --question "..." --budget 4 --policy LinUCB --verifier nli --out experiments/webapp_runs/run.jsonl`.

Tests: `python -m pytest tests/test_webapp.py` (scripted components, no network or model).
