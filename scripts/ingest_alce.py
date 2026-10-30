#!/usr/bin/env python3
"""Ingest ALCE-supplied evidence passages into a SEPARATE Qdrant collection.

ALCE (Gao et al., 2023) ships, for every question, the top-100 retrieved passages (`docs`) that systems
are supposed to answer from. This script loads the first N questions of an ALCE file (e.g. ASQA) and
stores their supplied passages, de-duplicated by passage id, so the unchanged live pipeline
(QdrantRetriever -> draft -> claims -> controller -> synthesis) can run against them.

Safety:
  * writes ONLY to collections whose name starts with 'alce_'; HotpotQA collections are never touched;
  * refuses to write into an existing non-empty collection unless --recreate (which only deletes an alce_ collection);
  * self-contained (stdlib + sentence-transformers + qdrant-client), so it can be run as `python -I scripts/ingest_alce.py ...`
    when the input JSON came from a download.

Embedding model matches the pipeline's QdrantRetriever default (BAAI/bge-large-en-v1.5, cosine, 1024-d),
so query and passage vectors live in the same space.
"""

import argparse
import json
import os
import sys
import time

MODEL_NAME = "BAAI/bge-large-en-v1.5"


def load_questions(path, n, start=0):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a JSON list of questions, got {type(data).__name__}")
    return data[start:start + n]


def build_documents(questions, top_docs):
    """De-duplicate supplied passages by ALCE doc id (fallback: title+text). Returns (documents, stats)."""
    docs, seen = [], {}
    n_supplied = 0
    for qi, q in enumerate(questions):
        sid = str(q.get("sample_id", qi))
        for d in (q.get("docs") or [])[:top_docs]:
            n_supplied += 1
            text = str(d.get("text") or "").strip()
            title = str(d.get("title") or "Untitled").strip()
            if not text:
                continue
            key = str(d["id"]) if d.get("id") is not None else f"{title}\x1f{text[:80]}"
            if key in seen:
                docs[seen[key]]["sample_ids"].append(sid)
                continue
            seen[key] = len(docs)
            docs.append({"source_doc_id": key, "title": title, "text": text, "url": d.get("url"), "sample_ids": [sid]})
    return docs, {"questions": len(questions), "passages_supplied": n_supplied, "unique_passages": len(docs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help="ALCE json, e.g. .../asqa_eval_gtr_top100.json")
    ap.add_argument("--dataset-name", default="asqa")
    ap.add_argument("--n-questions", type=int, default=10)
    ap.add_argument("--start", type=int, default=0, help="index of the first question to ingest (default 0)")
    ap.add_argument("--top-docs", type=int, default=100, help="passages per question to keep (ALCE supplies 100)")
    ap.add_argument("--collection", default=None, help="default: alce_<dataset>_bge-large-en-v1.5")
    ap.add_argument("--recreate", action="store_true", help="delete and rebuild the (alce_) collection")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted ingestion: skip the points already stored (ids are sequential and deterministic)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=6333)
    ap.add_argument("--dry-run", action="store_true", help="parse and count only; no embedding, no Qdrant writes")
    args = ap.parse_args()

    collection = args.collection or f"alce_{args.dataset_name}_{MODEL_NAME.split('/')[-1]}"
    if not collection.startswith("alce_"):
        sys.exit(f"Refusing to write to '{collection}': ALCE collections must start with 'alce_' (protects existing collections).")

    questions = load_questions(args.file, args.n_questions, args.start)
    docs, stats = build_documents(questions, args.top_docs)
    print(f"{args.file}: {stats}")
    print("sample question:", questions[0].get("question"))
    if args.dry_run:
        print("dry-run: nothing embedded or written")
        return

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from sentence_transformers import SentenceTransformer

    client = QdrantClient(host=args.host, port=args.port, timeout=300)
    if client.collection_exists(collection):
        count = client.count(collection).count
        if count and not args.recreate and not args.resume:
            sys.exit(f"Collection '{collection}' already has {count} points. Use --recreate to rebuild it (alce_ collections only).")
        if args.recreate:
            client.delete_collection(collection)

    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") else model.get_sentence_embedding_dimension()
    if not client.collection_exists(collection):
        client.create_collection(collection_name=collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    t0 = time.monotonic()
    first = client.count(collection).count if args.resume else 0
    if first:
        print(f"resuming after {first} stored points")
    for start in range(first, len(docs), args.batch_size):
        batch = docs[start:start + args.batch_size]
        texts = [f"Title: {d['title']}\n\n{d['text']}" for d in batch]
        vecs = model.encode(texts, normalize_embeddings=True)
        points = ([
            PointStruct(
                id=start + i, vector=v.tolist(),
                payload={"text": t, "title": d["title"], "source_doc_id": d["source_doc_id"], "url": d["url"],
                         "alce_dataset": args.dataset_name, "sample_ids": d["sample_ids"][:20]},
            )
            for i, (d, t, v) in enumerate(zip(batch, texts, vecs))
        ])
        for attempt in range(4):  # Qdrant can time out under memory pressure; upserts are idempotent
            try:
                client.upsert(collection_name=collection, points=points)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 3:
                    raise
                print(f"  upsert failed ({type(exc).__name__}); retry {attempt + 1}/3 in {10 * (attempt + 1)}s", flush=True)
                time.sleep(10 * (attempt + 1))
        done = start + len(batch)
        el = time.monotonic() - t0
        print(f"  {done}/{len(docs)} passages  {el:.0f}s  (~{(len(docs) - done) * el / max(done, 1):.0f}s left)", flush=True)

    print(f"Done: collection '{collection}' now has {client.count(collection).count} points (dim {dim}).")
    print("questions ingested (sample_id | question):")
    for q in questions:
        print(f"  {q.get('sample_id')} | {q.get('question')}")


if __name__ == "__main__":
    main()
