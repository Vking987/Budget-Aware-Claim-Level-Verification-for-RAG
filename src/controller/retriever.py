"""Read-only Qdrant evidence retriever adapter for the sequential verification controller.

Connects to the pre-existing Qdrant collection without mutating points, vectors,
or collection schemas.

CRITICAL INVARIANTS:
1. Performs strictly READ-ONLY search operations.
2. Reuses existing collection 'hotpotqa_bge-large-en-v1.5'.
3. Safely extracts payloads with fallback for missing metadata.
4. Deduplicates duplicate passages within retrieved batches.
5. Emits structured EvidencePassage and VerificationCost objects.
6. Operates offline when provided with a mock client and mock encoder.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.claims.models import ClaimRecord
from src.evidence.models import EvidencePassage, VerificationCost
from .interfaces import BaseRetriever


class QdrantRetriever(BaseRetriever):
    """Read-only evidence retriever querying pre-ingested Qdrant vector index."""

    def __init__(
        self,
        client: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        collection_name: str = "hotpotqa_bge-large-en-v1.5",
        model_name: str = "BAAI/bge-large-en-v1.5",
        host: str = "localhost",
        port: int = 6333,
    ) -> None:
        """Initialize QdrantRetriever adapter.

        Args:
            client: Pre-configured QdrantClient or mock object. If None, lazily connects.
            embedding_model: Pre-loaded SentenceTransformer or mock encoder. If None, lazily loads.
            collection_name: Target Qdrant collection name.
            model_name: HuggingFace model identifier for query embedding.
            host: Qdrant service hostname.
            port: Qdrant service port.
        """
        self._client = client
        self._embedding_model = embedding_model
        self.collection_name = collection_name
        self.model_name = model_name
        self.host = host
        self.port = port

    @property
    def client(self) -> Any:
        """Lazily initialize QdrantClient if not provided."""
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(host=self.host, port=self.port)
        return self._client

    @property
    def embedding_model(self) -> Any:
        """Lazily load SentenceTransformer embedding model if not provided."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.model_name)
        return self._embedding_model

    def _encode_query(self, query_text: str) -> List[float]:
        """Encode query text into a normalized dense embedding vector."""
        model = self.embedding_model
        if hasattr(model, "encode"):
            vec = model.encode(query_text, normalize_embeddings=True)
            if hasattr(vec, "tolist"):
                return vec.tolist()
            return list(vec)
        elif callable(model):
            return list(model(query_text))
        raise TypeError(f"Embedding model must have an .encode() method or be callable, got {type(model)}")

    def _execute_search(self, query_vector: List[float], limit: int) -> List[Any]:
        """Perform a read-only similarity search on the Qdrant collection."""
        client = self.client
        # Support standard QdrantClient.search()
        if hasattr(client, "search"):
            return client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )
        # Support newer QdrantClient.query_points()
        elif hasattr(client, "query_points"):
            res = client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            return getattr(res, "points", [])
        raise AttributeError(f"Client object does not support search() or query_points(): {type(client)}")

    def retrieve(
        self,
        claim: ClaimRecord,
        top_k: int = 3,
    ) -> Tuple[List[EvidencePassage], VerificationCost]:
        """Fetch candidate evidence passages for a claim proposition.

        Args:
            claim: Target claim record containing claim text.
            top_k: Number of candidate passages to retrieve.

        Returns:
            Tuple of (retrieved_passages, verification_cost).
        """
        start_time = time.monotonic()
        query_text = (claim.normalized_text or claim.original_text or "").strip()

        if not query_text:
            elapsed = time.monotonic() - start_time
            return [], VerificationCost(
                model_calls=0,
                latency_seconds=elapsed,
                model_id=self.model_name,
            )

        # 1. Embed query (request 2x top_k to account for intra-batch deduplication)
        query_vec = self._encode_query(query_text)
        fetch_limit = max(top_k * 2, 6)

        # 2. Query Qdrant
        hits = self._execute_search(query_vector=query_vec, limit=fetch_limit)

        # 3. Parse and deduplicate retrieved hits
        passages: List[EvidencePassage] = []
        seen_keys: set = set()

        for hit in hits:
            hit_id = getattr(hit, "id", None)
            score = float(getattr(hit, "score", 0.0))
            payload = getattr(hit, "payload", None) or {}

            # Robust payload extraction with defaults
            title = str(payload.get("title") or "Untitled").strip()
            text = str(payload.get("text") or "").strip()
            hotpot_id = payload.get("hotpot_id")

            # Ignore empty or placeholder text
            if not text:
                continue

            # Deduplication key based on normalized title and text prefix
            norm_title = title.lower()
            norm_text_prefix = " ".join(text.lower().split()[:20])
            dedup_key = (norm_title, norm_text_prefix)

            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            doc_id_str = str(hit_id) if hit_id is not None else f"doc_{len(passages)}"
            evidence_id = f"qdrant_{doc_id_str}"

            passage = EvidencePassage(
                evidence_id=evidence_id,
                doc_title=title,
                source_text=text,
                retrieval_score=round(score, 5),
                metadata={
                    "doc_id": doc_id_str,
                    "hotpot_id": hotpot_id,
                    "collection": self.collection_name,
                    "raw_point_id": hit_id,
                    # corpus-level id when the collection stores one (e.g. ALCE doc ids); None for HotpotQA
                    "source_doc_id": payload.get("source_doc_id"),
                    "url": payload.get("url"),
                },
            )
            passages.append(passage)

            if len(passages) >= top_k:
                break

        elapsed = time.monotonic() - start_time
        cost = VerificationCost(
            model_calls=1,
            prompt_tokens=len(query_text.split()),
            completion_tokens=0,
            total_tokens=len(query_text.split()),
            latency_seconds=elapsed,
            model_id=self.model_name,
        )

        return passages, cost
