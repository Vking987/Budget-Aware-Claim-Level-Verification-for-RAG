"""RAGTruth dataset adapter and claim-to-annotation mapping track.

Integrates ParticleMedia/RAGTruth data into the existing rag-research architecture
without modifying the controller, LinUCB policy, or HotpotQA baseline artifacts.

Features:
- Preserves official example IDs, split membership (train/test), models, quality, and tasks.
- Preserves source context, questions, prompts, and provenance.
- Decomposes responses into atomic claims using existing ContentBearingFilter and RuleBasedDecomposer.
- Maps word/span annotations onto decomposed claims with transparent, auditable overlap rules.
- Distinguishes mapping status/confidence from factual confidence.
- Marks ambiguous or multi-claim cross-cutting annotations as UNCERTAIN or MIXED.
- Computes claim-level evaluation metrics without calling LLM judges 'ground truth'.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.claims.decomposer import RuleBasedDecomposer
from src.claims.filter import ContentBearingFilter
from src.claims.models import ClaimRecord, ClaimState, ClaimType, DecompositionStatus, SentenceRecord
from src.claims.triage import ClaimTriageScorer


class AnnotationLabelType(str, Enum):
    """Official RAGTruth hallucination category types."""
    EVIDENT_CONFLICT = "Evident Conflict"
    SUBTLE_CONFLICT = "Subtle Conflict"
    EVIDENT_BASELESS = "Evident Baseless Info"
    SUBTLE_BASELESS = "Subtle Baseless Info"
    UNKNOWN = "Unknown"


class ClaimAnnotationLabel(str, Enum):
    """Mapped reference label assigned to a decomposed claim based on RAGTruth annotations.
    
    IMPORTANT: Dataset annotation labels must NOT be conflated with evidence-assessor verdicts
    (e.g., SUPPORTED, CONTRADICTED, INSUFFICIENT). The absence of a hallucination annotation
    means only that no annotator flagged the span; it does NOT constitute proof of factual
    grounding or evidence support.

    - UNFLAGGED: No hallucination annotations overlap this claim.
    - HALLUCINATED: Overlapped by valid hallucination annotations with high coverage/significance.
    - IMPLICIT_TRUE_ABSENT: Overlapped by annotations explicitly marked `implicit_true` (factually
      true proposition according to world knowledge, but absent from retrieved context; not an
      evident conflict or fabrication).
    - MIXED: Claim overlaps multiple conflicting annotations.
    - UNCERTAIN: Claim boundary or annotation overlap is ambiguous, malformed, or crosses multiple claims.
    """
    UNFLAGGED = "unflagged"
    HALLUCINATED = "hallucinated"
    IMPLICIT_TRUE_ABSENT = "implicit_true_absent"
    MIXED = "mixed"
    UNCERTAIN = "uncertain"


class MappingStatus(str, Enum):
    """Auditable mapping status reflecting character offset alignment reliability."""
    CLEAN_UNANNOTATED = "clean_unannotated"
    EXACT_SPAN_MATCH = "exact_span_match"
    CONTAINED_SPAN = "contained_span"
    MULTI_CLAIM_SPAN = "multi_claim_span"
    PARTIAL_OVERLAP = "partial_overlap"
    MALFORMED_SPAN = "malformed_span"
    OUT_OF_BOUNDS = "out_of_bounds"
    UNKNOWN = "unknown"


@dataclass
class RAGTruthLabelSpan:
    """Represents a single raw human annotation span from RAGTruth response.jsonl."""
    span_id: str
    text: str
    start: int
    end: int
    label_type: str
    due_to_null: bool = False
    implicit_true: bool = False
    meta: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], span_idx: int) -> RAGTruthLabelSpan:
        return cls(
            span_id=f"span_{span_idx}",
            text=str(data.get("text", "")),
            start=int(data.get("start", -1)),
            end=int(data.get("end", -1)),
            label_type=str(data.get("label_type", AnnotationLabelType.UNKNOWN.value)),
            due_to_null=bool(data.get("due_to_null", False)),
            implicit_true=bool(data.get("implicit_true", False)),
            meta=data.get("meta"),
        )


@dataclass
class RAGTruthSourceRecord:
    """Represents a raw source information record from source_info.jsonl."""
    source_id: str
    task_type: str
    source: str
    source_info: Dict[str, Any] | str
    prompt: Optional[str] = None
    question: Optional[str] = None
    passages: Optional[str | List[str]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RAGTruthSourceRecord:
        s_info = data.get("source_info", {})
        question = None
        passages = None
        if isinstance(s_info, dict):
            question = s_info.get("question")
            passages = s_info.get("passages")
        return cls(
            source_id=str(data.get("source_id", "")),
            task_type=str(data.get("task_type", "")),
            source=str(data.get("source", "")),
            source_info=s_info,
            prompt=data.get("prompt"),
            question=question,
            passages=passages,
        )

    def get_source_text(self) -> str:
        """Extract full source text context regardless of task type."""
        if isinstance(self.source_info, str):
            return self.source_info
        if isinstance(self.source_info, dict):
            if "passages" in self.source_info:
                p = self.source_info["passages"]
                if isinstance(p, list):
                    return "\n\n".join(str(x) for x in p)
                return str(p)
            return json.dumps(self.source_info, ensure_ascii=False)
        return str(self.source_info or "")

    def get_evidence_passages(self) -> List[Any]:
        """Convert source context into EvidencePassage objects for verification."""
        from src.evidence.models import EvidencePassage
        text = self.get_source_text()
        title = self.source or f"Source_{self.source_id}"
        if not text.strip():
            return []
        # Create chunked or single evidence passage
        # For evaluation, return passage with source_id and clean title
        return [
            EvidencePassage(
                evidence_id=f"ragtruth_{self.source_id}_p0",
                doc_title=title,
                source_text=text,
                retrieval_score=1.0,
                metadata={"source_id": self.source_id, "task_type": self.task_type},
            )
        ]


@dataclass
class RAGTruthResponseRecord:
    """Represents a raw LLM response record with annotations from response.jsonl."""
    response_id: str
    source_id: str
    model: str
    temperature: float
    split: str
    quality: str
    response: str
    labels: List[RAGTruthLabelSpan] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RAGTruthResponseRecord:
        raw_labels = data.get("labels", [])
        parsed_labels = [
            RAGTruthLabelSpan.from_dict(lbl, idx) for idx, lbl in enumerate(raw_labels)
        ]
        return cls(
            response_id=str(data.get("id", "")),
            source_id=str(data.get("source_id", "")),
            model=str(data.get("model", "")),
            temperature=float(data.get("temperature", 0.0)),
            split=str(data.get("split", "test")),
            quality=str(data.get("quality", "good")),
            response=str(data.get("response", "")),
            labels=parsed_labels,
        )


@dataclass
class MappedClaimAnnotation:
    """A decomposed ClaimRecord mapped to ground-truth RAGTruth annotations."""
    example_id: str  # source_id
    response_id: str
    claim_id: str
    claim_text: str
    char_start: int
    char_end: int
    mapped_label: ClaimAnnotationLabel
    mapping_status: MappingStatus
    mapping_confidence: float  # Quality of span alignment (distinct from model/factual confidence)
    mapped_span_ids: List[str] = field(default_factory=list)
    source_spans: List[Dict[str, Any]] = field(default_factory=list)
    claim_record: Optional[ClaimRecord] = None
    triage_score: Optional[float] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "response_id": self.response_id,
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "mapped_label": self.mapped_label.value,
            "mapping_status": self.mapping_status.value,
            "mapping_confidence": round(self.mapping_confidence, 4),
            "mapped_span_ids": self.mapped_span_ids,
            "source_spans": self.source_spans,
            "claim_record": self.claim_record.to_dict() if self.claim_record else None,
            "triage_score": self.triage_score,
            "notes": self.notes,
        }


class ClaimAnnotationMapper:
    """Maps decomposed claims to character-level hallucination span annotations."""

    def __init__(
        self,
        min_overlap_char_ratio: float = 0.20,
        content_filter: Optional[ContentBearingFilter] = None,
        decomposer: Optional[RuleBasedDecomposer] = None,
        triage_scorer: Optional[ClaimTriageScorer] = None,
    ) -> None:
        self.min_overlap_char_ratio = min_overlap_char_ratio
        self.content_filter = content_filter or ContentBearingFilter()
        self.decomposer = decomposer or RuleBasedDecomposer()
        self.triage_scorer = triage_scorer or ClaimTriageScorer()

    def map_response(
        self,
        response_rec: RAGTruthResponseRecord,
        source_rec: Optional[RAGTruthSourceRecord] = None,
    ) -> List[MappedClaimAnnotation]:
        """Decompose a response and map annotations to claims."""
        text = response_rec.response
        labels = response_rec.labels
        text_len = len(text)

        # 1. Validate raw annotation spans
        valid_labels: List[RAGTruthLabelSpan] = []
        malformed_labels: List[Tuple[RAGTruthLabelSpan, MappingStatus, str]] = []

        for lbl in labels:
            if lbl.start < 0 or lbl.end > text_len or lbl.start >= lbl.end:
                malformed_labels.append((lbl, MappingStatus.OUT_OF_BOUNDS, "Span offsets outside text boundaries"))
                continue
            actual_sub = text[lbl.start:lbl.end]
            if actual_sub != lbl.text:
                # Text mismatch: could be Unicode normalization or formatting anomaly
                malformed_labels.append((lbl, MappingStatus.MALFORMED_SPAN, f"Span text mismatch: expected '{lbl.text}', found '{actual_sub}'"))
                continue
            valid_labels.append(lbl)

        # 2. Extract sentences with character spans
        sentence_records = self.content_filter.process(text)
        if not sentence_records and text.strip():
            sentence_records = [
                SentenceRecord(
                    sentence_id="s_0",
                    original_text=text.strip(),
                    cleaned_text=text.strip(),
                    is_content_bearing=True,
                    filter_reason="fallback_whole_text",
                    char_start=0,
                    char_end=text_len,
                )
            ]

        # 3. Decompose sentences into claims with span tracking
        decomposed_claims_with_spans: List[Tuple[ClaimRecord, int, int]] = []
        for sent in sentence_records:
            if not sent.is_content_bearing:
                continue
            sub_claims = self.decomposer.decompose(sent)
            sent_start = sent.char_start
            sent_end = sent.char_end

            if len(sub_claims) <= 1:
                for c in sub_claims:
                    decomposed_claims_with_spans.append((c, sent_start, sent_end))
            else:
                # Multiple sub-claims: locate offsets within sentence or apportion
                for c_idx, c in enumerate(sub_claims):
                    c_text = c.original_text.strip().rstrip(".")
                    idx = sent.original_text.find(c_text)
                    if idx != -1:
                        c_start = sent_start + idx
                        c_end = c_start + len(c_text)
                    else:
                        # Shared subject or ellision: estimate span range
                        c_start = sent_start
                        c_end = sent_end
                    decomposed_claims_with_spans.append((c, c_start, c_end))

        # 4. Map annotations onto claims
        mapped_claims: List[MappedClaimAnnotation] = []
        context_str = ""
        if source_rec and source_rec.passages:
            context_str = str(source_rec.passages)

        for c, c_start, c_end in decomposed_claims_with_spans:
            c_len = max(1, c_end - c_start)
            overlapping_spans: List[Tuple[RAGTruthLabelSpan, int, float]] = []

            for lbl in valid_labels:
                ov_start = max(c_start, lbl.start)
                ov_end = min(c_end, lbl.end)
                ov_len = max(0, ov_end - ov_start)
                if ov_len > 0:
                    span_len = max(1, lbl.end - lbl.start)
                    span_coverage = ov_len / span_len
                    overlapping_spans.append((lbl, ov_len, span_coverage))

            # Compute CTS triage score
            # score_claim returns a float directly and mutates c.triage_score / c.triage_features in-place.
            try:
                c_score = self.triage_scorer.score_claim(c, context_text=context_str)
                # c.triage_score and c.triage_features are set in-place by score_claim
            except Exception as exc:  # noqa: BLE001
                c_score = None
                c.triage_score = None

            # Check if any malformed label affected this claim
            c_malformed = [
                m for m in malformed_labels
                if (m[0].start < c_end and m[0].end > c_start) or m[1] == MappingStatus.OUT_OF_BOUNDS
            ]

            if c_malformed and not overlapping_spans:
                mapped_claims.append(
                    MappedClaimAnnotation(
                        example_id=response_rec.source_id,
                        response_id=response_rec.response_id,
                        claim_id=c.claim_id,
                        claim_text=c.original_text,
                        char_start=c_start,
                        char_end=c_end,
                        mapped_label=ClaimAnnotationLabel.UNCERTAIN,
                        mapping_status=c_malformed[0][1],
                        mapping_confidence=0.0,
                        mapped_span_ids=[m[0].span_id for m in c_malformed],
                        source_spans=[asdict(m[0]) for m in c_malformed],
                        claim_record=c,
                        triage_score=c_score,
                        notes=f"Overlapped by malformed span: {c_malformed[0][2]}",
                    )
                )
                continue

            if not overlapping_spans:
                # No overlapping hallucination spans -> UNFLAGGED (not asserted as verified/supported)
                mapped_claims.append(
                    MappedClaimAnnotation(
                        example_id=response_rec.source_id,
                        response_id=response_rec.response_id,
                        claim_id=c.claim_id,
                        claim_text=c.original_text,
                        char_start=c_start,
                        char_end=c_end,
                        mapped_label=ClaimAnnotationLabel.UNFLAGGED,
                        mapping_status=MappingStatus.CLEAN_UNANNOTATED,
                        mapping_confidence=1.0,
                        mapped_span_ids=[],
                        source_spans=[],
                        claim_record=c,
                        triage_score=c_score,
                        notes="No hallucination annotations detected in claim span (unflagged; does not imply verified support)",
                    )
                )
                continue

            # Analyze overlapping spans
            span_ids = [sp[0].span_id for sp in overlapping_spans]
            spans_dict = [asdict(sp[0]) for sp in overlapping_spans]

            # Check if any span crosses multiple claims (coverage of the annotation span across multiple claims)
            # A span is multi-claim if the annotation span's length is strictly greater than the overlap and crosses boundaries
            is_multi_claim = any(sp[2] < 0.90 for sp in overlapping_spans)
            total_ov_chars = sum(sp[1] for sp in overlapping_spans)
            claim_ov_ratio = total_ov_chars / c_len

            # Check if all overlapping spans are fully contained within this claim
            all_spans_fully_contained = all(
                (lbl.start >= c_start and lbl.end <= c_end) for lbl, _, _ in overlapping_spans
            )

            if any(sp[0].implicit_true for sp in overlapping_spans):
                # Annotator marked span as factually true despite missing context
                # Kept distinct from outright hallucination and unflagged claims
                mapped_claims.append(
                    MappedClaimAnnotation(
                        example_id=response_rec.source_id,
                        response_id=response_rec.response_id,
                        claim_id=c.claim_id,
                        claim_text=c.original_text,
                        char_start=c_start,
                        char_end=c_end,
                        mapped_label=ClaimAnnotationLabel.IMPLICIT_TRUE_ABSENT,
                        mapping_status=MappingStatus.PARTIAL_OVERLAP,
                        mapping_confidence=0.8,
                        mapped_span_ids=span_ids,
                        source_spans=spans_dict,
                        claim_record=c,
                        triage_score=c_score,
                        notes="Contains implicit_true annotation (factually true world knowledge, but absent in retrieved context)",
                    )
                )
            elif is_multi_claim:
                # Annotation crosses multiple claim boundaries -> ambiguous attribution -> UNCERTAIN
                mapped_claims.append(
                    MappedClaimAnnotation(
                        example_id=response_rec.source_id,
                        response_id=response_rec.response_id,
                        claim_id=c.claim_id,
                        claim_text=c.original_text,
                        char_start=c_start,
                        char_end=c_end,
                        mapped_label=ClaimAnnotationLabel.UNCERTAIN,
                        mapping_status=MappingStatus.MULTI_CLAIM_SPAN,
                        mapping_confidence=round(claim_ov_ratio, 3),
                        mapped_span_ids=span_ids,
                        source_spans=spans_dict,
                        claim_record=c,
                        triage_score=c_score,
                        notes="Annotation span crosses multiple claims; marked UNCERTAIN to prevent ambiguous label attribution",
                    )
                )
            elif all_spans_fully_contained:
                # Annotation span is completely inside this single claim
                status = MappingStatus.EXACT_SPAN_MATCH if abs(c_len - total_ov_chars) <= 3 else MappingStatus.CONTAINED_SPAN
                mapped_claims.append(
                    MappedClaimAnnotation(
                        example_id=response_rec.source_id,
                        response_id=response_rec.response_id,
                        claim_id=c.claim_id,
                        claim_text=c.original_text,
                        char_start=c_start,
                        char_end=c_end,
                        mapped_label=ClaimAnnotationLabel.HALLUCINATED,
                        mapping_status=status,
                        mapping_confidence=1.0,
                        mapped_span_ids=span_ids,
                        source_spans=spans_dict,
                        claim_record=c,
                        triage_score=c_score,
                        notes=f"Hallucination span fully contained within claim ({total_ov_chars} chars, {claim_ov_ratio*100:.1f}% claim coverage)",
                    )
                )
            elif claim_ov_ratio >= 0.50:
                # Substantial overlap
                mapped_claims.append(
                    MappedClaimAnnotation(
                        example_id=response_rec.source_id,
                        response_id=response_rec.response_id,
                        claim_id=c.claim_id,
                        claim_text=c.original_text,
                        char_start=c_start,
                        char_end=c_end,
                        mapped_label=ClaimAnnotationLabel.HALLUCINATED,
                        mapping_status=MappingStatus.PARTIAL_OVERLAP,
                        mapping_confidence=round(claim_ov_ratio, 3),
                        mapped_span_ids=span_ids,
                        source_spans=spans_dict,
                        claim_record=c,
                        triage_score=c_score,
                        notes=f"Substantial overlap with hallucination annotation ({claim_ov_ratio*100:.1f}%)",
                    )
                )
            else:
                # Fragmentary / borderline partial overlap
                mapped_claims.append(
                    MappedClaimAnnotation(
                        example_id=response_rec.source_id,
                        response_id=response_rec.response_id,
                        claim_id=c.claim_id,
                        claim_text=c.original_text,
                        char_start=c_start,
                        char_end=c_end,
                        mapped_label=ClaimAnnotationLabel.UNCERTAIN,
                        mapping_status=MappingStatus.PARTIAL_OVERLAP,
                        mapping_confidence=round(claim_ov_ratio, 3),
                        mapped_span_ids=span_ids,
                        source_spans=spans_dict,
                        claim_record=c,
                        triage_score=c_score,
                        notes=f"Low overlap ratio ({claim_ov_ratio*100:.1f}%); marked uncertain to prevent false label transfer",
                    )
                )

        return mapped_claims


class RAGTruthDatasetAdapter:
    """Loads, indexes, and adapts official RAGTruth datasets."""

    def __init__(self, data_dir: Optional[Path | str] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else None
        self.sources: Dict[str, RAGTruthSourceRecord] = {}
        self.responses: List[RAGTruthResponseRecord] = []
        self.mapper = ClaimAnnotationMapper()

    def load_from_fixture(self, fixture_path: Path | str) -> None:
        """Load from a unified fixture JSON file."""
        with open(fixture_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("records", [])
        self.sources.clear()
        self.responses.clear()
        for item in records:
            src = RAGTruthSourceRecord.from_dict(item["source"])
            resp = RAGTruthResponseRecord.from_dict(item["response"])
            self.sources[src.source_id] = src
            self.responses.append(resp)

    def load_from_jsonl(
        self,
        source_info_path: Path | str,
        response_path: Path | str,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        """Load official source_info.jsonl and response.jsonl files."""
        self.sources.clear()
        self.responses.clear()

        # 1. Load source info
        with open(source_info_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                src = RAGTruthSourceRecord.from_dict(item)
                self.sources[src.source_id] = src

        # 2. Load responses
        count = 0
        with open(response_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                resp = RAGTruthResponseRecord.from_dict(item)
                if split and resp.split != split:
                    continue
                self.responses.append(resp)
                count += 1
                if max_samples and count >= max_samples:
                    break

    def process_all(self) -> List[MappedClaimAnnotation]:
        """Decompose all loaded responses and map annotations."""
        all_mapped: List[MappedClaimAnnotation] = []
        for resp in self.responses:
            src = self.sources.get(resp.source_id)
            claims = self.mapper.map_response(resp, src)
            all_mapped.extend(claims)
        return all_mapped


# ==============================================================================
# Claim-level Evaluation Metrics
# ==============================================================================

@dataclass
class ClaimEvaluationMetrics:
    """Evaluates claim extraction, mapping coverage, and verification performance.
    
    CRITICAL METHODOLOGICAL DISTINCTION:
    1. Unflagged claims (claims with zero overlapping hallucination annotations) indicate
       ONLY that human annotators found no conflict or baseless information in that span.
       Unflagged spans CANNOT be inferred as ground-truth 'supported' or 'factually verified'.
    2. Hallucination detection precision/recall/F1 are computed treating mapped hallucination
       annotations as positive reference instances and unflagged claims as negative reference
       instances (non-hallucinated within annotated scope), while strictly excluding uncertain,
       malformed, and implicit_true spans.
    3. Spans marked implicit_true (external world facts omitted in context) are tracked separately
       and excluded from binary hallucination scoring to avoid conflating missing context with falsehood.
    """
    total_responses: int
    total_claims: int
    total_annotations: int
    mapped_annotations_count: int
    mapped_annotations_percentage: float
    
    label_counts: Dict[str, int]
    uncertain_count: int
    mixed_count: int
    unflagged_count: int
    hallucinated_count: int
    implicit_true_absent_count: int
    
    # Coverage metrics
    claim_extraction_coverage: float
    mapping_coverage: float
    
    # Hallucination detection benchmark metrics against mapped reference annotations
    supervised_claim_count: int
    cts_triage_precision: Optional[float]
    cts_triage_recall: Optional[float]
    cts_triage_f1: Optional[float]
    
    # Operational rates
    hallucinated_rate_among_checked: Optional[float]
    hallucinated_rate_among_deferred: Optional[float]
    verification_coverage: float
    abstention_rate: float
    
    denominators: Dict[str, int]

    # Operational state quantities across all extracted claims
    claims_selected_for_verification: int = 0
    claims_verified_count: int = 0
    claims_unresolved_count: int = 0
    claims_deferred_count: int = 0
    claims_selected_count: int = 0
    claims_pending_count: int = 0

    # Operational rates
    verification_attempt_rate: float = 0.0
    hallucinated_rate_among_verified: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_responses": self.total_responses,
            "total_claims": self.total_claims,
            "total_annotations": self.total_annotations,
            "mapped_annotations_count": self.mapped_annotations_count,
            "mapped_annotations_percentage": round(self.mapped_annotations_percentage, 2),
            "label_counts": self.label_counts,
            "uncertain_count": self.uncertain_count,
            "mixed_count": self.mixed_count,
            "unflagged_count": self.unflagged_count,
            "hallucinated_count": self.hallucinated_count,
            "implicit_true_absent_count": self.implicit_true_absent_count,
            "claim_extraction_coverage": round(self.claim_extraction_coverage, 4),
            "mapping_coverage": round(self.mapping_coverage, 4),
            "supervised_claim_count": self.supervised_claim_count,
            "cts_triage_precision": round(self.cts_triage_precision, 4) if self.cts_triage_precision is not None else None,
            "cts_triage_recall": round(self.cts_triage_recall, 4) if self.cts_triage_recall is not None else None,
            "cts_triage_f1": round(self.cts_triage_f1, 4) if self.cts_triage_f1 is not None else None,
            "hallucinated_rate_among_checked": round(self.hallucinated_rate_among_checked, 4) if self.hallucinated_rate_among_checked is not None else None,
            "hallucinated_rate_among_verified": round(self.hallucinated_rate_among_verified, 4) if self.hallucinated_rate_among_verified is not None else None,
            "hallucinated_rate_among_deferred": round(self.hallucinated_rate_among_deferred, 4) if self.hallucinated_rate_among_deferred is not None else None,
            "verification_coverage": round(self.verification_coverage, 4),
            "verification_attempt_rate": round(self.verification_attempt_rate, 4),
            "abstention_rate": round(self.abstention_rate, 4),
            "claims_selected_for_verification": self.claims_selected_for_verification,
            "claims_verified": self.claims_verified_count,
            "claims_unresolved": self.claims_unresolved_count,
            "claims_deferred": self.claims_deferred_count,
            "claims_selected": self.claims_selected_count,
            "claims_pending": self.claims_pending_count,
            "denominators": self.denominators,
        }


def compute_claim_evaluation_metrics(
    mapped_claims: Sequence[MappedClaimAnnotation],
    raw_responses: Sequence[RAGTruthResponseRecord],
    cts_threshold: float = 0.50,
) -> ClaimEvaluationMetrics:
    """Computes transparent claim-level metrics with explicit denominators.
    
    NOTE: Uncertain labels are EXCLUDED from supervised evaluation metrics
    and reported separately.
    """
    total_responses = len(raw_responses)
    total_claims = len(mapped_claims)
    
    # Count total raw annotations across responses
    total_annotations = sum(len(r.labels) for r in raw_responses)
    
    # Set of annotation span IDs that successfully mapped to at least one claim
    mapped_span_ids: Set[str] = set()
    for c in mapped_claims:
        for sid in c.mapped_span_ids:
            mapped_span_ids.add(f"{c.response_id}_{sid}")
            
    # Count total unique raw annotation occurrences
    total_unique_raw_spans = sum(len(r.labels) for r in raw_responses)
    # Count mapped spans that had valid mapping (not malformed/out-of-bounds)
    valid_mapped_spans: Set[str] = set()
    for c in mapped_claims:
        if c.mapping_status not in (MappingStatus.MALFORMED_SPAN, MappingStatus.OUT_OF_BOUNDS):
            for sid in c.mapped_span_ids:
                valid_mapped_spans.add(f"{c.response_id}_{sid}")
                
    mapped_annotations_count = len(valid_mapped_spans)
    mapped_percentage = (
        (mapped_annotations_count / total_annotations * 100.0) if total_annotations > 0 else 100.0
    )
    
    # Label counts
    label_counts: Dict[str, int] = {
        ClaimAnnotationLabel.UNFLAGGED.value: 0,
        ClaimAnnotationLabel.HALLUCINATED.value: 0,
        ClaimAnnotationLabel.IMPLICIT_TRUE_ABSENT.value: 0,
        ClaimAnnotationLabel.MIXED.value: 0,
        ClaimAnnotationLabel.UNCERTAIN.value: 0,
    }
    for c in mapped_claims:
        label_counts[c.mapped_label.value] = label_counts.get(c.mapped_label.value, 0) + 1
        
    uncertain_cnt = label_counts[ClaimAnnotationLabel.UNCERTAIN.value]
    mixed_cnt = label_counts[ClaimAnnotationLabel.MIXED.value]
    unflagged_cnt = label_counts[ClaimAnnotationLabel.UNFLAGGED.value]
    hall_cnt = label_counts[ClaimAnnotationLabel.HALLUCINATED.value]
    implicit_true_cnt = label_counts[ClaimAnnotationLabel.IMPLICIT_TRUE_ABSENT.value]
    
    # Coverage metrics
    responses_with_claims = len(set(c.response_id for c in mapped_claims))
    claim_extraction_coverage = responses_with_claims / max(1, total_responses)
    mapping_coverage = (total_claims - uncertain_cnt) / max(1, total_claims)

    # --------------------------------------------------------------------------
    # Operational verification state counts across ALL extracted claims
    # --------------------------------------------------------------------------
    claims_verified_count = 0
    claims_unresolved_count = 0
    claims_deferred_count = 0
    claims_selected_count = 0
    claims_pending_count = 0
    claims_selected_for_verification = 0

    for c in mapped_claims:
        if c.claim_record:
            st = c.claim_record.state
            if st == ClaimState.VERIFIED:
                claims_verified_count += 1
            elif st == ClaimState.UNRESOLVED:
                claims_unresolved_count += 1
            elif st == ClaimState.DEFERRED:
                claims_deferred_count += 1
            elif st == ClaimState.SELECTED:
                claims_selected_count += 1
            else:
                claims_pending_count += 1

            # Selected for verification: any claim investigated or attempted by controller
            if st in (ClaimState.SELECTED, ClaimState.VERIFIED, ClaimState.UNRESOLVED) or any(
                h.get("to_state") == ClaimState.SELECTED.value for h in (c.claim_record.state_history or [])
            ):
                claims_selected_for_verification += 1
        else:
            claims_pending_count += 1

    # Primary completed-verification coverage metric: VERIFIED / total_claims
    ver_coverage = claims_verified_count / max(1, total_claims)
    # Verification attempt rate: claims_selected_for_verification / total_claims
    ver_attempt_rate = claims_selected_for_verification / max(1, total_claims)
    # Abstention rate: claims_deferred_count / total_claims
    abstention_rate = claims_deferred_count / max(1, total_claims)

    # --------------------------------------------------------------------------
    # Supervised hallucination detection evaluation:
    # Target: Detect reference hallucinated claims (Positive = HALLUCINATED, Negative = UNFLAGGED)
    # Excludes: UNCERTAIN, MIXED, and IMPLICIT_TRUE_ABSENT (world facts missing in context)
    # --------------------------------------------------------------------------
    supervised_claims = [
        c for c in mapped_claims
        if c.mapped_label in (ClaimAnnotationLabel.UNFLAGGED, ClaimAnnotationLabel.HALLUCINATED)
    ]
    sup_count = len(supervised_claims)
    
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    
    verified_supervised = 0
    verified_supervised_hallucinated = 0
    deferred_supervised = 0
    deferred_supervised_hallucinated = 0
    
    for c in supervised_claims:
        is_gold_hallucination = (c.mapped_label == ClaimAnnotationLabel.HALLUCINATED)
        pred_high_risk = (c.triage_score is not None and c.triage_score >= cts_threshold)
        
        if pred_high_risk and is_gold_hallucination:
            tp += 1
        elif pred_high_risk and not is_gold_hallucination:
            fp += 1
        elif not pred_high_risk and is_gold_hallucination:
            fn += 1
        else:
            tn += 1
            
        # Operational verification state check among supervised claims
        if c.claim_record:
            if c.claim_record.state == ClaimState.VERIFIED:
                verified_supervised += 1
                if is_gold_hallucination:
                    verified_supervised_hallucinated += 1
            elif c.claim_record.state == ClaimState.DEFERRED:
                deferred_supervised += 1
                if is_gold_hallucination:
                    deferred_supervised_hallucinated += 1
                    
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is None or recall is None:
        f1 = None  # undefined: no predicted positives or no gold positives
    elif precision + recall == 0:
        f1 = 0.0  # defined and zero (previously reported as None, which hid a real zero)
    else:
        f1 = (2.0 * precision * recall) / (precision + recall)
    
    # Annotation-based hallucination rates against mapped RAGTruth reference annotations
    # (Computed only when relevant denominator is nonzero; represents annotation flags,
    # NOT proof of evidence-grounding quality)
    hall_rate_verified = (
        verified_supervised_hallucinated / verified_supervised
        if verified_supervised > 0 else None
    )
    hall_rate_checked = hall_rate_verified
    hall_rate_deferred = (
        deferred_supervised_hallucinated / deferred_supervised
        if deferred_supervised > 0 else None
    )
    
    denominators = {
        "total_responses": total_responses,
        "total_claims": total_claims,
        "total_annotations": total_annotations,
        "mapped_annotations": mapped_annotations_count,
        "supervised_claims": sup_count,
        "excluded_uncertain_mixed": uncertain_cnt + mixed_cnt + implicit_true_cnt,
        "cts_predicted_positive": tp + fp,
        "cts_actual_positive": tp + fn,
        "checked_claims": claims_verified_count,
        "deferred_claims": claims_deferred_count,
        "verified_claims": claims_verified_count,
        "unresolved_claims": claims_unresolved_count,
        "selected_claims": claims_selected_count,
        "pending_claims": claims_pending_count,
        "claims_selected_for_verification": claims_selected_for_verification,
        "verified_supervised_claims": verified_supervised,
        "deferred_supervised_claims": deferred_supervised,
    }
    
    return ClaimEvaluationMetrics(
        total_responses=total_responses,
        total_claims=total_claims,
        total_annotations=total_annotations,
        mapped_annotations_count=mapped_annotations_count,
        mapped_annotations_percentage=mapped_percentage,
        label_counts=label_counts,
        uncertain_count=uncertain_cnt,
        mixed_count=mixed_cnt,
        unflagged_count=unflagged_cnt,
        hallucinated_count=hall_cnt,
        implicit_true_absent_count=implicit_true_cnt,
        claim_extraction_coverage=claim_extraction_coverage,
        mapping_coverage=mapping_coverage,
        supervised_claim_count=sup_count,
        cts_triage_precision=precision,
        cts_triage_recall=recall,
        cts_triage_f1=f1,
        hallucinated_rate_among_checked=hall_rate_checked,
        hallucinated_rate_among_deferred=hall_rate_deferred,
        verification_coverage=ver_coverage,
        abstention_rate=abstention_rate,
        denominators=denominators,
        claims_selected_for_verification=claims_selected_for_verification,
        claims_verified_count=claims_verified_count,
        claims_unresolved_count=claims_unresolved_count,
        claims_deferred_count=claims_deferred_count,
        claims_selected_count=claims_selected_count,
        claims_pending_count=claims_pending_count,
        verification_attempt_rate=ver_attempt_rate,
        hallucinated_rate_among_verified=hall_rate_verified,
    )


def run_ragtruth_controller_evaluation(
    adapter: RAGTruthDatasetAdapter,
    mapped_claims: Optional[List[MappedClaimAnnotation]] = None,
    policy_name: str = "ContextualMAB_LinUCB",
    use_llm_repairer: bool = False,
    dry_run: bool = True,
    max_responses: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute claim verification controller on RAGTruth records using real source context."""
    from src.claims.models import ClaimState
    from src.controller.budget import BudgetTracker, PerClaimBudget, VerificationBudget
    from src.controller.controller import VerificationController
    from src.controller.interfaces import LLMRepairer, MockRepairer, MockRetriever
    from src.controller.mab_policy import ContextualMABPolicy, MABPolicyConfig
    from src.controller.models import ControllerAction
    from src.controller.policy import ControllerPolicyConfig, SequentialDecisionPolicy
    from src.controller.reward import ProxyVerificationRewardModel
    from src.evidence.assessor import MockDeterministicAssessor
    from src.evidence.llama_assessor import LlamaAssessor
    from src.synthesis.synthesizer import LLMAnswerSynthesizer, MockAnswerSynthesizer

    if mapped_claims is None:
        mapped_claims = adapter.process_all()

    # Index mapped annotations by response_id to mutate original claim_records in-place
    mapped_by_response: Dict[str, List[MappedClaimAnnotation]] = {}
    for m in mapped_claims:
        mapped_by_response.setdefault(m.response_id, []).append(m)

    responses_to_process = adapter.responses[:max_responses] if max_responses else adapter.responses

    # Initialize policy
    if "mab" in policy_name.lower() or "linucb" in policy_name.lower():
        policy = ContextualMABPolicy(
            config=MABPolicyConfig(alpha=0.5, frozen=False),  # Allow genuine feedback updates
            reward_model=ProxyVerificationRewardModel(),
        )
    else:
        policy = SequentialDecisionPolicy(ControllerPolicyConfig())

    # Initialize assessor and repairer
    assessor = LlamaAssessor() if not dry_run else MockDeterministicAssessor()
    repairer = LLMRepairer() if use_llm_repairer else MockRepairer()
    synthesizer = LLMAnswerSynthesizer() if not dry_run else MockAnswerSynthesizer()

    action_counts = {a.value: 0 for a in ControllerAction}
    total_feedback_updates = 0
    repair_outcomes: Dict[str, int] = {"repaired": 0, "revisions_proposed": 0, "failed_or_empty": 0}
    eval_traces = []

    for resp in responses_to_process:
        src = adapter.sources.get(resp.source_id)
        if not src:
            continue

        # Extract source evidence passages
        source_passages = src.get_evidence_passages()
        
        # Retrieve the claims already mapped for this response
        resp_mapped = mapped_by_response.get(resp.response_id, [])
        if not resp_mapped:
            resp_mapped = adapter.mapper.map_response(resp, src)
            mapped_claims.extend(resp_mapped)
            mapped_by_response[resp.response_id] = resp_mapped

        claims = [m.claim_record for m in resp_mapped if m.claim_record]
        if not claims:
            continue

        # Setup mock retriever with canned source passages for this record's claims
        retriever = MockRetriever()
        evidence_store = {}
        for c in claims:
            retriever.set_evidence_for_claim(c.claim_id, source_passages)
            evidence_store[c.claim_id] = list(source_passages)

        global_budget = VerificationBudget(max_steps=5, max_model_calls=8)
        per_claim_budget = PerClaimBudget(max_retrievals=2, max_verifications=2, max_repairs=1)

        controller = VerificationController(
            retriever=retriever,
            assessor=assessor,
            repairer=repairer,
            policy=policy,
            global_budget=global_budget,
            per_claim_budget=per_claim_budget,
        )

        trace = controller.run(
            claims=claims,
            initial_evidence=evidence_store,
            run_id=f"ragtruth_{resp.response_id}",
        )

        # Track controller actions taken
        for step in trace.steps:
            if step.action in action_counts:
                action_counts[step.action] += 1
            if step.action == "REPAIR":
                rep_data = step.repair_data or {}
                # Non-empty proposed_revision indicates candidate revision generation,
                # NOT verified factual correctness. Repaired claims transition to UNRESOLVED.
                if rep_data.get("proposed_revision"):
                    repair_outcomes["repaired"] += 1
                    repair_outcomes["revisions_proposed"] += 1
                else:
                    repair_outcomes["failed_or_empty"] += 1

        eval_traces.append({
            "response_id": resp.response_id,
            "source_id": resp.source_id,
            "steps": trace.total_steps,
            "stopping_reason": trace.stopping_reason.value,
            "cost": trace.total_cost if isinstance(trace.total_cost, dict) else (trace.total_cost.to_dict() if trace.total_cost else {}),
        })

    if hasattr(policy, "total_feedback_updates"):
        total_feedback_updates = policy.total_feedback_updates

    # Compute operational state breakdown across all mapped claims post-controller
    verification_outcomes = {
        "VERIFIED": sum(1 for m in mapped_claims if m.claim_record and m.claim_record.state == ClaimState.VERIFIED),
        "UNRESOLVED": sum(1 for m in mapped_claims if m.claim_record and m.claim_record.state == ClaimState.UNRESOLVED),
        "DEFERRED": sum(1 for m in mapped_claims if m.claim_record and m.claim_record.state == ClaimState.DEFERRED),
        "SELECTED": sum(1 for m in mapped_claims if m.claim_record and m.claim_record.state == ClaimState.SELECTED),
        "PENDING": sum(1 for m in mapped_claims if m.claim_record and m.claim_record.state == ClaimState.PENDING),
    }
    claims_selected_for_verification = sum(
        1 for m in mapped_claims if m.claim_record and (
            m.claim_record.state in (ClaimState.SELECTED, ClaimState.VERIFIED, ClaimState.UNRESOLVED)
            or any(h.get("to_state") == ClaimState.SELECTED.value for h in (m.claim_record.state_history or []))
        )
    )

    return {
        "controller_executed": True,
        "policy_name": policy_name,
        "repairer_used": type(repairer).__name__,
        "assessor_used": type(assessor).__name__,
        "action_counts": action_counts,
        "feedback_updates": total_feedback_updates,
        "verification_outcomes": verification_outcomes,
        "claims_selected_for_verification": claims_selected_for_verification,
        "repair_outcomes": repair_outcomes,
        "traces_evaluated": len(eval_traces),
        "mapped_claims": mapped_claims,
    }


def main() -> None:
    """CLI entrypoint for running RAGTruth adapter on fixtures or data files."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="RAGTruth dataset adapter and claim-level evaluation harness.")
    parser.add_argument("--fixture", type=str, default="tests/fixtures/ragtruth_sample.json", help="Path to sample JSON fixture")
    parser.add_argument("--source-info", type=str, default=None, help="Path to official source_info.jsonl")
    parser.add_argument("--response-info", type=str, default=None, help="Path to official response.jsonl")
    parser.add_argument("--split", type=str, default=None, help="Filter by split (train/test)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of processed responses")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to serialize output report")
    parser.add_argument("--run-controller", action="store_true", help="Execute VerificationController on RAGTruth records")
    parser.add_argument("--policy", type=str, default="ContextualMAB_LinUCB", help="Policy for controller execution")
    parser.add_argument("--use-llm-repairer", action="store_true", help="Use live LLMRepairer instead of MockRepairer")
    parser.add_argument("--live-assessor", action="store_true", help="Use live LlamaAssessor instead of MockDeterministicAssessor")

    args = parser.parse_args()

    adapter = RAGTruthDatasetAdapter()
    if args.source_info and args.response_info:
        print(f"[RAGTruth] Loading dataset from: {args.source_info} and {args.response_info}...")
        adapter.load_from_jsonl(args.source_info, args.response_info, split=args.split, max_samples=args.max_samples)
    else:
        print(f"[RAGTruth] Loading fixture from: {args.fixture}...")
        adapter.load_from_fixture(args.fixture)

    print(f"[RAGTruth] Processing {len(adapter.responses)} responses...")
    mapped_claims = adapter.process_all()

    controller_summary = None
    if args.run_controller:
        print(f"[RAGTruth] Running controller evaluation (Policy: {args.policy}, LLMRepairer: {args.use_llm_repairer})...")
        controller_summary = run_ragtruth_controller_evaluation(
            adapter=adapter,
            mapped_claims=mapped_claims,
            policy_name=args.policy,
            use_llm_repairer=args.use_llm_repairer,
            dry_run=not args.live_assessor,
            max_responses=args.max_samples,
        )

    # Compute claim-level metrics from current claim states (post-controller if run_controller)
    metrics = compute_claim_evaluation_metrics(mapped_claims, adapter.responses)

    report: Dict[str, Any] = {
        "evaluation_mode": "controller_evaluation" if args.run_controller else "annotation_only_evaluation",
        "dataset_split": args.split or "all",
        "metrics": metrics.to_dict(),
        "sample_mappings": [c.to_dict() for c in mapped_claims[:5]],
    }
    if controller_summary:
        report["controller_execution"] = {
            k: v for k, v in controller_summary.items() if k != "mapped_claims"
        }

    print("\n" + "=" * 60)
    print("           RAGTruth Claim-Level Diagnostic Report           ")
    print("=" * 60)
    print(f"Evaluation Mode:                 {report['evaluation_mode']}")
    print(f"Total Responses Loaded:          {metrics.total_responses}")
    print(f"Total Claims Extracted:          {metrics.total_claims}")
    print(f"Total Reference Annotations:     {metrics.total_annotations}")
    print(f"Mapped Annotations:              {metrics.mapped_annotations_count} ({metrics.mapped_annotations_percentage:.1f}%)")
    print(f"Claim Extraction Coverage:       {metrics.claim_extraction_coverage * 100:.1f}%")
    print(f"Mapping Coverage:                {metrics.mapping_coverage * 100:.1f}%")
    print("-" * 60)
    print("Claim Label Distribution:")
    for label, count in metrics.label_counts.items():
        pct = (count / max(1, metrics.total_claims)) * 100
        print(f"  - {label.upper():<20}: {count:>3} ({pct:>5.1f}%)")
    print("-" * 60)
    print(f"Supervised Evaluation Claims:    {metrics.supervised_claim_count} (Excludes {metrics.uncertain_count} uncertain, {metrics.mixed_count} mixed, {metrics.implicit_true_absent_count} implicit_true)")
    if metrics.cts_triage_precision is not None:
        print(f"Initial CTS Triage Precision:    {metrics.cts_triage_precision:.4f}")
        print(f"Initial CTS Triage Recall:       {metrics.cts_triage_recall:.4f}")
        print(f"Initial CTS Triage F1:           {metrics.cts_triage_f1:.4f}")
    else:
        print("Initial CTS Triage Metrics:      N/A (insufficient positive instances)")
    
    print("-" * 60)
    print("Operational Verification Outcomes:")
    print(f"  - Selected for Verification:   {metrics.claims_selected_for_verification}")
    print(f"  - Reached VERIFIED:            {metrics.claims_verified_count}")
    print(f"  - Reached UNRESOLVED:          {metrics.claims_unresolved_count}")
    print(f"  - Reached DEFERRED:            {metrics.claims_deferred_count}")
    print(f"  - Remaining PENDING:           {metrics.claims_pending_count}")
    print(f"  - Verification Coverage:       {metrics.verification_coverage * 100:.1f}% (Completed VERIFIED / total)")
    print(f"  - Verification Attempt Rate:   {metrics.verification_attempt_rate * 100:.1f}% (Selected for verification / total)")
    print(f"  - Abstention Rate:             {metrics.abstention_rate * 100:.1f}%")
    if metrics.hallucinated_rate_among_checked is not None:
        print(f"  - Hallucination Rate (Checked): {metrics.hallucinated_rate_among_checked * 100:.1f}% (against mapped RAGTruth reference annotations)")
    if metrics.hallucinated_rate_among_deferred is not None:
        print(f"  - Hallucination Rate (Deferred):{metrics.hallucinated_rate_among_deferred * 100:.1f}% (against mapped RAGTruth reference annotations)")

    if controller_summary:
        print("-" * 60)
        print("Controller Execution Summary:")
        print(f"  - Policy:                      {controller_summary['policy_name']}")
        print(f"  - Repairer Used:               {controller_summary['repairer_used']}")
        print(f"  - Assessor Used:               {controller_summary['assessor_used']}")
        print(f"  - Total Feedback Updates:      {controller_summary['feedback_updates']}")
        print(f"  - Action Counts:               {controller_summary['action_counts']}")
        print(f"  - Verification Outcomes:       {controller_summary['verification_outcomes']}")
        print(f"  - Repair Outcomes:             {controller_summary['repair_outcomes']}")

    print("=" * 60 + "\n")

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[RAGTruth] Saved diagnostic report to: {args.output_json}")


if __name__ == "__main__":
    main()

