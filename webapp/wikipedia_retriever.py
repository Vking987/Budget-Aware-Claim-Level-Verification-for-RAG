"""Live Wikipedia retriever (public MediaWiki API; no key) implementing the repository's BaseRetriever interface.

Pipeline for one query (every step is recorded in `last_trace` and shown in the UI):
 1. Analyse the question: pull out the entity phrase ("VIT University") and build search queries around it. A whole question sent to a
    full-text search matches on words like "created" or "when" and returns unrelated pages.
 2. Search each query, fuse the rankings (reciprocal rank fusion, the entity query weighted double).
 3. Fetch lead sections; drop disambiguation pages; drop pages that do not mention the entity (an acronym also matches a title's initials).
 4. Fetch the full text of the two best pages and split pages into passages of at most `max_chars`.
 5. Score passages by page rank and question-word overlap (lexical, labelled as such), remove near-duplicates, cap passages per page.
Network failures raise WikipediaError; an empty or fully filtered result is an empty list (never invented text). Passages carry title, URL and
provenance; the score is a lexical combination, NOT an embedding similarity.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from src.claims.models import ClaimRecord
from src.controller.interfaces import BaseRetriever
from src.evidence.models import EvidencePassage, VerificationCost
from webapp.relevance import analyse_question, asks_when, has_year, mentions_entity, question_overlap, title_match

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "AdaptiveRAGLab/1.0 (research demo; contact: local user)"
_END_SECTIONS = re.compile(r"^==\s*(References|See also|External links|Notes|Further reading|Bibliography|Citations|Sources|Footnotes)\s*==\s*$", re.I | re.M)
_HEADING = re.compile(r"^=+\s*.+?\s*=+\s*$", re.M)


class WikipediaError(RuntimeError):
    pass


def split_passages(text: str, max_chars: int = 800) -> List[str]:
    """Pack whole sentences into passages of <= max_chars (an over-long sentence is hard-split)."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    out, cur = [], ""
    for s in sents:
        while len(s) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            out.append(s[:max_chars])
            s = s[max_chars:]
        if cur and len(cur) + 1 + len(s) > max_chars:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def clean_body(text: str) -> str:
    """Full plain text -> body prose: cut at References / See also / External links, drop heading lines."""
    m = _END_SECTIONS.search(text)
    if m:
        text = text[: m.start()]
    return _HEADING.sub("", text)


def rank_passages(query: str, pages: List[Dict[str, Any]], max_chars: int = 800) -> List[Dict[str, Any]]:
    """Passages from pages (dicts: title, url, text, rank), scored by query-word overlap; ties keep search rank then position.
    Kept for callers that already hold page text; `WikipediaRetriever.retrieve` uses the full pipeline above."""
    qa = analyse_question(query)
    cands = []
    for pg in pages:
        for j, ch in enumerate(split_passages(pg["text"], max_chars)):
            cands.append({"title": pg["title"], "url": pg["url"], "text": ch, "score": round(question_overlap(qa, pg["title"], ch), 4),
                          "search_rank": pg["rank"], "pos": j})
    cands.sort(key=lambda c: (-c["score"], c["search_rank"], c["pos"]))
    return cands


class WikipediaRetriever(BaseRetriever):
    def __init__(self, session: Optional[Any] = None, timeout: float = 10.0, n_hits: int = 8, max_chars: int = 800, full_text_pages: int = 2,
                 per_page_cap: int = 3, retries: int = 4, min_interval: float = 0.4) -> None:
        self._session = session
        self.timeout, self.n_hits, self.max_chars, self.full_text_pages, self.per_page_cap, self.retries = timeout, n_hits, max_chars, full_text_pages, per_page_cap, retries
        self.min_interval, self._last_call = min_interval, 0.0
        self._cache: Dict[str, Any] = {}
        self.last_trace: Dict[str, Any] = {}

    @property
    def session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers["User-Agent"] = USER_AGENT
        return self._session

    def _get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        key = json.dumps(params, sort_keys=True)
        if key in self._cache:
            return self._cache[key]
        last: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                gap = self.min_interval - (time.monotonic() - self._last_call)
                if gap > 0:
                    time.sleep(gap)
                self._last_call = time.monotonic()
                r = self.session.get(API, params={**params, "format": "json", "formatversion": 2}, timeout=self.timeout)
                status = getattr(r, "status_code", 200)
                if status in (429, 500, 502, 503, 504) and attempt < self.retries:
                    wait = float(getattr(r, "headers", {}).get("Retry-After", 0) or 0) or 2.0 * (2 ** attempt)
                    time.sleep(min(wait, 20.0))
                    continue
                r.raise_for_status()
                data = r.json()
                self._cache[key] = data
                return data
            except Exception as e:  # network error, timeout, bad status after the last retry
                last = e
                if attempt < self.retries:
                    time.sleep(1.0 * (attempt + 1))
        raise WikipediaError(f"Wikipedia API request failed ({type(last).__name__}: {str(last)[:150]})") from last

    # -- steps
    def search_pages(self, query: str) -> List[Dict[str, Any]]:
        """One API call: ranked search hits with lead text, URL and a disambiguation flag (generator=search)."""
        data = self._get({"action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": self.n_hits, "prop": "extracts|info|pageprops", "exintro": 1,
                          "explaintext": 1, "exlimit": "max", "inprop": "url", "ppprop": "disambiguation"}).get("query", {}).get("pages", [])
        out = []
        for p in sorted(data, key=lambda p: p.get("index", 999)):
            t = p.get("title")
            if t and not p.get("missing"):
                out.append({"title": t, "url": p.get("fullurl") or f"https://en.wikipedia.org/wiki/{quote(t.replace(' ', '_'))}", "lead": (p.get("extract") or "").strip(),
                            "disambiguation": "disambiguation" in (p.get("pageprops") or {})})
        return out

    def fetch_body(self, title: str) -> str:
        data = self._get({"action": "query", "prop": "extracts", "explaintext": 1, "exsectionformat": "wiki", "exlimit": 1, "titles": title}).get("query", {}).get("pages", [])
        return clean_body((data[0].get("extract") or "") if data else "")

    # -- main
    def retrieve(self, claim: ClaimRecord, top_k: int = 3) -> Tuple[List[EvidencePassage], VerificationCost]:
        t0 = time.monotonic()
        query = (claim.normalized_text or claim.original_text or "").strip()
        trace: Dict[str, Any] = {"query": query, "queries": [], "hits": {}, "pages_considered": [], "dropped_pages": [], "kept_pages": [], "full_text_pages": []}
        self.last_trace = trace
        if not query:
            return [], VerificationCost(model_calls=0, latency_seconds=0.0, model_id="wikipedia_api")
        qa = analyse_question(query)
        trace["entity_phrases"], trace["entity_tokens"], trace["queries"] = qa.entity_phrases, qa.entity_tokens, qa.queries
        rrf: Dict[str, float] = {}
        leads: Dict[str, Dict[str, Any]] = {}
        for vi, q in enumerate(qa.queries[:2]):
            pages = self.search_pages(q)
            trace["hits"][q] = [pg["title"] for pg in pages]
            for rank, pg in enumerate(pages):
                rrf[pg["title"]] = rrf.get(pg["title"], 0.0) + (2.0 if vi == 0 else 1.0) / (1 + rank)  # sharp rank decay: the top hit must clearly lead
                leads.setdefault(pg["title"], pg)
        order = sorted(rrf, key=lambda t: -rrf[t])[:10]
        top_rrf = max(rrf.values()) if rrf else 1.0
        kept: List[Dict[str, Any]] = []
        for t in order:
            pg = leads.get(t)
            if pg is None:
                continue
            trace["pages_considered"].append(pg["title"])
            if pg["disambiguation"]:
                trace["dropped_pages"].append({"title": pg["title"], "reason": "disambiguation page"})
            elif not pg["lead"]:
                trace["dropped_pages"].append({"title": pg["title"], "reason": "no text"})
            elif not mentions_entity(qa.entity_tokens, pg["title"], pg["lead"]):
                trace["dropped_pages"].append({"title": pg["title"], "reason": "does not mention " + " / ".join(qa.entity_phrases or qa.entity_tokens)})
            else:
                pg["rrf"] = rrf.get(t, 0.0) / top_rrf
                pg["title_match"] = title_match(qa.entity_tokens, pg["title"], pg["lead"], qa.content_tokens) if qa.entity_tokens else 1.0
                pg["page_score"] = 0.6 * pg["rrf"] + 0.4 * pg["title_match"]
                kept.append(pg)
        if kept and qa.entity_tokens and max(p["title_match"] for p in kept) >= 0.9:
            # one page clearly IS the entity: pages that name a different entity (extra distinctive title words) are not evidence about it
            for p in [p for p in kept if p["title_match"] < 0.7]:
                trace["dropped_pages"].append({"title": p["title"], "reason": "title names a different entity than the question (title match %.2f)" % p["title_match"]})
            kept = [p for p in kept if p["title_match"] >= 0.7]
            top_ps = max(p["page_score"] for p in kept)  # an exact-title page that the search ranked far below the best one (a different "VIT") is not evidence either
            for p in [p for p in kept if p["page_score"] < 0.55 * top_ps]:
                trace["dropped_pages"].append({"title": p["title"], "reason": "ranked far below the best matching page (score %.2f vs %.2f)" % (p["page_score"], top_ps)})
            kept = [p for p in kept if p["page_score"] >= 0.55 * top_ps]
        kept.sort(key=lambda p: -p["page_score"])
        trace["kept_pages"] = [p["title"] for p in kept]
        trace["page_scores"] = {p["title"]: round(p["page_score"], 3) for p in kept}
        cands: List[Dict[str, Any]] = []
        when_q = asks_when(query)
        for i, pg in enumerate(kept):
            lead_ps = split_passages(pg["lead"], self.max_chars)
            body_ps: List[str] = []
            if i < self.full_text_pages:
                body = self.fetch_body(pg["title"])
                body_ps = split_passages(body, self.max_chars) if body else []
                trace["full_text_pages"].append(pg["title"])
            for sect, plist in (("lead", lead_ps), ("body", body_ps)):
                for j, ch in enumerate(plist):
                    ov = question_overlap(qa, pg["title"], ch)
                    if not qa.entity_tokens and ov == 0:
                        continue
                    when_bonus = 0.1 if (when_q and has_year(ch)) else 0.0
                    cands.append({"title": pg["title"], "url": pg["url"], "text": ch, "section": sect, "pos": j, "page": pg["page_score"], "overlap": ov,
                                  "score": round(min(1.0, 0.3 * pg["page_score"] + 0.7 * ov + when_bonus), 4)})
        cands.sort(key=lambda c: (-c["score"], c["pos"]))
        chosen, seen, per_page = [], set(), {}
        best_page = cands[0]["title"] if cands else None
        best_ov = max((c["overlap"] for c in cands), default=0.0)
        min_best_page = min(3, top_k)  # the best page always supplies a few passages (enough context to write full sentences) even if they are weak
        for c in cands:  # the page of the best passage may supply up to ceil(top_k/2)+1 passages, any other page at most per_page_cap
            key = " ".join(c["text"].lower().split())[:90]
            cap = (top_k + 1) // 2 + 1 if c["title"] == best_page else self.per_page_cap
            weak = best_ov > 0 and c["overlap"] < 0.5 * best_ov  # answers far less of the question than the best passage: noise unless it is the best page's context
            if key in seen or per_page.get(c["title"], 0) >= cap:
                continue
            if weak and not (c["title"] == best_page and per_page.get(best_page, 0) < min_best_page):
                continue
            seen.add(key)
            per_page[c["title"]] = per_page.get(c["title"], 0) + 1
            chosen.append(c)
            if len(chosen) >= top_k:
                break
        passages = [EvidencePassage(evidence_id="wiki_" + hashlib.sha1((c["title"] + "\x1f" + c["text"]).encode("utf-8")).hexdigest()[:12], doc_title=c["title"], source_text=c["text"],
                                    retrieval_score=c["score"], metadata={"url": c["url"], "source": "wikipedia", "section": c["section"],
                                                                         "score_kind": "lexical: 0.3 x page score (search rank + title match) + 0.7 x question-word overlap (+0.1 if a 'when' question and the passage has a year)"}) for c in chosen]
        trace["n_passages"] = len(passages)
        return passages, VerificationCost(model_calls=0, latency_seconds=time.monotonic() - t0, model_id="wikipedia_api")
