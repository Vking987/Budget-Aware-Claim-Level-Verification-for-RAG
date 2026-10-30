"""Control policies for the question "does LinUCB add value beyond simple feature-weighted ranking?" (no changes to src/).

All policies are verify-first (verify eligible claims in a policy-specific order, then repair with leftover budget), use the same
eligibility and budget rules as the benchmark baselines, and expose `.config` / `select_next_action` / `record_feedback`:

  ScoredOrderPolicy          verify the eligible claim with the highest score (ties: arrival order)
  FeatureRanker              ScoredOrder with a LOGISTIC-REGRESSION score fit on TRAIN claims from (CTS, claim-type risk[, relative position])
  LexicalSupportPolicy       cheap, non-neural scorer: verify the claims least supported by their evidence passages first
  PositionMatchedRandom      random selection whose probability depends on relative claim position, matched to LinUCB's own TRAIN-time profile
  TypeThompson               context-free Thompson Sampling (Beta(1,1) per claim-type arm, reward in {0,1}, D2R-RAG's update rule) trained online on
                             TRAIN claims, then frozen (sampling continues, learning stops). This is an ADAPTATION: D2R-RAG's arms are repair actions

`blind(prepared)` returns a copy in which every claim has the same CTS, which removes CTS from features AND from LinUCB's tie-break (the confound
found in the ablation: ties among equal-score claims are broken by higher CTS).
"""

from __future__ import annotations

import copy
import random
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.claims.models import ClaimType
from src.controller.baselines import _EligibilityBaseline
from src.controller.models import ControllerAction
from src.controller.policy import ControllerPolicyConfig
from src.evidence.models import AssessmentLabel
from src.synthesis.grounding import support_for_claim

# Same values as src.controller.mab_policy.extract_context_features (feature 2); copied so the ranker uses the identical heuristic.
TYPE_RISK = {
    ClaimType.TEMPORAL_COMPARISON: 0.90, ClaimType.NUMERICAL_QUANTITATIVE: 0.85, ClaimType.FACTUAL_RELATION: 0.75,
    ClaimType.EXISTENCE_NEGATION: 0.65, ClaimType.ATTRIBUTION: 0.50, ClaimType.UNKNOWN_GENERAL: 0.40,
}
FLAG_LABELS = (AssessmentLabel.CONTRADICTED, AssessmentLabel.CONFLICTED)


def type_risk(claim) -> float:
    return TYPE_RISK.get(claim.claim_type, 0.50)


def cts(claim) -> float:
    return claim.triage_score if claim.triage_score is not None else 0.5


def blind(prepared: List[Dict[str, Any]], value: float = 0.5) -> List[Dict[str, Any]]:
    """Copy of the prepared responses with a constant CTS on every claim (gold labels and evidence untouched)."""
    out = []
    for p in prepared:
        q = dict(p)
        q["claims"] = copy.deepcopy(p["claims"])
        for c in q["claims"]:
            c.triage_score = value
        out.append(q)
    return out


class _VerifyFirst(_EligibilityBaseline):
    name = "VerifyFirst"

    def __init__(self, config: Optional[ControllerPolicyConfig] = None) -> None:
        super().__init__(config)

    def _pick(self, to_verify, claims, evidence_store):  # pragma: no cover - overridden
        raise NotImplementedError

    def select_next_action(self, claims, evidence_store, assessments, repairs, budget_tracker):
        if budget_tracker.is_exhausted():
            return ControllerAction.STOP, None, "Global verification budget is exhausted."
        to_verify, to_repair = self._eligible(claims, evidence_store, assessments, repairs, budget_tracker)
        if to_verify:
            c = self._pick(to_verify, claims, evidence_store)
            return ControllerAction.VERIFY, c, f"{self.name}: verifying '{c.claim_id}'."
        if to_repair:
            return ControllerAction.REPAIR, to_repair[0], f"{self.name}: all claims verified; repairing '{to_repair[0].claim_id}'."
        return ControllerAction.STOP, None, f"{self.name}: nothing eligible under the remaining budget."


class ScoredOrderPolicy(_VerifyFirst):
    def __init__(self, score_fn: Callable[[Any, int, int, Dict], float], name: str, config=None) -> None:
        super().__init__(config)
        self.score_fn, self.name = score_fn, name

    def _pick(self, to_verify, claims, evidence_store):
        pos = {c.claim_id: i for i, c in enumerate(claims)}
        n = len(claims)
        return max(to_verify, key=lambda c: (self.score_fn(c, pos[c.claim_id], n, evidence_store), -pos[c.claim_id]))


def rel_pos(i: int, n: int) -> float:
    return i / (n - 1) if n > 1 else 0.0


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 30) -> np.ndarray:
    """Newton / IRLS logistic regression with a small ridge term (numpy only)."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = np.clip(X @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def feature_vector(claim, i: int, n: int, features: Sequence[str]) -> List[float]:
    table = {"bias": 1.0, "cts": cts(claim), "type": type_risk(claim), "pos": rel_pos(i, n)}
    return [table[f] for f in features]


def fit_feature_ranker(train_prepared: List[Dict[str, Any]], features: Sequence[str]) -> Dict[str, Any]:
    """Fit P(gold positive | features) on TRAIN-split claims (scored gold labels only). Uses gold labels, so it is a SUPERVISED reference."""
    X, y = [], []
    for p in train_prepared:
        n = len(p["claims"])
        for i, c in enumerate(p["claims"]):
            g = p["gold"][c.claim_id]
            if g in ("HALLUCINATED", "UNFLAGGED"):
                X.append(feature_vector(c, i, n, ["bias", *features]))
                y.append(1.0 if g == "HALLUCINATED" else 0.0)
    w = fit_logistic(np.array(X), np.array(y))
    return {"features": list(features), "weights": [float(v) for v in w], "n_train_claims": len(y), "base_rate": float(np.mean(y))}


def FeatureRanker(model: Dict[str, Any], name: str, config=None) -> ScoredOrderPolicy:
    feats, w = model["features"], model["weights"]

    def score(c, i, n, _store):
        x = feature_vector(c, i, n, ["bias", *feats])
        return float(np.dot(w, x))

    return ScoredOrderPolicy(score, name, config)


def LexicalSupportPolicy(config=None) -> ScoredOrderPolicy:
    """Cheap non-neural uncertainty score: claims the evidence supports LEAST (low overlap / missing numbers) are verified first."""
    def score(c, i, n, store):
        sup = support_for_claim(c.normalized_text or c.original_text, store.get(c.claim_id, []))
        return (1.0 - sup["overlap"]) + (0.5 if sup["missing_facts"] else 0.0)

    return ScoredOrderPolicy(score, "LexicalSupport", config)


class PositionMatchedRandom(_VerifyFirst):
    """Random selection with P(pick) proportional to a relative-position profile (K bins) estimated from LinUCB's TRAIN-time choices."""

    def __init__(self, profile: Sequence[float], seed: int, config=None, name: str = "PositionMatchedRandom") -> None:
        super().__init__(config)
        self.profile, self.name, self._rng = list(profile), name, random.Random(seed)

    def _pick(self, to_verify, claims, evidence_store):
        pos = {c.claim_id: i for i, c in enumerate(claims)}
        n, K = len(claims), len(self.profile)
        w = [max(self.profile[min(K - 1, int(rel_pos(pos[c.claim_id], n) * K))], 1e-6) for c in to_verify]
        return self._rng.choices(to_verify, weights=w, k=1)[0]


def estimate_position_profile(rows: List[Dict[str, Any]], K: int = 5) -> List[float]:
    """P(claim verified | relative-position bin) from claim rows ({response_id, label}) in arrival order."""
    counts, sizes = {}, {}
    for r in rows:
        sizes[r["response_id"]] = sizes.get(r["response_id"], 0) + 1
    seen = {}
    ver, tot = [0] * K, [0] * K
    for r in rows:
        i = seen.get(r["response_id"], 0)
        seen[r["response_id"]] = i + 1
        b = min(K - 1, int(rel_pos(i, sizes[r["response_id"]]) * K))
        tot[b] += 1
        ver[b] += r["label"] is not None
    return [ver[b] / tot[b] if tot[b] else 0.0 for b in range(K)]


class TypeThompson(_VerifyFirst):
    """Context-free Thompson Sampling over claim-type arms. Reward 1 if verifying a claim of that type flags it (CONTRADICTED/CONFLICTED).
    Update rule as in D2R-RAG's bandits.py (alpha += reward, beta += 1 - reward); here learning happens on TRAIN claims and the policy is then frozen."""

    def __init__(self, posterior: Optional[Dict[str, List[float]]] = None, seed: int = 0, frozen: bool = False, config=None,
                 name: str = "TypeThompson") -> None:
        super().__init__(config)
        self.name, self.frozen = name, frozen
        self.posterior = {t.value: [1.0, 1.0] for t in ClaimType} if posterior is None else {k: list(v) for k, v in posterior.items()}
        self._rng = np.random.default_rng(seed)

    def _pick(self, to_verify, claims, evidence_store):
        theta = {t: self._rng.beta(a, b) for t, (a, b) in self.posterior.items()}
        pos = {c.claim_id: i for i, c in enumerate(claims)}
        return max(to_verify, key=lambda c: (theta[c.claim_type.value], -pos[c.claim_id]))

    def record_feedback(self, *args, **kwargs) -> None:
        a = kwargs.get("assessment")
        claim, action = kwargs.get("claim"), kwargs.get("action")
        if self.frozen or a is None or claim is None or action != ControllerAction.VERIFY or kwargs.get("action_failed"):
            return
        r = 1.0 if a.label in FLAG_LABELS else 0.0
        self.posterior[claim.claim_type.value][0] += r
        self.posterior[claim.claim_type.value][1] += 1.0 - r

    def frozen_copy(self, seed: int) -> "TypeThompson":
        return TypeThompson(self.posterior, seed=seed, frozen=True, config=self.config, name=self.name)
