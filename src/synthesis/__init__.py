"""Synthesis package: final evidence-grounded answer generation."""

from .models import SynthesisResult
from .synthesizer import BaseSynthesizer, LLMAnswerSynthesizer, MockAnswerSynthesizer

__all__ = [
    "SynthesisResult",
    "BaseSynthesizer",
    "LLMAnswerSynthesizer",
    "MockAnswerSynthesizer",
]
