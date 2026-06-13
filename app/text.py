"""Ukrainian text front-end: split → verbalize → normalize → stress → IPA.

This mirrors the preprocessing in the upstream StyleTTS2 Ukrainian Space so the
acoustic model receives exactly the phoneme strings it was trained on. The
output is a list of IPA strings (one per sentence chunk); ``tts.py`` tokenizes
and synthesizes each one.
"""
import logging
import re
from unicodedata import normalize as unicode_normalize

from . import verbalizer as verb

logger = logging.getLogger("app.text")

_DASHES = re.compile(r"[᠆‐‑‒–—―⁻₋−⸺⸻]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.?!:])\s+")

# Lazily-built singletons from the upstream Ukrainian front-end libraries.
_stressify = None
_ipa = None
_acute = None


def _ensure_frontend() -> None:
    global _stressify, _ipa, _acute
    if _stressify is not None:
        return
    from ipa_uk import ipa
    from ukrainian_word_stress import Stressifier, StressSymbol

    _stressify = Stressifier()
    _ipa = ipa
    _acute = StressSymbol.CombiningAcuteAccent


def split_to_parts(text: str) -> list[str]:
    """Split text into sentence-sized chunks, keeping terminal punctuation."""
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p for p in parts if p.strip()]


def _normalize_sentence(t: str) -> str:
    t = t.strip().replace('"', "")
    # '+' before a vowel is a manual stress marker → combining acute accent.
    t = t.replace("+", _acute)
    t = unicode_normalize("NFKC", t)
    t = _DASHES.sub("-", t)
    if t and t[-1] not in ".?!:-":
        t += "."
    t = re.sub(r" - ", ": ", t)
    return t


def to_ipa_chunks(text: str, verbalize: bool = True) -> list[str]:
    """Full pipeline → list of IPA phoneme strings ready for the tokenizer."""
    _ensure_frontend()
    chunks: list[str] = []
    for part in split_to_parts(text):
        part = part.strip()
        if not part:
            continue
        if verbalize:
            part = verb.verbalize(part)
        sentence = _normalize_sentence(part)
        sentence = _stressify(sentence)
        ps = _ipa(sentence)
        if ps:
            chunks.append(ps)
    return chunks
