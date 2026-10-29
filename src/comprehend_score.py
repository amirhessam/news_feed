"""Score extracted paper text with Amazon Comprehend.

Synchronous DetectSentiment accepts 5,000 UTF-8 bytes. Text is split on
character boundaries under that cap, then the chunk scores are averaged.
Key phrases are requested only for the first chunk to keep the weekly bill small.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Stays under the 5,000-byte limit with a margin.
COMPREHEND_MAX_BYTES = 4500
_LABELS = ("Positive", "Negative", "Neutral", "Mixed")
# Section labels in the extracted text, not topics.
_IGNORED_PHRASES = {"abstract"}
_MAX_PHRASES = 8
# Comprehend's confidence for a phrase, from 0 to 1. Lower ones are mostly noise.
_MIN_PHRASE_SCORE = 0.5


def chunk_utf8(text: str, max_bytes: int = COMPREHEND_MAX_BYTES) -> list[str]:
    """Split text into chunks of at most `max_bytes` UTF-8 bytes.

    Chunks end at whitespace when possible and never inside a character.
    Leading and trailing whitespace is dropped.
    """
    # The longest UTF-8 character is 4 bytes.
    if max_bytes < 4:
        raise ValueError("max_bytes must be at least 4")
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        while start < length and text[start].isspace():
            start += 1
        if start >= length:
            break
        end = start
        size = 0
        last_space_end: int | None = None
        while end < length:
            encoded_len = len(text[end].encode("utf-8"))
            if size + encoded_len > max_bytes:
                break
            size += encoded_len
            end += 1
            if text[end - 1].isspace():
                last_space_end = end
        # Back up to the last whitespace so a word is not split across chunks.
        if end < length and last_space_end and last_space_end > start:
            end = last_space_end
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == start:
            end = start + 1
        start = end
    return chunks


def score_text(client, text: str, *, language_code: str = "en", max_bytes: int = COMPREHEND_MAX_BYTES) -> dict | None:
    """Return the tone and key phrases of `text`, or None if nothing was scored.

    The result has `label` (the strongest sentiment), `scores` (each
    sentiment averaged over the chunks, weighted by chunk size), and up to
    eight `key_phrases`. A chunk that Comprehend rejects is skipped.
    """
    chunks = chunk_utf8(text, max_bytes=max_bytes)
    if not chunks:
        return None
    totals = {label: 0.0 for label in _LABELS}
    weight = 0
    for chunk in chunks:
        try:
            response = client.detect_sentiment(Text=chunk, LanguageCode=language_code)
        except Exception:
            logger.warning("Comprehend sentiment failed for a chunk", exc_info=True)
            continue
        scores = response.get("SentimentScore") or {}
        chunk_weight = len(chunk.encode("utf-8"))
        for label in _LABELS:
            totals[label] += _as_float(scores.get(label, 0.0)) * chunk_weight
        weight += chunk_weight
    if weight == 0:
        return None
    averaged = {label: totals[label] / weight for label in _LABELS}
    phrases: list[str] = []
    try:
        detected = client.detect_key_phrases(Text=chunks[0], LanguageCode=language_code)
    except Exception:
        logger.warning("Comprehend key phrases failed", exc_info=True)
        detected = {}
    seen: set[str] = set()
    for phrase in detected.get("KeyPhrases") or []:
        words = (phrase.get("Text") or "").strip()
        if not words or _as_float(phrase.get("Score")) < _MIN_PHRASE_SCORE:
            continue
        key = words.casefold()
        if key in seen or key in _IGNORED_PHRASES:
            continue
        seen.add(key)
        phrases.append(words)
        if len(phrases) >= _MAX_PHRASES:
            break
    return {
        "label": max(averaged, key=averaged.get),
        "scores": averaged,
        "key_phrases": phrases,
    }


def _as_float(value) -> float:
    """Return `value` as a float, or 0.0 if it is missing or not a number."""
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def tone_line(analysis: dict | None) -> str:
    """Describe the tone for the digest.

    For example: "mostly neutral (Positive 0.20, Negative 0.05, Neutral 0.70, Mixed 0.05)".
    """
    if not analysis:
        return "not available"
    raw_scores = analysis.get("scores")
    scores = raw_scores if isinstance(raw_scores, dict) else {}
    return (
        f"mostly {str(analysis.get('label') or 'neutral').lower()} "
        f"(Positive {_as_float(scores.get('Positive')):.2f}, "
        f"Negative {_as_float(scores.get('Negative')):.2f}, "
        f"Neutral {_as_float(scores.get('Neutral')):.2f}, "
        f"Mixed {_as_float(scores.get('Mixed')):.2f})"
    )
