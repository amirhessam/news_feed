"""Comprehend chunking and score aggregation."""

from __future__ import annotations

import re

from botocore.exceptions import ClientError

from comprehend_score import chunk_utf8, score_text, tone_line
from tests.fakes import FakeComprehend


def test_chunks_stay_inside_the_byte_cap_and_keep_text():
    text = "A measured result. " * 80 + "🚀🚀🚀" + " café " * 40
    chunks = chunk_utf8(text, max_bytes=50)
    assert chunks
    assert all(len(chunk.encode("utf-8")) <= 50 for chunk in chunks)
    assert re.sub(r"\s+", "", "".join(chunks)) == re.sub(r"\s+", "", text)


def test_score_text_averages_chunks_and_limits_key_phrases():
    client = FakeComprehend()
    text = "great " + ("word " * 20) + "plain " + ("word " * 20)
    analysis = score_text(client, text, max_bytes=80)
    recorded = list(client.sentiment)
    assert analysis is not None
    assert len(recorded) > 1
    assert len(client.phrases) == 1
    assert all(language == "en" for _chunk, language in recorded)
    assert all(len(chunk.encode("utf-8")) <= 80 for chunk, _language in recorded)

    weight = 0
    totals = {"Positive": 0.0, "Negative": 0.0, "Neutral": 0.0, "Mixed": 0.0}
    for chunk, _language in recorded:
        scores = (
            {"Positive": 0.8, "Negative": 0.05, "Neutral": 0.1, "Mixed": 0.05}
            if "great" in chunk
            else {"Positive": 0.2, "Negative": 0.05, "Neutral": 0.7, "Mixed": 0.05}
        )
        chunk_weight = len(chunk.encode("utf-8"))
        for label, value in scores.items():
            totals[label] += value * chunk_weight
        weight += chunk_weight
    expected = {label: totals[label] / weight for label in totals}
    for label, value in expected.items():
        assert analysis["scores"][label] == value
    assert analysis["label"] == max(expected, key=expected.get)
    assert analysis["key_phrases"] == ["language model"]
    assert "mostly" in tone_line(analysis)


def test_score_text_returns_none_when_sentiment_fails():
    class Broken:
        def detect_sentiment(self, Text, LanguageCode):
            raise ClientError({"Error": {"Code": "ThrottlingException", "Message": "slow"}}, "DetectSentiment")

        def detect_key_phrases(self, Text, LanguageCode):
            raise AssertionError("key phrases should not run without a scored chunk")

    assert score_text(Broken(), "A paper about models.") is None
    assert tone_line(None) == "not available"


def test_empty_text_returns_none():
    assert score_text(FakeComprehend(), "   ") is None


def test_non_numeric_scores_do_not_fail_the_paper():
    class Weird(FakeComprehend):
        def detect_key_phrases(self, Text, LanguageCode):
            return {"KeyPhrases": [{"Text": "kept phrase", "Score": "nope"}, {"Text": "kept phrase", "Score": 0.8}]}

        def detect_sentiment(self, Text, LanguageCode):
            return {"SentimentScore": {"Positive": "bad", "Negative": None, "Neutral": 1, "Mixed": 0}}

    analysis = score_text(Weird(), "A short paper about models.")
    assert analysis is not None
    assert analysis["label"] == "Neutral"
    assert analysis["key_phrases"] == ["kept phrase"]
    assert "mostly neutral" in tone_line({"label": "Neutral"})


def test_section_labels_are_not_key_phrases():
    class Labelled(FakeComprehend):
        def detect_key_phrases(self, Text, LanguageCode):
            return {"KeyPhrases": [{"Text": "Abstract", "Score": 0.99}, {"Text": "loop termination", "Score": 0.9}]}

    analysis = score_text(Labelled(), "Abstract: A paper about loop termination.")
    assert analysis["key_phrases"] == ["loop termination"]
