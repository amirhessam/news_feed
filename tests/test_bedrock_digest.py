"""Bedrock shortlist, ranking, and digest writing."""

from __future__ import annotations

import hashlib
import json
import threading
import time

import pytest

import bedrock_digest
from bedrock_digest import (
    MAX_OUTPUT_TOKENS,
    RANK_PASSES,
    SCORING_PASSES,
    SHORTLIST_BATCH,
    SHORTLIST_POOL,
    parse_json_object,
    rank_papers,
    select_shortlist,
    write_digest,
)
from tests.fakes import ScriptedBedrock
from tests.papers import make_paper

BLOCKED = "The generated text has been blocked by our content filters."


@pytest.fixture(autouse=True)
def one_call_at_a_time(monkeypatch):
    """Scripted replies are consumed in call order, so most tests run calls one by one."""
    monkeypatch.setattr(bedrock_digest, "WORKERS", 1)


class ScoringBedrock:
    """Scores whatever papers the prompt lists, so tests do not depend on the shuffle.

    Call n uses the nth score table, and the last table repeats. Unlisted ids get 3.
    """

    def __init__(self, *passes):
        self.passes = list(passes)
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        scores = self.passes[min(len(self.calls), len(self.passes)) - 1]
        numbered = {str(n): scores.get(identifier, 3) for n, identifier in enumerate(_prompt_ids(kwargs), start=1)}
        return _reply(json.dumps({"scores": numbered}))


class FilteredBedrock(ScoringBedrock):
    """Blocks every reply whose prompt includes paper 7, like Nova's content filter."""

    def converse(self, **kwargs):
        if '"Paper 7"' in kwargs["messages"][0]["content"][0]["text"]:
            self.calls.append(kwargs)
            return _reply(BLOCKED)
        return super().converse(**kwargs)


def _reply(text: str) -> dict:
    return {"output": {"message": {"content": [{"text": text}]}}}


def _prompt_ids(call) -> list[str]:
    """Base ids in the order shown. Shortlist prompts number papers, and test papers are titled "Paper N"."""
    user = call["messages"][0]["content"][0]["text"]
    shown = json.loads(user.split("Papers:\n", 1)[1])
    return [paper.get("id") or "2510.%05d" % int(paper["title"].split()[1]) for paper in shown]


def test_parse_json_object_accepts_fences_and_preamble():
    assert parse_json_object('```json\n{"ids": ["2510.00001"]}\n```') == {"ids": ["2510.00001"]}
    assert parse_json_object('Here you go {"ids": ["2510.00001"]} thanks') == {"ids": ["2510.00001"]}
    with pytest.raises(ValueError):
        parse_json_object("[1, 2, 3]")


def test_shortlist_returns_input_when_it_already_fits():
    class Boom:
        def converse(self, **kwargs):
            raise AssertionError("model should not be called")

    papers = [make_paper(1), make_paper(2)]
    assert select_shortlist(Boom(), papers, model_id="m", size=30) == papers


def test_shortlist_takes_the_highest_scores_wherever_papers_sit_in_the_list():
    papers = [make_paper(number) for number in range(1, 6)]
    client = ScoringBedrock({"2510.00005": 9, "2510.00004": 8, "2510.00001": 2})
    chosen = select_shortlist(client, papers, model_id="m", size=2)
    assert [paper.base_id for paper in chosen] == ["2510.00005", "2510.00004"]
    assert len(client.calls) == SCORING_PASSES
    assert sorted(_prompt_ids(client.calls[0])) == [paper.base_id for paper in papers]


def test_a_steady_score_beats_one_lucky_score():
    papers = [make_paper(number) for number in range(1, 31)]
    lucky, steady = papers[0].base_id, papers[1].base_id
    client = ScoringBedrock({lucky: 9, steady: 7}, {lucky: 3, steady: 7})
    chosen = select_shortlist(client, papers, model_id="m", size=1, seed="2026-W44")
    assert [paper.base_id for paper in chosen] == [steady]


def test_shortlist_ignores_scores_it_cannot_match_to_a_paper():
    papers = [make_paper(number) for number in range(1, 6)]
    reply = {"scores": {"1": "8", "0": 9, "6": 9, "x": 9, "2": True}}
    client = ScriptedBedrock([json.dumps(reply), '{"scores": {}}'])
    chosen = select_shortlist(client, papers, model_id="m", size=1, seed=7)
    assert [paper.base_id for paper in chosen] == [_prompt_ids(client.calls[0])[0]]


def test_a_batch_blocked_by_the_content_filter_is_split_and_retried():
    papers = [make_paper(number) for number in range(1, 41)]
    client = FilteredBedrock({"2510.00007": 10})
    chosen = select_shortlist(client, papers, model_id="m", size=5, seed="2026-W44")
    sizes = sorted(len(_prompt_ids(call)) for call in client.calls)
    assert sizes == sorted([40, 20, 20, 10, 10] * SCORING_PASSES)
    assert len(chosen) == 5
    assert "2510.00007" not in {paper.base_id for paper in chosen}


def test_bedrock_errors_are_not_split():
    papers = [make_paper(number) for number in range(1, 41)]
    client = ScriptedBedrock([RuntimeError("throttled")] * SCORING_PASSES)
    chosen = select_shortlist(client, papers, model_id="m", size=3)
    assert len(client.calls) == SCORING_PASSES
    assert chosen == papers[:3]


def test_shortlist_scores_every_batch_then_rescores_the_best():
    papers = [make_paper(number) for number in range(SHORTLIST_BATCH + 1)]
    oldest, middle = papers[-1].base_id, papers[50].base_id
    first_pass = {oldest: 9, middle: 8}
    stage_one = 2 * SCORING_PASSES
    client = ScoringBedrock(*[first_pass] * stage_one, {oldest: 9, middle: 10})
    chosen = select_shortlist(client, papers, model_id="m", size=2, seed="2026-W44")
    assert [paper.base_id for paper in chosen] == [middle, oldest]
    assert len(client.calls) == stage_one + SCORING_PASSES
    everyone = sorted(paper.base_id for paper in papers)
    for start in range(0, stage_one, 2):
        assert sorted(_prompt_ids(client.calls[start]) + _prompt_ids(client.calls[start + 1])) == everyone
    for call in client.calls[stage_one:]:
        pool = _prompt_ids(call)
        assert len(pool) == 2 * SHORTLIST_POOL
        assert {oldest, middle} <= set(pool)


def test_scoring_calls_run_in_parallel(monkeypatch):
    monkeypatch.setattr(bedrock_digest, "WORKERS", 4)

    class SlowBedrock(ScoringBedrock):
        def __init__(self):
            super().__init__({"2510.00123": 9})
            self.lock = threading.Lock()
            self.active = 0
            self.peak = 0

        def converse(self, **kwargs):
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            time.sleep(0.05)
            try:
                return super().converse(**kwargs)
            finally:
                with self.lock:
                    self.active -= 1

    papers = [make_paper(number) for number in range(350)]
    client = SlowBedrock()
    chosen = select_shortlist(client, papers, model_id="m", size=3, seed="2026-W44")
    assert client.peak > 1
    assert chosen[0].base_id == "2510.00123"
    assert len(client.calls) == 4 * SCORING_PASSES + SCORING_PASSES


def test_a_week_string_seeds_from_sha256_not_the_process_hash():
    digest = hashlib.sha256(b"2026-W44").digest()[:8]
    seeded = bedrock_digest._rng("2026-W44")
    direct = bedrock_digest.random.Random(int.from_bytes(digest, "big"))
    assert [seeded.randrange(1_000_000) for _ in range(5)] == [direct.randrange(1_000_000) for _ in range(5)]
    assert bedrock_digest._rng(7).randrange(1_000_000) == bedrock_digest.random.Random(7).randrange(1_000_000)


def test_shortlist_order_is_shuffled_and_repeatable_for_a_week():
    papers = [make_paper(number) for number in range(1, 41)]
    first, again = ScoringBedrock({}), ScoringBedrock({})
    select_shortlist(first, papers, model_id="m", size=5, seed="2026-W44")
    select_shortlist(again, papers, model_id="m", size=5, seed="2026-W44")
    presented = _prompt_ids(first.calls[0])
    assert presented == _prompt_ids(again.calls[0])
    assert presented != [paper.base_id for paper in papers]


def test_shortlist_keeps_the_newest_papers_when_bedrock_fails():
    papers = [make_paper(number) for number in range(SHORTLIST_BATCH + 1)]
    client = ScriptedBedrock([RuntimeError("down")] * (2 * SCORING_PASSES))
    chosen = select_shortlist(client, papers, model_id="m", size=4)
    assert len(client.calls) == 2 * SCORING_PASSES
    assert [paper.base_id for paper in chosen] == [paper.base_id for paper in papers[:4]]


def test_shortlist_stops_calling_the_model_when_time_is_short():
    papers = [make_paper(number) for number in range(SHORTLIST_BATCH * 3)]
    client = ScoringBedrock({})
    answers = iter([False])
    chosen = select_shortlist(
        client, papers, model_id="m", size=6, should_stop=lambda: next(answers, True), seed=1
    )
    assert len(client.calls) == 1
    scored = set(_prompt_ids(client.calls[0]))
    assert len(chosen) == 6
    assert {paper.base_id for paper in chosen} <= scored


def test_rank_and_write_skip_the_model_when_asked():
    class Boom:
        def converse(self, **kwargs):
            raise AssertionError("model should not be called")

    papers = [make_paper(1, abstract="First abstract."), make_paper(2)]
    ranked = rank_papers(Boom(), papers, model_id="m", digest_size=2, use_model=False)
    assert [paper.base_id for paper, _reason in ranked] == [papers[0].base_id, papers[1].base_id]
    digest = write_digest(Boom(), ranked, model_id="m", default_title="Fallback Title", should_stop=lambda: True)
    assert digest.title == "Fallback Title"
    assert digest.entries[0].summary == "First abstract."


def test_rank_sorts_by_score_and_fills_from_the_shortlist():
    papers = [make_paper(1), make_paper(2), make_paper(3), make_paper(4)]
    payload = {
        "scores": [
            {"id": papers[1].base_id + "v1", "score": 6, "reason": "A solid gain."},
            {"id": papers[2].base_id, "score": 9, "reason": "A new state of the art."},
            {"id": papers[3].base_id, "score": "high", "reason": "Not a number."},
            {"id": "9999.99999", "score": 10, "reason": "Not in the shortlist."},
        ]
    }
    client = ScriptedBedrock([json.dumps(payload)] * RANK_PASSES)
    ranked = rank_papers(client, papers, model_id="amazon.nova-lite-v1:0", digest_size=3, seed="2026-W44")
    assert [paper.base_id for paper, _reason in ranked] == [
        papers[2].base_id,
        papers[1].base_id,
        papers[0].base_id,
    ]
    assert [reason for _paper, reason in ranked] == [
        "A new state of the art.",
        "A solid gain.",
        "Included to complete the weekly shortlist.",
    ]
    assert sorted(_prompt_ids(client.calls[0])) == [paper.base_id for paper in papers]
    assert client.calls[0]["inferenceConfig"]["maxTokens"] <= MAX_OUTPUT_TOKENS
    system = client.calls[0]["system"][0]["text"]
    assert "Do not rank a paper higher because its tone is more positive" in system


def test_rank_uses_shortlist_order_when_the_model_fails():
    papers = [make_paper(1), make_paper(2)]
    client = ScriptedBedrock([RuntimeError("bedrock down")] * RANK_PASSES)
    ranked = rank_papers(client, papers, model_id="m", digest_size=2)
    assert [paper.base_id for paper, _reason in ranked] == [papers[0].base_id, papers[1].base_id]
    assert ranked[0][1] == "Selected for the weekly digest from the papers read this run."


def test_rank_averages_its_passes():
    papers = [make_paper(1), make_paper(2)]
    first = {"scores": [{"id": "2510.00001", "score": 9, "reason": "Lucky."}, {"id": "2510.00002", "score": 7}]}
    later = {"scores": [{"id": "2510.00001", "score": 4}, {"id": "2510.00002", "score": 7, "reason": "Steady."}]}
    client = ScriptedBedrock([json.dumps(first)] + [json.dumps(later)] * (RANK_PASSES - 1))
    ranked = rank_papers(client, papers, model_id="m", digest_size=1)
    assert [(paper.base_id, reason) for paper, reason in ranked] == [("2510.00002", "Steady.")]
    assert len(client.calls) == RANK_PASSES


def test_each_summary_is_written_from_its_own_paper_only():
    first = make_paper(1)
    first.extract = "Only the first paper mentions the falcon benchmark."
    second = make_paper(2)
    second.extract = "Only the second paper mentions the heron dataset."
    client = ScriptedBedrock(
        [
            json.dumps({"summary": "Falcon summary.", "why_it_matters": "Falcon matters."}),
            json.dumps({"summary": "Heron summary.", "why_it_matters": "Heron matters."}),
            json.dumps({"title": "Birds of AI", "lead": "Two papers this week."}),
        ]
    )
    digest = write_digest(client, [(first, "r1"), (second, "r2")], model_id="m", default_title="Fallback")
    users = [call["messages"][0]["content"][0]["text"] for call in client.calls]
    assert "falcon" in users[0] and "heron" not in users[0]
    assert "heron" in users[1] and "falcon" not in users[1]
    assert client.calls[2]["system"][0]["text"].startswith("You write the opening")
    assert [entry.summary for entry in digest.entries] == ["Falcon summary.", "Heron summary."]
    assert (digest.title, digest.lead) == ("Birds of AI", "Two papers this week.")
    assert all(call["inferenceConfig"]["maxTokens"] <= MAX_OUTPUT_TOKENS for call in client.calls)


def test_write_digest_fills_missing_summaries_from_abstracts():
    first = make_paper(1, abstract="First abstract stays.")
    second = make_paper(2, abstract="Second abstract stays.")
    client = ScriptedBedrock(
        [
            "not json",
            json.dumps(
                {
                    "summary": "A plain-language summary of the second paper.",
                    "why_it_matters": "It improves a public benchmark.",
                }
            ),
            json.dumps({"title": "Models worth reading", "lead": "Two results stood out."}),
        ]
    )
    digest = write_digest(
        client,
        [(first, "fill reason"), (second, "model reason")],
        model_id="m",
        default_title="AI Research Weekly: October 28, 2026",
    )
    assert digest.title == "Models worth reading"
    assert digest.lead == "Two results stood out."
    assert digest.entries[0].summary == "First abstract stays."
    assert digest.entries[0].why_it_matters == "fill reason"
    assert digest.entries[1].summary == "A plain-language summary of the second paper."


def test_write_digest_uses_abstracts_when_the_model_fails():
    paper = make_paper(1, abstract="Unique abstract sentence.")
    digest = write_digest(
        ScriptedBedrock([RuntimeError("write failed"), RuntimeError("write failed")]),
        [(paper, "because it is new")],
        model_id="m",
        default_title="Fallback Title",
    )
    assert digest.title == "Fallback Title"
    assert digest.entries[0].summary == "Unique abstract sentence."
    assert digest.entries[0].why_it_matters == "because it is new"


def test_write_digest_stops_summarizing_when_time_is_short():
    first = make_paper(1)
    second = make_paper(2, abstract="Second abstract is kept.")
    client = ScriptedBedrock([json.dumps({"summary": "First summary.", "why_it_matters": "Why."})])
    answers = iter([False])
    digest = write_digest(
        client,
        [(first, "r1"), (second, "r2")],
        model_id="m",
        default_title="Fallback Title",
        should_stop=lambda: next(answers, True),
    )
    assert len(client.calls) == 1
    assert [entry.summary for entry in digest.entries] == ["First summary.", "Second abstract is kept."]
    assert digest.title == "Fallback Title"
