"""End-to-end weekly job with AWS clients replaced."""

from __future__ import annotations

import base64
import gzip
import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

import handler
from handler import Config, run_digest, serve_feed
from rss_feed import empty_feed, parse_feed, upsert_item
from storage import FeedStore, iso_week_key, run_key
from tests.fakes import FakeComprehend, FakeS3
from tests.papers import make_paper

NOW = datetime(2026, 10, 28, 6, 0, tzinfo=timezone.utc)
WEEK = iso_week_key(NOW)


def _config() -> Config:
    return Config(
        bucket="bucket",
        model_id="amazon.nova-lite-v1:0",
        categories=["cs.LG", "cs.AI"],
        shortlist_size=30,
        digest_size=3,
        max_chars=20000,
        contact_email="editor@example.com",
        feed_public_url="https://feed.example/rss",
        feed_title="AI Paper News",
        pdf_delay_seconds=3,
        rank_model_id="amazon.nova-pro-v1:0",
    )


class StableBedrock:
    def __init__(self):
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        system = kwargs["system"][0]["text"]
        user = kwargs["messages"][0]["content"][0]["text"]
        if system.startswith("You rank"):
            payload = {
                "scores": [
                    {"id": "2510.00002v1", "score": 8, "reason": "Reported a clear gain."},
                    {"id": "9999.99999", "score": 10, "reason": "Missing paper."},
                ]
            }
        elif system.startswith("You summarize"):
            payload = {}
            if '"2510.00002"' in user:
                payload = {
                    "summary": "A plain-language summary of paper 2.",
                    "why_it_matters": "It improves a public benchmark.",
                }
        else:
            payload = {"title": "Week of new models", "lead": "Three papers stood out."}
        return {"output": {"message": {"content": [{"text": json.dumps(payload)}]}}}


def _paper_text(_pdf: bytes) -> str:
    results = "measured " * 800
    return (
        "Abstract\nWe propose a measured model.\n\n"
        "1 Introduction\nIntro text stays in the reading set.\n\n"
        "2 Related Work\nUnrelated prior papers should be dropped from the extract.\n\n"
        "3 Proposed Planner\nThe custom-named method section is read.\n\n"
        "4 Results\n"
        f"{results}\n\n"
        "References\n[1] Drop this citation.\n\n"
        "5 Conclusion\nThe conclusion claim is readable.\n"
    )


def _run(s3, papers, bedrock=None, fetch_error=None, *, request_id=None, time_left=None, on_download=None):
    bedrock = bedrock or StableBedrock()
    comprehend = FakeComprehend()
    sleeps = []
    fetched = []
    downloads = []

    def fetch_papers(categories, **kwargs):
        fetched.append((categories, kwargs))
        if fetch_error:
            raise fetch_error
        return list(papers)

    def download(url, contact_email, timeout=60):
        assert contact_email == "editor@example.com"
        downloads.append(url)
        if on_download:
            on_download()
        if "00003" in url:
            raise TimeoutError("pdf missing")
        return b"%PDF-1.4\n"

    result = run_digest(
        _config(),
        s3,
        comprehend,
        bedrock,
        now=NOW,
        fetch_papers=fetch_papers,
        download_pdf_fn=download,
        sleep=sleeps.append,
        request_id=request_id,
        time_left=time_left,
    )
    return result, bedrock, comprehend, sleeps, fetched, downloads


def test_publish_reads_papers_and_appends_one_digest(monkeypatch):
    monkeypatch.setattr(handler, "extract_text", _paper_text)
    papers = [
        make_paper(1, abstract="Abstract one stays as the fallback summary."),
        make_paper(2, abstract="Abstract two is replaced by the model."),
        make_paper(3, abstract="Abstract three is used because the pdf failed."),
    ]
    s3 = FakeS3()
    result, bedrock, comprehend, sleeps, fetched, _downloads = _run(s3, papers)

    assert result == {"status": "published", "week": WEEK, "papers": ["2510.00002", "2510.00001", "2510.00003"]}
    assert sleeps == [3, 3]
    categories, options = fetched[0]
    assert categories == ["cs.LG", "cs.AI"]
    assert options["until"] == NOW - timedelta(days=3)
    assert options["days"] == 7
    assert json.loads(s3.objects[run_key(WEEK)])["status"] == "published"
    assert FeedStore(s3, "bucket").get_seen() == {"2510.00001", "2510.00002", "2510.00003"}
    assert s3.objects["tmp/feed.xml"] == s3.objects["feed.xml"]

    feed = parse_feed(FeedStore(s3, "bucket").get_feed())
    assert feed["channel"]["link"] == "https://feed.example/rss?week=latest"
    assert len(feed["items"]) == 1
    item = feed["items"][0]
    description = item["description"]
    assert item["guid"] == WEEK
    assert item["title"] == "Week of new models"
    assert item["link"] == f"https://feed.example/rss?week={WEEK}"
    assert "Papers first submitted from October 18, 2026 to October 25, 2026 (UTC)." in description
    assert "Three papers stood out." in description
    assert "A plain-language summary of paper 2." in description
    assert "Abstract one stays as the fallback summary." in description
    assert "Abstract three is used because the pdf failed." in description
    assert "mostly neutral" in description
    assert "language model" in description
    assert description.count("Full text was unavailable") == 1
    assert "https://arxiv.org/abs/2510.00002v1" in description

    prompt = "\n".join(part["text"] for call in bedrock.calls for part in call["messages"][0]["content"])
    assert "Intro text stays in the reading set." in prompt
    assert "The custom-named method section is read." in prompt
    assert "The conclusion claim is readable." in prompt
    assert "Unrelated prior papers should be dropped" not in prompt
    assert "[1] Drop this citation." not in prompt
    models = {(call["system"][0]["text"].split()[1], call["modelId"]) for call in bedrock.calls}
    assert models == {
        ("rank", "amazon.nova-pro-v1:0"),
        ("summarize", "amazon.nova-lite-v1:0"),
        ("write", "amazon.nova-lite-v1:0"),
    }
    assert all(len(text.encode("utf-8")) <= 5000 for text, _language in comprehend.sentiment)
    assert len(comprehend.sentiment) > len(comprehend.phrases) == 3

    again, _, _, _, fetched_again, _ = _run(s3, papers)
    assert again == {"status": "skipped", "week": WEEK, "reason": "published"}
    assert fetched_again == []
    assert len(parse_feed(FeedStore(s3, "bucket").get_feed())["items"]) == 1


def test_republish_replaces_the_same_week_instead_of_duplicating(monkeypatch):
    monkeypatch.setattr(handler, "extract_text", _paper_text)
    papers = [make_paper(1), make_paper(2), make_paper(3)]
    s3 = FakeS3()
    _run(s3, papers)
    del s3.objects[run_key(WEEK)]
    del s3.objects["state/seen.json"]
    second, *_ = _run(s3, papers)
    assert second["status"] == "published"
    items = parse_feed(FeedStore(s3, "bucket").get_feed())["items"]
    assert len(items) == 1
    assert items[0]["guid"] == WEEK


def test_failed_run_releases_the_lock():
    s3 = FakeS3()
    with pytest.raises(RuntimeError):
        _run(s3, [], fetch_error=RuntimeError("arxiv down"))
    assert run_key(WEEK) not in s3.objects


def test_an_empty_arxiv_result_fails_so_lambda_retries():
    s3 = FakeS3()
    with pytest.raises(RuntimeError, match="no papers"):
        _run(s3, [])
    assert run_key(WEEK) not in s3.objects
    assert "feed.xml" not in s3.objects


def test_previously_featured_papers_are_not_repeated(monkeypatch):
    monkeypatch.setattr(handler, "extract_text", _paper_text)
    papers = [make_paper(1), make_paper(2)]
    s3 = FakeS3()
    FeedStore(s3, "bucket").put_seen({paper.base_id for paper in papers})
    result, bedrock, *_ = _run(s3, papers)
    assert result["papers"] == []
    assert result["note"] == "no new papers"
    assert bedrock.calls == []
    assert "feed.xml" not in s3.objects


def test_a_retry_of_a_crashed_attempt_publishes_the_week(monkeypatch):
    monkeypatch.setattr(handler, "extract_text", _paper_text)
    papers = [make_paper(1), make_paper(2)]
    s3 = FakeS3()
    store = FeedStore(s3, "bucket")
    store.claim_week(WEEK, NOW, request_id="event-1", expires_at=NOW + timedelta(minutes=15))

    other, *_ = _run(s3, papers, request_id="event-2")
    assert other == {"status": "skipped", "week": WEEK, "reason": "in_progress"}

    retry, *_ = _run(s3, papers, request_id="event-1")
    assert retry["status"] == "published"
    assert json.loads(s3.objects[run_key(WEEK)])["status"] == "published"


def test_a_short_time_budget_publishes_from_abstracts(monkeypatch):
    monkeypatch.setattr(handler, "extract_text", _paper_text)
    papers = [make_paper(1), make_paper(2), make_paper(4)]
    s3 = FakeS3()
    state = {}

    def time_left():
        if run_key(WEEK) in s3.objects and "lock" not in state:
            state["lock"] = json.loads(s3.objects[run_key(WEEK)])
        return 200.0

    result, bedrock, _comprehend, sleeps, fetched, downloads = _run(s3, papers, time_left=time_left)
    assert result["status"] == "published"
    assert fetched[0][1]["should_stop"]()
    assert downloads == []
    assert sleeps == []
    assert [call["system"][0]["text"].split()[1] for call in bedrock.calls] == ["summarize"] * 3 + ["write"]
    description = parse_feed(FeedStore(s3, "bucket").get_feed())["items"][0]["description"]
    assert description.count("Full text was unavailable") == 3
    assert state["lock"]["expires_at"] == (NOW + timedelta(seconds=230)).isoformat()


def test_a_corrupt_stored_feed_is_backed_up_and_replaced(monkeypatch):
    monkeypatch.setattr(handler, "extract_text", _paper_text)
    s3 = FakeS3()
    s3.objects["feed.xml"] = b"<rss><channel><item>"
    result, *_ = _run(s3, [make_paper(1), make_paper(2)])
    assert result["status"] == "published"
    assert s3.objects[f"state/feed-broken-{WEEK}.xml"] == b"<rss><channel><item>"
    items = parse_feed(FeedStore(s3, "bucket").get_feed())["items"]
    assert [item["guid"] for item in items] == [WEEK]


def test_losing_the_lock_mid_run_skips_without_writing_the_feed(monkeypatch):
    monkeypatch.setattr(handler, "extract_text", _paper_text)
    s3 = FakeS3()

    def take_over():
        lock = json.loads(s3.objects[run_key(WEEK)])
        lock["token"] = "another-invocation"
        s3.objects[run_key(WEEK)] = json.dumps(lock).encode()

    result, *_ = _run(s3, [make_paper(1), make_paper(2)], on_download=take_over)
    assert result == {"status": "skipped", "week": WEEK, "reason": "lost_lock"}
    assert "feed.xml" not in s3.objects
    assert json.loads(s3.objects[run_key(WEEK)])["token"] == "another-invocation"


def test_publish_digest_passes_the_request_id_and_deadline(monkeypatch):
    monkeypatch.setenv("BUCKET_NAME", "bucket")
    clients = {}
    captured = {}

    def client(service, **kwargs):
        clients[service] = kwargs
        return object()

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return {"status": "skipped"}

    class Context:
        aws_request_id = "request-1"

        def get_remaining_time_in_millis(self):
            return 899_500

    monkeypatch.setattr("boto3.client", client)
    monkeypatch.setattr(handler, "run_digest", fake_run)
    assert handler.publish_digest({}, Context()) == {"status": "skipped"}
    assert captured["request_id"] == "request-1"
    assert captured["time_left"]() == 899.5
    assert clients["bedrock-runtime"]["config"].read_timeout == 120


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("BUCKET_NAME", "bucket")
    monkeypatch.setenv("CATEGORIES", "cs.AI, cs.CL")
    monkeypatch.setenv("DIGEST_SIZE", "10")
    monkeypatch.setenv("SHORTLIST_SIZE", "30")
    monkeypatch.setenv("ANNOUNCE_LAG_DAYS", "4")
    monkeypatch.delenv("RANK_MODEL_ID", raising=False)
    config = Config.from_env()
    assert config.categories == ["cs.AI", "cs.CL"]
    assert config.digest_size == 10
    assert config.shortlist_size == 30
    assert config.announce_lag_days == 4
    assert config.max_papers == 5000
    assert config.rank_model_id == ""
    monkeypatch.setenv("RANK_MODEL_ID", "amazon.nova-pro-v1:0")
    assert Config.from_env().rank_model_id == "amazon.nova-pro-v1:0"


def _serve(monkeypatch, s3, event):
    monkeypatch.setenv("BUCKET_NAME", "bucket")
    monkeypatch.setenv("FEED_TITLE", "AI Paper News")
    monkeypatch.setattr("boto3.client", lambda service: s3)
    return serve_feed(event, None)


def _stored_feed() -> FakeS3:
    s3 = FakeS3()
    feed = upsert_item(
        empty_feed("AI Paper News", "https://feed.example/rss?week=latest", NOW),
        {"title": "Week 44", "guid": WEEK, "pubDate": "Wed", "description": "<p>Digest body</p>"},
    )
    FeedStore(s3, "bucket").put_feed(feed)
    return s3


def test_serve_feed_returns_stored_xml_with_validators(monkeypatch):
    s3 = _stored_feed()
    response = _serve(monkeypatch, s3, {})
    assert response["statusCode"] == 200
    assert response["headers"]["Content-Type"].startswith("application/rss+xml")
    assert response["headers"]["ETag"] == "W/" + s3.etag("feed.xml")
    assert "Last-Modified" in response["headers"]
    assert response["body"] == s3.objects["feed.xml"].decode()
    assert response["isBase64Encoded"] is False


def test_serve_feed_gzips_when_the_reader_accepts_it(monkeypatch):
    s3 = _stored_feed()
    response = _serve(monkeypatch, s3, {"headers": {"Accept-Encoding": "br, gzip;q=0.8"}})
    assert response["isBase64Encoded"] is True
    assert response["headers"]["Content-Encoding"] == "gzip"
    assert gzip.decompress(base64.b64decode(response["body"])) == s3.objects["feed.xml"]
    refused = _serve(monkeypatch, s3, {"headers": {"Accept-Encoding": "gzip;q=0"}})
    assert refused["isBase64Encoded"] is False


def test_serve_feed_answers_conditional_requests_with_304(monkeypatch):
    s3 = _stored_feed()
    first = _serve(monkeypatch, s3, {})
    by_etag = _serve(monkeypatch, s3, {"headers": {"If-None-Match": first["headers"]["ETag"]}})
    assert by_etag["statusCode"] == 304
    assert by_etag["body"] == ""
    later = format_datetime(datetime.now(timezone.utc) + timedelta(hours=1), usegmt=True)
    by_date = _serve(monkeypatch, s3, {"headers": {"If-Modified-Since": later}})
    assert by_date["statusCode"] == 304
    changed = _serve(monkeypatch, s3, {"headers": {"If-None-Match": 'W/"old"', "If-Modified-Since": later}})
    assert changed["statusCode"] == 200
    garbage = _serve(monkeypatch, s3, {"headers": {"If-Modified-Since": "not a date"}})
    assert garbage["statusCode"] == 200


def test_week_links_open_an_html_page(monkeypatch):
    s3 = _stored_feed()
    page = _serve(monkeypatch, s3, {"queryStringParameters": {"week": WEEK}})
    assert page["statusCode"] == 200
    assert page["headers"]["Content-Type"].startswith("text/html")
    assert "default-src 'none'" in page["headers"]["Content-Security-Policy"]
    assert "<p>Digest body</p>" in page["body"]
    latest = _serve(monkeypatch, s3, {"queryStringParameters": {"week": "latest"}})
    assert "<h1>Week 44</h1>" in latest["body"]
    missing = _serve(monkeypatch, s3, {"queryStringParameters": {"week": "<script>"}})
    assert missing["statusCode"] == 404
    assert "<script>" not in missing["body"]
    assert _serve(monkeypatch, FakeS3(), {"queryStringParameters": {"week": WEEK}})["statusCode"] == 404


def test_serve_feed_returns_an_empty_channel_before_the_first_run(monkeypatch):
    empty = FakeS3()
    response = _serve(
        monkeypatch,
        empty,
        {"headers": {"Host": "abc.lambda-url.us-east-1.on.aws", "X-Forwarded-Proto": "https"}},
    )
    assert response["statusCode"] == 200
    assert "https://abc.lambda-url.us-east-1.on.aws/?week=latest" in response["body"]
    assert "<rss" in response["body"]

    missing_headers = _serve(monkeypatch, empty, {"headers": None})
    assert missing_headers["statusCode"] == 200
    assert "<rss" in missing_headers["body"]

    poisoned = _serve(monkeypatch, empty, {"headers": {"Host": "evil.com", "X-Forwarded-Proto": "javascript"}})
    assert "evil.com" not in poisoned["body"]
    assert "javascript" not in poisoned["body"]
