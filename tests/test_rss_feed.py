"""RSS construction, escaping, and the 52-week cap."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pytest

from bedrock_digest import Digest, DigestEntry
from rss_feed import digest_item, empty_feed, find_item, parse_feed, render_item_page, upsert_item
from tests.papers import make_paper

NOW = datetime(2026, 10, 28, 6, 0, tzinfo=timezone.utc)


def test_round_trip_escapes_text_and_replaces_the_same_guid():
    feed = empty_feed("AI Paper News", "https://feed.example/rss", NOW)
    first = {
        "title": "A < B",
        "link": "https://example.com?a=1&b=2",
        "guid": "2026-W44",
        "pubDate": "Wed, 28 Oct 2026 06:00:00 +0000",
        "description": "<p>Fish & chips</p>",
    }
    updated = upsert_item(feed, first)
    parsed = parse_feed(updated)
    assert parsed["items"][0]["title"] == "A < B"
    assert parsed["items"][0]["description"] == "<p>Fish & chips</p>"
    assert parsed["channel"]["link"] == "https://feed.example/rss"

    replacement = dict(first, description="<p>Replaced</p>")
    replaced = upsert_item(updated, replacement)
    items = parse_feed(replaced)["items"]
    assert len(items) == 1
    assert items[0]["description"] == "<p>Replaced</p>"
    assert 'isPermaLink="false"' in replaced


def test_feed_keeps_the_newest_52_items():
    feed = empty_feed("AI Paper News", "https://feed.example/rss", NOW)
    for number in range(53):
        feed = upsert_item(
            feed,
            {
                "title": f"Week {number}",
                "link": "https://feed.example/rss",
                "guid": str(number),
                "pubDate": "Wed, 28 Oct 2026 06:00:00 +0000",
                "description": "body",
            },
        )
    items = parse_feed(feed)["items"]
    assert len(items) == 52
    assert items[0]["guid"] == "52"
    assert all(item["guid"] != "0" for item in items)


def test_digest_html_shows_tone_links_and_a_missing_pdf_note():
    readable = make_paper(1)
    readable.full_text_available = True
    readable.sentiment_label = "Neutral"
    readable.sentiment_scores = {"Positive": 0.2, "Negative": 0.05, "Neutral": 0.7, "Mixed": 0.05}
    readable.key_phrases = ["language model"]
    missing = make_paper(2)
    missing.full_text_available = False
    digest = Digest(
        title="AI Research Weekly: October 28, 2026",
        lead="Two papers stood out.",
        entries=[
            DigestEntry(readable, "reason", "Readable summary.", "It matters."),
            DigestEntry(missing, "reason", "Abstract summary.", "It also matters."),
        ],
    )
    item = digest_item(
        digest,
        guid="2026-W44",
        link="https://feed.example/rss",
        published=NOW,
        covered="from October 15, 2026 to October 22, 2026",
    )
    html = item["description"]
    assert "mostly neutral" in html
    assert "language model" in html
    assert readable.abs_url in html
    assert readable.pdf_url in html
    assert html.count("Full text was unavailable") == 1
    assert "Read the linked paper before citing a result." in html
    assert "Papers first submitted from October 15, 2026 to October 22, 2026 (UTC)." in html


def test_control_characters_from_pdfs_or_the_model_cannot_break_the_feed():
    paper = make_paper(1, title="Ligature e\x0ecient title")
    paper.key_phrases = ["e\x0ecient model"]
    digest = Digest(
        title="Week\x01 of models",
        lead="Lead\x1b text.",
        entries=[DigestEntry(paper, "reason", "Summary\x00 text.", "Matters\ufffe.")],
    )
    item = digest_item(digest, guid="2026-W44", link="https://feed.example/rss", published=NOW)
    feed = upsert_item(empty_feed("AI Paper News", "https://feed.example/rss", NOW), item)
    ET.fromstring(feed)
    parsed = parse_feed(feed)["items"][0]
    assert parsed["title"] == "Week of models"
    assert "ecient model" in parsed["description"]
    assert "Summary text." in parsed["description"]


def test_a_stored_feed_with_raw_control_characters_is_repaired():
    feed = empty_feed("AI Paper News", "https://feed.example/rss", NOW).replace(
        "</channel>",
        "<item><title>Bad\x0e title</title><guid>2026-W43</guid><description>x</description></item></channel>",
    )
    with pytest.raises(ET.ParseError):
        ET.fromstring(feed)
    items = parse_feed(feed)["items"]
    assert items[0]["title"] == "Bad title"
    assert "2026-W43" in upsert_item(feed, {"guid": "2026-W44", "pubDate": "", "title": "New"})


def test_an_unreadable_feed_raises_value_error():
    with pytest.raises(ValueError):
        parse_feed("<rss><channel><item>")


def test_find_item_by_guid_or_latest_and_render_a_page():
    feed = empty_feed("AI Paper News", "https://feed.example/rss", NOW)
    for guid in ("2026-W43", "2026-W44"):
        feed = upsert_item(
            feed,
            {"title": f"<b>{guid}</b>", "guid": guid, "pubDate": "Mon", "description": f"<p>{guid} body</p>"},
        )
    assert find_item(feed, "2026-W43")["description"] == "<p>2026-W43 body</p>"
    assert find_item(feed, "latest")["guid"] == "2026-W44"
    assert find_item(feed, "2020-W01") is None

    page = render_item_page("AI Paper News", find_item(feed, "latest"))
    assert "<title>&lt;b&gt;2026-W44&lt;/b&gt;</title>" in page
    assert "<p>2026-W44 body</p>" in page
