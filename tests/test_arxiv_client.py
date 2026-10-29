"""arXiv API parsing, date window, and PDF download."""

from __future__ import annotations

import logging
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import pytest

from arxiv_client import (
    MAX_API_BYTES,
    MAX_PDF_BYTES,
    PAGE_ATTEMPTS,
    _ArxivRedirect,
    base_id,
    download_pdf,
    fetch_recent_papers,
    normalize_arxiv_url,
    parse_atom,
    parse_page,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_atom.xml"
NOW = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)


def _entry(arxiv_id: str, published: str, title: str = "A result") -> str:
    return f"""
    <entry>
      <id>http://arxiv.org/abs/{arxiv_id}</id>
      <title>{title}</title>
      <summary>We report a result.</summary>
      <published>{published}</published>
      <author><name>Ada Lovelace</name></author>
      <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
      <link href="https://arxiv.org/pdf/{arxiv_id}" title="pdf" type="application/pdf"/>
    </entry>
    """


def _atom(entries: list[str], total: int | None = None) -> bytes:
    body = "\n".join(entries)
    count = "" if total is None else f"<opensearch:totalResults>{total}</opensearch:totalResults>\n"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom"'
        ' xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">\n'
        f"{count}{body}\n</feed>"
    ).encode()


def _paged(pages: dict[int, list[bytes]], requested: list):
    """Opener that serves queued responses per start index, then an empty page."""

    def opener(request, timeout=60):
        requested.append(request)
        query = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)
        queue = pages.get(int(query["start"][0])) or [_atom([], total=0)]
        response = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(response, Exception):
            raise response
        return Response(response)

    return opener


def _fetch(opener, sleeps=None, **overrides):
    options = {
        "until": NOW,
        "days": 7,
        "contact_email": "editor@example.com",
        "pause_seconds": 3,
        "page_size": 3,
        "max_papers": 20,
        "sleep": (sleeps.append if sleeps is not None else lambda seconds: None),
        "opener": opener,
    }
    options.update(overrides)
    return fetch_recent_papers(["cs.AI"], **options)


class Response:
    def __init__(self, data: bytes):
        self.data = data

    def read(self, size=-1):
        if size is None or size < 0:
            return self.data
        return self.data[:size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_parse_sample_atom_collapses_whitespace_and_ids():
    papers = parse_atom(FIXTURE.read_bytes())
    assert len(papers) == 1
    paper = papers[0]
    assert paper.arxiv_id == "2510.12345v2"
    assert paper.base_id == "2510.12345"
    assert paper.title == "A New Result for Models"
    assert paper.abstract == "We show a measurable gain on a public benchmark."
    assert paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert paper.categories == ["cs.LG", "cs.AI"]
    assert paper.pdf_url == "https://arxiv.org/pdf/2510.12345v2"
    assert paper.published == datetime(2026, 10, 26, tzinfo=timezone.utc)


def test_base_id_strips_version_and_abs_url():
    assert base_id("2510.12345v2") == "2510.12345"
    assert base_id("http://arxiv.org/abs/2510.12345v1") == "2510.12345"


def test_invalid_atom_raises():
    with pytest.raises(ValueError):
        parse_atom(b"this is not xml")


def test_fetch_filters_to_the_lookback_window_and_dedupes_versions():
    pages = {
        0: [
            _atom(
                [
                    _entry("2510.20001v1", "2026-10-27T00:00:00Z", "Recent"),
                    _entry("2510.20002v2", "2026-10-22T00:00:00Z", "Also recent"),
                    _entry("2510.20002v1", "2026-10-22T00:00:00Z", "Older version"),
                ],
                total=4,
            )
        ],
        3: [_atom([_entry("2510.10001v1", "2026-10-01T00:00:00Z", "Too old")], total=4)],
    }
    requested = []
    sleeps = []
    papers = fetch_recent_papers(
        ["cs.AI", "cs.LG"],
        until=NOW,
        days=7,
        contact_email="editor@example.com",
        pause_seconds=3,
        page_size=3,
        max_papers=20,
        sleep=sleeps.append,
        opener=_paged(pages, requested),
    )

    assert [paper.title for paper in papers] == ["Recent", "Also recent"]
    assert papers[1].arxiv_id == "2510.20002v2"
    assert len(requested) == 2
    assert sleeps == [3]
    url = urllib.parse.unquote_plus(requested[0].full_url)
    assert "cat:cs.AI" in url
    assert "cat:cs.LG" in url
    assert "submittedDate:[202610211200 TO 202610281200]" in url
    assert "editor@example.com" in requested[0].get_header("User-agent")


def test_parse_page_reads_total_results_and_counts_raw_entries():
    page = parse_page(
        _atom(
            [_entry("2510.20001v1", "2026-10-27T00:00:00Z"), _entry("../bad", "2026-10-27T00:00:00Z")],
            total=812,
        )
    )
    assert page.entries == 2
    assert len(page.papers) == 1
    assert page.total == 812


def test_an_empty_page_is_retried_before_the_results_are_trusted():
    good = _atom([_entry("2510.20001v1", "2026-10-27T00:00:00Z", "Recent")], total=1)
    requested = []
    sleeps = []
    papers = _fetch(_paged({0: [_atom([], total=0), good]}, requested), sleeps)
    assert [paper.title for paper in papers] == ["Recent"]
    assert len(requested) == 2
    assert sleeps == [3]


def test_a_week_with_no_readable_page_raises_instead_of_returning_nothing():
    requested = []
    with pytest.raises(RuntimeError, match="no results"):
        _fetch(_paged({0: [_atom([], total=0)]}, requested))
    assert len(requested) == PAGE_ATTEMPTS


def test_an_api_error_entry_is_not_read_as_an_empty_week():
    error = _atom(
        [
            "<entry><id>http://arxiv.org/api/errors#incorrect_id_format</id>"
            "<title>Error</title><summary>incorrect id format</summary></entry>"
        ],
        total=1,
    )
    with pytest.raises(RuntimeError, match="incorrect id format"):
        _fetch(_paged({0: [error]}, []))


def test_paging_follows_raw_entries_when_some_entries_are_skipped():
    first = _atom(
        [
            _entry("2510.20001v1", "2026-10-27T00:00:00Z", "One"),
            _entry("../bad", "2026-10-27T00:00:00Z", "Skipped"),
            _entry("2510.20003v1", "2026-10-27T00:00:00Z", "Three"),
        ],
        total=4,
    )
    second = _atom([_entry("2510.20004v1", "2026-10-26T00:00:00Z", "Four")], total=4)
    papers = _fetch(_paged({0: [first], 3: [second]}, []))
    assert sorted(paper.title for paper in papers) == ["Four", "One", "Three"]


def test_a_failing_later_page_keeps_the_papers_already_read(caplog):
    first = _atom(
        [_entry(f"2510.2000{n}v1", "2026-10-27T00:00:00Z", f"P{n}") for n in range(1, 4)],
        total=6,
    )
    pages = {0: [first], 3: [urllib.error.URLError("offline")]}
    caplog.set_level(logging.WARNING)
    papers = _fetch(_paged(pages, []))
    assert len(papers) == 3
    assert "stopped early" in caplog.text


def test_hitting_the_cap_logs_a_warning(caplog):
    first = _atom(
        [_entry(f"2510.2000{n}v1", "2026-10-27T00:00:00Z", f"P{n}") for n in range(1, 4)],
        total=900,
    )
    caplog.set_level(logging.WARNING)
    papers = _fetch(_paged({0: [first]}, []), max_papers=3)
    assert len(papers) == 3
    assert "900 papers in the window" in caplog.text


def test_should_stop_ends_paging_with_what_was_read():
    first = _atom([_entry("2510.20001v1", "2026-10-27T00:00:00Z", "One")] * 3, total=9)
    requested = []
    calls = iter([False, True])
    papers = _fetch(_paged({0: [first]}, requested), should_stop=lambda: next(calls, True))
    assert [paper.title for paper in papers] == ["One"]
    assert len(requested) == 1


def test_fetch_api_failure_raises_runtime_error():
    def opener(request, timeout=60):
        raise urllib.error.URLError("offline")

    with pytest.raises(RuntimeError, match="offline"):
        fetch_recent_papers(
            ["cs.AI"],
            until=NOW,
            days=7,
            contact_email="editor@example.com",
            pause_seconds=0,
            opener=opener,
        )


def test_fetch_rejects_oversized_api_response():
    def opener(request, timeout=60):
        return Response(b"x" * (MAX_API_BYTES + 2))

    with pytest.raises(RuntimeError, match="too large"):
        fetch_recent_papers(
            ["cs.AI"],
            until=NOW,
            days=7,
            contact_email="editor@example.com",
            pause_seconds=0,
            opener=opener,
        )


def test_download_pdf_checks_header_size_and_user_agent():
    seen = {}

    def opener(request, timeout=60):
        seen["agent"] = request.get_header("User-agent")
        return Response(b"%PDF-1.4\nbody")

    assert download_pdf("https://arxiv.org/pdf/1.pdf", "editor@example.com", opener=opener) == b"%PDF-1.4\nbody"
    assert "mailto:editor@example.com" in seen["agent"]

    def not_pdf(request, timeout=60):
        return Response(b"hello")

    with pytest.raises(ValueError):
        download_pdf("https://arxiv.org/pdf/1.pdf", "editor@example.com", opener=not_pdf)

    def too_big(request, timeout=60):
        return Response(b"%PDF" + b"a" * (MAX_PDF_BYTES + 5))

    with pytest.raises(ValueError):
        download_pdf("https://arxiv.org/pdf/1.pdf", "editor@example.com", opener=too_big)


def test_download_pdf_rejects_non_arxiv_hosts():
    def opener(request, timeout=60):
        raise AssertionError("should not fetch")

    with pytest.raises(ValueError, match="arXiv"):
        download_pdf("https://evil.example/paper.pdf", "editor@example.com", opener=opener)
    with pytest.raises(ValueError, match="arXiv"):
        download_pdf("https://user:pass@arxiv.org/pdf/1.pdf", "editor@example.com", opener=opener)


def test_http_arxiv_url_is_upgraded_and_redirects_must_stay_on_arxiv():
    seen = {}

    def opener(request, timeout=60):
        seen["url"] = request.full_url
        return Response(b"%PDF-1.4\n")

    download_pdf("http://arxiv.org/pdf/2510.00001v1.pdf", "editor@example.com", opener=opener)
    assert seen["url"] == "https://arxiv.org/pdf/2510.00001v1.pdf"
    assert normalize_arxiv_url("http://export.arxiv.org/pdf/2510.00001.pdf").startswith("https://")
    with pytest.raises(ValueError):
        _ArxivRedirect().redirect_request(None, None, 302, "Found", {}, "https://evil.example/secret")


def test_parse_atom_rejects_dtd_and_unexpected_ids():
    with pytest.raises(ValueError, match="DTD"):
        parse_atom(b'<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY x "y">]><feed></feed>')
    papers = parse_atom(
        _atom([_entry("../secret", "2026-10-27T00:00:00Z", "Bad id")])
    )
    assert papers == []


def test_unsafe_categories_are_not_queried():
    def opener(request, timeout=60):
        raise AssertionError(request.full_url)

    assert (
        fetch_recent_papers(
            ["cs.AI) OR cat:cs.CR"],
            until=NOW,
            contact_email="editor@example.com",
            pause_seconds=0,
            opener=opener,
        )
        == []
    )
