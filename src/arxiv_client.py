"""Fetch recent AI papers from the arXiv API and download their PDFs.

arXiv asks automated clients to identify themselves and to pause about three
seconds between requests. Every request here sends a User-Agent with a contact
address, and callers pass the pause. Responses are size-capped, XML with a DTD
is rejected, and PDFs are only fetched from arXiv hosts, including after
redirects.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"
NS = {"atom": ATOM_NS, "arxiv": ARXIV_NS, "opensearch": OPENSEARCH_NS}
# Papers with many figures can be over 20 MB.
MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_API_BYTES = 10 * 1024 * 1024
# The arXiv API sometimes answers a valid query with an empty page.
PAGE_ATTEMPTS = 3
ALLOWED_PDF_HOSTS = {"arxiv.org", "export.arxiv.org", "www.arxiv.org"}
_VERSION = re.compile(r"v\d+$")
# New-style ids such as 2510.12345 and old-style ids such as hep-th/9901001,
# each with an optional version suffix.
_ARXIV_ID = re.compile(
    r"^(?:\d{4}\.\d{4,5}|(?:[A-Za-z-]+(?:\.[A-Za-z]{2})?)/\d{7})(?:v\d+)?$"
)
# Categories are placed in the search query, so only plain names are allowed.
_CATEGORY = re.compile(r"^[A-Za-z][A-Za-z0-9.-]{0,31}$")


@dataclass
class Paper:
    """One arXiv paper.

    The fields from `extract` on are filled in by the handler after the PDF
    is read and scored.
    """

    # With the version suffix, such as 2510.12345v2.
    arxiv_id: str
    # Without the version. Featured papers are tracked by this id.
    base_id: str
    title: str
    authors: list[str]
    abstract: str
    # The primary category first.
    categories: list[str]
    # When the first version was submitted, in UTC.
    published: datetime
    pdf_url: str
    abs_url: str
    # The sections of the PDF the digest reads, or the abstract if the PDF failed.
    extract: str = ""
    full_text_available: bool = False
    sentiment_label: str | None = None
    sentiment_scores: dict | None = None
    key_phrases: list[str] = field(default_factory=list)

    def author_line(self, limit: int = 8) -> str:
        """Return the authors for display, shortened after `limit` names."""
        if not self.authors:
            return "Unknown authors"
        if len(self.authors) <= limit:
            return ", ".join(self.authors)
        return ", ".join(self.authors[:limit]) + ", and others"


def base_id(arxiv_id: str) -> str:
    """Return the id without its version suffix. Accepts an id or an abs URL."""
    cleaned = arxiv_id.strip()
    if "/abs/" in cleaned:
        cleaned = cleaned.split("/abs/")[-1]
    cleaned = cleaned.split("?")[0].strip("/")
    return _VERSION.sub("", cleaned)


def user_agent(contact_email: str) -> str:
    """Return the User-Agent arXiv uses to reach the operator of this client."""
    return f"ai-paper-rss/1.0 (weekly digest; mailto:{contact_email})"


@dataclass
class AtomPage:
    """One page of API results."""

    papers: list[Paper]
    # Every entry on the page, including ones that could not be parsed, so
    # paging advances by the right amount.
    entries: int
    # The API's total result count for the query, if it sent one.
    total: int | None


def fetch_recent_papers(
    categories: list[str],
    *,
    until: datetime,
    days: int = 7,
    contact_email: str,
    pause_seconds: float = 3,
    page_size: int = 500,
    max_papers: int = 5000,
    sleep=time.sleep,
    opener=None,
    should_stop=None,
) -> list[Paper]:
    """Return papers whose first version was submitted in the `days` before `until`.

    The API is queried by submittedDate, then filtered on the Atom `published`
    field so a revision of an older paper is not treated as new. Raises
    RuntimeError when no page could be read. Once `should_stop()` is true,
    paging stops and the papers read so far are returned.
    """
    categories = [category for category in categories if _CATEGORY.fullmatch(category)]
    if not categories:
        return []
    page_size = max(1, page_size)
    opener = opener or _open_arxiv
    stop = should_stop or (lambda: False)
    window_end = _as_utc(until)
    window_start = window_end - timedelta(days=days)
    found: dict[str, Paper] = {}
    start_index = 0
    total: int | None = None
    requests = 0
    while start_index < max_papers and (total is None or start_index < total):
        size = min(page_size, max_papers - start_index)
        page = None
        problem = "empty page"
        for attempt in range(1, PAGE_ATTEMPTS + 1):
            if stop():
                break
            if requests:
                sleep(pause_seconds)
            requests += 1
            try:
                page = _fetch_page(
                    categories,
                    window_start,
                    window_end,
                    start_index,
                    size,
                    contact_email,
                    opener,
                )
            except RuntimeError as exc:
                problem = str(exc)
                logger.warning("arXiv request at %s failed (attempt %s): %s", start_index, attempt, exc)
                continue
            if page.entries:
                break
            problem = "empty page"
            logger.warning("arXiv returned an empty page at %s (attempt %s)", start_index, attempt)
            page = None
        if page is None:
            if stop():
                logger.warning("stopped reading arXiv at %s results to leave time for the digest", start_index)
                break
            if start_index == 0:
                raise RuntimeError(f"arXiv returned no results after {PAGE_ATTEMPTS} attempts: {problem}")
            logger.warning("arXiv paging stopped early at %s of %s results", start_index, total)
            break
        for paper in page.papers:
            # submittedDate also matches new versions of older papers.
            if window_start <= paper.published <= window_end:
                # A paper can appear on two pages if the index shifts while paging.
                found.setdefault(paper.base_id, paper)
        if page.total is not None:
            total = page.total
        start_index += page.entries
        # Without a total, a short page is the last one.
        if total is None and page.entries < size:
            break
    if total is not None and max_papers <= start_index < total:
        logger.warning(
            "arXiv has %s papers in the window and only the newest %s were read. "
            "Raise MAX_PAPERS or narrow CATEGORIES.",
            total,
            start_index,
        )
    papers = list(found.values())
    papers.sort(key=lambda paper: paper.published, reverse=True)
    logger.info("arXiv returned %s new papers", len(papers))
    return papers


def download_pdf(
    url: str,
    contact_email: str,
    timeout: int = 60,
    opener=None,
) -> bytes:
    """Download a PDF from arXiv.

    Raises ValueError if the URL is not on arXiv, the file is larger than
    MAX_PDF_BYTES, or the response is not a PDF.
    """
    opener = opener or _open_arxiv
    url = normalize_arxiv_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": user_agent(contact_email)})
    with opener(request, timeout=timeout) as response:
        data = response.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES:
        raise ValueError(f"PDF exceeds {MAX_PDF_BYTES} bytes")
    if not data.startswith(b"%PDF"):
        raise ValueError("response was not a PDF")
    return data


def normalize_arxiv_url(url: str) -> str:
    """Return an https arXiv URL, or raise if the URL points somewhere else."""
    parsed = urllib.parse.urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if parsed.username or parsed.password or host not in ALLOWED_PDF_HOSTS:
        raise ValueError("PDF URL is not an arXiv address")
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("PDF URL is not an arXiv address")
    return urllib.parse.urlunparse(parsed._replace(scheme="https"))


class _ArxivRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects that leave arXiv."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        normalize_arxiv_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_ArxivRedirect)


def _open_arxiv(request, timeout=60):
    return _OPENER.open(request, timeout=timeout)


def parse_atom(xml_bytes: bytes) -> list[Paper]:
    """Return the papers in one Atom response."""
    return parse_page(xml_bytes).papers


def parse_page(xml_bytes: bytes) -> AtomPage:
    """Parse one Atom response.

    Raises ValueError for invalid XML, a DTD, or an arXiv API error entry.
    """
    # ElementTree expands entities declared in a DTD, which a hostile
    # response could use to exhaust memory. The arXiv API does not send one.
    lowered = xml_bytes.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise ValueError("arXiv response contained a DTD")
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise ValueError("arXiv response was not valid XML") from exc
    entries = root.findall("atom:entry", NS)
    papers = []
    for entry in entries:
        if "/api/errors" in _text(entry, "atom:id"):
            raise ValueError(f"arXiv API error: {_collapse(_text(entry, 'atom:summary'))}")
        paper = _parse_entry(entry)
        if paper is not None:
            papers.append(paper)
    total_text = (root.findtext("opensearch:totalResults", namespaces=NS) or "").strip()
    total = int(total_text) if total_text.isdigit() else None
    return AtomPage(papers=papers, entries=len(entries), total=total)


def _fetch_page(categories, start, end, start_index, page_size, contact_email, opener) -> AtomPage:
    """Request one page of results, newest submissions first.

    Raises RuntimeError if the request fails or the response cannot be read.
    """
    cat_query = " OR ".join(f"cat:{category}" for category in categories)
    # submittedDate takes YYYYMMDDHHMM in UTC.
    date_query = (
        f"submittedDate:[{start.strftime('%Y%m%d%H%M')} TO {end.strftime('%Y%m%d%H%M')}]"
    )
    query = urllib.parse.urlencode(
        {
            "search_query": f"({cat_query}) AND {date_query}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": start_index,
            "max_results": page_size,
        }
    )
    request = urllib.request.Request(
        f"{API_URL}?{query}",
        headers={"User-Agent": user_agent(contact_email)},
    )
    try:
        with opener(request, timeout=60) as response:
            payload = response.read(MAX_API_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"arXiv API request failed: {exc}") from exc
    if len(payload) > MAX_API_BYTES:
        raise RuntimeError("arXiv API response too large")
    try:
        return parse_page(payload)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _parse_entry(entry: ET.Element) -> Paper | None:
    """Build a Paper from one Atom entry, or return None if it is unusable."""
    raw_id = _text(entry, "atom:id")
    published_text = _text(entry, "atom:published")
    if not raw_id or not published_text:
        return None
    try:
        published = _parse_time(published_text)
    except ValueError:
        logger.warning("skipping arXiv entry with bad date: %s", raw_id)
        return None
    identifier = base_id(raw_id)
    versioned = raw_id.rstrip("/").split("/abs/")[-1]
    if not _ARXIV_ID.fullmatch(versioned):
        logger.warning("skipping arXiv entry with unexpected id: %s", raw_id)
        return None
    title = _collapse(_text(entry, "atom:title"))
    abstract = _collapse(_text(entry, "atom:summary"))
    authors = [
        _collapse(name.text or "")
        for name in entry.findall("atom:author/atom:name", NS)
        if name.text and name.text.strip()
    ]
    categories: list[str] = []
    primary = entry.find("arxiv:primary_category", NS)
    if primary is not None and primary.attrib.get("term"):
        categories.append(primary.attrib["term"])
    for node in entry.findall("atom:category", NS):
        term = node.attrib.get("term")
        if term and term not in categories:
            categories.append(term)
    pdf_url = f"https://arxiv.org/pdf/{versioned}.pdf"
    abs_url = f"https://arxiv.org/abs/{versioned}"
    # Prefer the entry's own PDF link, but only if it points at arXiv.
    for link in entry.findall("atom:link", NS):
        href = link.attrib.get("href", "")
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
            if not href:
                continue
            try:
                pdf_url = normalize_arxiv_url(href)
            except ValueError:
                logger.warning("ignoring non-arXiv PDF link for %s", versioned)
    return Paper(
        arxiv_id=versioned,
        base_id=identifier,
        title=title,
        authors=authors,
        abstract=abstract,
        categories=categories,
        published=published,
        pdf_url=pdf_url,
        abs_url=abs_url,
    )


def _text(entry: ET.Element, path: str) -> str:
    node = entry.find(path, NS)
    if node is None or node.text is None:
        return ""
    return node.text


def _collapse(value: str) -> str:
    """Collapse whitespace, including the line breaks arXiv puts in titles."""
    return re.sub(r"\s+", " ", value).strip()


def _parse_time(value: str) -> datetime:
    """Parse an ISO 8601 timestamp and return it in UTC."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _as_utc(now: datetime) -> datetime:
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)
