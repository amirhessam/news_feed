"""RSS 2.0 feed for the weekly digest. The feed keeps one year of items.

Each week is one item. Its description is the digest as HTML and its guid is
the ISO week, such as 2026-W44, so publishing a week again replaces its item.
Every value from a paper or the model is escaped before it goes into HTML.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import format_datetime
from xml.sax.saxutils import escape

from bedrock_digest import Digest
from comprehend_score import tone_line

CHANNEL_DESCRIPTION = (
    "Weekly digest of important new AI papers from arXiv. "
    "Summaries are written by a model from the paper text."
)
# One year of weekly items.
KEEP_ITEMS = 52
# XML 1.0 forbids these characters even when escaped. ElementTree writes them
# as-is, and a single one makes feed.xml unreadable.
_XML_INVALID = re.compile("[^\t\n\r\u0020-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")


def xml_safe(value: str) -> str:
    """Remove characters that XML 1.0 does not allow."""
    return _XML_INVALID.sub("", value)


def empty_feed(title: str, link: str, now: datetime | None = None) -> str:
    """Return a feed with channel details and no items."""
    moment = _as_utc(now or datetime.now(timezone.utc))
    return render_feed(
        {
            "title": title,
            "link": link or "https://arxiv.org",
            "description": CHANNEL_DESCRIPTION,
            "language": "en-us",
            "lastBuildDate": format_datetime(moment),
        },
        [],
    )


def digest_item(
    digest: Digest,
    *,
    guid: str,
    link: str,
    published: datetime,
    covered: str = "",
) -> dict:
    """Build the RSS item for one week.

    `link` is the week's HTML page and `covered` describes the submission
    window, such as "from October 18, 2026 to October 25, 2026".
    """
    return {
        "title": digest.title,
        "link": link or "https://arxiv.org",
        "guid": guid,
        "pubDate": format_datetime(_as_utc(published)),
        "description": render_digest_html(digest, covered=covered),
    }


def render_digest_html(digest: Digest, *, covered: str = "") -> str:
    """Render the digest as the HTML that goes in the item description."""
    parts = []
    if digest.lead:
        parts.append(f"<p>{escape(digest.lead)}</p>")
    if covered:
        parts.append(f"<p><small>Papers first submitted {escape(covered)} (UTC).</small></p>")
    parts.append(
        "<p><em>AI-generated summaries from the paper text. "
        "Read the linked paper before citing a result.</em></p>"
    )
    for index, entry in enumerate(digest.entries, start=1):
        paper = entry.paper
        parts.append(f"<h2>{index}. {escape(paper.title)}</h2>")
        parts.append(f"<p>{escape(paper.author_line())}</p>")
        if paper.categories:
            parts.append(f"<p>{escape(', '.join(paper.categories))}</p>")
        tone = tone_line(
            None
            if paper.sentiment_scores is None
            else {"label": paper.sentiment_label or "Neutral", "scores": paper.sentiment_scores}
        )
        phrase_text = ""
        if paper.key_phrases:
            phrase_text = " Key phrases: " + escape(", ".join(paper.key_phrases)) + "."
        parts.append(f"<p>Tone: {escape(tone)}.{phrase_text}</p>")
        parts.append(f"<p>{escape(entry.summary)}</p>")
        parts.append(f"<p><strong>Why it matters:</strong> {escape(entry.why_it_matters)}</p>")
        if not paper.full_text_available:
            parts.append(
                "<p><small>Full text was unavailable; this summary is based on the abstract.</small></p>"
            )
        abs_url = _attr(paper.abs_url)
        pdf_url = _attr(paper.pdf_url)
        parts.append(f'<p><a href="{abs_url}">arXiv abstract</a> · <a href="{pdf_url}">PDF</a></p>')
    return xml_safe("\n".join(parts))


def upsert_item(feed_xml: str, item: dict, *, keep: int = KEEP_ITEMS) -> str:
    """Put `item` first, replacing any item with the same guid, and keep the newest `keep`.

    Raises ValueError if `feed_xml` cannot be read as RSS.
    """
    parsed = parse_feed(feed_xml)
    items = [existing for existing in parsed["items"] if existing["guid"] != item["guid"]]
    items.insert(0, item)
    channel = dict(parsed["channel"])
    channel["lastBuildDate"] = item["pubDate"]
    return render_feed(channel, items[:keep])


def parse_feed(feed_xml: str) -> dict:
    """Parse a stored feed. Raises ValueError when it cannot be read as RSS."""
    try:
        root = ET.fromstring(feed_xml)
    except ET.ParseError:
        # Repair a stored feed that holds characters XML forbids.
        try:
            root = ET.fromstring(xml_safe(feed_xml))
        except ET.ParseError as exc:
            raise ValueError("stored feed is not valid XML") from exc
    channel = root.find("channel")
    if channel is None:
        raise ValueError("RSS document has no channel")
    items = []
    for node in channel.findall("item"):
        items.append(
            {
                "title": node.findtext("title") or "",
                "link": node.findtext("link") or "",
                "guid": node.findtext("guid") or "",
                "pubDate": node.findtext("pubDate") or "",
                "description": node.findtext("description") or "",
            }
        )
    return {
        "channel": {
            "title": channel.findtext("title") or "AI Paper News",
            "link": channel.findtext("link") or "https://arxiv.org",
            "description": channel.findtext("description") or CHANNEL_DESCRIPTION,
            "language": channel.findtext("language") or "en-us",
            "lastBuildDate": channel.findtext("lastBuildDate") or "",
        },
        "items": items,
    }


def find_item(feed_xml: str, guid: str) -> dict | None:
    """Return the item with this guid, or the newest item for `latest`."""
    items = parse_feed(feed_xml)["items"]
    if guid == "latest":
        return items[0] if items else None
    return next((item for item in items if item["guid"] == guid), None)


def render_item_page(feed_title: str, item: dict) -> str:
    """Render one stored digest as a standalone HTML page.

    The description is HTML this module generated, with every model and paper
    value already escaped.
    """
    title = escape(xml_safe(item.get("title") or feed_title))
    published = escape(xml_safe(item.get("pubDate") or ""))
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n"
        "<style>body{max-width:46rem;margin:2rem auto;padding:0 1rem;"
        "font:16px/1.6 system-ui,sans-serif}</style>\n"
        f"</head>\n<body>\n<h1>{title}</h1>\n<p><small>{published}</small></p>\n"
        f"{xml_safe(item.get('description') or '')}\n</body>\n</html>\n"
    )


def render_feed(channel: dict, items: list[dict]) -> str:
    """Serialize the channel and its items as an RSS 2.0 document."""
    rss = ET.Element("rss", version="2.0")
    node = ET.SubElement(rss, "channel")
    for tag in ("title", "link", "description", "language", "lastBuildDate"):
        child = ET.SubElement(node, tag)
        child.text = xml_safe(channel.get(tag) or "")
    for item in items:
        item_node = ET.SubElement(node, "item")
        for tag in ("title", "link", "pubDate", "description"):
            child = ET.SubElement(item_node, tag)
            child.text = xml_safe(item.get(tag) or "")
        guid = ET.SubElement(item_node, "guid", isPermaLink="false")
        guid.text = xml_safe(item.get("guid") or "")
        category = ET.SubElement(item_node, "category")
        category.text = "Artificial Intelligence"
    ET.indent(rss, space="  ")
    body = ET.tostring(rss, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n"


def _attr(value: str) -> str:
    """Escape a value for a double-quoted HTML attribute."""
    return escape(value, {'"': "&quot;"})


def _as_utc(now: datetime) -> datetime:
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)
