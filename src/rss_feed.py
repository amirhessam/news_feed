"""RSS 2.0 feed for the weekly digest. The feed keeps one year of items.

Each week is one item. Its description is the digest as HTML and its guid is
the ISO week, such as 2026-W44, so publishing a week again replaces its item.
Every value from a paper or the model is escaped before it goes into HTML.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import format_datetime, parsedate_to_datetime
from xml.sax.saxutils import escape

from bedrock_digest import Digest
from comprehend_score import tone_line

CHANNEL_DESCRIPTION = (
    "Weekly digest of important new AI papers from arXiv. "
    "Summaries are written by a model from the paper text."
)
# One year of weekly items.
KEEP_ITEMS = 52
# English month names. strftime("%B") follows the process locale.
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
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
    value already escaped. Papers are grouped from the stored headings, so a
    digest published earlier still picks up this layout.
    """
    title = escape(xml_safe(item.get("title") or feed_title))
    name = escape(xml_safe(feed_title or "AI Paper News"))
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n"
        f"<style>{_PAGE_STYLE}</style>\n"
        "</head>\n<body>\n"
        '<div class="page">\n'
        f'<header><p class="mark">{name}</p><h1>{title}</h1>'
        f"<p class=\"when\">{_pretty_when(item.get('pubDate') or '')}</p></header>\n"
        f"<main>{_layout_digest(item.get('description') or '')}</main>\n"
        "</div>\n</body>\n</html>\n"
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


# The week page styles the description HTML already stored in the feed.
# No images or web fonts: the page's content security policy blocks them.
_PAGE_STYLE = """
:root {
  color-scheme: light;
  --ink: #241c16;
  --muted: #6f6256;
  --line: #e4d8c8;
  --paper: #fffdf9;
  --field: #f3ecdf;
  --teal: #0e5f5c;
  --teal-soft: #e5f2f1;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background:
    radial-gradient(900px 420px at 0% -10%, #f8f1e4, transparent 55%),
    var(--field);
  color: var(--ink);
  font: 17px/1.6 Georgia, "Iowan Old Style", Palatino, "Palatino Linotype", serif;
}
.page { max-width: 42rem; margin: 0 auto; padding: 2.75rem 1.25rem 4rem; }
.mark {
  margin: 0;
  font: 600 0.72rem/1 system-ui, sans-serif;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--teal);
}
h1 {
  margin: 0.45rem 0 0.2rem;
  font-size: clamp(1.85rem, 4.2vw, 2.45rem);
  font-weight: 500;
  letter-spacing: -0.03em;
  line-height: 1.15;
  text-wrap: balance;
}
.when { margin: 0; color: var(--muted); font: 0.9rem/1.4 system-ui, sans-serif; }
.intro > p:first-child { margin: 1.6rem 0 0.7rem; font-size: 1.18rem; line-height: 1.5; }
.intro small, .intro em {
  color: var(--muted);
  font: 0.92rem/1.5 system-ui, sans-serif;
}
.intro em { font-style: normal; }
.paper {
  margin: 1.15rem 0;
  padding: 1.2rem 1.25rem 1rem;
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 18px;
}
.paper h2 {
  display: flex;
  gap: 0.75rem;
  align-items: flex-start;
  margin: 0 0 0.35rem;
  font-size: 1.28rem;
  text-wrap: balance;
  font-weight: 500;
  letter-spacing: -0.02em;
  line-height: 1.3;
}
.index {
  flex: none;
  margin-top: 0.15rem;
  min-width: 1.7rem;
  height: 1.7rem;
  padding: 0 0.4rem;
  border-radius: 999px;
  background: var(--teal-soft);
  color: var(--teal);
  font: 600 0.78rem/1.7rem system-ui, sans-serif;
  text-align: center;
}
.paper h2 + p, .paper h2 + p + p {
  margin: 0 0 0.15rem 2.45rem;
  color: var(--muted);
  font: 0.92rem/1.45 system-ui, sans-serif;
}
.paper h2 + p + p {
  margin-bottom: 0.85rem;
  font-size: 0.8rem;
}
.paper .tone {
  margin: 0.4rem 0 0.8rem;
  color: var(--muted);
  font: 0.92rem/1.5 system-ui, sans-serif;
  letter-spacing: normal;
  text-transform: none;
}
.paper .why {
  margin: 1rem 0;
  padding: 0.75rem 0.95rem;
  background: #f4f8f7;
  border-left: 3px solid var(--teal);
  border-radius: 0 12px 12px 0;
}
.paper .why strong { font-family: system-ui, sans-serif; font-size: 0.78rem; letter-spacing: 0.04em; text-transform: uppercase; color: var(--teal); }
.paper .note { margin: 0.4rem 0; }
.paper .note small { color: var(--muted); font: 0.85rem/1.45 system-ui, sans-serif; }
.links { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0 0.15rem; }
.links a {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0.28rem 0.75rem;
  color: var(--teal);
  font: 600 0.84rem/1.4 system-ui, sans-serif;
  text-decoration: none;
}
.links a:hover { background: var(--teal); color: #fff; border-color: var(--teal); }
"""
_PAPER_SPLIT = re.compile(r"(?=<h2\b)", re.IGNORECASE)
_PAPER_NUMBER = re.compile(r"<h2>(\d+)\.\s*", re.IGNORECASE)


def _layout_digest(description: str) -> str:
    """Group each stored paper heading into its own article.

    The feed keeps the description as ordinary HTML. This only changes the
    page, including digests that were published before this layout.
    """
    html = xml_safe(description)
    pieces = _PAPER_SPLIT.split(html)
    if len(pieces) == 1:
        return f'<div class="intro">{html}</div>'
    intro, *papers = pieces
    articles = "\n".join(f'<article class="paper">{_decorate_paper(paper)}</article>' for paper in papers)
    return f'<div class="intro">{intro}</div>\n{articles}'


def _decorate_paper(paper: str) -> str:
    """Mark the parts of one paper so the page style can tell them apart."""
    paper = _PAPER_NUMBER.sub(r'<h2><span class="index">\1</span> ', paper, count=1)
    paper = paper.replace("<p>Tone:", '<p class="tone">Tone:', 1)
    paper = paper.replace("<p><strong>Why it matters:</strong>", '<p class="why"><strong>Why it matters:</strong>', 1)
    paper = paper.replace(
        "<p><small>Full text was unavailable",
        '<p class="note"><small>Full text was unavailable',
        1,
    )
    paper = paper.replace('<p><a href="', '<p class="links"><a href="', 1)
    return paper.replace("</a> · <a ", "</a><a ")


def _pretty_when(value: str) -> str:
    """Return a publication date as 'October 29, 2026', or the original text."""
    text = xml_safe(value)
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        parsed = None
    if parsed is None:
        return escape(text)
    return escape(f"{_MONTHS[parsed.month - 1]} {parsed.day}, {parsed.year}")


def _attr(value: str) -> str:
    """Escape a value for a double-quoted HTML attribute."""
    return escape(value, {'"': "&quot;"})


def _as_utc(now: datetime) -> datetime:
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)
