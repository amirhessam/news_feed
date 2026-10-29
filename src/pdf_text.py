"""Extract the sections of a paper that a digest needs to read.

pypdf extracts the text of every page. `select_sections` then finds the
section headings and keeps the paper's own content: abstract, introduction,
method, results, discussion, and conclusion. Related work, references,
acknowledgements, and appendices are dropped. They describe other papers or
add length, and Comprehend and Bedrock are billed by the amount of text.
"""

from __future__ import annotations

import io
import logging
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError

# pypdf warns about every malformed font or form object, dozens per paper.
# Unreadable PDFs still raise, and the handler logs those.
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Names are case-insensitive. The tail stays case-sensitive so a sentence
# such as "results are strong today" is not treated as a heading.
_HEADING = re.compile(
    r"""(?mx)
    ^[ \t]*
    (?:(?P<num>(?:\d+|[IVXLC]+)(?:\.\d+)*)[.)]?[ \t]+)?
    (?P<name>(?i:
        experimental[ \t]+results
        | related[ \t]+work
        | methodology
        | acknowledgements?
        | acknowledgments?
        | introduction
        | experiments?
        | conclusions?
        | bibliography
        | appendices
        | evaluation
        | discussion
        | background
        | references
        | appendix
        | abstract
        | approach
        | methods?
        | results
    ))
    \b
    (?:[ \t]+(?:and[ \t]+)?[A-Z0-9][\w-]*){0,6}
    [ \t]*$
    """
)
# Numbered headings with any name, such as "3 Proposed Framework". Roman
# numerals need a period so a sentence that starts with "I" is not a heading.
_NUMBERED = re.compile(
    r"(?m)^[ \t]*(?:(?P<num>\d{1,2}(?:\.\d{1,2}){0,2})\.?|(?P<roman>[IVX]{1,5})\.)"
    r"[ \t]+(?P<title>[A-Z][^\n]{0,70}?)[ \t]*$"
)
# Table rows look like numbered headings, so a title containing a number
# such as 12.5, 3%, or (4) is not a heading.
_NUMBER_WORD = re.compile(r"[-+(]?\d+(?:[.,]\d+)*%?\)?")
# A numbered line that starts with one of these words is a sentence.
_SENTENCE_STARTS = {"we", "this", "these", "it", "in", "here", "there", "to", "for", "as", "if"}
# PDF fonts often map ligatures and symbols to control characters.
_UNSAFE_TEXT = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff\ufffe\uffff]")

# Section names the parser recognizes. Any other heading is named `other`.
_KNOWN = (
    "experimental results",
    "related work",
    "methodology",
    "acknowledgements",
    "acknowledgement",
    "acknowledgments",
    "acknowledgment",
    "introduction",
    "experiments",
    "experiment",
    "conclusions",
    "conclusion",
    "bibliography",
    "appendices",
    "evaluation",
    "discussion",
    "background",
    "references",
    "appendix",
    "abstract",
    "approach",
    "methods",
    "method",
    "results",
)
# Sections left out of the extract, together with their numbered subsections.
_DROP = {
    "related work",
    "references",
    "bibliography",
    "acknowledgement",
    "acknowledgements",
    "acknowledgment",
    "acknowledgments",
    "appendix",
    "appendices",
}
# Unnamed numbered sections after these are proofs, tables, or citations.
_BACK_MATTER = {"references", "bibliography", "appendix", "appendices"}
# The name given to headings that match no known section, such as "3 Proposed Framework".
_OTHER = "other"


def extract_text(pdf_bytes: bytes) -> str:
    """Return the text of every page with control characters removed.

    Raises ValueError if the PDF cannot be opened. Pages that fail to extract
    are skipped.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except (PdfReadError, OSError, ValueError) as exc:
        raise ValueError("could not read PDF") from exc
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except (PdfReadError, OSError, ValueError):
            continue
    return clean_text("\n".join(pages))


def clean_text(text: str) -> str:
    """Remove control characters and unpaired surrogates from extracted text."""
    return _UNSAFE_TEXT.sub("", text)


def select_sections(text: str, max_chars: int) -> str:
    """Keep the body of the paper and drop related work and back matter.

    Sections with names the parser does not know, such as "3 Proposed
    Framework", are kept. Numbered subsections such as "2.1" are dropped with
    their section. If no section headings survive PDF extraction, keep the
    start and end of the text instead.
    """
    if max_chars <= 0:
        return ""
    normalized = clean_text(text).replace("\r\n", "\n").replace("\r", "\n")
    headings = _find_headings(normalized)
    parts = []
    in_back_matter = False
    parent_number, parent_name = "", ""
    for index, (_start, end, name, number) in enumerate(headings):
        stop = headings[index + 1][0] if index + 1 < len(headings) else len(normalized)
        top = number.split(".")[0]
        # A subsection such as 2.1 under a dropped section is dropped with it.
        if "." in number and top == parent_number:
            if parent_name in _DROP:
                name = parent_name
        else:
            parent_number, parent_name = top, name
        # After references or an appendix, unnamed sections are back matter
        # until a known body section such as "Conclusion" appears.
        if name in _BACK_MATTER:
            in_back_matter = True
        elif name not in _DROP and name != _OTHER:
            in_back_matter = False
        if name in _DROP or (in_back_matter and name == _OTHER):
            continue
        body = normalized[end:stop].strip()
        if body:
            parts.append(body)
    combined = "\n\n".join(parts).strip()
    if not combined:
        return _head_and_tail(normalized, max_chars)
    return _truncate(combined, max_chars)


def _find_headings(text: str) -> list[tuple[int, int, str, str]]:
    """Return (start, end, section name, section number) for each heading."""
    found: dict[int, tuple[int, str, str]] = {}
    # A known section name wins when both patterns match the same line.
    for match in _HEADING.finditer(text):
        name = re.sub(r"\s+", " ", match.group("name")).lower()
        found[match.start()] = (match.end(), _known_name(name), match.group("num") or "")
    for match in _NUMBERED.finditer(text):
        title = match.group("title").strip()
        if match.start() in found or not _looks_like_title(title):
            continue
        number = match.group("num") or match.group("roman") or ""
        found[match.start()] = (match.end(), _known_name(title.lower()), number)
    return [(start, end, name, number) for start, (end, name, number) in sorted(found.items())]


def _known_name(title: str) -> str:
    """Return the known section name a lowercase title starts with, or `other`."""
    compact = re.sub(r"\s+", " ", title).strip()
    for name in _KNOWN:
        if compact == name or compact.startswith(name + " "):
            return name
    return _OTHER


def _looks_like_title(title: str) -> bool:
    """Return False for numbered lines that read like sentences or table rows."""
    words = title.split()
    if not words or len(words) > 10 or title[-1] in ".,;:":
        return False
    if words[0].lower() in _SENTENCE_STARTS:
        return False
    return not any(_NUMBER_WORD.fullmatch(word) for word in words)


def _head_and_tail(text: str, max_chars: int) -> str:
    """Keep the first two thirds and the last third of the allowed length.

    The start usually holds the abstract and introduction, and the end the
    conclusion.
    """
    compact = text.strip()
    if len(compact) <= max_chars:
        return compact
    tail_len = max(1, max_chars // 3)
    head_len = max_chars - tail_len
    head = _truncate(compact, head_len)
    tail_source = compact[-tail_len:]
    tail = tail_source.split(" ", 1)[-1].strip() if " " in tail_source else tail_source
    return f"{head}\n\n{tail}".strip()


def _truncate(text: str, max_chars: int) -> str:
    """Cut text to `max_chars`, at a word boundary when there is one."""
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    if " " not in clipped:
        return clipped
    return clipped.rsplit(" ", 1)[0].strip() or clipped
