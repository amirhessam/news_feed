"""Paper fixtures shared by unit tests."""

from __future__ import annotations

from datetime import datetime, timezone

from arxiv_client import Paper


def make_paper(number: int, **overrides) -> Paper:
    identifier = f"2510.{number:05d}"
    paper = Paper(
        arxiv_id=f"{identifier}v1",
        base_id=identifier,
        title=f"Paper {number}",
        authors=["Ada Lovelace", "Alan Turing"],
        abstract=f"Abstract {number} reports a concrete result on a public benchmark.",
        categories=["cs.LG", "cs.AI"],
        published=datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc),
        pdf_url=f"https://arxiv.org/pdf/{identifier}v1.pdf",
        abs_url=f"https://arxiv.org/abs/{identifier}v1",
    )
    for key, value in overrides.items():
        setattr(paper, key, value)
    return paper
