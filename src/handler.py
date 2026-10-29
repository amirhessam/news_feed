"""Lambda entry points: the weekly digest publisher and the public feed reader.

`publish_digest` runs once a week. It loads the week's arXiv papers, has
Bedrock pick a shortlist, reads those PDFs, scores them with Comprehend, has
Bedrock rank them and write the digest, and adds one item to feed.xml.

`serve_feed` answers the public function URL with feed.xml, or with one
week's digest as an HTML page for `?week=2026-W44` or `?week=latest`.
"""

from __future__ import annotations

import base64
import gzip
import logging
import os
import re
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime, parsedate_to_datetime

from arxiv_client import download_pdf, fetch_recent_papers
from bedrock_digest import rank_papers, select_shortlist, write_digest
from comprehend_score import score_text
from pdf_text import extract_text, select_sections
from rss_feed import _MONTHS, digest_item, empty_feed, find_item, render_item_page, upsert_item
from storage import FeedStore, LostWeekLock, iso_week_key

logger = logging.getLogger(__name__)

# Seconds that must remain in the run before each stage starts, and before each
# summary is written. Below these, the run falls back to abstracts and
# shortlist order so it can still publish before the 15-minute Lambda timeout.
FETCH_MIN_SECONDS = 660
SHORTLIST_MIN_SECONDS = 540
PDF_MIN_SECONDS = 300
RANK_MIN_SECONDS = 240
WRITE_MIN_SECONDS = 150
# Lambda stops an attempt at its deadline. The margin covers clock skew.
LOCK_MARGIN = timedelta(seconds=30)
# Readers and proxies may reuse the feed for five minutes.
FEED_CACHE = "public, max-age=300"
# The week page needs no scripts, images, or outside resources, only its inline style.
PAGE_SECURITY = "default-src 'none'; style-src 'unsafe-inline'"


@dataclass
class Config:
    """Settings for both functions, from the environment variables in template.yaml."""

    bucket: str
    model_id: str
    categories: list[str]
    shortlist_size: int
    digest_size: int
    max_chars: int
    contact_email: str
    feed_public_url: str
    feed_title: str
    lookback_days: int = 7
    announce_lag_days: int = 3
    pdf_delay_seconds: float = 3.0
    max_papers: int = 5000
    # Empty means the shortlist is ranked with model_id.
    rank_model_id: str = ""

    @classmethod
    def from_env(cls) -> "Config":
        """Read the settings, using the template defaults for missing variables."""
        categories = [
            item.strip()
            for item in os.environ.get("CATEGORIES", "cs.AI,cs.LG,cs.CL,cs.CV,stat.ML").split(",")
            if item.strip()
        ]
        return cls(
            bucket=os.environ["BUCKET_NAME"],
            model_id=os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0"),
            categories=categories,
            shortlist_size=max(1, int(os.environ.get("SHORTLIST_SIZE", "30"))),
            digest_size=max(1, int(os.environ.get("DIGEST_SIZE", "10"))),
            max_chars=max(500, int(os.environ.get("MAX_CHARS_PER_PAPER", "20000"))),
            contact_email=os.environ.get("CONTACT_EMAIL", "anonymous@example.com"),
            feed_public_url=os.environ.get("FEED_PUBLIC_URL", ""),
            feed_title=os.environ.get("FEED_TITLE", "AI Paper News"),
            lookback_days=max(1, int(os.environ.get("LOOKBACK_DAYS", "7"))),
            announce_lag_days=max(0, int(os.environ.get("ANNOUNCE_LAG_DAYS", "3"))),
            pdf_delay_seconds=max(0.0, float(os.environ.get("PDF_DELAY_SECONDS", "3"))),
            max_papers=max(1, int(os.environ.get("MAX_PAPERS", "5000"))),
            rank_model_id=os.environ.get("RANK_MODEL_ID", "").strip(),
        )


def publish_digest(event, context):
    """Lambda handler for the weekly schedule and for manual invokes. The event is ignored."""
    import boto3
    from botocore.config import Config as BotoConfig

    logging.getLogger().setLevel(logging.INFO)
    remaining_ms = getattr(context, "get_remaining_time_in_millis", None)
    # Bounded timeouts and one retry, so a stuck Bedrock call cannot use up the run.
    bedrock = boto3.client(
        "bedrock-runtime",
        config=BotoConfig(
            connect_timeout=10,
            read_timeout=120,
            retries={"mode": "standard", "max_attempts": 2},
        ),
    )
    return run_digest(
        Config.from_env(),
        boto3.client("s3"),
        boto3.client("comprehend"),
        bedrock,
        now=datetime.now(timezone.utc),
        fetch_papers=fetch_recent_papers,
        download_pdf_fn=download_pdf,
        sleep=time.sleep,
        request_id=getattr(context, "aws_request_id", None),
        time_left=(lambda: remaining_ms() / 1000) if remaining_ms else None,
    )


def serve_feed(event, context):
    """Lambda handler for the public function URL."""
    import boto3

    config = Config.from_env()
    return feed_response(FeedStore(boto3.client("s3"), config.bucket), config, event or {})


def feed_response(store: FeedStore, config: Config, event: dict) -> dict:
    """Answer a function URL request with the feed, or one week as HTML."""
    headers = _headers(event)
    accept = str(headers.get("accept-encoding") or "")
    params = event.get("queryStringParameters")
    week = params.get("week") if isinstance(params, dict) else None
    if week:
        return _week_page(store, config, str(week), accept)
    # HTTP says If-None-Match takes precedence over If-Modified-Since.
    etag = _client_etag(headers.get("if-none-match"))
    stored = store.read_feed(
        if_none_match=etag,
        if_modified_since=None if etag else _client_date(headers.get("if-modified-since")),
    )
    validators = {}
    # S3's ETag describes the stored bytes, and the body may be sent gzipped,
    # so it is passed on as a weak ETag.
    if stored.etag:
        validators["ETag"] = stored.etag if stored.etag.startswith("W/") else f"W/{stored.etag}"
    if stored.last_modified:
        validators["Last-Modified"] = format_datetime(
            stored.last_modified.astimezone(timezone.utc), usegmt=True
        )
    if stored.not_modified:
        return {
            "statusCode": 304,
            "headers": {"Cache-Control": FEED_CACHE, "Vary": "Accept-Encoding", **validators},
            "body": "",
        }
    body = stored.body
    # Nothing is published yet. Serve a valid empty feed so readers can subscribe.
    if not body:
        base = _request_url(event) or config.feed_public_url
        body = empty_feed(config.feed_title, _week_url(base, "latest"))
    return _response(
        {"Content-Type": "application/rss+xml; charset=utf-8", "Cache-Control": FEED_CACHE, **validators},
        body,
        accept,
    )


def run_digest(
    config: Config,
    s3,
    comprehend,
    bedrock,
    *,
    now: datetime,
    fetch_papers,
    download_pdf_fn,
    sleep,
    request_id: str | None = None,
    time_left=None,
) -> dict:
    """Publish the digest for the ISO week of `now`.

    The AWS clients and the arXiv functions are passed in so tests can replace
    them. `time_left` returns the seconds left before Lambda stops this
    attempt. Returns {"status": "published", ...} with the featured paper ids,
    or {"status": "skipped", "reason": ...} when the week is already published
    or another invocation is running it. Errors are raised after the lock is
    released, so Lambda's retry can run the week again.
    """
    logging.getLogger().setLevel(logging.INFO)

    def seconds_left() -> float:
        return time_left() if time_left else float("inf")

    store = FeedStore(s3, config.bucket)
    week = iso_week_key(now)
    # The lock expires just after Lambda would stop this attempt, so a run
    # that crashed or timed out does not block the week.
    expires_at = now + timedelta(seconds=time_left()) + LOCK_MARGIN if time_left else None
    claim, token = store.claim_week(week, now, request_id=request_id, expires_at=expires_at)
    if claim != "claimed" or not token:
        logger.info("skipping week %s (%s)", week, claim)
        return {"status": "skipped", "week": week, "reason": claim}

    # arXiv can take three days to announce a submission (Friday afternoon to
    # Monday evening, US Eastern), so the window ends that far back.
    until = now - timedelta(days=config.announce_lag_days)
    since = until - timedelta(days=config.lookback_days)
    published = False
    try:
        papers = fetch_papers(
            config.categories,
            until=until,
            days=config.lookback_days,
            contact_email=config.contact_email,
            pause_seconds=config.pdf_delay_seconds,
            max_papers=config.max_papers,
            should_stop=lambda: seconds_left() < FETCH_MIN_SECONDS,
        )
        # A normal week has thousands of papers, so none means arXiv failed.
        # Raising lets Lambda retry.
        if not papers:
            raise RuntimeError(f"arXiv returned no papers submitted {_covered(since, until)}")
        seen = store.get_seen()
        fresh = [paper for paper in papers if paper.base_id not in seen]
        if not fresh:
            store.complete_week(week, {"paper_ids": [], "note": "no new papers"}, now, token)
            published = True
            return {"status": "published", "week": week, "papers": [], "note": "no new papers"}

        shortlist = select_shortlist(
            bedrock,
            fresh,
            model_id=config.model_id,
            size=config.shortlist_size,
            should_stop=lambda: seconds_left() < SHORTLIST_MIN_SECONDS,
            seed=week,
        )
        _read_papers(
            shortlist,
            config,
            comprehend,
            download_pdf_fn,
            sleep,
            has_time=lambda: seconds_left() >= PDF_MIN_SECONDS,
        )
        ranked = rank_papers(
            bedrock,
            shortlist,
            model_id=config.rank_model_id or config.model_id,
            digest_size=min(config.digest_size, len(shortlist)),
            use_model=seconds_left() >= RANK_MIN_SECONDS,
            seed=week,
        )
        digest = write_digest(
            bedrock,
            ranked,
            model_id=config.model_id,
            default_title=f"AI Research Weekly: {_date(now)}",
            should_stop=lambda: seconds_left() < WRITE_MIN_SECONDS,
        )
        item = digest_item(
            digest,
            guid=week,
            link=_week_url(config.feed_public_url, week),
            published=now,
            covered=_covered(since, until),
        )
        feed = _updated_feed(store, config, item, week, now)
        # A retry may have taken the lock over while this run was working.
        if not store.owns_week(week, token):
            raise LostWeekLock(week)
        store.put_feed(feed)
        featured = {paper.base_id for paper, _reason in ranked}
        store.put_seen(seen | featured)
        store.complete_week(
            week,
            {
                "paper_ids": [paper.base_id for paper, _reason in ranked],
                "title": digest.title,
            },
            datetime.now(timezone.utc),
            token,
        )
        published = True
        logger.info("published %s with %s papers", week, len(ranked))
        return {
            "status": "published",
            "week": week,
            "papers": [paper.base_id for paper, _reason in ranked],
        }
    except LostWeekLock:
        logger.warning("another invocation took over %s; skipping this one", week)
        return {"status": "skipped", "week": week, "reason": "lost_lock"}
    except Exception:
        if not published:
            try:
                store.release_week(week, token)
            except Exception:
                logger.warning("could not release the lock for %s", week, exc_info=True)
        raise


def _read_papers(papers, config: Config, comprehend, download_pdf_fn, sleep, has_time=None) -> None:
    """Fill in each paper's extract and Comprehend scores, in place.

    A paper whose PDF fails, and every paper after `has_time()` turns false,
    uses its abstract instead. Downloads are spaced by the arXiv pause.
    """
    reading = True
    for index, paper in enumerate(papers):
        if reading and has_time is not None and not has_time():
            logger.warning("time is short; using abstracts for the last %s papers", len(papers) - index)
            reading = False
        paper.extract = ""
        paper.full_text_available = False
        if reading:
            if index:
                sleep(config.pdf_delay_seconds)
            try:
                pdf_bytes = download_pdf_fn(paper.pdf_url, config.contact_email)
                paper.extract = select_sections(extract_text(pdf_bytes), config.max_chars)
                paper.full_text_available = bool(paper.extract.strip())
            except Exception:
                logger.warning("PDF unavailable for %s", paper.base_id, exc_info=True)
                paper.extract = ""
                paper.full_text_available = False
        if not paper.extract.strip():
            paper.extract = paper.abstract[: config.max_chars]
            paper.full_text_available = False
        # Put the abstract first when the PDF text lost it, so the models always see it.
        elif paper.abstract and paper.abstract[:80] not in paper.extract:
            paper.extract = f"Abstract: {paper.abstract}\n\n{paper.extract}"
            paper.extract = paper.extract[: config.max_chars]
        try:
            analysis = score_text(comprehend, paper.extract) if paper.extract.strip() else None
        except Exception:
            logger.warning("Comprehend failed for %s", paper.base_id, exc_info=True)
            analysis = None
        if analysis:
            paper.sentiment_label = analysis["label"]
            paper.sentiment_scores = analysis["scores"]
            paper.key_phrases = analysis["key_phrases"]


def _updated_feed(store: FeedStore, config: Config, item: dict, week: str, now: datetime) -> str:
    """Return feed.xml with this week's item added.

    A stored feed that cannot be parsed is backed up and replaced by a new feed.
    """
    current = store.get_feed()
    if current:
        try:
            return upsert_item(current, item)
        except ValueError:
            backup = store.backup_feed(week)
            logger.error("feed.xml could not be read; saved it to %s and started a new feed", backup, exc_info=True)
    fresh = empty_feed(config.feed_title, _week_url(config.feed_public_url, "latest"), now)
    return upsert_item(fresh, item)


def _week_page(store: FeedStore, config: Config, week: str, accept: str) -> dict:
    """Return one week's digest as an HTML page, or a 404."""
    body = store.get_feed()
    item = find_item(body, week) if body else None
    if item is None:
        return {"statusCode": 404, "headers": {"Content-Type": "text/plain; charset=utf-8"}, "body": "Not found"}
    return _response(
        {
            "Content-Type": "text/html; charset=utf-8",
            "Cache-Control": FEED_CACHE,
            "Content-Security-Policy": PAGE_SECURITY,
            "X-Content-Type-Options": "nosniff",
        },
        render_item_page(config.feed_title, item),
        accept,
    )


def _response(headers: dict, body: str, accept_encoding: str) -> dict:
    """Build a function URL response, gzipped when the client accepts it.

    Function URLs need binary bodies base64-encoded.
    """
    headers = {**headers, "Vary": "Accept-Encoding"}
    if not _accepts_gzip(accept_encoding):
        return {"statusCode": 200, "headers": headers, "body": body, "isBase64Encoded": False}
    headers["Content-Encoding"] = "gzip"
    packed = base64.b64encode(gzip.compress(body.encode("utf-8"))).decode("ascii")
    return {"statusCode": 200, "headers": headers, "body": packed, "isBase64Encoded": True}


def _accepts_gzip(value: str) -> bool:
    """Return True if Accept-Encoding allows gzip. `gzip;q=0` refuses it."""
    for part in value.split(","):
        name, _, params = part.partition(";")
        if name.strip().lower() != "gzip":
            continue
        quality = params.strip().lower()
        if not quality.startswith("q="):
            return True
        try:
            return float(quality[2:]) > 0
        except ValueError:
            return False
    return False


def _client_etag(value) -> str | None:
    """Return the first ETag in If-None-Match, without the weak prefix."""
    text = str(value or "").split(",")[0].strip()
    if text.startswith("W/"):
        text = text[2:]
    return text or None


def _client_date(value) -> datetime | None:
    """Parse an If-Modified-Since date, or return None if it is missing or invalid."""
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _week_url(base: str, week: str) -> str:
    """Return the URL of a week's HTML page, such as FEED_URL?week=2026-W44."""
    if not base:
        return ""
    return f"{base}?{urllib.parse.urlencode({'week': week})}"


def _covered(since: datetime, until: datetime) -> str:
    """Describe the submission window, such as "from October 18, 2026 to October 25, 2026"."""
    return f"from {_date(since)} to {_date(until)}"


def _date(moment: datetime) -> str:
    """Format a date in UTC, such as "October 29, 2026"."""
    moment = moment.astimezone(timezone.utc) if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    return f"{_MONTHS[moment.month - 1]} {moment.day}, {moment.year}"


# The empty feed links to the host the request came in on. Only function URL
# hosts are used, so a forged Host header cannot put another site in the feed.
_LAMBDA_HOST = re.compile(
    r"^[a-z0-9-]+\.lambda-url\.[a-z0-9-]+\.on\.aws$",
    re.IGNORECASE,
)


def _headers(event) -> dict:
    """Return the request headers with lowercase names."""
    raw = (event or {}).get("headers") or {}
    try:
        items = raw.items()
    except AttributeError:
        return {}
    return {str(key).lower(): value for key, value in items}


def _request_url(event) -> str:
    """Return this function URL's base URL, or "" if the Host is not a function URL."""
    headers = _headers(event)
    host = str(headers.get("host") or "").strip()
    host_only = host
    if host.startswith("["):
        return ""
    if ":" in host:
        host_only, port = host.rsplit(":", 1)
        if not port.isdigit():
            return ""
    if not _LAMBDA_HOST.fullmatch(host_only):
        return ""
    proto = str(headers.get("x-forwarded-proto") or "https").split(",")[0].strip().lower()
    if proto not in {"http", "https"}:
        proto = "https"
    return f"{proto}://{host}/"
