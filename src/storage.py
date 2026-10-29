"""S3 state for the RSS feed, seen papers, and the weekly run lock.

Bucket layout:

    feed.xml              the public feed
    tmp/feed.xml          staging copy of the next feed.xml
    state/seen.json       arXiv ids already featured, so no paper appears twice
    runs/YYYY-Www.json    one record per ISO week, and the lock while it runs

A run claims its week by creating the run record with a conditional write
that fails if the record exists, so only one invocation can hold it. The
record carries a random token. The run checks the token again just before it
writes the feed, so an invocation whose lock was taken over stops there.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from botocore.exceptions import ClientError

FEED_KEY = "feed.xml"
TMP_FEED_KEY = "tmp/feed.xml"
SEEN_KEY = "state/seen.json"
# Used only for locks that carry no deadline. Longer than the 15-minute
# Lambda timeout so a live run is not treated as abandoned.
RUN_STALE_AFTER = timedelta(minutes=20)


class LostWeekLock(RuntimeError):
    """The caller no longer owns the weekly run record."""


@dataclass
class StoredFeed:
    """feed.xml as read from S3."""

    # None when there is no feed yet, or when the reader's copy is current.
    body: str | None
    etag: str | None = None
    last_modified: datetime | None = None
    # True when S3 answered a conditional read with 304 Not Modified.
    not_modified: bool = False


class FeedStore:
    """Read and write the feed, seen papers, and run records in one bucket."""

    def __init__(self, s3, bucket: str):
        self.s3 = s3
        self.bucket = bucket

    def get_seen(self) -> set[str]:
        """Return the base ids of papers already featured."""
        payload = self._get_json(SEEN_KEY)
        if not payload:
            return set()
        ids = payload.get("ids") or []
        if not isinstance(ids, list):
            return set()
        return {str(identifier) for identifier in ids}

    def put_seen(self, ids: set[str]) -> None:
        """Replace the list of featured paper ids."""
        body = json.dumps({"ids": sorted(ids)}, indent=2).encode("utf-8")
        self.s3.put_object(
            Bucket=self.bucket,
            Key=SEEN_KEY,
            Body=body,
            ContentType="application/json",
        )

    def get_feed(self) -> str | None:
        """Return feed.xml, or None if nothing has been published yet."""
        return self.read_feed().body

    def read_feed(
        self,
        *,
        if_none_match: str | None = None,
        if_modified_since: datetime | None = None,
    ) -> StoredFeed:
        """Read feed.xml, passing a feed reader's cache validators to S3.

        If the reader's copy is current, S3 sends no body and the result has
        `not_modified` set.
        """
        params = {"Bucket": self.bucket, "Key": FEED_KEY}
        if if_none_match:
            params["IfNoneMatch"] = if_none_match
        if if_modified_since:
            params["IfModifiedSince"] = if_modified_since
        try:
            response = self.s3.get_object(**params)
        except ClientError as exc:
            if _not_modified(exc):
                return StoredFeed(body=None, etag=if_none_match, not_modified=True)
            if _missing(exc):
                return StoredFeed(body=None)
            raise
        return StoredFeed(
            body=response["Body"].read().decode("utf-8"),
            etag=response.get("ETag"),
            last_modified=response.get("LastModified"),
        )

    def put_feed(self, xml_text: str) -> None:
        """Upload the feed to tmp/, then copy it over feed.xml.

        feed.xml is only replaced by a server-side copy of a complete upload.
        """
        body = xml_text.encode("utf-8")
        self.s3.put_object(
            Bucket=self.bucket,
            Key=TMP_FEED_KEY,
            Body=body,
            ContentType="application/rss+xml; charset=utf-8",
        )
        self.s3.copy_object(
            Bucket=self.bucket,
            Key=FEED_KEY,
            CopySource={"Bucket": self.bucket, "Key": TMP_FEED_KEY},
            ContentType="application/rss+xml; charset=utf-8",
            MetadataDirective="REPLACE",
        )

    def backup_feed(self, name: str) -> str:
        """Copy the live feed to `state/` before it is replaced."""
        key = f"state/feed-broken-{name}.xml"
        self.s3.copy_object(
            Bucket=self.bucket,
            Key=key,
            CopySource={"Bucket": self.bucket, "Key": FEED_KEY},
        )
        return key

    def claim_week(
        self,
        week: str,
        now: datetime,
        *,
        request_id: str | None = None,
        expires_at: datetime | None = None,
    ) -> tuple[str, str | None]:
        """Create the weekly run record if this invocation may own it.

        Returns (`claimed`, token), (`published`, None), or (`in_progress`, None).
        A running record can be replaced by a retry of the same Lambda request,
        or by anyone once the earlier attempt's deadline has passed.
        """
        key = run_key(week)
        token = uuid.uuid4().hex
        payload = _running_body(week, token, now, request_id, expires_at)
        try:
            self._create(key, payload)
            return "claimed", token
        except ClientError as exc:
            if not _is_precondition_failed(exc):
                raise
        current = self._get_json(key) or {}
        if current.get("status") == "published":
            return "published", None
        if current.get("status") == "running" and not _abandoned(current, now, request_id):
            return "in_progress", None
        self.s3.delete_object(Bucket=self.bucket, Key=key)
        try:
            self._create(key, payload)
            return "claimed", token
        except ClientError as exc:
            if _is_precondition_failed(exc):
                return "in_progress", None
            raise

    def owns_week(self, week: str, token: str) -> bool:
        """Return True while the week's record is still this invocation's lock."""
        return self._owned_lock(week, token) is not None

    def release_week(self, week: str, token: str) -> None:
        """Delete the lock only when this invocation still owns it."""
        if self.owns_week(week, token):
            self.s3.delete_object(Bucket=self.bucket, Key=run_key(week))

    def complete_week(self, week: str, details: dict, now: datetime, token: str) -> None:
        """Replace the lock with the published record for the week.

        `details` is stored with the record, for example the featured paper
        ids. Raises LostWeekLock if another invocation holds the record now.
        """
        lock = self._owned_lock(week, token)
        if lock is None:
            raise LostWeekLock(week)
        body = {
            **details,
            "status": "published",
            "week": week,
            "started_at": lock.get("started_at"),
            "finished_at": _as_utc(now).isoformat(),
        }
        self.s3.put_object(
            Bucket=self.bucket,
            Key=run_key(week),
            Body=json.dumps(body, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    def _owned_lock(self, week: str, token: str) -> dict | None:
        """Return the running record if it carries `token`, else None."""
        current = self._get_json(run_key(week)) or {}
        if current.get("status") == "running" and token and current.get("token") == token:
            return current
        return None

    def _create(self, key: str, payload: bytes) -> None:
        """Write `key` only if it does not exist yet.

        S3 rejects the write with PreconditionFailed when the key exists.
        """
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=payload,
            ContentType="application/json",
            IfNoneMatch="*",
        )

    def _get_text(self, key: str) -> str | None:
        """Return the object's text, or None if the key does not exist."""
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if _missing(exc):
                return None
            raise
        return response["Body"].read().decode("utf-8")

    def _get_json(self, key: str) -> dict | None:
        """Return the object as a JSON object, or None if it is missing or unreadable."""
        text = self._get_text(key)
        if text is None:
            return None
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None


def run_key(week: str) -> str:
    """Return the S3 key of a week's run record."""
    return f"runs/{week}.json"


def iso_week_key(now: datetime) -> str:
    """Return the ISO week of `now` in UTC, such as 2026-W44."""
    moment = _as_utc(now)
    year, week, _day = moment.isocalendar()
    return f"{year}-W{week:02d}"


def _running_body(
    week: str,
    token: str,
    now: datetime,
    request_id: str | None,
    expires_at: datetime | None,
) -> bytes:
    """Return the JSON for a lock held by a run in progress."""
    return json.dumps(
        {
            "status": "running",
            "week": week,
            "token": token,
            "started_at": _as_utc(now).isoformat(),
            "expires_at": _as_utc(expires_at).isoformat() if expires_at else None,
            "request_id": request_id,
        }
    ).encode("utf-8")


def _abandoned(lock: dict, now: datetime, request_id: str | None) -> bool:
    """Return True if a running lock may be taken over."""
    # Lambda keeps the request id when it retries a failed async event. A rare
    # duplicate delivery keeps it too, so the run that loses the lock must
    # skip the week rather than fail and be retried.
    if request_id and lock.get("request_id") == request_id:
        return True
    moment = _as_utc(now)
    deadline = _parse_time(lock.get("expires_at"))
    if deadline is not None:
        return moment >= deadline
    started = _parse_time(lock.get("started_at"))
    return started is None or moment - started >= RUN_STALE_AFTER


def _is_precondition_failed(exc: ClientError) -> bool:
    """Return True if a conditional write lost to an existing or concurrent write."""
    code = exc.response.get("Error", {}).get("Code", "")
    return code in {"PreconditionFailed", "ConditionalRequestConflict"}


def _missing(exc: ClientError) -> bool:
    """Return True if S3 says the key does not exist.

    S3 reports a missing key as 403 AccessDenied to a caller that may not
    list the bucket, so the functions are granted s3:ListBucket.
    """
    return exc.response.get("Error", {}).get("Code", "") in {"NoSuchKey", "404", "NotFound"}


def _not_modified(exc: ClientError) -> bool:
    """Return True if S3 answered a conditional read with 304 Not Modified."""
    code = exc.response.get("Error", {}).get("Code", "")
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"304", "NotModified"} or status == 304


def _parse_time(value) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _as_utc(parsed)


def _as_utc(now: datetime) -> datetime:
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)
