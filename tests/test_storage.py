"""Weekly run lock and feed storage."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from botocore.exceptions import ClientError

from storage import FeedStore, LostWeekLock, iso_week_key, run_key
from tests.fakes import FakeS3

NOW = datetime(2026, 10, 28, 6, 0, tzinfo=timezone.utc)
WEEK = "2026-W44"
DEADLINE = NOW + timedelta(minutes=15, seconds=30)


def test_iso_week_covers_monday_through_sunday():
    monday = datetime(2026, 10, 26, tzinfo=timezone.utc)
    sunday = datetime(2026, 11, 1, 23, tzinfo=timezone.utc)
    next_monday = datetime(2026, 11, 2, tzinfo=timezone.utc)
    assert iso_week_key(monday) == iso_week_key(sunday) == iso_week_key(NOW)
    assert iso_week_key(next_monday) != iso_week_key(monday)


def test_a_live_run_blocks_other_requests_until_its_deadline():
    store = FeedStore(FakeS3(), "bucket")
    status, token = store.claim_week(WEEK, NOW, request_id="first", expires_at=DEADLINE)
    assert status == "claimed"
    assert store.claim_week(WEEK, NOW + timedelta(minutes=1), request_id="second")[0] == "in_progress"
    assert store.claim_week(WEEK, NOW + timedelta(minutes=15), request_id="second")[0] == "in_progress"

    status, late_token = store.claim_week(WEEK, DEADLINE, request_id="second")
    assert status == "claimed"
    assert late_token != token
    assert not store.owns_week(WEEK, token)


def test_a_retry_of_the_same_request_takes_over_after_a_crash():
    store = FeedStore(FakeS3(), "bucket")
    _status, crashed = store.claim_week(WEEK, NOW, request_id="event-1", expires_at=DEADLINE)
    status, retry = store.claim_week(WEEK, NOW + timedelta(minutes=1), request_id="event-1")
    assert status == "claimed"
    assert store.owns_week(WEEK, retry)
    with pytest.raises(LostWeekLock):
        store.complete_week(WEEK, {"paper_ids": []}, NOW, crashed)


def test_published_weeks_are_skipped_and_release_needs_the_token():
    store = FeedStore(FakeS3(), "bucket")
    _status, token = store.claim_week(WEEK, NOW)
    store.release_week(WEEK, "someone-else")
    assert store.owns_week(WEEK, token)
    with pytest.raises(LostWeekLock):
        store.complete_week(WEEK, {"paper_ids": []}, NOW, "someone-else")

    store.complete_week(WEEK, {"paper_ids": ["2510.00001"]}, NOW, token)
    assert store.claim_week(WEEK, NOW, request_id="any")[0] == "published"
    saved = json.loads(store.s3.objects[run_key(WEEK)])
    assert saved["status"] == "published"
    assert saved["paper_ids"] == ["2510.00001"]
    assert saved["started_at"] == NOW.isoformat()

    store.release_week(WEEK, token)
    assert run_key(WEEK) in store.s3.objects
    store.s3.delete_object(Bucket="bucket", Key=run_key(WEEK))
    status, token = store.claim_week(WEEK, NOW)
    assert status == "claimed"
    store.release_week(WEEK, token)
    assert run_key(WEEK) not in store.s3.objects


def test_locks_without_a_deadline_go_stale_after_20_minutes():
    store = FeedStore(FakeS3(), "bucket")
    store.claim_week(WEEK, NOW)
    assert store.claim_week(WEEK, NOW + timedelta(minutes=19))[0] == "in_progress"
    assert store.claim_week(WEEK, NOW + timedelta(minutes=21))[0] == "claimed"


def test_corrupt_lock_times_are_treated_as_stale():
    store = FeedStore(FakeS3(), "bucket")
    store.s3.objects[run_key(WEEK)] = json.dumps(
        {"status": "running", "week": WEEK, "started_at": "not-a-date", "expires_at": "also-bad"}
    ).encode()
    assert store.claim_week(WEEK, NOW)[0] == "claimed"


def test_seen_ids_must_be_a_list():
    store = FeedStore(FakeS3(), "bucket")
    store.s3.objects["state/seen.json"] = b'{"ids": "2510.00001"}'
    assert store.get_seen() == set()


def test_put_feed_writes_a_temp_object_then_the_live_key():
    store = FeedStore(FakeS3(), "bucket")
    assert store.get_feed() is None
    assert store.get_seen() == set()
    store.put_feed("<rss>digest</rss>")
    assert store.get_feed() == "<rss>digest</rss>"
    assert store.s3.objects["tmp/feed.xml"] == store.s3.objects["feed.xml"]
    store.put_seen({"2510.00002", "2510.00001"})
    assert store.get_seen() == {"2510.00001", "2510.00002"}


def test_missing_keys_fail_loudly_without_list_bucket():
    store = FeedStore(FakeS3(list_bucket=False), "bucket")
    with pytest.raises(ClientError, match="AccessDenied"):
        store.get_seen()
    with pytest.raises(ClientError, match="AccessDenied"):
        store.get_feed()


def test_read_feed_reports_not_modified_for_a_matching_etag_or_date():
    store = FeedStore(FakeS3(), "bucket")
    store.put_feed("<rss>digest</rss>")
    fresh = store.read_feed()
    assert fresh.body == "<rss>digest</rss>"
    assert fresh.etag == store.s3.etag("feed.xml")

    cached = store.read_feed(if_none_match=fresh.etag)
    assert cached.not_modified
    assert cached.body is None
    assert store.read_feed(if_none_match='"stale"').body == "<rss>digest</rss>"
    assert store.read_feed(if_modified_since=fresh.last_modified).not_modified
    earlier = fresh.last_modified - timedelta(days=1)
    assert store.read_feed(if_modified_since=earlier).body == "<rss>digest</rss>"


def test_backup_feed_copies_the_live_feed_into_state():
    store = FeedStore(FakeS3(), "bucket")
    store.put_feed("<rss>broken")
    key = store.backup_feed(WEEK)
    assert key == f"state/feed-broken-{WEEK}.xml"
    assert store.s3.objects[key] == b"<rss>broken"
