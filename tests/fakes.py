"""In-memory AWS stand-ins for unit tests."""

from __future__ import annotations

import hashlib
import io
from datetime import datetime, timezone

from botocore.exceptions import ClientError

_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _error(code: str, status: int, operation: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        operation,
    )


class FakeS3:
    """S3 objects in memory.

    Real S3 reports a missing key as 403 AccessDenied unless the caller may
    list the bucket, so `list_bucket=False` models a role without ListBucket.
    """

    def __init__(self, *, list_bucket: bool = True):
        self.objects: dict[str, bytes] = {}
        self.modified: dict[str, datetime] = {}
        self.list_bucket = list_bucket

    def put_object(self, Bucket, Key, Body, ContentType=None, IfNoneMatch=None, **kwargs):
        if IfNoneMatch == "*" and Key in self.objects:
            raise _error("PreconditionFailed", 412, "PutObject")
        if isinstance(Body, str):
            Body = Body.encode("utf-8")
        self.objects[Key] = bytes(Body)
        self.modified[Key] = datetime.now(timezone.utc).replace(microsecond=0)

    def get_object(self, Bucket, Key, IfNoneMatch=None, IfModifiedSince=None):
        if Key not in self.objects:
            if self.list_bucket:
                raise _error("NoSuchKey", 404, "GetObject")
            raise _error("AccessDenied", 403, "GetObject")
        etag = self.etag(Key)
        modified = self.modified.get(Key, _EPOCH)
        if IfNoneMatch is not None and IfNoneMatch in {etag, "*"}:
            raise _error("304", 304, "GetObject")
        if IfNoneMatch is None and IfModifiedSince is not None and modified <= IfModifiedSince:
            raise _error("304", 304, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key]), "ETag": etag, "LastModified": modified}

    def etag(self, key: str) -> str:
        return '"' + hashlib.md5(self.objects[key]).hexdigest() + '"'

    def delete_object(self, Bucket, Key, **kwargs):
        self.objects.pop(Key, None)
        self.modified.pop(Key, None)

    def copy_object(self, Bucket, Key, CopySource, ContentType=None, MetadataDirective=None, **kwargs):
        self.objects[Key] = self.objects[CopySource["Key"]]
        self.modified[Key] = datetime.now(timezone.utc).replace(microsecond=0)


class FakeComprehend:
    """Comprehend with fixed answers. Every call is recorded.

    Text containing "great" is positive and anything else is neutral. Key
    phrases include a duplicate and a low-confidence phrase, which the code
    under test should drop.
    """

    def __init__(self):
        self.sentiment: list[tuple[str, str]] = []
        self.phrases: list[str] = []

    def detect_sentiment(self, Text, LanguageCode):
        self.sentiment.append((Text, LanguageCode))
        if "great" in Text:
            scores = {"Positive": 0.8, "Negative": 0.05, "Neutral": 0.1, "Mixed": 0.05}
        else:
            scores = {"Positive": 0.2, "Negative": 0.05, "Neutral": 0.7, "Mixed": 0.05}
        return {"SentimentScore": scores}

    def detect_key_phrases(self, Text, LanguageCode):
        self.phrases.append(Text)
        return {
            "KeyPhrases": [
                {"Text": "language model", "Score": 0.91},
                {"Text": "language model", "Score": 0.8},
                {"Text": "weak phrase", "Score": 0.2},
            ]
        }


class ScriptedBedrock:
    """Bedrock that answers Converse calls with scripted replies, in order.

    An exception in the script is raised instead of returned. The arguments
    of every call are recorded in `calls`.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise RuntimeError("no scripted response left")
        text = self.responses.pop(0)
        if isinstance(text, Exception):
            raise text
        return {"output": {"message": {"content": [{"text": text}]}}}
