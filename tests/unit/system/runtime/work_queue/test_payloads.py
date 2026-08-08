"""Work payload 稳定 JSON bytes 与 versioned codec 契约测试。"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from hivememory.system.runtime.work_queue import (
    DuplicateWorkPayloadCodecError,
    UnknownWorkPayloadCodecError,
    WorkPayloadCodecRegistry,
    WorkPayloadDecodeError,
    WorkPayloadEncodeError,
    decode_canonical_json,
    encode_canonical_json,
)


@dataclass
class _Submission:
    interaction_id: str
    events: list[str]


class _SubmissionCodec:
    kind = "test.submission"
    schema_version = 1

    def encode(self, payload: _Submission) -> object:
        return {
            "interaction_id": payload.interaction_id,
            "events": payload.events,
        }

    def decode(self, payload: object) -> _Submission:
        if not isinstance(payload, dict):
            raise TypeError("submission payload must be an object")
        interaction_id = payload.get("interaction_id")
        events = payload.get("events")
        if not isinstance(interaction_id, str):
            raise TypeError("interaction_id must be a string")
        if not isinstance(events, list) or not all(isinstance(item, str) for item in events):
            raise TypeError("events must be a string array")
        return _Submission(interaction_id=interaction_id, events=events)


def _registry() -> WorkPayloadCodecRegistry:
    registry = WorkPayloadCodecRegistry()
    registry.register(_SubmissionCodec())
    return registry


def test_json_encoding_is_deterministic_and_normalizes_sequences() -> None:
    first = encode_canonical_json({"z": 1, "items": ("a", "b"), "nested": {"x": True}})
    second = encode_canonical_json({"nested": {"x": True}, "items": ["a", "b"], "z": 1})

    assert first == second
    assert first == b'{"items":["a","b"],"nested":{"x":true},"z":1}'
    assert decode_canonical_json(first) == {
        "items": ["a", "b"],
        "nested": {"x": True},
        "z": 1,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"unsupported": object()},
        {"unordered": {"a", "b"}},
        {"not_finite": float("nan")},
    ],
)
def test_json_encoding_rejects_values_without_stable_projection(payload: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        encode_canonical_json(payload)


def test_json_decoder_accepts_valid_non_canonical_bytes() -> None:
    payload = b'{"z": 1, "items": ["a", "b"]}'

    assert decode_canonical_json(payload) == {"items": ["a", "b"], "z": 1}


def test_versioned_codec_creates_snapshot_and_fresh_decode_per_attempt() -> None:
    registry = _registry()
    source = _Submission(interaction_id="interaction-1", events=["created"])

    payload = registry.encode("test.submission", 1, source)
    source.events.append("source-changed")
    first = registry.decode("test.submission", 1, payload)
    first.events.append("handler-changed")
    second = registry.decode("test.submission", 1, payload)

    assert first.events == ["created", "handler-changed"]
    assert second == _Submission(interaction_id="interaction-1", events=["created"])


def test_registry_rejects_duplicate_and_unknown_codec_versions() -> None:
    registry = _registry()

    with pytest.raises(DuplicateWorkPayloadCodecError):
        registry.register(_SubmissionCodec())
    with pytest.raises(UnknownWorkPayloadCodecError):
        registry.encode("test.submission", 2, _Submission("interaction-1", []))
    with pytest.raises(UnknownWorkPayloadCodecError):
        registry.decode("unknown", 1, b"{}")


def test_registry_wraps_unsupported_business_values_as_safe_encode_error() -> None:
    registry = _registry()
    source = _Submission(interaction_id="interaction-1", events=[])
    source.events = object()  # type: ignore[assignment]

    with pytest.raises(WorkPayloadEncodeError) as captured:
        registry.encode("test.submission", 1, source)

    assert "interaction-1" not in str(captured.value)


def test_registry_wraps_invalid_json_as_safe_decode_error() -> None:
    registry = _registry()

    with pytest.raises(WorkPayloadDecodeError):
        registry.decode("test.submission", 1, b"not-json")
