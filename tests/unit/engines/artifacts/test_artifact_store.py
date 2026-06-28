"""
Artifact 存储层与 ArtifactEngine 单元测试

覆盖：
- FilesystemArtifactStore put/get/exists
- ArtifactRef.sha256 内容校验
- 旧 MemoryAtom Artifacts payload 反序列化兼容性
- ArtifactEngine 可实例化并持有三个 builder 引用
"""

import json
import hashlib
import pytest
from datetime import datetime
from pathlib import Path

from hivememory.core.models.artifact import (
    ArtifactType,
    ArtifactRef,
    InteractionArtifact,
    MemoryEventLog,
    MemoryEventType,
)
from hivememory.core.models.memory import Artifacts
from hivememory.engines.artifacts.engine import ArtifactEngine
from hivememory.patchouli.memory_library.adapters.artifact import FilesystemArtifactStorageAdapter
from hivememory.patchouli.memory_library.stores import ArtifactStore


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(FilesystemArtifactStorageAdapter(root_dir=str(tmp_path)))


def _make_artifact(artifact_id: str = "test-id") -> InteractionArtifact:
    return InteractionArtifact(
        artifact_id=artifact_id,
        created_at=datetime(2026, 6, 14, 12, 0, 0),
        owner_user_id="u1",
        topic_id="topic-1",
        captured_at=datetime(2026, 6, 14, 12, 0, 0),
    )


# ── FilesystemArtifactStore ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_put_and_get_roundtrip(store):
    artifact = _make_artifact()
    ref = await store.put(artifact)

    assert ref.artifact_id == "test-id"
    assert ref.artifact_type == ArtifactType.INTERACTION
    assert ref.uri is not None

    data = await store.get(ref)
    assert data["artifact_id"] == "test-id"
    assert data["topic_id"] == "topic-1"


@pytest.mark.asyncio
async def test_sha256_matches_content(store):
    artifact = _make_artifact()
    ref = await store.put(artifact)

    data = await store.get(ref)
    stored_hash = data["content_hash"]

    # put_json 在 content_hash=null 时计算 sha256，之后才写入真实值
    # 还原为 null 以重现原始序列化
    data["content_hash"] = None
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    expected = hashlib.sha256(payload.encode()).hexdigest()
    assert stored_hash == expected
    assert ref.sha256 == stored_hash


@pytest.mark.asyncio
async def test_get_json_rejects_tampered_content(store):
    artifact = _make_artifact()
    ref = await store.put(artifact)

    path = Path(ref.uri)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["topic_id"] = "tampered-topic"
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        await store.get(ref)


@pytest.mark.asyncio
async def test_exists(store):
    artifact = _make_artifact("exists-id")
    assert not await store.exists("exists-id")
    await store.put(artifact)
    assert await store.exists("exists-id")


@pytest.mark.asyncio
async def test_get_by_id_string(store):
    artifact = _make_artifact("str-lookup")
    await store.put(artifact)
    data = await store.get("str-lookup")
    assert data["artifact_id"] == "str-lookup"


# ── 旧 MemoryAtom payload 兼容性 ─────────────────────────────────────────────

def test_old_artifacts_payload_ignores_removed_legacy_fields():
    """v0.4 旧字段已被结构化 artifact 体系替代，反序列化时忽略。"""
    old = {
        "raw_source_url": "https://example.com",
        "file_path": "/tmp/doc.md",
        "context_ref": [{"session_id": "s1", "msg_id": "m1"}],
        "full_history": [{"timestamp": "2026-01-01", "content": "v1", "reason": "init"}],
    }
    artifacts = Artifacts.model_validate(old)
    assert not hasattr(artifacts, "raw_source_url")
    assert not hasattr(artifacts, "file_path")
    assert not hasattr(artifacts, "context_ref")
    assert not hasattr(artifacts, "full_history")
    assert artifacts.refs == []
    assert artifacts.events == []
    assert artifacts.cold_archive_uri is None
    assert artifacts.cold_archive_hash is None
    assert artifacts.revival_keys == []


def test_artifacts_payload_deserializes_refs_and_events_as_models():
    payload = {
        "refs": [
            {
                "artifact_id": "interaction-1",
                "artifact_type": "interaction",
                "uri": "/tmp/interaction.json",
                "sha256": "abc",
            }
        ],
        "events": [
            {
                "event_type": "versioned",
                "artifact_refs": [
                    {
                        "artifact_id": "interaction-1",
                        "artifact_type": "interaction",
                    }
                ],
            }
        ],
    }

    artifacts = Artifacts.model_validate(payload)

    assert isinstance(artifacts.refs[0], ArtifactRef)
    assert isinstance(artifacts.events[0], MemoryEventLog)
    assert artifacts.events[0].event_type == MemoryEventType.VERSIONED
    assert isinstance(artifacts.events[0].artifact_refs[0], ArtifactRef)



# ── ArtifactEngine ────────────────────────────────────────────────────────────

def test_artifact_engine_holds_all_builders(tmp_path):
    store = ArtifactStore(FilesystemArtifactStorageAdapter(root_dir=str(tmp_path)))
    engine = ArtifactEngine(store)
    assert engine.interaction is not None
    assert engine.document is not None
    assert engine.memory is not None
