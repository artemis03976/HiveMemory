"""MemoryLibrary store 暴露的读模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hivememory.core.models import TopicData
    from hivememory.memory_library.topic_log import TopicRetiredRecord


@dataclass
class ArtifactIntegrityResult:
    artifact_id: str
    ok: bool
    stored_hash: str | None = None
    actual_hash: str | None = None


@dataclass(frozen=True)
class StorageHealthComponent:
    name: str
    healthy: bool
    required: bool = True
    detail: str | None = None


@dataclass(frozen=True)
class StorageHealthReport:
    components: tuple[StorageHealthComponent, ...]

    @property
    def healthy(self) -> bool:
        return all(component.healthy for component in self.components if component.required)


@dataclass(frozen=True)
class LoadedTopicLogs:
    """启动时读回话题日志的结果。

    ``topics`` 是没有退役、已经回到话题池的话题；``retired`` 是已经退役、最后一次沉淀没有
    完成的话题及其退役记录，日志保留，等待重新发起这次沉淀；``rejected_files`` 是读不回、
    移到一旁的日志。
    """

    topics: list[TopicData]
    retired: list[tuple[TopicData, TopicRetiredRecord]]
    rejected_files: list[str]


__all__ = [
    "ArtifactIntegrityResult",
    "LoadedTopicLogs",
    "StorageHealthComponent",
    "StorageHealthReport",
]
