"""话题句柄的读取结果 → MemoryUnitIR 构建器。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.models import format_topic_handle
from hivememory.engines.memory_compiler.ir import (
    MemoryContentIR,
    MemoryIdentityIR,
    MemoryStatusIR,
    MemoryUnitIR,
)

if TYPE_CHECKING:
    from hivememory.core.models.reference import ReferenceResolution


def build_topic_ir(resolution: ReferenceResolution) -> MemoryUnitIR:
    """单元的 alias 是规范写法的句柄（话题 id 加条目序号），与渲染中签发的句柄一致。"""
    topic = resolution.topic
    if resolution.kind != "topic" or topic is None:
        raise ValueError(f"build_topic_ir only handles 'topic', got '{resolution.kind}'.")
    return MemoryUnitIR(
        identity=MemoryIdentityIR(
            source_kind="topic",
            alias=format_topic_handle(topic.topic_id, resolution.topic_entry),
        ),
        content=MemoryContentIR(title=topic.topic_title),
        status=MemoryStatusIR(),
        topic=topic,
        topic_entry=resolution.topic_entry,
    )
