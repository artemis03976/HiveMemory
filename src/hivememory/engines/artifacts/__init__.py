"""
engines/artifacts - Artifact 构建引擎

ArtifactEngine 是统一的组合容器，注入 ArtifactStore 后
持有三个 builder 的单一入口。
"""

from hivememory.engines.artifacts.engine import ArtifactEngine
from hivememory.engines.artifacts.interaction import (
    InteractionArtifactBuilder,
    NoOpInteractionArtifactBuilder,
    create_interaction_builder,
)
from hivememory.engines.artifacts.document import (
    DocumentArtifactBuilder,
    NoOpDocumentArtifactBuilder,
    create_document_builder,
)
from hivememory.engines.artifacts.memory import (
    MemoryArtifactBuilder,
    MemoryCreationBundle,
    NoOpMemoryArtifactBuilder,
    create_memory_builder,
)

__all__ = [
    "ArtifactEngine",
    "InteractionArtifactBuilder",
    "NoOpInteractionArtifactBuilder",
    "create_interaction_builder",
    "DocumentArtifactBuilder",
    "NoOpDocumentArtifactBuilder",
    "create_document_builder",
    "MemoryArtifactBuilder",
    "MemoryCreationBundle",
    "NoOpMemoryArtifactBuilder",
    "create_memory_builder",
]
