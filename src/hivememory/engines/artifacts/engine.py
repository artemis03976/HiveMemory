"""ArtifactEngine - 极薄的组合根，不包含业务逻辑。"""

from hivememory.engines.artifacts.document import (
    DocumentArtifactBuilder,
    NoOpDocumentArtifactBuilder,
    create_document_builder,
)
from hivememory.engines.artifacts.interaction import (
    InteractionArtifactBuilder,
    NoOpInteractionArtifactBuilder,
    create_interaction_builder,
)
from hivememory.engines.artifacts.memory import (
    MemoryArtifactBuilder,
    NoOpMemoryArtifactBuilder,
    create_memory_builder,
)
from hivememory.system.config.patchouli import ArtifactConfig

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from hivememory.patchouli.memory_library.stores import ArtifactStore


class ArtifactEngine:
    """统一 artifact 构建入口，调用方只需注入一个实例。"""

    def __init__(
        self,
        *,
        config: ArtifactConfig,
        interaction: InteractionArtifactBuilder | NoOpInteractionArtifactBuilder,
        document: DocumentArtifactBuilder | NoOpDocumentArtifactBuilder,
        memory: MemoryArtifactBuilder | NoOpMemoryArtifactBuilder,
    ) -> None:
        self.config = config
        self.interaction = interaction
        self.document = document
        self.memory = memory

    @classmethod
    def from_store(
        cls,
        store: "ArtifactStore | None",
        config: ArtifactConfig | None = None,
    ) -> "ArtifactEngine":
        config = config or ArtifactConfig()
        if not config.enabled:
            store = None
        return cls(
            config=config,
            interaction=create_interaction_builder(config.interaction, store),
            document=create_document_builder(config.document, store),
            memory=create_memory_builder(config.memory, store),
        )

    @classmethod
    def noop(cls) -> "ArtifactEngine":
        return cls.from_store(store=None, config=ArtifactConfig(enabled=False))
