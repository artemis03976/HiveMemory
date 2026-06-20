"""ArtifactEngine - 极薄的组合根，不包含业务逻辑。"""

from hivememory.engines.artifacts.document import DocumentArtifactBuilder
from hivememory.engines.artifacts.interaction import InteractionArtifactBuilder
from hivememory.engines.artifacts.memory import MemoryArtifactBuilder

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from hivememory.patchouli.memory_library.stores import ArtifactStore


class ArtifactEngine:
    """统一 artifact 构建入口，调用方只需注入一个实例。"""

    def __init__(self, store: "ArtifactStore") -> None:
        self.interaction = InteractionArtifactBuilder(store)
        self.document = DocumentArtifactBuilder(store)
        self.memory = MemoryArtifactBuilder(store)