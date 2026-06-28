"""DocumentArtifactBuilder - 从外源文档元数据构建 DocumentArtifact。"""

from datetime import datetime
from typing import List, Optional

from hivememory.core.models.artifact import ArtifactRef, DocumentArtifact, DocumentLocator
from hivememory.patchouli.memory_library import ArtifactStore
from hivememory.system.config.patchouli import ArtifactComponentConfig

_DOC_FIELDS = set(DocumentArtifact.model_fields)


class DocumentArtifactBuilder:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    async def build_and_store(
        self,
        *,
        source_type: str,
        source_uri: Optional[str],
        content_hash: Optional[str],
        retrieved_at: datetime,
        locators: Optional[List[DocumentLocator]] = None,
        **kwargs,
    ) -> ArtifactRef | None:
        artifact = DocumentArtifact(
            source_type=source_type,
            source_uri=source_uri,
            content_hash=content_hash,
            retrieved_at=retrieved_at,
            locators=locators or [],
            **{k: v for k, v in kwargs.items() if k in _DOC_FIELDS},
        )
        return await self._store.put(artifact)


class NoOpDocumentArtifactBuilder:
    async def build_and_store(
        self,
        *,
        source_type: str,
        source_uri: Optional[str],
        content_hash: Optional[str],
        retrieved_at: datetime,
        locators: Optional[List[DocumentLocator]] = None,
        **kwargs,
    ) -> ArtifactRef | None:
        return None


def create_document_builder(
    config: ArtifactComponentConfig,
    store: ArtifactStore | None,
) -> DocumentArtifactBuilder | NoOpDocumentArtifactBuilder:
    if store is None or not config.enabled:
        return NoOpDocumentArtifactBuilder()
    return DocumentArtifactBuilder(store)
