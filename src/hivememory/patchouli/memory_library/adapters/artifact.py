"""FilesystemArtifactStorageAdapter — ArtifactStoragePort 的文件系统实现。

从 infrastructure/storage/artifact_store.py 迁入，接口对齐 §2.7。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from hivememory.core.models.artifact import ArtifactRef, ArtifactType, BaseArtifact
from hivememory.patchouli.memory_library.models import ArtifactIntegrityResult
from hivememory.patchouli.memory_library.ports import ArtifactStoragePort

logger = logging.getLogger(__name__)


class FilesystemArtifactStorageAdapter(ArtifactStoragePort):
    """文件系统 Artifact 仓库。布局：{root}/{type}/{YYYY}/{MM}/{DD}/{id}.json"""

    def __init__(self, root_dir: str = ".hivememory/artifacts") -> None:
        self._root = Path(root_dir)

    def _artifact_path(self, artifact: BaseArtifact) -> Path:
        ts = artifact.created_at
        return (
            self._root
            / artifact.artifact_type.value
            / str(ts.year)
            / f"{ts.month:02d}"
            / f"{ts.day:02d}"
            / f"{artifact.artifact_id}.json"
        )

    def _find_by_id(self, artifact_id: str) -> Optional[Path]:
        matches = list(self._root.rglob(f"{artifact_id}.json"))
        return matches[0] if matches else None

    async def put(self, artifact: BaseArtifact) -> ArtifactRef:
        def _write() -> ArtifactRef:
            data = artifact.model_dump(mode="json")
            data["content_hash"] = None
            payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            sha = hashlib.sha256(payload.encode()).hexdigest()
            data["content_hash"] = sha
            payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            path = self._artifact_path(artifact)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8")
            logger.debug("artifact written: %s → %s", artifact.artifact_id, path)
            return ArtifactRef(
                artifact_id=artifact.artifact_id,
                artifact_type=artifact.artifact_type,
                uri=str(path),
                sha256=sha,
                created_at=artifact.created_at,
                summary=artifact.summary[:500] if artifact.summary else "",
            )
        return await asyncio.to_thread(_write)

    async def get(self, ref_or_id: "ArtifactRef | str") -> Dict[str, Any]:
        def _read() -> Dict[str, Any]:
            expected_hash = None
            if isinstance(ref_or_id, ArtifactRef):
                path = Path(ref_or_id.uri)
                expected_hash = ref_or_id.sha256 or None
            else:
                path = self._find_by_id(ref_or_id)
                if path is None:
                    raise FileNotFoundError(f"artifact not found: {ref_or_id}")
            data = json.loads(path.read_text(encoding="utf-8"))
            stored_hash = data.get("content_hash")
            hash_payload = {**data, "content_hash": None}
            actual_hash = hashlib.sha256(
                json.dumps(hash_payload, ensure_ascii=False, separators=(",", ":")).encode()
            ).hexdigest()
            if stored_hash and stored_hash != actual_hash:
                raise ValueError(f"artifact hash mismatch for {data.get('artifact_id')}")
            if expected_hash and expected_hash != actual_hash:
                raise ValueError(f"artifact ref hash mismatch for {data.get('artifact_id')}")
            return data
        return await asyncio.to_thread(_read)

    async def exists(self, artifact_id: str) -> bool:
        return await asyncio.to_thread(lambda: self._find_by_id(artifact_id) is not None)

    async def list_by_memory(
        self,
        memory_id: str,
        artifact_type: Optional[ArtifactType] = None,
    ) -> List[ArtifactRef]:
        # TODO: requires an artifact index for efficient lookup; stub returns []
        return []

    async def verify(self, ref: ArtifactRef) -> ArtifactIntegrityResult:
        def _verify() -> ArtifactIntegrityResult:
            path = self._find_by_id(ref.artifact_id)
            if path is None:
                return ArtifactIntegrityResult(artifact_id=ref.artifact_id, ok=False)
            data = json.loads(path.read_text(encoding="utf-8"))
            stored_hash = data.get("content_hash")
            hash_payload = {**data, "content_hash": None}
            actual_hash = hashlib.sha256(
                json.dumps(hash_payload, ensure_ascii=False, separators=(",", ":")).encode()
            ).hexdigest()
            ok = stored_hash == actual_hash
            return ArtifactIntegrityResult(
                artifact_id=ref.artifact_id,
                ok=ok,
                stored_hash=stored_hash,
                actual_hash=actual_hash,
            )
        return await asyncio.to_thread(_verify)


__all__ = ["FilesystemArtifactStorageAdapter"]
