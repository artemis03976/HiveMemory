"""Filesystem Artifact 存储适配器。

Artifact 的物理路径只是持久化细节，真正的访问入口是持久化索引中的
``(owner_user_id, workspace_id, artifact_id)`` 复合键。调用方提供的 URI
只作为返回信息保存，永远不能改变寻址或授权结果。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from hivememory.core.models import (
    WorkspaceAccessContext,
    WorkspaceArtifactKey,
    WorkspaceIdentity,
    require_workspace_access_context,
)
from hivememory.core.models.artifact import ArtifactRef, ArtifactType, BaseArtifact
from hivememory.patchouli.memory_library.models import (
    ArtifactIntegrityResult,
    StorageHealthComponent,
)
from hivememory.patchouli.memory_library.ports import ArtifactStoragePort

logger = logging.getLogger(__name__)

_INDEX_NAME = ".artifact_index.json"
_INDEX_VERSION = 1


class FilesystemArtifactStorageAdapter(ArtifactStoragePort):
    """按 Workspace 分区的文件系统 Artifact 仓库。

    新文件布局为 ``{root}/{owner_digest}/{workspace_digest}/{type}/date``。
    索引会在首次读取 miss 时执行一次受控 legacy 扫描，扫描结果持久化后不再
    对每次读取执行全局 ``rglob``。
    """

    def __init__(
        self,
        root_dir: str = ".hivememory/artifacts",
        *,
        max_inline_summary_chars: int = 500,
    ) -> None:
        self._root = Path(root_dir)
        self._max_inline_summary_chars = max(0, max_inline_summary_chars)
        self._lock = threading.RLock()
        self._index: dict[str, list[str]] | None = None
        self._legacy_scan_complete = False

    @property
    def _index_path(self) -> Path:
        return self._root / _INDEX_NAME

    @staticmethod
    def _digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _index_key(workspace: WorkspaceIdentity, artifact_id: str) -> str:
        key = WorkspaceArtifactKey(
            workspace_identity=workspace,
            artifact_id=artifact_id,
        )
        return json.dumps(
            [
                key.workspace_identity.owner_user_id,
                key.workspace_identity.workspace_id,
                key.artifact_id,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _decode_index_key(value: str) -> tuple[str, str, str]:
        decoded = json.loads(value)
        if not isinstance(decoded, list) or len(decoded) != 3:
            raise ValueError("Artifact 索引键格式无效")
        owner_user_id, workspace_id, artifact_id = decoded
        if not all(isinstance(item, str) and item for item in decoded):
            raise ValueError("Artifact 索引键字段不能为空")
        return owner_user_id, workspace_id, artifact_id

    def _artifact_path(self, artifact: BaseArtifact) -> Path:
        workspace = artifact.workspace_identity
        ts = artifact.created_at
        return (
            self._root
            / self._digest(workspace.owner_user_id)
            / self._digest(workspace.workspace_id)
            / artifact.artifact_type.value
            / str(ts.year)
            / f"{ts.month:02d}"
            / f"{ts.day:02d}"
            / f"{self._digest(artifact.artifact_id)}.json"
        )

    def _resolve_relative_path(self, relative_path: str) -> Path:
        root = self._root.resolve()
        candidate = (self._root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("Artifact 索引路径越过存储根目录") from exc
        return candidate

    @staticmethod
    def _relative_path(root: Path, path: Path) -> str:
        return path.resolve().relative_to(root.resolve()).as_posix()

    def _load_index_locked(self) -> None:
        if self._index is not None:
            return
        if not self._index_path.exists():
            self._index = {}
            self._legacy_scan_complete = False
            return

        raw = json.loads(self._index_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != _INDEX_VERSION:
            raise ValueError("Artifact 索引版本不受支持")
        entries = raw.get("entries")
        if not isinstance(entries, dict):
            raise ValueError("Artifact 索引 entries 格式无效")
        normalized: dict[str, list[str]] = {}
        for key, paths in entries.items():
            self._decode_index_key(key)
            if isinstance(paths, str):
                paths = [paths]
            if not isinstance(paths, list) or not all(
                isinstance(path, str) and path for path in paths
            ):
                raise ValueError("Artifact 索引路径格式无效")
            normalized[key] = list(dict.fromkeys(paths))
        self._index = normalized
        self._legacy_scan_complete = bool(raw.get("legacy_scan_complete", False))

    def _save_index_locked(self) -> None:
        assert self._index is not None
        self._root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _INDEX_VERSION,
            "legacy_scan_complete": self._legacy_scan_complete,
            "entries": self._index,
        }
        temp_path = self._index_path.with_name(f".{_INDEX_NAME}.{uuid4().hex}.tmp")
        try:
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            temp_path.replace(self._index_path)
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    def _canonical_hash(data: dict[str, Any]) -> str:
        hash_payload = {**data, "content_hash": None}
        return hashlib.sha256(
            json.dumps(
                hash_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _workspace_from_data(data: dict[str, Any]) -> WorkspaceIdentity:
        """把新格式、扁平格式和 legacy owner 字段归一为唯一 Workspace。"""
        nested_present = "workspace_identity" in data
        nested = data.get("workspace_identity")
        flat_keys = {"workspace_key", "workspace_id"}
        has_flat_workspace = bool(flat_keys & data.keys())
        owner = data.get("owner_user_id")

        if nested_present:
            if not isinstance(nested, dict):
                raise ValueError("Artifact workspace_identity 格式无效")
            try:
                identity = WorkspaceIdentity.model_validate(nested)
            except Exception as exc:
                raise ValueError("Artifact workspace_identity 不完整") from exc
            if has_flat_workspace and not (
                data.get("workspace_key") == identity.workspace_key
                and data.get("workspace_id") == identity.workspace_id
            ):
                raise ValueError("Artifact Workspace 投影不一致")
            if owner is not None and owner != identity.owner_user_id:
                raise ValueError("Artifact owner_user_id 与 Workspace 投影不一致")
            return identity

        if has_flat_workspace:
            if not isinstance(owner, str) or not owner:
                raise ValueError("Artifact Workspace 投影不完整")
            if not all(
                isinstance(data.get(key), str) and data.get(key)
                for key in flat_keys
            ):
                raise ValueError("Artifact Workspace 投影不完整")
            return WorkspaceIdentity(
                owner_user_id=owner,
                workspace_key=data["workspace_key"],
                workspace_id=data["workspace_id"],
            )

        if isinstance(owner, str) and owner.strip():
            # v0.5 文件没有 Workspace 坐标，只能兼容到 owner 的 main_workspace。
            return WorkspaceIdentity(
                owner_user_id=owner,
                workspace_key="main_workspace",
                workspace_id="main_workspace",
            )
        raise ValueError("Artifact 缺少完整 Workspace 归属")

    def _scan_legacy_once_locked(self) -> None:
        self._load_index_locked()
        if self._legacy_scan_complete:
            return
        assert self._index is not None
        if self._root.exists():
            for path in sorted(self._root.rglob("*.json")):
                if path.resolve() == self._index_path.resolve():
                    continue
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    artifact_id = data.get("artifact_id")
                    artifact_type = data.get("artifact_type")
                    if not isinstance(artifact_id, str) or not artifact_id:
                        continue
                    if not isinstance(artifact_type, str) or not artifact_type:
                        continue
                    workspace = self._workspace_from_data(data)
                    key = self._index_key(workspace, artifact_id)
                    relative = self._relative_path(self._root, path)
                    paths = self._index.setdefault(key, [])
                    if relative not in paths:
                        paths.append(relative)
                except Exception as exc:
                    logger.warning("忽略无法归一化的 legacy Artifact: %s (%s)", path, exc)
        self._legacy_scan_complete = True
        self._save_index_locked()

    def _paths_for_locked(
        self,
        workspace: WorkspaceIdentity,
        artifact_id: str,
    ) -> list[Path]:
        self._load_index_locked()
        self._scan_legacy_once_locked()
        assert self._index is not None
        key = self._index_key(workspace, artifact_id)
        paths = self._index.get(key, [])
        if len(paths) != 1:
            return []
        try:
            return [self._resolve_relative_path(paths[0])]
        except ValueError:
            return []

    def _read_path_locked(
        self,
        access_context: WorkspaceAccessContext,
        artifact_id: str,
    ) -> tuple[Path, dict[str, Any]]:
        paths = self._paths_for_locked(access_context.workspace_identity, artifact_id)
        if not paths or not paths[0].is_file():
            raise FileNotFoundError(f"artifact not found: {artifact_id}")
        path = paths[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("artifact_id") != artifact_id:
            raise FileNotFoundError(f"artifact not found: {artifact_id}")
        try:
            workspace = self._workspace_from_data(data)
        except ValueError as exc:
            raise FileNotFoundError(f"artifact not found: {artifact_id}") from exc
        if workspace != access_context.workspace_identity:
            # 控制面不泄漏其他 Workspace 是否存在同 ID 资源。
            raise FileNotFoundError(f"artifact not found: {artifact_id}")
        return path, data

    def _ref_from_data(
        self,
        path: Path,
        data: dict[str, Any],
    ) -> ArtifactRef:
        artifact_type = ArtifactType(data["artifact_type"])
        actual_hash = self._canonical_hash(data)
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if not isinstance(created_at, datetime):
            created_at = datetime.now()
        return ArtifactRef(
            artifact_id=data["artifact_id"],
            artifact_type=artifact_type,
            workspace_identity=self._workspace_from_data(data),
            uri=str(path),
            sha256=actual_hash,
            created_at=created_at,
            summary=(data.get("summary") or "")[: self._max_inline_summary_chars],
        )

    async def put(self, artifact: BaseArtifact) -> ArtifactRef:
        def _write() -> ArtifactRef:
            with self._lock:
                self._load_index_locked()
                self._scan_legacy_once_locked()
                assert self._index is not None
                key = self._index_key(artifact.workspace_identity, artifact.artifact_id)
                existing = self._index.get(key, [])

                data = artifact.model_dump(mode="json")
                data["content_hash"] = None
                content_hash = self._canonical_hash(data)
                data["content_hash"] = content_hash

                if len(existing) > 1:
                    raise ValueError("Artifact 复合键存在冲突记录")
                if existing:
                    path = self._resolve_relative_path(existing[0])
                    if not path.is_file():
                        raise ValueError("Artifact 索引指向不存在的记录")
                    current = json.loads(path.read_text(encoding="utf-8"))
                    if current.get("artifact_id") != artifact.artifact_id:
                        raise ValueError("Artifact 索引与记录 ID 不一致")
                    if self._workspace_from_data(current) != artifact.workspace_identity:
                        raise ValueError("Artifact 索引与记录 Workspace 不一致")
                    current_hash = self._canonical_hash(current)
                    stored_hash = current.get("content_hash")
                    if stored_hash and stored_hash != current_hash:
                        raise ValueError("Artifact 已存记录 hash 校验失败")
                    if current_hash != content_hash:
                        raise ValueError("Artifact 为 append-only，禁止覆盖同作用域不同内容")
                    return self._ref_from_data(path, current)

                path = self._artifact_path(artifact)
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
                temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
                try:
                    temp_path.write_text(payload, encoding="utf-8")
                    temp_path.replace(path)
                    relative = self._relative_path(self._root, path)
                    self._index[key] = [relative]
                    try:
                        self._save_index_locked()
                    except Exception:
                        self._index.pop(key, None)
                        path.unlink(missing_ok=True)
                        raise
                finally:
                    temp_path.unlink(missing_ok=True)
                logger.debug("artifact written: %s → %s", artifact.artifact_id, path)
                return self._ref_from_data(path, data)

        return await asyncio.to_thread(_write)

    async def get(
        self,
        access_context: WorkspaceAccessContext,
        ref_or_id: ArtifactRef | str,
    ) -> dict[str, Any]:
        access_context = require_workspace_access_context(access_context)

        def _read() -> dict[str, Any]:
            with self._lock:
                expected_hash = None
                artifact_id: str
                if isinstance(ref_or_id, ArtifactRef):
                    if ref_or_id.workspace_identity != access_context.workspace_identity:
                        raise FileNotFoundError(
                            f"artifact not found: {ref_or_id.artifact_id}"
                        )
                    artifact_id = ref_or_id.artifact_id
                    expected_hash = ref_or_id.sha256 or None
                else:
                    artifact_id = ref_or_id
                _, data = self._read_path_locked(access_context, artifact_id)
                stored_hash = data.get("content_hash")
                actual_hash = self._canonical_hash(data)
                if stored_hash and stored_hash != actual_hash:
                    raise ValueError(f"artifact hash mismatch for {artifact_id}")
                if expected_hash and expected_hash != actual_hash:
                    raise ValueError(f"artifact ref hash mismatch for {artifact_id}")
                return data

        return await asyncio.to_thread(_read)

    async def exists(
        self,
        access_context: WorkspaceAccessContext,
        artifact_id: str,
    ) -> bool:
        access_context = require_workspace_access_context(access_context)

        def _exists() -> bool:
            with self._lock:
                try:
                    _, data = self._read_path_locked(access_context, artifact_id)
                    stored_hash = data.get("content_hash")
                    return not stored_hash or stored_hash == self._canonical_hash(data)
                except (FileNotFoundError, ValueError, json.JSONDecodeError):
                    return False

        return await asyncio.to_thread(_exists)

    async def list_by_memory(
        self,
        access_context: WorkspaceAccessContext,
        memory_id: str,
        artifact_type: ArtifactType | None = None,
    ) -> list[ArtifactRef]:
        access_context = require_workspace_access_context(access_context)

        def _list() -> list[ArtifactRef]:
            with self._lock:
                self._load_index_locked()
                self._scan_legacy_once_locked()
                assert self._index is not None
                result: list[ArtifactRef] = []
                for key, paths in self._index.items():
                    owner, workspace_id, artifact_id = self._decode_index_key(key)
                    if (
                        owner != access_context.workspace_identity.owner_user_id
                        or workspace_id != access_context.workspace_identity.workspace_id
                        or len(paths) != 1
                    ):
                        continue
                    try:
                        path = self._resolve_relative_path(paths[0])
                        data = json.loads(path.read_text(encoding="utf-8"))
                        if data.get("artifact_id") != artifact_id:
                            continue
                        if self._workspace_from_data(data) != access_context.workspace_identity:
                            continue
                        if data.get("memory_id") != memory_id:
                            continue
                        actual_hash = self._canonical_hash(data)
                        stored_hash = data.get("content_hash")
                        if stored_hash and stored_hash != actual_hash:
                            logger.warning("忽略校验失败的 Artifact: %s", artifact_id)
                            continue
                        actual_type = ArtifactType(data["artifact_type"])
                        if artifact_type is not None and actual_type != artifact_type:
                            continue
                        result.append(self._ref_from_data(path, data))
                    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                        logger.warning("忽略无法读取的 Artifact 索引项 %s: %s", artifact_id, exc)
                return result

        return await asyncio.to_thread(_list)

    async def verify(
        self,
        access_context: WorkspaceAccessContext,
        ref: ArtifactRef,
    ) -> ArtifactIntegrityResult:
        access_context = require_workspace_access_context(access_context)
        if ref.workspace_identity != access_context.workspace_identity:
            return ArtifactIntegrityResult(artifact_id=ref.artifact_id, ok=False)

        def _verify() -> ArtifactIntegrityResult:
            with self._lock:
                try:
                    _, data = self._read_path_locked(access_context, ref.artifact_id)
                    stored_hash = data.get("content_hash")
                    actual_hash = self._canonical_hash(data)
                    ok = bool(stored_hash) and stored_hash == actual_hash
                    if ref.sha256:
                        ok = ok and ref.sha256 == actual_hash
                    return ArtifactIntegrityResult(
                        artifact_id=ref.artifact_id,
                        ok=ok,
                        stored_hash=stored_hash,
                        actual_hash=actual_hash,
                    )
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    return ArtifactIntegrityResult(artifact_id=ref.artifact_id, ok=False)

        return await asyncio.to_thread(_verify)

    async def check_health(self) -> StorageHealthComponent:
        def _check() -> StorageHealthComponent:
            try:
                self._root.mkdir(parents=True, exist_ok=True)
                probe = self._root / ".healthcheck"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink(missing_ok=True)
                return StorageHealthComponent(name="artifact", healthy=True, required=False)
            except Exception as exc:
                return StorageHealthComponent(
                    name="artifact",
                    healthy=False,
                    required=False,
                    detail=str(exc),
                )

        return await asyncio.to_thread(_check)


__all__ = ["FilesystemArtifactStorageAdapter"]
