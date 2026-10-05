"""为记忆物化桥接尚未拆分身份的 WorkspaceAsset 读取端口。"""

from hivememory.core.models import IdentityScope, WorkspaceIdentity, system_actor_for_workspace
from hivememory.core.models.workspace_asset import (
    RepresentationLease,
    RepresentationPreference,
    WorkspaceAssetRef,
)
from hivememory.core.ports.workspace_assets import WorkspaceAssetReaderPort


class AssetMaterializationReader:
    """组合根的兼容适配器；资产 owner 的归属检查仍由原读取端口执行。

    WorkspaceAssetStore 的身份拆分另行处理。本适配器只在一次租借调用内
    组装旧端口所需的 scope，不把它交回 Patchouli 或保存到后台任务。
    """

    def __init__(self, reader: WorkspaceAssetReaderPort) -> None:
        self._reader = reader

    def acquire_ready_representation(
        self,
        belong_to: WorkspaceIdentity,
        asset_ref: WorkspaceAssetRef,
        preference: RepresentationPreference | None = None,
    ) -> RepresentationLease:
        scope = IdentityScope(
            workspace_identity=belong_to,
            actor_identity=system_actor_for_workspace(belong_to),
        )
        return self._reader.acquire_ready_representation(scope, asset_ref, preference)

    def release_representation_lease(self, lease_id: str) -> bool:
        return self._reader.release_representation_lease(lease_id)
