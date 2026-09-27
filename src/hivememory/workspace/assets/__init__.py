"""Workspace 资产：进程内 WorkspaceAsset working set、解析交接与上传接收。

AssetStore 是网络共享设施，按 ``core.ports.workspace_assets`` 的读取/命令
端口向其他子系统提供窄化能力；文件格式的确定性解析器位于
``infrastructure.attachments``。
"""

from hivememory.workspace.assets.parse_service import AttachmentParseService
from hivememory.workspace.assets.store import InMemoryWorkspaceAssetStore

__all__ = [
    "AttachmentParseService",
    "InMemoryWorkspaceAssetStore",
]
