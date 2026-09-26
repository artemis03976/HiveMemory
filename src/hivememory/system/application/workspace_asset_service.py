"""迁移期 re-export shim：实现已迁至 ``hivememory.workspace.capability.assets``。

A2 §1.2：``system/application`` 的资源能力部分改造为 workspace 能力层；本模块
只为既有导入路径保留转发，不维护第二份实现，A6 完成消费者切换后删除。
"""

from hivememory.workspace.capability.assets import (
    WorkspaceAssetApplicationService,
)

__all__ = [
    "WorkspaceAssetApplicationService",
]
