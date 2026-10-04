"""resolver 的 L2 冷读端口：Patchouli backing 读取契约（A2 §2.1 / §8 D-2）。

resolver 只依赖本协议，不导入 System 路由常量或 Patchouli 实现；基于
``GlobalSystemBus`` 的实现位于能力层（``workspace.capability.backing``），
由组合根注入。冷读取经公共 backing 路由（宪章 §4.4 过线契约），库侧资源
归属与 policy 校验独立成立，构成纵深防御。

``scope`` 是能力层完成操作授权后由 guard 组装的可信坐标；resolver 与
backing 位于授权点以下，不接收访问 context。
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from hivememory.core.models import IdentityScope, MemoryAtom, ResolvedAgentProfile
from hivememory.core.protocol.models import RetrievalRequest


class CanonicalReadBackend(Protocol):
    """Patchouli canonical/Profile 读取的 backing 端口。

    不可达或路由缺失时抛出 ``ResourceUnavailableError``；存储错误按原结构化
    错误传播；未知或不可见资源按 A1 防泄露规则返回 ``None`` / 不出现在列表中。
    """

    async def read(
        self,
        memory_id: UUID,
        *,
        scope: IdentityScope,
    ) -> MemoryAtom | None:
        """Actor-visible 的 UUID 点读。"""
        ...

    async def retrieve_by_aliases(
        self,
        aliases: list[str],
        *,
        scope: IdentityScope,
    ) -> list[MemoryAtom]:
        """按 alias 批量读取实际可读的完整原子。"""
        ...

    async def retrieve(
        self,
        request: RetrievalRequest,
    ) -> list[MemoryAtom]:
        """语义检索，按领域排序返回完整原子列表。"""
        ...

    async def get_agent_profile(
        self,
        agent_alias: str | None,
        *,
        scope: IdentityScope,
    ) -> ResolvedAgentProfile:
        """Profile 定义解析：AgentProfile + 源原子 policy 依据与关联。"""
        ...


__all__ = ["CanonicalReadBackend"]
