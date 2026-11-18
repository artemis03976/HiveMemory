"""Profile 读取器：经 Patchouli backing 解析 Profile 与列出全部 Profile。

Profile 是 AgentProfileStore 中的文件，Patchouli 每次都直接读文件，文件的改动在下一次
读取时生效；这里因此不缓存，也不订阅变更。Profile 是全局的，不属于任何 Workspace，也
没有可见性：谁能在哪个 Workspace 执行或被调用，由访问登记的准入与 CALL 授权决定，授权
在调用本读取器之前完成。解析的规则（内置 Profile、缺失与不合格、已停用）归 Patchouli，
失败按原错误传播。

读取 Profile 不等于获得其描述的权限；选择 Profile、绑定模型/工具与应用执行限制归
Actor runtime。读取视图关闭后，与记忆读取一样显式失败。
"""

from __future__ import annotations

from hivememory.core.models import AgentProfile
from hivememory.workspace.resolution.backing import CanonicalReadBackend
from hivememory.workspace.resolution.guard import ColdReadGuard


class ProfileResolver:
    """读取全局的 Profile，不缓存。"""

    def __init__(self, *, guard: ColdReadGuard, backing: CanonicalReadBackend) -> None:
        self._guard = guard
        self._backing = backing

    async def get(self, agent_alias: str | None) -> AgentProfile:
        """解析可以执行的 Profile；缺失、文件不合格或已停用均显式失败，不降级为内置 Profile。"""
        self._guard.ensure_open()
        alias = agent_alias.strip() if agent_alias else ""
        return await self._backing.get_agent_profile(alias or None)

    async def list(self) -> list[AgentProfile]:
        """列出全部合格的 Profile，含停用的；不合格的文件不在其中。"""
        self._guard.ensure_open()
        return [stored.profile for stored in await self._backing.list_agent_profiles()]


__all__ = ["ProfileResolver"]
