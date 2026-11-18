from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hivememory.core.errors import AgentProfileExistsError, ResourceNotFoundError
from hivememory.core.models import AgentProfile, StoredAgentProfile
from hivememory.memory_library.agent_profiles import (
    FRONTMATTER_FIELDS,
    AgentProfileStore,
    profile_from_fields,
)
from hivememory.patchouli.application.preset_agent_profiles import PRESET_AGENT_PROFILES
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.runtime.bus import PatchouliBus


class AgentProfileManagementService:
    """Patchouli 面向 Agent Profile 管理与解析的应用服务。

    新建、编辑、删除与列出直接读写 MemoryLibrary 的 AgentProfileStore，不经
    ``MemoryGenerationFamiliar``：Profile 不是记忆。解析经本地路由交给检索使魔，
    内置 Profile 与失败原因的规则只维护在那里。Profile 是全局的，不属于任何
    Workspace，因此这里不接收 ``IdentityScope``：管理用例由 workspace root API、
    CALL 的解析由 actor API 在调用之前授权，agent 能在哪个 Workspace 工作由访问登记决定。
    """

    def __init__(self, *, store: AgentProfileStore, bus: PatchouliBus) -> None:
        self._store = store
        self._bus = bus

    async def create_agent_profile(
        self, agent_id: str, fields: Mapping[str, Any]
    ) -> StoredAgentProfile:
        """新建 Profile；同名文件已存在时以 ``AgentProfileExistsError`` 拒绝，不覆盖。"""
        profile = _profile_from_submission(agent_id, fields)
        # 查重与写入之间没有 await，同一进程中的两次新建不会交错。
        if self._store.exists(agent_id):
            raise AgentProfileExistsError(details={"agent_id": agent_id})
        return self._store.save(profile)

    async def update_agent_profile(
        self, agent_id: str, changes: Mapping[str, Any]
    ) -> StoredAgentProfile:
        """以给出的字段覆盖已有 Profile 的对应字段，其余不变；agent_id 不可改。"""
        current = self._store.get(agent_id)
        if current is None:
            raise ResourceNotFoundError(details={"resource": "agent_profile", "agent_id": agent_id})
        base = current.profile.model_dump(include={*FRONTMATTER_FIELDS, "persona"})
        return self._store.save(_profile_from_submission(agent_id, {**base, **changes}))

    async def delete_agent_profile(self, agent_id: str) -> bool:
        return self._store.delete(agent_id)

    async def list_agent_profiles(self) -> list[StoredAgentProfile]:
        return self._store.list()

    async def seed_preset_agent_profiles(self) -> list[str]:
        """写入尚不存在的预设 Profile，返回写入的 agent_id；已有同名文件的跳过。

        只用于初次部署，不经过授权：这时还没有任何调用方，写入的也只是随程序发布的
        预设内容。
        """
        created: list[str] = []
        for preset in PRESET_AGENT_PROFILES:
            agent_id = preset.agent_id or ""
            if self._store.exists(agent_id):
                continue
            self._store.save(preset)
            created.append(agent_id)
        return created

    async def get_agent_profile(self, agent_alias: str | None) -> AgentProfile:
        """解析可以执行的 Profile：内置 omni_doll，或已启用的 Profile 文件。"""
        return await self._bus.request(PatchouliLocalRoutes.GET_AGENT_PROFILE, agent_alias)


def _profile_from_submission(agent_id: str, fields: Mapping[str, Any]) -> AgentProfile:
    """管理接口提交的字段中人设是正文，其余是 frontmatter，与读取文件走同一套校验。"""
    frontmatter = dict(fields)
    persona = frontmatter.pop("persona", "")
    return profile_from_fields(agent_id, frontmatter, persona=persona)
