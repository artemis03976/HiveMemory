"""Profile 读取 resolver：解析结果缓存与交付授权。

这里只负责缓存与交付授权；Profile 解析能力（builtin 规则、``from_atom``、
类型检查）归 Patchouli。流程：

- L1 命中 ``(Workspace, agent_alias)``：按条目随存的源原子 policy 对当前
  Actor 授权，通过则交付 AgentProfile 独立副本，不回源；
- L1 未命中：受代次守护调用一次 ``GET_AGENT_PROFILE`` backing，得到
  AgentProfile + 源原子 policy 依据与关联；builtin 结果不进缓存；
- 全程不触 AtomCache，避免两套缓存共同持有 profile 信息、各自过期。

读取 Profile 不等于获得其描述的权限；选择 Profile、绑定模型/工具与应用
执行限制归 Actor runtime。

管理读取（``enforce_actor_visibility=False``）供任务进程以 root scope 加载
执行者自己的图纸：缓存命中与冷读都只检查 Workspace 归属，不按 scope 的
actor 检查可见性，与 root API 读取记忆的管理语义相同。CALL 解析被调用方的
图纸仍按调用方的可见性交付。
"""

from __future__ import annotations

from hivememory.core.errors import ResourceNotFoundError
from hivememory.core.memory_access import access_policy_permits
from hivememory.core.models import AgentProfile, IdentityScope
from hivememory.workspace.cache.profile import ProfileCache, ProfileCacheEntry
from hivememory.workspace.resolution.backing import CanonicalReadBackend
from hivememory.workspace.resolution.guard import ColdReadGuard


class ProfileResolver:
    """按 ``(Workspace, agent_alias)`` 缓存 Profile 解析结果并逐次授权交付。"""

    def __init__(
        self,
        *,
        cache: ProfileCache,
        guard: ColdReadGuard,
        backing: CanonicalReadBackend,
    ) -> None:
        self._cache = cache
        self._guard = guard
        self._backing = backing

    async def get(
        self,
        agent_alias: str | None,
        *,
        scope: IdentityScope,
        enforce_actor_visibility: bool = True,
    ) -> AgentProfile:
        """解析并交付 AgentProfile 独立副本。

        缺失、不可见、类型不符或配置损坏均按 backing 的既有错误失败，不降级
        为默认配置；命中后对当前 Actor 不可见时与"不存在"同样表达，不泄露资源是否存在。
        ``enforce_actor_visibility=False`` 是管理读取，命中与冷读都不检查可见性。
        """
        self._guard.ensure_open()
        workspace = scope.workspace_identity
        alias = agent_alias.strip() if agent_alias else ""
        if alias:
            cached = self._cache.get(workspace, alias)
            if cached is not None:
                return self._deliver(cached, alias, scope, enforce_actor_visibility)

        resolved, fill = await self._guard.load(
            workspace,
            lambda: self._backing.get_agent_profile(
                alias or None, scope=scope, enforce_actor_visibility=enforce_actor_visibility
            ),
        )
        policy = resolved.access_policy
        source_memory_id = resolved.source_memory_id
        source_version = resolved.source_version
        if policy is None or source_memory_id is None or source_version is None:
            # builtin 无源原子（ResolvedAgentProfile 保证关联字段全有或全无）：
            # 由 Patchouli 规则解析，每次返回独立副本，不进缓存。
            return resolved.profile.model_copy(deep=True)

        entry = ProfileCacheEntry(
            profile=resolved.profile,
            access_policy=policy,
            source_memory_id=source_memory_id,
            source_version=source_version,
        )
        if fill:
            self._cache.put(workspace, alias, entry)
        return self._deliver(entry, alias, scope, enforce_actor_visibility)

    @staticmethod
    def _deliver(
        entry: ProfileCacheEntry,
        alias: str,
        scope: IdentityScope,
        enforce_actor_visibility: bool,
    ) -> AgentProfile:
        """交付边界逐次授权：按条目随存的源原子 policy 判定当前 Actor；管理读取跳过。

        缓存按 ``(Workspace, alias)`` 分区，归属已由键保证，管理读取不需要另查。
        """
        if enforce_actor_visibility and not access_policy_permits(
            entry.access_policy, scope.actor_identity
        ):
            raise ResourceNotFoundError(details={"resource": "agent_profile", "agent_alias": alias})
        return entry.profile.model_copy(deep=True)


__all__ = ["ProfileResolver"]
