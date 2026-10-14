"""测试用操作请求装配：真实 workspace 授权、读取视图、登记与入口。"""

from __future__ import annotations

from dataclasses import replace

from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.access import RunBinding
from hivememory.core.errors import ResourceUnavailableError
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    IdentityScope,
    MemoryAtom,
    ResolvedAgentProfile,
)
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.workspace.capability.actor.agent_profiles import ActorAgentService
from hivememory.workspace.capability.actor.memory import ActorMemoryService
from hivememory.workspace.capability.operations import WorkspaceOperationEntry
from hivememory.workspace.contracts import (
    OperationSubmitter,
    ThreadOperationSubmitter,
)
from hivememory.workspace.runtime import WorkspaceRuntime
from tests.helpers.intent_dispatch import install_intent_admission
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_identity_scope,
)


class MemoryBackend:
    """仅替代持久化边界，读取视图与授权仍使用生产实现。"""

    def __init__(self, bus=None) -> None:
        self.memories: dict[str, MemoryAtom] = {}
        self.bus = bus

    async def read(self, memory_id, *, scope):
        return next((atom for atom in self.memories.values() if atom.id == memory_id), None)

    async def retrieve_by_aliases(self, aliases, *, scope):
        from hivememory.patchouli.contracts import PatchouliRoutes

        found = [self.memories[alias] for alias in aliases if alias in self.memories]
        missing = [alias for alias in aliases if alias not in self.memories]
        if self.bus is not None and missing:
            try:
                found.extend(
                    await self.bus.request(
                        PatchouliRoutes.MEMORY_RETRIEVE_BY_ALIASES,
                        aliases=missing,
                        identity_scope=scope,
                    )
                )
            except KeyError as error:
                raise ResourceUnavailableError("测试存储读取路由不可达") from error
        return found

    async def retrieve(self, request):
        if self.bus is not None:
            return await self.bus.request(PatchouliRoutes.MEMORY_RETRIEVE, request=request)
        return list(self.memories.values())

    async def get_agent_profile(self, alias, *, scope):
        # 内存 backing 沿用 Patchouli 的内置图纸规则；入口仍必须先执行
        # agent.call 授权，测试不在 Alice 内另设默认配置回退。
        if alias is None or alias.strip() in ("", "default", "omni_doll"):
            return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE.model_copy(deep=True))
        if self.bus is None:
            raise NotImplementedError("本测试未装配 Profile backing")
        return await self.bus.request(
            PatchouliRoutes.GET_AGENT_PROFILE, alias, identity_scope=scope
        )


class OperationsHarness:
    """为测试身份装配真实入口，跨调用保留同一份进程级登记。"""

    def __init__(self, bus=None, *, operation_authorizer=None) -> None:
        # 缺省认证声明独立于 CPU 观测标签；自定义身份必须显式签发提交函数。
        self.default_scope = make_identity_scope()
        self.bus = bus or GlobalSystemBus()
        self.admission = install_intent_admission(self.bus)
        self.citations: list[dict] = []
        if bus is None:
            self.bus.register(PatchouliRoutes.RECORD_MEMORY_CITATION, self._record_citation)
        self.backing = MemoryBackend(bus)
        self.runtime = WorkspaceRuntime(backing=self.backing, atom_capacity=32, profile_capacity=16)
        self.registry = self.runtime.intents
        self.memories = self.backing.memories
        self.compositions = {}
        empty_access = make_access_composition([])
        self.entry = self.make_entry(
            operation_authorizer or empty_access.authorizer,
            empty_access.thread_spawner,
        )

    async def _record_citation(self, *, memory_id, source, identity_scope):
        """持久化替身记录公共引用路由的副作用，读取与授权边界保持真实。"""
        self.citations.append(
            {"memory_id": memory_id, "source": source, "identity_scope": identity_scope}
        )

    def make_entry(self, operation_authorizer, thread_spawner) -> WorkspaceOperationEntry:
        """入口与所有测试线程 context 共享同一份登记，替身仅位于持久化边界。"""
        return WorkspaceOperationEntry(
            ActorMemoryService(
                self.bus,
                operation_authorizer=operation_authorizer,
                memory_reader=self.runtime.aliases,
                dispatch_topics=thread_spawner,
            ),
            agent=ActorAgentService(
                operation_authorizer=operation_authorizer,
                profile_reader=self.runtime.profiles,
                thread_spawner=thread_spawner,
            ),
            operation_authorizer=operation_authorizer,
        )

    async def submitter(
        self,
        scope: IdentityScope,
        process_id: str = "test_run",
        *,
        allowed_operations=None,
        callee_operations=None,
    ) -> OperationSubmitter:
        """完成真实认证并构造绑定线程 context 的泛型提交函数。"""
        workspace = scope.workspace_identity
        actor = scope.actor_identity
        composition = make_access_composition(
            [
                make_actor_access_record(
                    owner_user_id=workspace.owner_user_id,
                    workspace_id=workspace.workspace_id,
                    user_id=actor.user_id,
                    agent_id=None,
                    allowed_operations=callee_operations,
                ),
                make_actor_access_record(
                    owner_user_id=workspace.owner_user_id,
                    workspace_id=workspace.workspace_id,
                    user_id=actor.user_id,
                    agent_id=actor.agent_id,
                    allowed_operations=allowed_operations,
                ),
            ],
            default_workspace=workspace,
        )
        access = await composition.gateway.authenticate(
            adapter="local",
            principal=composition.principal,
            actor=actor,
            workspace=workspace,
            binding=RunBinding.for_task_process(process_id),
        )
        self.compositions.setdefault(process_id, []).append(composition)
        entry = self.make_entry(composition.authorizer, composition.thread_spawner)
        thread_access = composition.register_main_thread(access)
        return ThreadOperationSubmitter(entry, thread_access)

    def revoke(self, process_id: str = "test_run") -> None:
        """同步撤销测试进程的全部线程 context，不取消调用方任务。"""
        for composition in self.compositions.pop(process_id, []):
            process = composition.process_table.get(process_id)
            composition.thread_spawner.revoke_threads(process.working_set)
            composition.gateway.invalidate_context(process.record.access)
            composition.process_table.close(process)


class HarnessKoakumaRuntime(KoakumaRuntime):
    """测试显式装配提交函数，复用生产 MTP handler，不模拟解析或登记。"""

    def __init__(self, *, bus=None, config=None):
        super().__init__(config=config)
        self.harness = OperationsHarness(bus)
        self.memories = self.harness.memories
        self.registry = self.harness.registry

    async def execute_mtp(self, text, context=None):
        if context is not None and context.submit_operation is None:
            submit_operation = await self.harness.submitter(
                self.harness.default_scope, context.runtime_scope.run_id
            )
            context = replace(context, submit_operation=submit_operation)
        return await super().execute_mtp(text, context=context)
