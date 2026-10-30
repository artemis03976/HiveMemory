"""测试用操作请求装配：真实 workspace 授权、读取视图、登记与入口。"""

from __future__ import annotations

from hivememory.agent_runtime.frame import FrameExecutionPolicy, RuntimeScope
from hivememory.agent_runtime.mtp.context import MTPExecutionContext
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
from hivememory.workspace.read_view import WorkspaceReadView
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
        # 持久化边界依次收到的读取（点读、alias 批量读、检索与 Profile 读取），
        # 用来观察读取视图是否回源。
        self.reads: list[str] = []

    async def read(self, memory_id, *, scope):
        self.reads.append("read")
        return next((atom for atom in self.memories.values() if atom.id == memory_id), None)

    async def retrieve_by_aliases(self, aliases, *, scope):
        from hivememory.patchouli.contracts import PatchouliRoutes

        self.reads.append("retrieve_by_aliases")
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
        self.reads.append("retrieve")
        if self.bus is not None:
            return await self.bus.request(PatchouliRoutes.MEMORY_RETRIEVE, request=request)
        return list(self.memories.values())

    async def get_agent_profile(self, alias, *, scope):
        # 内存 backing 沿用 Patchouli 的内置图纸规则；入口仍必须先执行
        # agent.call 授权，测试不在 Alice 内另设默认配置回退。
        self.reads.append("get_agent_profile")
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
        self.runtime = WorkspaceReadView(
            backing=self.backing, atom_capacity=32, profile_capacity=16
        )
        self.registry = self.runtime.intents
        self.memories = self.backing.memories
        self.compositions = {}
        empty_access = make_access_composition([])
        self.entry = self.make_entry(
            operation_authorizer or empty_access.authorizer,
            empty_access.threads,
        )

    async def _record_citation(self, *, memory_id, source, identity_scope):
        """持久化替身记录公共引用路由的副作用，读取与授权边界保持真实。"""
        self.citations.append(
            {"memory_id": memory_id, "source": source, "identity_scope": identity_scope}
        )

    def make_entry(self, operation_authorizer, threads) -> WorkspaceOperationEntry:
        """入口与所有测试线程 context 共享同一份登记，替身仅位于持久化边界。"""
        return WorkspaceOperationEntry(
            ActorMemoryService(
                self.bus,
                operation_authorizer=operation_authorizer,
                memory_reader=self.runtime.memory,
                threads=threads,
            ),
            agent=ActorAgentService(
                operation_authorizer=operation_authorizer,
                profile_reader=self.runtime.profiles,
            ),
            threads=threads,
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
            credential=None,
            actor=actor,
            workspace=workspace,
            binding=RunBinding.for_task_process(process_id),
        )
        self.compositions.setdefault(process_id, []).append(composition)
        entry = self.make_entry(composition.authorizer, composition.threads)
        return ThreadOperationSubmitter(entry, composition.register_main_thread(access))

    def revoke(self, process_id: str = "test_run") -> None:
        """同步结束测试进程的全部线程并撤销进程 context，不取消调用方任务。"""
        for composition in self.compositions.pop(process_id, []):
            process = composition.process_table.get(process_id)
            composition.threads.end_all(process)
            composition.gateway.revoke_context(process.record.access)
            composition.process_table.close(process)


class HarnessKoakumaRuntime(KoakumaRuntime):
    """测试显式装配提交函数，复用生产 MTP handler，不模拟解析或登记。"""

    def __init__(self, *, bus=None, config=None):
        super().__init__(config=config)
        self.harness = OperationsHarness(bus)
        self.memories = self.harness.memories
        self.registry = self.harness.registry

    async def make_context(
        self,
        *,
        runtime_scope: RuntimeScope,
        execution_policy: FrameExecutionPolicy,
        language: str = "zh",
    ) -> MTPExecutionContext:
        """用测试身份完成认证，再把真实线程提交函数交给 MTP 上下文。"""
        submit_operation = await self.harness.submitter(
            self.harness.default_scope, runtime_scope.run_id
        )
        return MTPExecutionContext(
            runtime_scope=runtime_scope,
            execution_policy=execution_policy,
            submit_operation=submit_operation,
            language=language,
        )
