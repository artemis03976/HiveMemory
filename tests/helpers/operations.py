"""测试用进程操作通道装配：真实 workspace 授权、读取视图、登记与通道。"""

from __future__ import annotations

from dataclasses import replace

from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.errors import ResourceUnavailableError
from hivememory.core.models import IdentityScope, MemoryAtom
from hivememory.workspace.capability.memory import MemoryApplicationService
from hivememory.workspace.process.operations import ProcessOperationChannel
from hivememory.workspace.runtime import WorkspaceRuntime
from tests.helpers.workspace import make_access_composition, make_actor_access_record


class MemoryBackend:
    """仅替代持久化边界，读取视图与授权仍使用生产实现。"""

    def __init__(self, bus=None) -> None:
        self.memories: dict[str, MemoryAtom] = {}
        self.bus = bus

    async def read(self, memory_id, *, scope):
        return next((atom for atom in self.memories.values() if atom.id == memory_id), None)

    async def retrieve_by_aliases(self, aliases, *, scope):
        from hivememory.core.contracts.routes import GlobalRoutes

        found = [self.memories[alias] for alias in aliases if alias in self.memories]
        missing = [alias for alias in aliases if alias not in self.memories]
        if self.bus is not None and missing:
            try:
                found.extend(
                    await self.bus.request(
                        GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE_BY_ALIASES,
                        aliases=missing,
                        identity_scope=scope,
                    )
                )
            except KeyError as error:
                raise ResourceUnavailableError("测试存储读取路由不可达") from error
        return found

    async def retrieve(self, request):
        return list(self.memories.values())

    async def get_agent_profile(self, alias, *, scope):
        from hivememory.core.contracts.routes import GlobalRoutes

        if self.bus is None:
            raise NotImplementedError("本测试未装配 Profile backing")
        return await self.bus.request(
            GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, alias, identity_scope=scope
        )


class OperationsHarness:
    """为测试身份装配真实通道，跨调用保留同一份进程级登记。"""

    def __init__(self, bus=None) -> None:
        self.backing = MemoryBackend(bus)
        self.runtime = WorkspaceRuntime(backing=self.backing, atom_capacity=32, profile_capacity=16)
        self.registry = self.runtime.intents
        self.memories = self.backing.memories

    async def channel(
        self, scope: IdentityScope, process_id: str = "test_run", *, allowed_operations=None
    ):
        workspace = scope.workspace_identity
        actor = scope.actor_identity
        composition = make_access_composition(
            [
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
        access = await composition.authenticate(agent_id=actor.agent_id, user_id=actor.user_id)
        service = MemoryApplicationService(
            GlobalSystemBus(),
            operation_authorizer=composition.authorizer,
            memory_reader=self.runtime.aliases,
        )
        return ProcessOperationChannel(
            service, access=access, target_workspace=workspace, process_id=process_id
        )


class HarnessKoakumaRuntime(KoakumaRuntime):
    """测试显式装配通道，复用生产 MTP handler，不模拟解析或登记行为。"""

    def __init__(self, *, bus=None, config=None):
        super().__init__(bus=bus, config=config)
        self.harness = OperationsHarness(bus)
        self.memories = self.harness.memories
        self.registry = self.harness.registry

    async def execute_mtp(self, text, context=None):
        if context is not None and context.operations is None:
            operations = await self.harness.channel(
                context.identity_scope, context.runtime_scope.run_id
            )
            context = replace(context, operations=operations)
        return await super().execute_mtp(text, context=context)
