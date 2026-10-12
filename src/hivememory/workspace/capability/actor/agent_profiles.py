"""actor CALL API：操作授权后经 Profile 读取视图解析目标，保留资源可见性。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.access import WorkspaceOperation
from hivememory.workspace.contracts.operations import DeriveCallThreadRequest, DerivedCallThread
from hivememory.workspace.contracts.threads import ChildThreadSpawner

if TYPE_CHECKING:
    from hivememory.core.access import WorkspaceAccessContext
    from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
    from hivememory.workspace.resolution.profile import ProfileResolver


class ActorAgentService:
    """actor 资源入口，访问 context 只由本层授权并转为本次操作 scope。"""

    def __init__(
        self,
        *,
        operation_authorizer: WorkspaceOperationAuthorizer,
        profile_reader: ProfileResolver,
        thread_spawner: ChildThreadSpawner,
    ) -> None:
        self._authorizer = operation_authorizer
        self._profile_reader = profile_reader
        self._thread_spawner = thread_spawner

    async def derive_call_thread(
        self,
        request: DeriveCallThreadRequest,
        *,
        access: WorkspaceAccessContext,
    ) -> DerivedCallThread:
        """CALL 派生（``agent.call``）：以调用方可见的图纸确定被调用方。

        未指定 alias 或显式选择 ``default`` / ``omni_doll`` 时返回内置 Profile；
        自定义 alias 缺失、不可见、类型不符或配置损坏均显式失败，不降级为
        默认配置。Profile 冷读返回后重新确认线程寿命，再由进程校验主线程
        与阶段；被调用方的准入、签发与操作快照均由进程侧处理。
        """
        scope = self._authorizer.authorize_thread_operation(access, WorkspaceOperation.AGENT_CALL)
        profile = await self._profile_reader.get(request.agent_alias, scope=scope)
        if not profile.agent_id:
            raise ValueError("CALL 目标 Profile 缺少 agent_id")
        process_id = self._authorizer.get_thread_process_id(access)
        child = self._thread_spawner.derive_child_thread(
            access=access, process_id=process_id, agent_id=profile.agent_id
        )
        return DerivedCallThread(
            agent_profile=profile, access=child.access, allowed_operations=child.allowed_operations
        )
