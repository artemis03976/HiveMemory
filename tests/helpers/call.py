"""CALL 编排单元测试的端口替身；不模拟 workspace 内部派生与授权。"""

from hivememory.alice.orchestration.sub_agent.call_context_provider import CallContext
from hivememory.core.access import WorkspaceOperation
from hivememory.core.models import OMNI_DOLL_PROFILE
from hivememory.workspace.contracts import CallThread, DeriveCallThreadRequest


class StubOperationSubmitter:
    """只在测试边界之外按脚本返回结果，缺失脚本时显式失败。"""

    def __init__(self, *, operation=None, derive=None):
        self.operation = operation
        self.derive = derive

    async def __call__(self, request):
        if isinstance(request, DeriveCallThreadRequest):
            raise TypeError("派生请求必须通过 derive_call_thread 提交")
        if self.operation is None:
            raise AssertionError(f"意外的操作请求: {request}")
        return await self.operation(request)

    async def derive_call_thread(self, request):
        if self.derive is None:
            raise AssertionError(f"意外的派生请求: {request}")
        return await self.derive(request)


def make_call_context(
    *,
    agent_profile=OMNI_DOLL_PROFILE,
    allowed_operations=frozenset(WorkspaceOperation),
    submit_operation=None,
):
    """只供 Coordinator 单元边界注入已完成派生的上下文。"""
    return CallContext(
        agent_profile=agent_profile,
        allowed_operations=allowed_operations,
        submit_operation=submit_operation or StubOperationSubmitter(),
    )


def make_call_thread(profile=OMNI_DOLL_PROFILE, *, submit_operation=None):
    """Provider 单元测试以端口结果替代边界外的线程派生。"""
    return CallThread(
        profile, frozenset(WorkspaceOperation), submit_operation or StubOperationSubmitter()
    )
