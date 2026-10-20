"""CALL 编排单元测试的端口替身；不模拟 workspace 内部派生与授权。"""

from hivememory.alice.orchestration.call.context import CallContext
from hivememory.core.access import WorkspaceOperation
from hivememory.core.models import OMNI_DOLL_PROFILE, ExecutionLabels
from tests.helpers.chat_handoff import make_thread_descriptor
from tests.helpers.workspace import make_execution_labels


class StubOperationSubmitter:
    """线程句柄替身：按脚本返回结果，缺失脚本时显式失败；记录派生出的子线程与结束次数。"""

    def __init__(self, *, thread=None, operation=None, derive=None):
        self.thread = thread or make_thread_descriptor()
        self.operation = operation
        self.derive = derive
        self.children = []
        self.end_calls = 0

    async def __call__(self, request):
        if self.operation is None:
            raise AssertionError(f"意外的操作请求: {request}")
        return await self.operation(request)

    async def derive_child(self, request):
        if self.derive is None:
            raise AssertionError(f"意外的派生请求: {request}")
        child = await self.derive(request)
        self.children.append(child)
        return child

    def end(self):
        self.end_calls += 1


def make_callee(
    profile=OMNI_DOLL_PROFILE,
    *,
    allowed_operations=frozenset(WorkspaceOperation),
    operation=None,
):
    """以端口结果替代边界外的子线程派生：子线程描述的标签取被调用方身份。"""
    workspace_id = make_execution_labels().workspace_id
    return StubOperationSubmitter(
        thread=make_thread_descriptor(
            profile,
            thread_id="child-1",
            labels=ExecutionLabels(
                agent_id=profile.agent_id or "callee", workspace_id=workspace_id
            ),
            allowed_operations=allowed_operations,
        ),
        operation=operation,
    )


def make_call_context(
    *,
    agent_profile=OMNI_DOLL_PROFILE,
    allowed_operations=frozenset(WorkspaceOperation),
):
    """只供 Coordinator 单元边界注入已完成派生的上下文。"""
    return CallContext(callee=make_callee(agent_profile, allowed_operations=allowed_operations))
