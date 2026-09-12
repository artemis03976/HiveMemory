"""附件上传串行门的隔离、等待取消与退出行为。"""

import asyncio

import pytest

from hivememory.system.services.attachments.serial_gate import AttachmentUploadSerialGate
from tests.helpers.workspace import make_identity_scope


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_break_exclusion_or_block_next_holder() -> None:
    """等待取消不能释放持有者的门，也不能使下一请求永久阻塞。"""
    gate = AttachmentUploadSerialGate()
    key = (make_identity_scope().workspace_identity, "op-1")
    attempting = asyncio.Event()
    entered: list[str] = []
    tasks = []

    async def enter(label: str) -> None:
        attempting.set()
        async with gate.hold(key):
            entered.append(label)

    try:
        async with gate.hold(key):
            cancelled = asyncio.create_task(enter("cancelled"))
            tasks.append(cancelled)
            await asyncio.wait_for(attempting.wait(), 1)
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            attempting.clear()
            survivor = asyncio.create_task(enter("survivor"))
            tasks.append(survivor)
            await asyncio.wait_for(attempting.wait(), 1)
            assert entered == []
        await asyncio.wait_for(survivor, 1)
        await asyncio.wait_for(enter("reused"), 1)
        assert entered == ["survivor", "reused"]
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "other_workspace, other_operation",
    [
        ("workspace-2", "op-1"),
        ("workspace-1", "op-2"),
    ],
)
async def test_unrelated_operation_can_enter_while_first_is_held(
    other_workspace: str,
    other_operation: str,
) -> None:
    """门必须使用 Workspace 与 operation 组合键，不能串行化所有上传。"""
    gate = AttachmentUploadSerialGate()
    first = (make_identity_scope(workspace_id="workspace-1").workspace_identity, "op-1")
    other = (make_identity_scope(workspace_id=other_workspace).workspace_identity, other_operation)

    async def enter_other() -> str:
        async with gate.hold(other):
            return "entered"

    async with gate.hold(first):
        assert await asyncio.wait_for(enter_other(), 1) == "entered"
