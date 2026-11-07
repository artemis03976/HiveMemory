"""PatchouliService 的检索命中记录：归属取 scope 的 Workspace，同一批去重，单条失败尽力而为。

生命周期 owner 位于本次边界之外，以记录型 handler 挂在真实本地总线的命中路由上。
"""

from __future__ import annotations

import logging
from uuid import uuid4

import pytest

from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.service import PatchouliService
from tests.helpers.workspace import make_identity_scope

# 命中由任务进程经 root API 记录：授权点以 system 的 root 授权组装 scope。
ROOT_SCOPE = make_identity_scope(user_id="u1", agent_id=SYSTEM_AGENT_ID)
SOURCE = "retrieval.pre_retrieval"


def _service_recording_hits(*, failing: frozenset = frozenset()):
    """返回挂好命中 handler 的服务与命中记录；``failing`` 中的记忆记录时失败。"""
    recorded: list[tuple] = []

    async def record_hit(memory_id, *, belong_to, source):
        if memory_id in failing:
            raise ConnectionError("lifecycle store unavailable")
        recorded.append((memory_id, belong_to, source))

    bus = PatchouliBus()
    bus.register(PatchouliLocalRoutes.MEMORY_RECORD_HIT, record_hit)
    return PatchouliService(bus), recorded


@pytest.mark.asyncio
async def test_record_retrieval_hits_records_each_memory_once_in_scope_workspace() -> None:
    """同一记忆以 UUID 与字符串重复出现时只记一次命中，归属取 scope 的 Workspace。"""
    first, second = uuid4(), uuid4()
    service, recorded = _service_recording_hits()

    await service.record_retrieval_hits(
        [first, str(first), second, second], identity_scope=ROOT_SCOPE, source=SOURCE
    )

    belong_to = ROOT_SCOPE.workspace_identity
    assert recorded == [(first, belong_to, SOURCE), (second, belong_to, SOURCE)]


@pytest.mark.asyncio
async def test_record_retrieval_hits_failure_is_logged_and_does_not_stop_the_batch(
    caplog,
) -> None:
    """单条记录失败只记日志、不交回调用方，其余记忆照常记录命中。"""
    failed, kept = uuid4(), uuid4()
    service, recorded = _service_recording_hits(failing=frozenset({failed}))

    with caplog.at_level(logging.WARNING, logger="hivememory.patchouli.service"):
        await service.record_retrieval_hits(
            [failed, kept], identity_scope=ROOT_SCOPE, source=SOURCE
        )

    assert [memory_id for memory_id, _, _ in recorded] == [kept]
    assert [record.levelno for record in caplog.records] == [logging.WARNING]
    assert str(failed) in caplog.records[0].getMessage()
