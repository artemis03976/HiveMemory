"""会话取消收口与 finalize 生命周期；保留真实注册入口、权限与进程编排。"""

import asyncio
import logging
from dataclasses import dataclass, field

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.workspace import WorkspaceSessionConfig
from hivememory.core.models import OMNI_DOLL_PROFILE, ActorIdentity, ResolvedAgentProfile
from hivememory.core.models.interaction import TurnEvent
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.contracts import CPUResultSlot, CPURunHandle
from hivememory.workspace.sessions import ConversationSessionService
from tests.helpers.chat_handoff import make_gateway_decision
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.process import make_task_process_service
from tests.helpers.workspace import make_access_composition, make_actor_access_record


@dataclass
class CompletionProbe:
    """记录外部 Patchouli 边界的交互接管与清理，提供可控提交屏障。"""

    submitted: list = field(default_factory=list)
    cleaned: list = field(default_factory=list)
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)


async def completion_chain(cpu, *, block_finalize=False, shutdown_wait_seconds=10.0):
    """组装进程与会话，只有 Gateway/Patchouli 服务边界使用确定性替身。"""
    access = make_access_composition([make_actor_access_record(agent_id="omni_doll")])
    sessions = ConversationSessionService(WorkspaceSessionConfig())
    bus = GlobalSystemBus()
    probe = CompletionProbe()
    if not block_finalize:
        probe.release.set()

    async def gateway(**_kwargs):
        return GatewayDecisionOutcome(decision=make_gateway_decision())

    async def profile(*_args, **_kwargs):
        return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)

    async def prepare(*, identity_scope, interaction_id, **_kwargs):
        return PreparedAgentRun(
            belong_to=identity_scope.workspace_identity,
            interaction_id=interaction_id,
            topic_id="topic-shared",
            is_new_topic=True,
        )

    async def finalize(*, payload, **_kwargs):
        probe.entered.set()
        await probe.release.wait()
        probe.submitted.append(payload)

    async def cleanup(*, prepared_run, **_kwargs):
        probe.cleaned.append(prepared_run.interaction_id)

    for route, handler in (
        (GatewayPublicRoutes.PROCESS, gateway),
        (PatchouliRoutes.GET_AGENT_PROFILE, profile),
        (PatchouliRoutes.PREPARE_AGENT_RUN, prepare),
        (PatchouliRoutes.FINALIZE_AGENT_RUN, finalize),
        (PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN, cleanup),
    ):
        bus.register(route, handler)
    service = make_task_process_service(
        bus,
        cpu=cpu,
        sessions=sessions,
        access_gateway=access.gateway,
        operation_authorizer=access.authorizer,
        shutdown_wait_seconds=shutdown_wait_seconds,
    )
    handle = await service.register_process(
        adapter="local",
        principal=access.principal,
        actor=ActorIdentity(user_id=access.default_workspace.owner_user_id, agent_id="omni_doll"),
        workspace=access.default_workspace,
        process_id="partial",
        message="请继续分析",
    )
    return service, handle, sessions, access.default_workspace, probe


@pytest.mark.asyncio
@pytest.mark.parametrize("during_pull", [True, False])
async def test_user_stop_seals_partial_slot_even_between_pulls(during_pull):
    """正在拉取和两次拉取之间的 stop 都交回半截结果，并在 done 前接管交互。"""
    cpu = ScriptedCPU(
        events=[{"event": "token", "data": {"content": "只说到这里"}}],
        hang_before_result=True,
        cancelled_result=make_cpu_result(status="cancelled", final_text="只说到这里"),
    )
    service, handle, sessions, workspace, probe = await completion_chain(cpu)
    stream = service.run_process(handle, stream=True)
    while (await anext(stream))["event"] != "token":
        pass
    pending = None
    if during_pull:
        pending = asyncio.create_task(anext(stream))
        await cpu.hang_entered.wait()
    assert service.cancel_process(handle).cancelled is True
    if pending is not None:
        assert (await pending)["data"]["status"] == "finalizing"
    events = [event async for event in stream]
    done = events[-1]["data"]
    assert (done["status"], done["final_text"]) == ("cancelled", "只说到这里")
    history = sessions.get(workspace, handle.session_id)
    assert [(t.payload.outcome, t.payload.assistant_final_text) for t in history.turns] == [
        ("cancelled", "只说到这里")
    ]
    assert probe.submitted == [history.turns[0].payload]
    assert probe.cleaned == []
    assert history.active_process_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "completed", "cancelled"])
async def test_output_with_each_outcome_is_submitted_before_terminal(status):
    """三种结局统一封口，失败/取消不被改写为 completed。"""
    service, handle, sessions, workspace, probe = await completion_chain(
        ScriptedCPU(result=make_cpu_result(status=status, final_text="已有输出"))
    )
    events = [event async for event in service.run_process(handle, stream=True)]
    assert events[-1]["data"]["status"] == status
    assert [p.outcome for p in probe.submitted] == [status]
    assert sessions.get(workspace, handle.session_id).turns[0].payload == probe.submitted[0]
    assert probe.cleaned == []


@pytest.mark.asyncio
async def test_alice_context_guard_seals_tool_output_as_failed_turn(tmp_path, monkeypatch):
    """真实 Alice 读入大文件后由守卫结束，经 CPU 端口保留 failed 会话和工具原文。"""
    from hivememory.agent_runtime.execution.worker import WorkerAgentService
    from hivememory.agent_runtime.models import GenerationResult, StreamChunk
    from hivememory.alice.system import AliceSystem
    from hivememory.config.alice import AliceConfig, CompactionConfig
    from tests.helpers.operations import OperationsHarness

    content = "x" * 20000
    (tmp_path / "large.txt").write_text(content)
    config = AliceConfig(compaction=CompactionConfig(default_context_window=36000, safety_margin=0))
    config.koakuma.workspace_path = str(tmp_path)
    config.koakuma.mtp_prompt.enabled = False
    calls = []

    async def generate(_self, _messages, **_kwargs):
        calls.append(True)
        text = '⟪ RUN | sys_read_file | path="large.txt" ⟫'
        yield StreamChunk(
            is_final=True,
            result=GenerationResult(text=text, was_mtp_interrupted=True, mtp_fragment=text),
        )

    monkeypatch.setattr(WorkerAgentService, "generate_stream", generate)
    alice = AliceSystem(config, GlobalSystemBus(), operation_entry=OperationsHarness().entry)
    await alice.start()
    try:
        service, handle, sessions, workspace, probe = await completion_chain(alice.cpu_port)
        events = [event async for event in service.run_process(handle, stream=True)]
        done = events[-1]["data"]
        assert (done["status"], done["failure_reason"]) == (
            "failed",
            "turn_context_budget_exceeded",
        )
        turns = sessions.get(workspace, handle.session_id).turns
        assert len(turns) == 1
        assert turns[0].payload.outcome == "failed"
        assert content in turns[0].payload.turn_events[-1].content
        assert probe.submitted == [turns[0].payload]
        assert probe.cleaned == []
        assert calls == [True]
    finally:
        await alice.stop()


@pytest.mark.asyncio
async def test_empty_terminal_does_not_create_turn_and_done_omits_text():
    """无输出结果不能污染历史；空 final_text 不得覆盖客户端已显示的内容。

    无输出判断位于终态分派之前，各终态的 done 投影由流程单测逐一覆盖；
    这里取 cancelled 代表，它同时经过取消终态省略 topic_id 的分支。
    """
    service, handle, sessions, workspace, probe = await completion_chain(
        ScriptedCPU(
            result=make_cpu_result(
                status="cancelled",
                final_text="",
                turn_events=[
                    TurnEvent(kind="user_message", sequence=0, role="user", content="请继续分析")
                ],
            )
        )
    )
    events = [event async for event in service.run_process(handle, stream=True)]
    assert "final_text" not in events[-1]["data"]
    assert sessions.get(workspace, handle.session_id).turns == ()
    assert probe.submitted == []
    assert probe.cleaned == [handle.process_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_close", [False, True])
async def test_close_at_finalizing_drains_submission_and_never_cleans_prepared(cancel_close):
    """finalizing 事件刚交出就关闭交付，甚至再取消关闭任务，也不能丢失提交。"""
    service, handle, sessions, workspace, probe = await completion_chain(
        ScriptedCPU(result=make_cpu_result()), block_finalize=True
    )
    stream = service.run_process(handle, stream=True)
    while (await anext(stream))["event"] != "run_status":
        pass
    assert service.cancel_process(handle).reason == "already_finalizing"
    closing = asyncio.create_task(stream.aclose())
    await probe.entered.wait()
    assert sessions.get(workspace, handle.session_id).active_process_id == handle.process_id
    if cancel_close:
        closing.cancel()
    probe.release.set()
    if cancel_close:
        with pytest.raises(asyncio.CancelledError):
            await closing
    else:
        await closing
    assert [p.outcome for p in probe.submitted] == ["completed"]
    assert probe.cleaned == []
    assert sessions.get(workspace, handle.session_id).active_process_id is None


class SlowCloseCPU:
    """交出 completed 终态后，关闭输出流要等待测试放行，用于在此窗口注入停止。"""

    def __init__(self) -> None:
        self.closing = asyncio.Event()
        self.release = asyncio.Event()

    def execute(self, manifest, *, access, generation_options, stream):
        slot = CPUResultSlot()
        return CPURunHandle(output=self._iterate(slot), result_slot=slot)

    async def _iterate(self, slot):
        slot.result = make_cpu_result(final_text="完整答案")
        try:
            yield slot.result
        finally:
            self.closing.set()
            await self.release.wait()


@pytest.mark.asyncio
async def test_stop_after_cpu_terminal_keeps_completed_outcome():
    """CPU 已交出 completed 后才到达的停止不把完整回复改记为中断。"""
    cpu = SlowCloseCPU()
    service, handle, sessions, workspace, probe = await completion_chain(cpu)
    run = asyncio.create_task(_collect(service.run_process(handle, stream=True)))
    await cpu.closing.wait()
    assert service.cancel_process(handle).cancelled is True
    cpu.release.set()
    done = (await run)[-1]["data"]
    assert (done["status"], done["final_text"]) == ("completed", "完整答案")
    turns = sessions.get(workspace, handle.session_id).turns
    assert [t.payload.outcome for t in turns] == ["completed"]
    assert [p.outcome for p in probe.submitted] == ["completed"]


@pytest.mark.asyncio
async def test_stop_between_pulls_keeps_completed_slot_outcome():
    """停止落在两次拉取之间而 CPU 已完成时，沿用结果槽中的 completed。"""
    cpu = ScriptedCPU(
        events=[{"event": "token", "data": {"content": "完整答案"}}],
        hang_before_result=True,
        cancelled_result=make_cpu_result(final_text="完整答案"),
    )
    service, handle, sessions, workspace, probe = await completion_chain(cpu)
    stream = service.run_process(handle, stream=True)
    while (await anext(stream))["event"] != "token":
        pass
    assert service.cancel_process(handle).cancelled is True
    done = [event async for event in stream][-1]["data"]
    assert done["status"] == "completed"
    assert [t.payload.outcome for t in sessions.get(workspace, handle.session_id).turns] == [
        "completed"
    ]
    assert [p.outcome for p in probe.submitted] == ["completed"]


async def _collect(stream):
    return [event async for event in stream]


async def _drive_like_sse(service, handle):
    """与 SSE 路由相同：每次拉取都在新 task 中进行，结束后经注册入口关闭。"""
    stream = service.run_process(handle, stream=True)
    events = []
    try:
        while True:
            try:
                events.append(await asyncio.create_task(anext(stream)))
            except StopAsyncIteration:
                return events
    finally:
        await service.close_process(handle)


@pytest.mark.asyncio
async def test_shutdown_stops_sse_driven_process_and_waits_for_its_own_close(caplog):
    """关闭不与仍在拉取的交付方并发收口；部分输出不入会话、不提交。"""
    cpu = ScriptedCPU(
        events=[{"event": "token", "data": {"content": "半截"}}],
        hang_before_result=True,
        cancelled_result=make_cpu_result(status="cancelled", final_text="半截"),
    )
    service, handle, sessions, workspace, probe = await completion_chain(cpu)
    driver = asyncio.create_task(_drive_like_sse(service, handle))
    await cpu.hang_entered.wait()
    with caplog.at_level(logging.WARNING):
        await service.shutdown()
    session = sessions.get(workspace, handle.session_id)
    assert (session.active_process_id, session.turns) == (None, ())
    done = (await driver)[-1]["data"]
    assert (done["status"], done["reason"]) == ("cancelled", "system_shutdown")
    assert "final_text" not in done
    assert probe.submitted == []
    assert probe.cleaned == [handle.process_id]
    assert cpu.closed is True
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.mark.asyncio
async def test_shutdown_force_closes_process_whose_driver_stopped_pulling(caplog):
    """交付方停在两次拉取之间不再拉取时，等待超时后兜底关闭并释放会话认领。"""
    cpu = ScriptedCPU(events=[{"event": "token", "data": {"content": "半截"}}])
    service, handle, sessions, workspace, probe = await completion_chain(
        cpu, shutdown_wait_seconds=0
    )
    stream = service.run_process(handle, stream=True)
    while (await anext(stream))["event"] != "token":
        pass
    with caplog.at_level(logging.WARNING):
        await service.shutdown()
    assert sessions.get(workspace, handle.session_id).active_process_id is None
    assert probe.cleaned == [handle.process_id]
    assert cpu.closed is True
    assert service.cancel_process(handle).status == "not_found"
    assert any("兜底关闭" in r.getMessage() for r in caplog.records)
    await stream.aclose()


@pytest.mark.asyncio
async def test_shutdown_closes_registered_process_that_never_ran():
    """注册后从未运行的进程没有交付方，关闭时直接收口并释放认领。"""
    service, handle, sessions, workspace, probe = await completion_chain(
        ScriptedCPU(result=make_cpu_result())
    )
    await service.shutdown()
    assert sessions.get(workspace, handle.session_id).active_process_id is None
    assert service.cancel_process(handle).status == "not_found"
    assert probe.cleaned == []
