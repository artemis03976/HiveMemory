"""会话取消收口与 finalize 生命周期；保留真实注册入口、权限与进程编排。"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.workspace import WorkspaceSessionConfig
from hivememory.core.models import OMNI_DOLL_PROFILE, ActorIdentity, ResolvedAgentProfile
from hivememory.core.models.interaction import TurnEvent
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.core.protocol.models import InteractionPayload
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.process.table import CancelResult
from hivememory.workspace.sessions import ConversationSessionService
from tests.helpers.chat_handoff import make_gateway_decision
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.process import collect_events, done_data, make_task_process_service
from tests.helpers.sessions import make_compaction_checkpoint
from tests.helpers.workspace import make_access_composition, make_actor_access_record


@dataclass
class CompletionProbe:
    """记录外部 Patchouli 边界的交互接管与清理，提供可控提交屏障与用户停止入口。

    ``stop`` 以同一 Workspace 已获准入的请求方 context 对本进程发起停止，
    与 ``/chat/stop`` 走同一个取消方法。
    """

    submitted: list = field(default_factory=list)
    cleaned: list = field(default_factory=list)
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    stop: Callable[[], CancelResult] | None = None


async def completion_chain(
    cpu, *, block_finalize=False, shutdown_wait_seconds=10.0, seed_history=False
):
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

    async def prefetch_models() -> None:
        return None

    for route, handler in (
        (PatchouliRoutes.PREFETCH_MODELS, prefetch_models),
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
    session_id = None
    if seed_history:
        session_id = sessions.create(access.default_workspace).session_id
        sessions.claim(access.default_workspace, session_id, "old")
        sessions.append(
            access.default_workspace,
            session_id,
            process_id="old",
            agent_id="omni_doll",
            payload=InteractionPayload(user_message="旧问题", assistant_final_text="旧回复"),
            topic_id="old-topic",
        )
        sessions.release(access.default_workspace, session_id, "old")
    handle = await service.register_process(
        adapter="local",
        principal=access.principal,
        actor=ActorIdentity(user_id=access.default_workspace.owner_user_id, agent_id="omni_doll"),
        workspace=access.default_workspace,
        process_id="partial",
        session_id=session_id,
        message="请继续分析",
    )
    requestor = await access.authenticate(agent_id="omni_doll")
    probe.stop = partial(service.cancel_process, handle.process_id, access=requestor)
    return service, handle, sessions, access.default_workspace, probe


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reported", "final_text"), [("cancelled", "只说到这里"), ("completed", "完整答案")]
)
async def test_user_stop_seals_outcome_returned_by_cpu(reported, final_text):
    """用户停止时沿用 CPU 取消后返回的结局：部分输出按 cancelled、停止前已完成按
    completed，在 done 前封口、写入会话并由 Patchouli 接管。"""
    cpu = ScriptedCPU(
        events=[{"event": "token", "data": {"content": final_text}}],
        hang_before_result=True,
        cancelled_result=make_cpu_result(status=reported, final_text=final_text),
    )
    service, handle, sessions, workspace, probe = await completion_chain(cpu)
    stream = service.run_process(handle)
    first = await anext(stream)
    await asyncio.wait_for(cpu.hang_entered.wait(), timeout=2)
    assert probe.stop().cancelled is True
    events = [first, *[event async for event in stream]]
    done = events[-1]["data"]
    assert (done["status"], done["final_text"]) == (reported, final_text)
    history = sessions.get(workspace, handle.session_id)
    assert [(t.payload.outcome, t.payload.assistant_final_text) for t in history.turns] == [
        (reported, final_text)
    ]
    assert probe.submitted == [history.turns[0].payload]
    assert probe.cleaned == []
    assert history.active_process_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "completed", "cancelled"])
async def test_output_with_each_outcome_is_submitted_before_terminal(status):
    """三种结局统一封口，失败/取消不被改写为 completed。"""
    service, handle, sessions, workspace, probe = await completion_chain(
        ScriptedCPU(
            result=make_cpu_result(
                status=status,
                final_text="已有输出",
                compaction_checkpoint=make_compaction_checkpoint(reduced_through="old"),
            )
        ),
        seed_history=True,
    )
    events = [event async for event in service.run_process(handle)]
    assert events[-1]["data"]["status"] == status
    assert [p.outcome for p in probe.submitted] == [status]
    saved = sessions.get(workspace, handle.session_id)
    assert saved.turns[-1].payload == probe.submitted[0]
    assert saved.compaction_checkpoint.reduced_through == "old"
    assert "compaction_checkpoint" not in events[-1]["data"]
    assert probe.cleaned == []


@pytest.mark.asyncio
async def test_alice_context_guard_seals_tool_output_as_failed_turn(tmp_path):
    """真实 Alice 读入大文件后由守卫结束，经 CPU 端口保留 failed 会话和工具原文。"""
    from hivememory.alice.system import AliceSystem
    from hivememory.config.alice import AgentRuntimeConfig, AliceConfig
    from tests.helpers.alice import StaticModelResolver
    from tests.helpers.llm import CallbackLLMService
    from tests.helpers.operations import OperationsHarness

    content = "x" * 20000
    (tmp_path / "large.txt").write_text(content)
    config = AliceConfig(runtime=AgentRuntimeConfig(default_context_window=36000, safety_margin=0))
    config.koakuma.workspace_path = str(tmp_path)
    config.koakuma.mtp_prompt.enabled = False

    async def read_large_file(_messages, _params):
        yield '⟪ RUN | sys_read_file | path="large.txt" ⟫'

    llm = CallbackLLMService(stream=read_large_file)
    alice = AliceSystem(
        config,
        operation_entry=OperationsHarness().entry,
        llm_service=llm,
        model_registry=StaticModelResolver(),
    )
    await alice.start()
    try:
        service, handle, sessions, workspace, probe = await completion_chain(alice.cpu_port)
        events = [event async for event in service.run_process(handle)]
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
        assert len(llm.calls) == 1
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
                compaction_checkpoint=make_compaction_checkpoint(reduced_through="old"),
                turn_events=[
                    TurnEvent(kind="user_message", sequence=0, role="user", content="请继续分析")
                ],
            )
        ),
        seed_history=True,
    )
    events = [event async for event in service.run_process(handle)]
    assert "final_text" not in events[-1]["data"]
    saved = sessions.get(workspace, handle.session_id)
    assert [turn.interaction_id for turn in saved.turns] == ["old"]
    assert saved.compaction_checkpoint.reduced_through == "old"
    assert "compaction_checkpoint" not in events[-1]["data"]
    assert probe.submitted == []
    assert probe.cleaned == [handle.process_id]


@pytest.mark.asyncio
async def test_delivery_leaving_after_finalizing_never_skips_submission():
    """交付方在 finalizing 之后离开：退订立即返回，提交照常完成，已接管的交互不被 cleanup。"""
    service, handle, sessions, workspace, probe = await completion_chain(
        ScriptedCPU(result=make_cpu_result()), block_finalize=True
    )
    stream = service.run_process(handle)
    while (await anext(stream))["event"] != "run_status":
        pass
    assert probe.stop().reason == "already_finalizing"
    await stream.aclose()
    await probe.entered.wait()
    assert probe.submitted == []
    assert sessions.get(workspace, handle.session_id).active_process_id == handle.process_id
    probe.release.set()
    await service.close_process(handle)
    assert [p.outcome for p in probe.submitted] == ["completed"]
    assert probe.cleaned == []
    assert sessions.get(workspace, handle.session_id).active_process_id is None


@pytest.mark.asyncio
async def test_stop_after_cpu_terminal_keeps_completed_outcome():
    """CPU 已返回 completed、进程尚未恢复时到达的停止，不把完整回复改记为中断。"""
    cpu = ScriptedCPU(result=make_cpu_result(final_text="完整答案"))
    service, handle, sessions, workspace, probe = await completion_chain(cpu)
    run = asyncio.create_task(_collect(service.run_process(handle)))
    await asyncio.wait_for(cpu.result_returning.wait(), timeout=2)
    assert probe.stop().cancelled is True
    done = (await run)[-1]["data"]
    assert (done["status"], done["final_text"]) == ("completed", "完整答案")
    turns = sessions.get(workspace, handle.session_id).turns
    assert [t.payload.outcome for t in turns] == ["completed"]
    assert [p.outcome for p in probe.submitted] == ["completed"]


@pytest.mark.asyncio
async def test_delivery_leaving_after_cpu_terminal_discards_turn_but_accepts_checkpoint():
    """CPU 已返回带检查点的终态后交付方离开：本轮丢弃不写会话，检查点仍被接受。"""
    checkpoint = make_compaction_checkpoint(reduced_through="old")
    cpu = ScriptedCPU(result=make_cpu_result(compaction_checkpoint=checkpoint))
    service, handle, sessions, workspace, probe = await completion_chain(cpu, seed_history=True)
    stream = service.run_process(handle)
    await anext(stream)
    await asyncio.wait_for(cpu.result_returning.wait(), timeout=2)
    await stream.aclose()
    await service.close_process(handle)
    saved = sessions.get(workspace, handle.session_id)
    assert saved.compaction_checkpoint == checkpoint
    assert [turn.interaction_id for turn in saved.turns] == ["old"]
    assert saved.active_process_id is None
    assert probe.submitted == []
    assert probe.cleaned == [handle.process_id]


async def _collect(stream):
    return [event async for event in stream]


class _HardCancelSignal(logging.Handler):
    """系统关闭记录硬取消告警时置位，用于在进程 task 被取消之后再放行提交。"""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.hard_cancelled = asyncio.Event()

    def emit(self, record: logging.LogRecord) -> None:
        if "硬取消" in record.getMessage():
            self.hard_cancelled.set()


@pytest.mark.asyncio
async def test_shutdown_stops_streaming_process_and_discards_partial_output(caplog):
    """关闭以 system_shutdown 停止进程并等待它在自己的 task 中收尾；部分输出不入会话、不提交。"""
    cpu = ScriptedCPU(
        events=[{"event": "token", "data": {"content": "半截"}}],
        hang_before_result=True,
        cancelled_result=make_cpu_result(status="cancelled", final_text="半截"),
    )
    service, handle, sessions, workspace, probe = await completion_chain(cpu)
    subscriber = asyncio.create_task(_collect(service.run_process(handle)))
    await asyncio.wait_for(cpu.hang_entered.wait(), timeout=2)
    with caplog.at_level(logging.WARNING):
        await service.shutdown()
    session = sessions.get(workspace, handle.session_id)
    assert (session.active_process_id, session.turns) == (None, ())
    done = (await subscriber)[-1]["data"]
    assert (done["status"], done["reason"]) == ("cancelled", "system_shutdown")
    assert "final_text" not in done
    assert probe.submitted == []
    assert probe.cleaned == [handle.process_id]
    assert cpu.finished is True
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.mark.asyncio
async def test_shutdown_hard_cancels_process_that_does_not_stop_in_time(caplog):
    """停止后未在等待上限内结束的进程被硬取消，照常收尾、补偿 prepare 并释放会话认领。"""
    cpu = ScriptedCPU(
        events=[{"event": "token", "data": {"content": "半截"}}],
        hang_before_result=True,
        linger_on_cancel=True,
    )
    service, handle, sessions, workspace, probe = await completion_chain(
        cpu, shutdown_wait_seconds=0
    )
    subscriber = asyncio.create_task(_collect(service.run_process(handle)))
    await asyncio.wait_for(cpu.hang_entered.wait(), timeout=2)
    with caplog.at_level(logging.WARNING):
        await service.shutdown()
    assert sessions.get(workspace, handle.session_id).active_process_id is None
    assert probe.cleaned == [handle.process_id]
    assert cpu.finished is True
    assert probe.stop().status == "not_found"
    assert any("硬取消" in r.getMessage() for r in caplog.records)
    assert [event["event"] for event in await subscriber] == ["process_id", "memory_refs", "token"]


@pytest.mark.asyncio
async def test_shutdown_hard_cancel_still_waits_for_finalize_submission():
    """进入 finalize 的进程拒绝停止；被硬取消后仍等提交结束，不 cleanup 已接管交互。"""
    service, handle, sessions, workspace, probe = await completion_chain(
        ScriptedCPU(result=make_cpu_result()), block_finalize=True, shutdown_wait_seconds=0
    )
    stream = service.run_process(handle)
    while (await anext(stream))["event"] != "run_status":
        pass
    await probe.entered.wait()
    signal = _HardCancelSignal()
    service_logger = logging.getLogger("hivememory.workspace.process.service")
    service_logger.addHandler(signal)
    try:
        shutdown = asyncio.create_task(service.shutdown())
        await asyncio.wait_for(signal.hard_cancelled.wait(), timeout=2)
        probe.release.set()
        await asyncio.wait_for(shutdown, timeout=2)
    finally:
        service_logger.removeHandler(signal)
    assert [p.outcome for p in probe.submitted] == ["completed"]
    assert probe.cleaned == []
    saved = sessions.get(workspace, handle.session_id)
    assert [t.payload.outcome for t in saved.turns] == ["completed"]
    assert saved.active_process_id is None
    await stream.aclose()


@pytest.mark.asyncio
async def test_shutdown_closes_registered_process_that_never_ran():
    """注册后从未运行的进程没有交付方，关闭时直接收口并释放认领。"""
    service, handle, sessions, workspace, probe = await completion_chain(
        ScriptedCPU(result=make_cpu_result())
    )
    await service.shutdown()
    assert sessions.get(workspace, handle.session_id).active_process_id is None
    assert probe.stop().status == "not_found"
    assert probe.cleaned == []


@pytest.mark.asyncio
@pytest.mark.parametrize("close_mode", ["delivery_left", "user_stop", "hard_cancel"])
async def test_closing_cpu_accepts_old_history_checkpoint_before_releasing_claim(close_mode):
    """交付方离开、用户停止与硬取消都保存 CPU 返回的旧轮次视图，且不新增空轮次。

    硬取消时 CPU 在最外层把取消转换为返回值：进程仍接受其中的检查点，并把
    取消继续向上传播，不交付终态。
    """
    checkpoint = make_compaction_checkpoint(
        reduced_through="old", summarized_through="old", summary="已完成的接力摘要"
    )
    cpu = ScriptedCPU(
        events=[{"event": "context_compaction", "data": {"status": "completed"}}],
        hang_before_result=True,
        linger_on_cancel=close_mode == "hard_cancel",
        cancelled_result=make_cpu_result(
            status="cancelled", final_text="", compaction_checkpoint=checkpoint
        ),
    )
    service, handle, sessions, workspace, probe = await completion_chain(
        cpu, seed_history=True, shutdown_wait_seconds=0
    )
    stream = service.run_process(handle)
    while (await anext(stream))["event"] != "context_compaction":
        pass
    await asyncio.wait_for(cpu.hang_entered.wait(), timeout=2)
    if close_mode == "delivery_left":
        await stream.aclose()
        await service.close_process(handle)
    elif close_mode == "hard_cancel":
        await service.shutdown()
        assert [event["event"] async for event in stream] == []
    else:
        probe.stop()
        events = [event async for event in stream]
        assert events[-1]["data"]["status"] == "cancelled"
    saved = sessions.get(workspace, handle.session_id)
    assert saved.compaction_checkpoint == checkpoint
    assert [turn.interaction_id for turn in saved.turns] == ["old"]
    assert saved.active_process_id is None
    assert probe.submitted == []


@pytest.mark.asyncio
async def test_rejected_checkpoint_does_not_prevent_current_turn_submission(caplog):
    """拒绝覆盖本轮或外部轮次的游标只发告警，正常回复仍提交。"""
    cpu = ScriptedCPU(
        result=make_cpu_result(
            compaction_checkpoint=make_compaction_checkpoint(reduced_through="partial")
        )
    )
    service, handle, sessions, workspace, probe = await completion_chain(cpu, seed_history=True)
    with caplog.at_level(logging.WARNING):
        events = await collect_events(service, handle)
    assert done_data(events)["status"] == "completed"
    saved = sessions.get(workspace, handle.session_id)
    assert saved.compaction_checkpoint is None
    assert [turn.interaction_id for turn in saved.turns] == ["old", "partial"]
    assert probe.submitted == [saved.turns[-1].payload]
    assert any("拒绝会话压缩检查点" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_unavailable_alice_fails_process_and_releases_prepared_resources():
    """CPU 未启动时执行失败仍补偿 prepare、释放会话，并且不提交交互。"""
    from hivememory.alice import AliceSystem
    from hivememory.config.alice import AliceConfig
    from tests.helpers.llm import ScriptedLLMService
    from tests.helpers.operations import OperationsHarness

    alice = AliceSystem(
        AliceConfig(), operation_entry=OperationsHarness().entry, llm_service=ScriptedLLMService()
    )
    service, handle, sessions, workspace, probe = await completion_chain(alice.cpu_port)
    events = [event async for event in service.run_process(handle)]
    assert events[-1] == {"event": "error", "data": {"message": "系统错误，请检查后端服务器"}}
    saved = sessions.get(workspace, handle.session_id)
    assert saved.turns == ()
    assert saved.active_process_id is None
    assert probe.cleaned == [handle.process_id]
    assert probe.submitted == []
    assert probe.stop().status == "not_found"
