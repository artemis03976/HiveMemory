"""真实能力层、提交服务与生成队列的实时接纳和取消边界。

生成执行端用可控替身提供结算；授权、进程话题、登记、MTP、公共意图服务、
Coordinator、Controller 和 work queue 均使用生产对象。
"""

import asyncio
import logging
from types import SimpleNamespace

import pytest
import pytest_asyncio

from hivememory.agent_runtime.frame import FrameExecutionPolicy
from hivememory.agent_runtime.mtp.context import MTPExecutionContext
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.core.errors import ScopeRequiredError
from hivememory.core.models import PendingAtomResolution, PendingAtomSettlement, WriteFocus
from hivememory.core.models.pending import PendingAtomStatus
from hivememory.patchouli.application.memory_intent_submission_service import (
    MemoryIntentSubmissionService,
)
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.memory_generation.controller import MemoryGenerationTaskController
from hivememory.patchouli.control.memory_generation.coordinator import MemoryGenerationCoordinator
from hivememory.patchouli.control.memory_generation.models import MemoryGenerationResult
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.workspace.contracts import SubmitWriteIntentRequest
from hivememory.workspace.process.table import ProcessPhase
from tests.helpers.operations import OperationsHarness
from tests.helpers.workspace import make_identity_scope, make_runtime_scope


@pytest_asyncio.fixture
async def chain():
    """控制话题读取与生成完成两个边界，不依赖固定延时等待状态。"""
    harness = OperationsHarness()
    scope = make_identity_scope()
    submit = await harness.submitter(scope)
    bus = PatchouliBus()
    controller = MemoryGenerationTaskController(bus=bus)
    coordinator = MemoryGenerationCoordinator(bus=bus, recent_block_limit=5)
    service = MemoryIntentSubmissionService(bus=bus)
    reading, admit, generating, finish = (asyncio.Event() for _ in range(4))
    intents, specs = [], []

    async def topic(*_args, **_kwargs):
        reading.set()
        await admit.wait()
        return None

    async def execute(spec):
        specs.append(spec)
        generating.set()
        await finish.wait()
        return [
            MemoryGenerationResult(
                settlement=PendingAtomSettlement(
                    pending_alias=spec.pending_alias,
                    intent_id=spec.intent_id,
                    resolution=PendingAtomResolution.DISCARDED,
                )
            )
        ]

    async def dispatch(*, intent, identity_scope):
        intents.append(intent)
        return await service.submit_memory_intent(intent=intent, identity_scope=identity_scope)

    harness.bus.register(PatchouliRoutes.MEMORY_INTENT_SUBMIT, dispatch)
    bus.register(PatchouliLocalRoutes.TOPIC_GET, topic)
    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, coordinator.submit_active)
    bus.register(
        PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY, controller.submit_generation_many
    )
    bus.register(PatchouliLocalRoutes.GENERATION_EXECUTE_SPEC, execute)
    # 本边界验证结算载荷与句柄的对应；Bridge 的全局转发另由装配集成测试守护。
    bus.subscribe(PatchouliLocalEvents.PENDING_ATOM_SETTLED, harness.registry.on_settled)
    bus.subscribe(PatchouliLocalEvents.PENDING_ATOM_FAILED, harness.registry.on_failed)
    bus.subscribe(PatchouliLocalEvents.PENDING_ATOM_CANCELLED, harness.registry.on_cancelled)
    await controller.start()
    value = SimpleNamespace(
        harness=harness,
        scope=scope,
        submit=submit,
        bus=bus,
        controller=controller,
        service=service,
        reading=reading,
        admit=admit,
        generating=generating,
        finish=finish,
        intents=intents,
        specs=specs,
    )
    try:
        yield value
    finally:
        admit.set()
        finish.set()
        await controller.stop()
        harness.revoke()
        harness.runtime.close()


async def _write(chain):
    """以真实 MTP handler 提交，错误和 ACK 都由运行时生成。"""
    return await KoakumaRuntime().execute_mtp(
        '⟪ WRITE | * | content="需要立即生成的内容" ⟫',
        MTPExecutionContext(
            execution_policy=FrameExecutionPolicy.from_verbs(
                {"SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"}
            ),
            runtime_scope=make_runtime_scope(),
            submit_operation=chain.submit,
            language="zh",
        ),
    )


@pytest.mark.asyncio
async def test_ack_waits_for_admission_and_settlement_reaches_registered_handle(chain):
    """队列接纳前没有 ACK；载荷标识、派发话题与结算使用同一个句柄。"""
    task = asyncio.create_task(_write(chain))
    async with asyncio.timeout(2):
        await chain.reading.wait()
        assert task.done() is False
        intent = chain.intents[0]
        assert (
            chain.harness.registry.get(intent.pending_alias, chain.scope.workspace_identity).status
            == PendingAtomStatus.MATERIALIZING
        )
        chain.admit.set()
        result = await task
        await chain.generating.wait()
    assert (result.response_status, result.pending_alias) == ("ack", intent.pending_alias)
    (spec,) = chain.specs
    assert (spec.intent_id, spec.pending_alias, spec.topic_id, spec.from_actor) == (
        intent.intent_id,
        intent.pending_alias,
        "test-topic",
        chain.scope.actor_identity,
    )
    chain.finish.set()
    await chain.controller.wait_task(f"active:{intent.intent_id}", timeout=2)
    assert (
        chain.harness.registry.get(intent.pending_alias, chain.scope.workspace_identity).status
        == PendingAtomStatus.SETTLED
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_caller", [False, True])
async def test_revocation_during_dispatch_preserves_submission(chain, cancel_caller):
    """派发中撤销线程或重复取消调用方，都不能取消生成接纳。"""
    task = asyncio.create_task(_write(chain))
    async with asyncio.timeout(2):
        await chain.reading.wait()
        chain.harness.revoke()
        if cancel_caller:
            task.cancel()
            # 以事件循环回调交付第二次取消，不靠固定 sleep 推进。
            asyncio.get_running_loop().call_soon(task.cancel)
        chain.admit.set()
        if cancel_caller:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert (await task).response_status == "ack"
        await chain.generating.wait()
    intent = chain.intents[0]
    assert (
        chain.harness.registry.get(intent.pending_alias, chain.scope.workspace_identity).status
        == PendingAtomStatus.MATERIALIZING
    )
    chain.finish.set()
    await chain.controller.wait_task(f"active:{intent.intent_id}", timeout=2)
    assert (
        chain.harness.registry.get(intent.pending_alias, chain.scope.workspace_identity).status
        == PendingAtomStatus.SETTLED
    )


@pytest.mark.asyncio
async def test_stopped_queue_returns_structured_rejection_without_ack(chain):
    """队列明确拒绝时，MTP 报错且同步标记 FAILED，即使失败事件丢失。"""
    await chain.controller.stop()
    chain.bus.unsubscribe(
        PatchouliLocalEvents.PENDING_ATOM_FAILED, chain.harness.registry.on_failed
    )
    chain.admit.set()
    result = await _write(chain)
    intent = chain.intents[0]
    assert result.response_status == "error"
    assert "mtp.memory.intent_rejected" in result.formatted_response
    assert (
        chain.harness.registry.get(intent.pending_alias, chain.scope.workspace_identity).status
        == PendingAtomStatus.FAILED
    )
    assert chain.specs == []


@pytest.mark.asyncio
async def test_lost_admission_response_keeps_handle_until_later_settlement(chain):
    """队列已接纳但应答丢失时返回 unknown，随后仍可正常结算。"""

    async def lost_response(*, intent, identity_scope):
        chain.intents.append(intent)
        await chain.service.submit_memory_intent(intent=intent, identity_scope=identity_scope)
        raise OSError("lost response")

    chain.harness.bus.register(PatchouliRoutes.MEMORY_INTENT_SUBMIT, lost_response)
    chain.admit.set()
    result = await _write(chain)
    intent = chain.intents[0]
    assert result.response_status == "error"
    assert "mtp.memory.intent_unknown" in result.formatted_response
    assert (
        chain.harness.registry.get(intent.pending_alias, chain.scope.workspace_identity).status
        == PendingAtomStatus.MATERIALIZING
    )
    chain.finish.set()
    await chain.controller.wait_task(f"active:{intent.intent_id}", timeout=2)
    assert (
        chain.harness.registry.get(intent.pending_alias, chain.scope.workspace_identity).status
        == PendingAtomStatus.SETTLED
    )


@pytest.mark.asyncio
async def test_unknown_dispatch_logs_root_cause_hidden_from_agent(chain, caplog):
    """Agent 只得到 unknown 错误码，派发失败的原始异常必须留在日志中可诊断。"""
    failure = RuntimeError("patchouli submit route crashed")

    async def crash(*, intent, identity_scope):
        chain.intents.append(intent)
        raise failure

    chain.harness.bus.register(PatchouliRoutes.MEMORY_INTENT_SUBMIT, crash)
    with caplog.at_level(logging.ERROR, logger="hivememory.workspace.capability.actor.memory"):
        result = await _write(chain)
    assert "mtp.memory.intent_unknown" in result.formatted_response
    assert "patchouli submit route crashed" not in result.formatted_response
    logged = [
        record
        for record in caplog.records
        if record.name == "hivememory.workspace.capability.actor.memory"
    ]
    assert [(record.levelno, record.exc_info[1]) for record in logged] == [(logging.ERROR, failure)]
    assert chain.intents[0].pending_alias in logged[0].getMessage()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", [ProcessPhase.PREPARE, ProcessPhase.FINALIZE])
async def test_non_actor_phase_rejects_before_registration(chain, phase):
    """有效线程仍不能越过进程阶段检查，拒绝发生在登记和公共派发之前。"""
    composition = chain.harness.compositions["test_run"][0]
    composition.process_table.get("test_run").record.enter_phase(phase)
    with pytest.raises(ScopeRequiredError, match="Actor"):
        await chain.submit(SubmitWriteIntentRequest(WriteFocus(content="禁止派发")))
    assert chain.harness.registry.size == 0
    assert chain.intents == []


@pytest.mark.asyncio
async def test_invalid_payload_is_rejected_without_stranding_a_handle(chain):
    """操作请求绕过 MTP 参数检查时，本地校验失败也必须是确定性拒绝。"""
    from hivememory.workspace.contracts.errors import IntentDispatchRejectedError

    with pytest.raises(IntentDispatchRejectedError) as exc_info:
        await chain.submit(SubmitWriteIntentRequest(WriteFocus(content="")))
    alias = exc_info.value.details["pending_alias"]
    assert (
        chain.harness.registry.get(alias, chain.scope.workspace_identity).status
        == PendingAtomStatus.FAILED
    )
    assert chain.intents == []


@pytest.mark.asyncio
async def test_update_base_disappearing_before_generation_returns_rejection(chain):
    """workspace 冷读成功后基础原子被删除，生成侧拒绝，agent 不会得到 ACK。"""
    from hivememory.core.models import IndexLayer, MemoryAtom, MemoryType, PayloadLayer
    from tests.helpers.memory import make_memory_metadata

    chain.harness.memories["base"] = MemoryAtom(
        meta=make_memory_metadata(source_agent_id="test_agent", user_id="test_user"),
        index=IndexLayer(alias="base", title="基础", memory_type=MemoryType.FACT),
        payload=PayloadLayer(content="旧内容"),
    )

    async def missing(*_args, **_kwargs):
        return None

    chain.bus.register(PatchouliLocalRoutes.MEMORY_GET, missing)
    chain.admit.set()
    result = await KoakumaRuntime().execute_mtp(
        '⟪ UPDATE | base | instruction="更新内容" ⟫',
        MTPExecutionContext(
            execution_policy=FrameExecutionPolicy.from_verbs(
                {"SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"}
            ),
            runtime_scope=make_runtime_scope(),
            submit_operation=chain.submit,
            language="zh",
        ),
    )
    assert result.response_status == "error"
    assert "mtp.memory.intent_rejected" in result.formatted_response
    intent = chain.intents[0]
    assert (
        chain.harness.registry.get(intent.pending_alias, chain.scope.workspace_identity).status
        == PendingAtomStatus.FAILED
    )
    assert chain.specs == []
