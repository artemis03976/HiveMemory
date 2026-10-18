"""CALL 与 workspace Profile 能力、真实管理变更和读取缓存失效的协作。"""

from __future__ import annotations

import pytest

from hivememory.agent_runtime.frame import (
    ExecutionFrame,
    FrameExecutionPolicy,
    FrameExecutionResult,
    FrameExecutionStatus,
)
from hivememory.agent_runtime.mtp.context import MTPExecutionContext
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.agent_runtime.runtime import build_agent_runtime
from hivememory.alice.orchestration.call import CallContextProvider, CallCoordinator
from hivememory.alice.orchestration.call.coordinator import DispatchCallee, ResumeCaller
from hivememory.alice.orchestration.frames import FrameFactory
from hivememory.alice.orchestration.output import NullAgentRunOutput
from hivememory.alice.orchestration.session import RunSession
from hivememory.config.alice import AliceConfig
from hivememory.config.memory_compiler import MemoryCompilerConfig
from hivememory.core.access import WorkspaceOperation
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    AgentProfile,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    PendingAtomStatus,
    ResolvedAgentProfile,
    TurnEvent,
    WriteFocus,
)
from hivememory.core.mtp import MTPCallRequest
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.prompts.assembler import AgentPromptAssembler
from hivememory.workspace.capability.operations import WorkspaceOperationEntry
from hivememory.workspace.contracts import SubmitWriteIntentRequest, ThreadOperationSubmitter
from tests.helpers.memory import make_memory_metadata
from tests.helpers.operations import OperationsHarness
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_identity_scope,
    make_runtime_scope,
)


def _caller(submit_operation, *, agent_id="parent") -> ExecutionFrame:
    """创建已产生 CALL suspension 的调用方，保持真实响应回填入口。"""
    caller = ExecutionFrame(
        execution_policy=FrameExecutionPolicy.from_verbs(
            {"SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"}
        ),
        runtime_scope=make_runtime_scope(
            agent_id=agent_id, run_id="call-process", frame_id="caller"
        ),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[],
        topic_id="topic-parent",
        submit_operation=submit_operation,
    )
    caller.progress.turn_events.append(
        TurnEvent(
            kind="tool_call",
            sequence=0,
            role="assistant",
            content="<CALL>",
            action_id="call-1",
            tool_kind="CALL",
            tool_name="CALL",
            status="pending",
        )
    )
    caller.progress.sequence = 1
    return caller


async def _begin_call(caller, target_alias, *, context_refs=()):
    """运行真实 CALL 准备、frame 装配和失败回填，不触发外部 LLM。"""
    config = AliceConfig()
    runtime = build_agent_runtime(
        runtime_config=config.runtime,
        koakuma_config=config.koakuma,
        memory_compiler_config=MemoryCompilerConfig(),
    )
    coordinator = CallCoordinator(
        runtime,
        CallContextProvider(),
        frame_factory=FrameFactory(),
        prompt_assembler=AgentPromptAssembler(config.koakuma),
    )
    session = RunSession(agent_run_id=caller.runtime_scope.run_id, process_id="process-1")
    session.register_root_frame(caller)
    suspension = FrameExecutionResult(
        status=FrameExecutionStatus.SUSPENDED,
        call_request=MTPCallRequest(
            target_alias=target_alias, task="总结资料", context_refs=list(context_refs)
        ),
        suspend_action_id="call-1",
    )
    return (
        await coordinator.begin_call(
            caller, suspension, session=session, run_output=NullAgentRunOutput()
        ),
        session,
        coordinator,
    )


def _profile_atom(*, visibility="PUBLIC", agent_config=None) -> MemoryAtom:
    """Profile 源原子经过真实读取、policy 校验与 AgentProfile 解析。"""
    return MemoryAtom(
        meta=make_memory_metadata(source_agent_id="writer", user_id="u1", visibility=visibility),
        index=IndexLayer(
            title="CALL 图纸",
            summary="目标 Agent 配置",
            tags=["profile"],
            alias="custom_agent",
            memory_type=MemoryType.AGENT_PROFILE,
        ),
        payload=PayloadLayer(
            content="旧的人设",
            agent_config={"model_name": "old-model"} if agent_config is None else agent_config,
        ),
    )


async def _profile_submitter(chain):
    """操作入口与管理服务共用同一读取视图，线程 context 绑定主线程身份。"""
    entry = WorkspaceOperationEntry(
        chain.memories,
        agent=chain.profiles,
        operation_authorizer=chain.composition.authorizer,
    )
    workspace = make_identity_scope(user_id="u1", agent_id="reader").workspace_identity
    thread_access = await chain.composition.authenticate_thread(
        agent_id="reader", process_id="call-process", register_process=True
    )

    return ThreadOperationSubmitter(entry, thread_access), workspace


@pytest.mark.asyncio
async def test_call_reloads_profile_after_management_updates_source_atom(profile_chain):
    """管理更新发布 canonical 事件后，下一次 CALL 使用新模型和人设。"""
    atom = _profile_atom()
    await profile_chain.store.upsert(atom)
    submit_operation, workspace = await _profile_submitter(profile_chain)
    first, _, _ = await _begin_call(_caller(submit_operation, agent_id="reader"), "custom_agent")
    assert isinstance(first, DispatchCallee)
    assert (first.frame.agent_profile.model_name, first.frame.agent_profile.persona) == (
        "old-model",
        "旧的人设",
    )

    await profile_chain.root_memories.update_memory(
        atom.id,
        content="更新后的人设",
        agent_config={"model_name": "new-model"},
        target_workspace=workspace,
        access=profile_chain.manager,
    )
    second, _, _ = await _begin_call(_caller(submit_operation, agent_id="reader"), "custom_agent")

    assert isinstance(second, DispatchCallee)
    assert (second.frame.agent_profile.model_name, second.frame.agent_profile.persona) == (
        "new-model",
        "更新后的人设",
    )


@pytest.mark.asyncio
async def test_builtin_call_uses_workspace_profile_read_and_keeps_builtin_configuration(
    profile_chain,
):
    """显式内置 alias 经真实 backing 解析，交付内置能力配置且不进缓存。

    空 alias 与 default 的归一化由 Patchouli 的 Profile 读取单测覆盖。
    """
    submit_operation, _ = await _profile_submitter(profile_chain)
    transition, _, _ = await _begin_call(_caller(submit_operation, agent_id="reader"), "omni_doll")

    assert isinstance(transition, DispatchCallee)
    assert transition.frame.agent_profile == OMNI_DOLL_PROFILE
    assert profile_chain.runtime.stats()["profile_size"] == 0


@pytest.mark.asyncio
async def test_call_without_call_permission_returns_permission_error_without_dispatch():
    """只有 READ 许可不能执行 CALL，内置 Profile 同样受 agent.call 约束。

    agent.call 授权先于 Profile 解析，任何 alias 都在同一处被拒，取内置目标代表。
    """
    harness = OperationsHarness()
    scope = make_identity_scope(agent_id="parent")
    submit_operation = await harness.submitter(
        scope,
        "call-process",
        allowed_operations={WorkspaceOperation.MEMORY_READ},
    )
    caller = _caller(submit_operation)

    transition, session, _ = await _begin_call(caller, "omni_doll")

    assert isinstance(transition, ResumeCaller)
    assert set(session.frames) == {"caller"}
    response = caller.progress.turn_events[-1]
    assert response.status == "error"
    assert 'code="mtp.permission.denied"' in response.content
    assert (harness.runtime.stats()["cold_reads"], harness.registry.size) == (0, 0)


@pytest.mark.asyncio
async def test_unusable_custom_profile_returns_preparation_error_without_fallback(profile_chain):
    """调用方不可见的自定义图纸回填具体错误，不降级执行内置 Agent。

    缺失、不可见与配置损坏的完整矩阵归 Patchouli 的 Profile 读取单测；这里取
    经真实 adapter 可见性过滤的不可见图纸作代表。
    """
    expected_code = "mtp.alias.not_found"
    await profile_chain.store.upsert(_profile_atom(visibility="PRIVATE"))
    submit_operation, _ = await _profile_submitter(profile_chain)
    caller = _caller(submit_operation, agent_id="reader")

    transition, session, _ = await _begin_call(caller, "custom_agent")

    assert isinstance(transition, ResumeCaller)
    assert set(session.frames) == {"caller"}
    response = caller.progress.turn_events[-1]
    assert response.status == "error"
    assert f'code="{expected_code}"' in response.content
    assert profile_chain.runtime.stats()["profile_size"] == 0


@pytest.mark.asyncio
async def test_call_context_refs_record_citations_for_delivered_formal_atoms():
    """CALL 给子帧共享正式原子时，引用记录由真实 workspace 读取能力完成。"""
    harness = OperationsHarness()
    atom = MemoryAtom(
        meta=make_memory_metadata(source_agent_id="parent", user_id="test_user"),
        index=IndexLayer(
            title="共享资料",
            summary="资料摘要",
            tags=["context"],
            alias="context_atom",
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="被共享的正式原子"),
    )
    harness.memories["context_atom"] = atom
    submit_operation = await harness.submitter(make_identity_scope(agent_id="parent"))
    transition, _, _ = await _begin_call(
        _caller(submit_operation), "omni_doll", context_refs=["context_atom"]
    )

    assert isinstance(transition, DispatchCallee)
    assert "被共享的正式原子" in str(transition.frame.working_history)
    assert [citation["memory_id"] for citation in harness.citations] == [atom.id]
    assert [
        citation["identity_scope"].actor_identity.agent_id for citation in harness.citations
    ] == ["parent"]


@pytest.mark.asyncio
async def test_call_permission_without_read_dispatches_child_without_shared_memory():
    """只有 CALL 许可仍能创建子帧，缺少 READ 时 context_refs 不交付记忆或记录引用。"""
    harness = OperationsHarness()
    atom = MemoryAtom(
        meta=make_memory_metadata(source_agent_id="parent", user_id="test_user"),
        index=IndexLayer(
            title="需要 READ 许可的共享资料",
            alias="context_guarded",
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="没有 READ 就不能共享的正文"),
    )
    harness.memories[atom.get_alias()] = atom
    submit_operation = await harness.submitter(
        make_identity_scope(agent_id="parent"),
        "call-process",
        allowed_operations={WorkspaceOperation.AGENT_CALL},
    )
    caller = _caller(submit_operation)

    transition, session, _ = await _begin_call(caller, "omni_doll", context_refs=[atom.get_alias()])

    assert isinstance(transition, DispatchCallee)
    assert set(session.frames) == {"caller", transition.frame.runtime_scope.frame_id}
    assert transition.frame.agent_profile == OMNI_DOLL_PROFILE
    assert atom.payload.content not in str(transition.frame.working_history)
    assert atom.get_alias() not in str(transition.frame.working_history)
    assert harness.runtime.stats()["atom_size"] == 0
    assert harness.citations == []


@pytest.mark.asyncio
async def test_callee_can_write_with_its_own_permission_even_when_caller_cannot():
    """被调用方自己的访问记录允许 WRITE，调用方仅有 CALL 不限制子线程写入。"""
    harness = OperationsHarness()
    submit_operation = await harness.submitter(
        make_identity_scope(agent_id="parent"),
        "call-process",
        allowed_operations={WorkspaceOperation.AGENT_CALL},
    )
    caller = _caller(submit_operation)
    transition, _, _ = await _begin_call(caller, "omni_doll")
    assert isinstance(transition, DispatchCallee)
    child = transition.frame

    result = await KoakumaRuntime().execute_mtp(
        '⟪ WRITE | * | content="子帧使用自己的写入许可" ⟫',
        MTPExecutionContext(
            execution_policy=FrameExecutionPolicy.from_verbs(
                {"SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"}
            ),
            runtime_scope=child.runtime_scope,
            agent_profile=child.agent_profile,
            submit_operation=child.submit_operation,
        ),
    )

    assert child.submit_operation is not caller.submit_operation
    assert result.response_status == "ack"
    task = harness.registry.get(result.pending_alias, make_identity_scope().workspace_identity)
    assert task.from_actor.agent_id == "omni_doll"
    intent, scope = harness.admission.submissions[-1]
    assert (
        intent.pending_alias,
        intent.intent_id,
        intent.topic_id,
        scope.actor_identity.agent_id,
    ) == (result.pending_alias, task.intent_id, "test-topic", "omni_doll")


async def _profile_named_by_alias(alias, **_kwargs):
    """Profile 路由替身：自定义 alias 解析为同名 Agent 的图纸；内置图纸不经路由。"""
    return ResolvedAgentProfile(profile=AgentProfile(agent_id=alias))


@pytest.mark.asyncio
async def test_unadmitted_call_target_returns_structured_error_without_creating_child():
    """目标准入失败时只回填结构化权限错误，进程登记里没有子线程。

    缺失、禁用与保留 system 三种准入拒绝由认证一侧的子线程派生测试覆盖。
    """
    target = "omni_doll"
    access = make_access_composition([make_actor_access_record(agent_id="parent")])
    harness = OperationsHarness()
    harness.bus.register(PatchouliRoutes.GET_AGENT_PROFILE, _profile_named_by_alias)
    harness.backing.bus = harness.bus
    context = await access.authenticate_thread(
        agent_id="parent", process_id="call-process", register_process=True
    )
    submit = ThreadOperationSubmitter(
        harness.make_entry(access.authorizer, access.thread_spawner), context
    )
    caller = _caller(submit)
    try:
        transition, session, _ = await _begin_call(caller, target)
        assert isinstance(transition, ResumeCaller)
        assert set(session.frames) == {"caller"}
        response = caller.progress.turn_events[-1].content
        assert 'code="mtp.permission.denied"' in response
        assert "[Agent Unavailable]" in response
        assert set(access.process_table.get("call-process").working_set.thread_contexts) == {"main"}
    finally:
        access.gateway.revoke_all_contexts()
        harness.runtime.close()


@pytest.mark.asyncio
async def test_failed_child_preserves_dispatched_intents():
    """子帧失败不撤回已派发意图，主线程、其他进程和子线程写入均保留。

    子帧终态只决定 CALL 响应，不触及意图登记；响应映射另有单测，取 FAILED 代表。
    """
    status = FrameExecutionStatus.FAILED
    harness = OperationsHarness()
    submit = await harness.submitter(make_identity_scope(agent_id="parent"), "call-process")
    foreign = await harness.submitter(make_identity_scope(agent_id="parent"), "other-process")
    try:
        claimed = await submit(SubmitWriteIntentRequest(WriteFocus(content="已认领")))
        parent_pending = await submit(SubmitWriteIntentRequest(WriteFocus(content="父线程")))
        foreign_pending = await foreign(SubmitWriteIntentRequest(WriteFocus(content="另一进程")))
        caller = _caller(submit)
        transition, session, coordinator = await _begin_call(caller, "omni_doll")
        assert isinstance(transition, DispatchCallee)
        child = transition.frame
        child_pending = await child.submit_operation(
            SubmitWriteIntentRequest(WriteFocus(content="子线程"))
        )
        assert child_pending.from_actor.agent_id == "omni_doll"
        for pending in (child_pending, claimed, foreign_pending):
            child.add_harvested_alias(pending.pending_alias)
        await coordinator.complete_call(
            caller,
            FrameExecutionResult(
                status=FrameExecutionStatus.SUSPENDED,
                call_request=MTPCallRequest(target_alias="omni_doll", task="总结资料"),
                suspend_action_id="call-1",
            ),
            child,
            FrameExecutionResult(status=status),
            session=session,
            run_output=NullAgentRunOutput(),
        )
        assert [
            harness.registry.get(p.pending_alias, make_identity_scope().workspace_identity).status
            for p in (child_pending, parent_pending, foreign_pending, claimed)
        ] == [
            PendingAtomStatus.MATERIALIZING,
            PendingAtomStatus.MATERIALIZING,
            PendingAtomStatus.MATERIALIZING,
            PendingAtomStatus.MATERIALIZING,
        ]
        # CALL 返回并不提前撤销子线程，统一寿命仍以 CPU 执行结束为界。
        later = await child.submit_operation(
            SubmitWriteIntentRequest(WriteFocus(content="线程仍有效"))
        )
        assert later.from_actor.agent_id == "omni_doll"
        assert later.process_id == "call-process"
    finally:
        harness.revoke("call-process")
        harness.revoke("other-process")
        harness.runtime.close()
