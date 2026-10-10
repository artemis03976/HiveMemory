"""CALL 与 workspace Profile 能力、真实管理变更和读取缓存失效的协作。"""

from __future__ import annotations

import pytest

from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
)
from hivememory.alice.orchestration.frame_factory import FrameFactory
from hivememory.alice.orchestration.run_session import RunSession
from hivememory.alice.orchestration.sub_agent import CallContextProvider, CallCoordinator
from hivememory.alice.orchestration.sub_agent.call_coordinator import DispatchCallee, ResumeCaller
from hivememory.alice.runtime.core import AliceRuntime
from hivememory.config.alice import AliceConfig
from hivememory.config.memory_compiler import MemoryCompilerConfig
from hivememory.core.access import WorkspaceOperation
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    TurnEvent,
)
from hivememory.core.mtp import MTPCallRequest
from hivememory.prompts.assembler import AgentPromptAssembler
from hivememory.workspace.capability.operations import WorkspaceOperationEntry
from hivememory.workspace.contracts import OperationRequest
from hivememory.workspace.credentials import ExecutionCredentialRegistry
from tests.helpers.memory import make_memory_metadata
from tests.helpers.operations import OperationsHarness
from tests.helpers.workspace import make_identity_scope, make_runtime_scope


def _caller(submit_operation, *, agent_id="parent") -> ExecutionFrame:
    """创建已产生 CALL suspension 的调用方，保持真实响应回填入口。"""
    caller = ExecutionFrame(
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
    runtime = AliceRuntime(config, MemoryCompilerConfig())
    coordinator = CallCoordinator(
        runtime.agent_runtime,
        CallContextProvider(),
        frame_factory=FrameFactory(),
        prompt_assembler=AgentPromptAssembler(config.koakuma),
    )
    session = RunSession(agent_run_id=caller.runtime_scope.run_id)
    session.register_root_frame(caller)
    suspension = FrameExecutionResult(
        status=FrameExecutionStatus.SUSPENDED,
        call_request=MTPCallRequest(
            target_alias=target_alias, task="总结资料", context_refs=list(context_refs)
        ),
        suspend_action_id="call-1",
    )
    return await coordinator.begin_call(caller, suspension, session=session), session


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
    """操作入口与管理服务共用同一读取视图，凭据绑定主线程身份。"""
    credentials = ExecutionCredentialRegistry()
    entry = WorkspaceOperationEntry(
        chain.memories,
        agent=chain.profiles,
        credential_registry=credentials,
        intent_registry=chain.runtime.intents,
    )
    workspace = make_identity_scope(user_id="u1", agent_id="reader").workspace_identity
    credential = credentials.issue(
        access=chain.reader, target_workspace=workspace, process_id="call-process"
    )

    async def submit_operation[R](request: OperationRequest[R]) -> R:
        return await entry.execute(request, credential=credential)

    return submit_operation, workspace


@pytest.mark.asyncio
async def test_call_reloads_profile_after_management_updates_source_atom(profile_chain):
    """管理更新发布 canonical 事件后，下一次 CALL 使用新模型和人设。"""
    atom = _profile_atom()
    await profile_chain.store.upsert(atom)
    submit_operation, workspace = await _profile_submitter(profile_chain)
    first, _ = await _begin_call(_caller(submit_operation, agent_id="reader"), "custom_agent")
    assert isinstance(first, DispatchCallee)
    assert (first.frame.agent_profile.model_name, first.frame.agent_profile.persona) == (
        "old-model",
        "旧的人设",
    )

    await profile_chain.memories.update_memory(
        atom.id,
        content="更新后的人设",
        agent_config={"model_name": "new-model"},
        target_workspace=workspace,
        access=profile_chain.manager,
    )
    second, _ = await _begin_call(_caller(submit_operation, agent_id="reader"), "custom_agent")

    assert isinstance(second, DispatchCallee)
    assert (second.frame.agent_profile.model_name, second.frame.agent_profile.persona) == (
        "new-model",
        "更新后的人设",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ["", "default", "omni_doll"])
async def test_builtin_call_uses_workspace_profile_read_and_keeps_builtin_configuration(
    profile_chain, alias
):
    """未指定与显式内置 alias 经真实 backing 解析，仍交付同一能力配置。"""
    submit_operation, _ = await _profile_submitter(profile_chain)
    transition, _ = await _begin_call(_caller(submit_operation, agent_id="reader"), alias)

    assert isinstance(transition, DispatchCallee)
    assert transition.frame.agent_profile == OMNI_DOLL_PROFILE
    assert profile_chain.runtime.stats()["profile_size"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ["", "default", "omni_doll", "custom_agent"])
async def test_call_without_profile_read_returns_permission_error_without_dispatch(alias):
    """缺少 profile.read 的凭据无法执行 CALL，内置 Profile 同样不能旁路。"""
    harness = OperationsHarness()
    scope = make_identity_scope(agent_id="parent")
    submit_operation = await harness.submitter(
        scope,
        "call-process",
        allowed_operations={WorkspaceOperation.RESOURCE_READ},
    )
    caller = _caller(submit_operation)

    transition, session = await _begin_call(caller, alias)

    assert isinstance(transition, ResumeCaller)
    assert set(session.frames) == {"caller"}
    response = caller.progress.turn_events[-1]
    assert response.status == "error"
    assert 'code="mtp.permission.denied"' in response.content
    assert (harness.runtime.stats()["cold_reads"], harness.registry.size) == (0, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "expected_code"),
    [
        ("missing", "mtp.alias.not_found"),
        ("private", "mtp.alias.not_found"),
        ("invalid", "mtp.argument.invalid"),
    ],
)
async def test_unusable_custom_profile_returns_preparation_error_without_fallback(
    profile_chain, scenario, expected_code
):
    """缺失、不可见与损坏的自定义图纸回填具体错误，不降级执行内置 Agent。"""
    if scenario != "missing":
        atom = _profile_atom(
            visibility="PRIVATE" if scenario == "private" else "PUBLIC",
            agent_config={"temperature": 9} if scenario == "invalid" else None,
        )
        await profile_chain.store.upsert(atom)
    submit_operation, _ = await _profile_submitter(profile_chain)
    caller = _caller(submit_operation, agent_id="reader")

    transition, session = await _begin_call(caller, "custom_agent")

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
    transition, _ = await _begin_call(
        _caller(submit_operation), "omni_doll", context_refs=["context_atom"]
    )

    assert isinstance(transition, DispatchCallee)
    assert "被共享的正式原子" in str(transition.frame.working_history)
    assert [citation["memory_id"] for citation in harness.citations] == [atom.id]
    assert [
        citation["identity_scope"].actor_identity.agent_id for citation in harness.citations
    ] == ["parent"]
