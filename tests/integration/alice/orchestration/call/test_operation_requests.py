"""CALL 子帧与真实 workspace 操作入口协作，验证被调用方的独立授权与记账。"""

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
from hivememory.alice.orchestration.call.coordinator import DispatchCallee
from hivememory.alice.orchestration.frames import FrameFactory
from hivememory.alice.orchestration.output import NullAgentRunOutput
from hivememory.alice.orchestration.session import RunSession
from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.alice import AliceConfig
from hivememory.config.memory_compiler import MemoryCompilerConfig
from hivememory.core.access import WorkspaceOperation
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    ActorIdentity,
    AgentProfile,
    ResolvedAgentProfile,
)
from hivememory.core.mtp import MTPCallRequest
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.prompts.assembler import AgentPromptAssembler
from tests.helpers.operations import OperationsHarness
from tests.helpers.workspace import make_identity_scope, make_runtime_scope


@pytest.mark.asyncio
async def test_callee_frame_uses_callee_verbs_without_call():
    """子帧按被调用方的操作快照翻译动词并去掉 CALL，不继承父帧 READ。"""
    harness = OperationsHarness(GlobalSystemBus())
    profile = AgentProfile(agent_id="callee", persona="被调用人设", language="en")

    async def profile_route(_alias, **_kwargs):
        return ResolvedAgentProfile(profile=profile)

    harness.bus.register(PatchouliRoutes.GET_AGENT_PROFILE, profile_route)
    scope = make_runtime_scope(agent_id="parent", run_id="process-inherit")
    caller = ExecutionFrame(
        runtime_scope=scope,
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[],
        topic_id="topic-parent",
        execution_policy=FrameExecutionPolicy.from_verbs({"READ", "RUN", "CALL"}),
        submit_operation=await harness.submitter(
            make_identity_scope(agent_id="parent"),
            scope.run_id,
            allowed_operations={WorkspaceOperation.MEMORY_READ, WorkspaceOperation.AGENT_CALL},
            callee_operations={WorkspaceOperation.MEMORY_WRITE, WorkspaceOperation.AGENT_CALL},
        ),
    )
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
    session = RunSession(agent_run_id=scope.run_id, process_id="process-1")
    session.register_root_frame(caller)

    transition = await coordinator.begin_call(
        caller,
        FrameExecutionResult(
            status=FrameExecutionStatus.SUSPENDED,
            call_request=MTPCallRequest(target_alias="callee", task="按人设读取"),
            suspend_action_id="call-inherit",
        ),
        session=session,
        run_output=NullAgentRunOutput(),
    )

    assert isinstance(transition, DispatchCallee)
    child = transition.frame
    assert child.execution_policy.permitted_verbs == {"WRITE", "RUN"}
    assert child.submit_operation is not caller.submit_operation
    prompt = child.working_history[0]["content"]
    assert "被调用人设" in prompt and "WRITE" in prompt
    assert "CALL" not in prompt and "READ" not in prompt and "SEARCH" not in prompt
    assert "sys_write_file" in prompt


@pytest.mark.asyncio
async def test_callee_submits_with_own_thread_context_and_keeps_callee_as_intent_actor() -> None:
    """真实子帧取得独立提交函数，WRITE 归被调用方且仍绑定同一进程，父帧可回读。"""
    harness = OperationsHarness()
    scope = make_runtime_scope(agent_id="parent", run_id="process-parent")
    identity_scope = make_identity_scope(agent_id="parent")
    submit_operation = await harness.submitter(identity_scope, "process-parent")
    caller = ExecutionFrame(
        execution_policy=FrameExecutionPolicy.from_verbs(
            {"SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"}
        ),
        runtime_scope=scope,
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[{"role": "user", "content": "委托写入"}],
        topic_id="topic-parent",
        submit_operation=submit_operation,
    )
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
    session = RunSession(agent_run_id=scope.run_id, process_id="process-1")
    session.register_root_frame(caller)
    suspension = FrameExecutionResult(
        status=FrameExecutionStatus.SUSPENDED,
        call_request=MTPCallRequest(target_alias="omni_doll", task="写入共享草稿"),
        suspend_action_id="call-1",
    )

    transition = await coordinator.begin_call(
        caller, suspension, session=session, run_output=NullAgentRunOutput()
    )

    assert isinstance(transition, DispatchCallee)
    child = transition.frame
    assert child.submit_operation is not caller.submit_operation
    koakuma = KoakumaRuntime()
    write = await koakuma.execute_mtp(
        '⟪ WRITE | * | content="子帧创建的共享草稿" ⟫',
        MTPExecutionContext(
            execution_policy=FrameExecutionPolicy.from_verbs(
                {"SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"}
            ),
            runtime_scope=child.runtime_scope,
            submit_operation=child.submit_operation,
        ),
    )
    assert write.response_status == "ack"
    pending = harness.registry.get(write.pending_alias, identity_scope.workspace_identity)
    assert pending.from_actor == ActorIdentity(user_id="test_user", agent_id="omni_doll")
    assert pending.process_id == "process-parent"
    read = await koakuma.execute_mtp(
        f"⟪ READ | {write.pending_alias} | ⟫",
        MTPExecutionContext(
            execution_policy=FrameExecutionPolicy.from_verbs(
                {"SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"}
            ),
            runtime_scope=scope,
            submit_operation=caller.submit_operation,
        ),
    )
    assert read.response_status == "success"
    assert "子帧创建的共享草稿" in read.response_content


@pytest.mark.asyncio
async def test_child_reads_only_callee_private_memory_and_records_callee_citations():
    """同一缓存中的 PRIVATE 原子仍按每次读取的子线程 actor 过滤。"""
    from hivememory.core.models import IndexLayer, MemoryAtom, MemoryType, PayloadLayer
    from hivememory.workspace.contracts import DeriveCallThreadRequest, ResolveReferencesRequest
    from tests.helpers.memory import make_memory_metadata

    harness = OperationsHarness()
    for owner in ("parent", "omni_doll"):
        harness.memories[owner] = MemoryAtom(
            meta=make_memory_metadata(
                user_id="test_user", source_agent_id=owner, visibility="PRIVATE"
            ),
            index=IndexLayer(alias=owner, title=owner, memory_type=MemoryType.FACT),
            payload=PayloadLayer(content=f"{owner} 的私有正文"),
        )
    submit = await harness.submitter(make_identity_scope(agent_id="parent"))
    try:
        parent_reads = await submit(ResolveReferencesRequest(("parent", "omni_doll")))
        child = await submit.derive_call_thread(DeriveCallThreadRequest())
        child_reads = await child.submit_operation(
            ResolveReferencesRequest(("parent", "omni_doll"))
        )
        assert [item.kind for item in parent_reads] == ["atom", "not_found"]
        assert [item.kind for item in child_reads] == ["not_found", "atom"]
        assert child_reads[1].atom.payload.content == "omni_doll 的私有正文"
        assert [item["identity_scope"].actor_identity.agent_id for item in harness.citations] == [
            "parent",
            "omni_doll",
        ]
    finally:
        harness.revoke()
        harness.runtime.close()


@pytest.mark.asyncio
async def test_callee_permissions_do_not_inherit_caller_read_and_cannot_recursively_derive():
    """调用方的 READ 不授予子线程；即使被调用方拥有 CALL，进程仍禁止递归派生。"""
    from hivememory.core.errors import OperationDeniedError, ScopeRequiredError
    from hivememory.workspace.contracts import DeriveCallThreadRequest, ResolveReferencesRequest

    harness = OperationsHarness()
    submit = await harness.submitter(
        make_identity_scope(agent_id="parent"),
        allowed_operations={WorkspaceOperation.AGENT_CALL, WorkspaceOperation.MEMORY_READ},
        callee_operations={WorkspaceOperation.AGENT_CALL},
    )
    try:
        child = await submit.derive_call_thread(DeriveCallThreadRequest())
        with pytest.raises(OperationDeniedError) as denied:
            await child.submit_operation(ResolveReferencesRequest(("missing",)))
        assert denied.value.details["operation"] == "memory.read"
        with pytest.raises(ScopeRequiredError, match="主线程"):
            await child.submit_operation.derive_call_thread(DeriveCallThreadRequest())
        assert harness.runtime.stats()["cold_reads"] == 2
    finally:
        harness.revoke()
        harness.runtime.close()


@pytest.mark.asyncio
async def test_generic_submission_cannot_deliver_child_context():
    """普通提交拒绝派生请求，既不读取图纸也不交付含凭据的结果。"""
    from hivememory.workspace.contracts import DeriveCallThreadRequest

    harness = OperationsHarness()
    submit = await harness.submitter(make_identity_scope(agent_id="parent"))
    try:
        with pytest.raises(TypeError, match="derive_call_thread"):
            await submit(DeriveCallThreadRequest())
        assert harness.runtime.stats()["cold_reads"] == 0
    finally:
        harness.revoke()
        harness.runtime.close()
