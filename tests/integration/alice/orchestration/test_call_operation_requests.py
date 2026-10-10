"""CALL 子帧与真实 workspace 操作入口协作，验证主线程凭据的继承。"""

from __future__ import annotations

import pytest

from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
    MTPExecutionContext,
)
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.alice.orchestration.frame_factory import FrameFactory
from hivememory.alice.orchestration.run_session import RunSession
from hivememory.alice.orchestration.sub_agent import CallContextProvider, CallCoordinator
from hivememory.alice.orchestration.sub_agent.call_coordinator import DispatchCallee
from hivememory.alice.runtime.core import AliceRuntime
from hivememory.config.alice import AliceConfig
from hivememory.config.memory_compiler import MemoryCompilerConfig
from hivememory.core.models import OMNI_DOLL_PROFILE
from hivememory.core.mtp import MTPCallRequest
from hivememory.prompts.assembler import AgentPromptAssembler
from tests.helpers.operations import OperationsHarness
from tests.helpers.workspace import make_identity_scope, make_runtime_scope


@pytest.mark.asyncio
async def test_callee_submits_with_parent_credential_and_keeps_parent_as_intent_actor() -> None:
    """真实创建的子帧继承提交函数，WRITE 意图仍归主线程发起者，父帧可以回读。"""
    harness = OperationsHarness()
    scope = make_runtime_scope(agent_id="parent", run_id="process-parent")
    identity_scope = make_identity_scope(agent_id="parent")
    submit_operation = await harness.submitter(identity_scope, "process-parent")
    caller = ExecutionFrame(
        runtime_scope=scope,
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[{"role": "user", "content": "委托写入"}],
        topic_id="topic-parent",
        submit_operation=submit_operation,
    )
    config = AliceConfig()
    runtime = AliceRuntime(
        alice_config=config,
        memory_compiler_config=MemoryCompilerConfig(),
    )
    coordinator = CallCoordinator(
        runtime.agent_runtime,
        CallContextProvider(),
        frame_factory=FrameFactory(),
        prompt_assembler=AgentPromptAssembler(config.koakuma),
    )
    session = RunSession(agent_run_id=scope.run_id)
    session.register_root_frame(caller)
    suspension = FrameExecutionResult(
        status=FrameExecutionStatus.SUSPENDED,
        call_request=MTPCallRequest(target_alias="omni_doll", task="写入共享草稿"),
        suspend_action_id="call-1",
    )

    transition = await coordinator.begin_call(caller, suspension, session=session)

    assert isinstance(transition, DispatchCallee)
    child = transition.frame
    assert child.submit_operation is caller.submit_operation
    koakuma = KoakumaRuntime()
    write = await koakuma.execute_mtp(
        '⟪ WRITE | * | content="子帧创建的共享草稿" ⟫',
        MTPExecutionContext(
            runtime_scope=child.runtime_scope,
            submit_operation=child.submit_operation,
        ),
    )
    assert write.response_status == "ack"
    pending = harness.registry.get(write.pending_alias, identity_scope.workspace_identity)
    assert pending.from_actor == identity_scope.actor_identity
    assert pending.process_id == "process-parent"
    read = await koakuma.execute_mtp(
        f"⟪ READ | {write.pending_alias} | ⟫",
        MTPExecutionContext(runtime_scope=scope, submit_operation=caller.submit_operation),
    )
    assert read.response_status == "success"
    assert "子帧创建的共享草稿" in read.response_content
