"""
AgentRunService 单元测试（unit 保留集）

真实装配链协作测试已迁移至 tests/integration/alice/application/
test_agent_run_service.py。本文件保留防御路径测试：
- test_run_agent_stream_without_executor_terminal_fails_cleanly
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hivememory.agent_runtime.models import FrameExecutionResult, FrameExecutionStatus
from hivememory.alice.application.agent_run_service import AgentRunService
from hivememory.alice.orchestration.frame_factory import FrameFactory
from hivememory.alice.orchestration.sub_agent import CallContextProvider, CallCoordinator
from hivememory.alice.runtime.core import AliceRuntime
from hivememory.alice.runtime.runtime_events import AgentRunEventEmitter
from hivememory.alice.runtime.streaming import AgentRunStreamAdapter
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
)
from hivememory.core.protocol.models import AgentRunContext, RetrievalResponse
from hivememory.prompts.assembler import AgentPromptAssembler
from hivememory.system.config import HiveMemoryConfig
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.events import NullRuntimeEventSink, RecordingRuntimeEventSink
from hivememory.system.runtime.publisher import RuntimeEventPublisher
from tests.helpers.workspace import make_access_context
from tests.helpers.memory import make_memory_metadata


def _build_memory_atom() -> MemoryAtom:
    return MemoryAtom(
        meta=make_memory_metadata(
            source_agent_id="agent-1",
            user_id="u1",
            confidence_score=0.9,
        ),
        index=IndexLayer(
            title="test memory",
            summary="summary text",
            tags=["tag"],
            memory_type=MemoryType.FACT,
            alias="mem_alias",
        ),
        payload=PayloadLayer(content="memory content"),
    )


def _build_agent_run_context(memory: MemoryAtom) -> AgentRunContext:
    return AgentRunContext(
        identity_scope=make_access_context(user_id="u1", agent_id="omni_doll"),
        interaction_id="test-interaction",
        topic_id="topic_1",
        user_message="hello",
        topic_context=None,
        retrieval_result=RetrievalResponse(memories=[memory]),
        memory_context="ctx",
        agent_profile=OMNI_DOLL_PROFILE,
        storage_available=True,
    )


def _build_service(*, runtime_events=None) -> tuple[AliceRuntime, AgentRunService]:
    config = HiveMemoryConfig()
    runtime = AliceRuntime(
        alice_config=config.alice,
        memory_compiler_config=config.memory_compiler,
    )
    frame_factory = FrameFactory()
    prompt_assembler = AgentPromptAssembler(config.alice.koakuma)
    coordinator = CallCoordinator(
        runtime.agent_runtime,
        CallContextProvider(runtime.profile_resolver, runtime.alias_resolver),
        frame_factory=frame_factory,
        prompt_assembler=prompt_assembler,
    )
    service = AgentRunService(
        agent_runtime=runtime.agent_runtime,
        call_coordinator=coordinator,
        frame_factory=frame_factory,
        prompt_assembler=prompt_assembler,
        atom_cache=runtime.atom_cache,
        stream_adapter=AgentRunStreamAdapter(),
        agent_run_events=AgentRunEventEmitter(
            RuntimeEventPublisher(runtime_events or NullRuntimeEventSink())
        ),
    )
    return runtime, service


@pytest.mark.asyncio
async def test_run_agent_stream_without_executor_terminal_fails_cleanly():
    recorder = RecordingRuntimeEventSink()
    _runtime, service = _build_service(runtime_events=recorder)
    context = _build_agent_run_context(_build_memory_atom())
    executor = MagicMock()
    executor.run = AsyncMock(
        return_value=FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)
    )
    executor.terminal_result = None
    with patch(
        "hivememory.alice.application.agent_run_service.RunExecutor",
        return_value=executor,
    ):
        with pytest.raises(RuntimeError, match="ended without done"):
            async for _ in service.run_agent_stream(context):
                pass

    assert recorder.events[-1].event_type == RuntimeEventType.AGENT_RUN_FAILED
    assert recorder.events[-1].message == "Agent stream ended without done event."
