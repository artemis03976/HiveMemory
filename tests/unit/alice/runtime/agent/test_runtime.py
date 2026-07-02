from unittest.mock import MagicMock

from hivememory.alice.runtime.agent.runtime import AgentRuntime
from hivememory.agent_runtime.loop_executor import AgentLoopExecutor
from hivememory.system.config import HiveMemoryConfig


def test_agent_runtime_builds_engine_facade():
    """AgentRuntime 作为单 Agent 运行时门面，内部装配 loop_executor 引擎。"""
    config = HiveMemoryConfig()
    mtp_executor = MagicMock()

    runtime = AgentRuntime(
        mtp_executor=mtp_executor,
        alice_config=config.alice,
    )

    assert isinstance(runtime._loop_executor, AgentLoopExecutor)
    assert runtime._loop_executor._mtp_executor is mtp_executor
    assert runtime._max_iterations == config.alice.runtime.max_loop_iterations


def test_agent_runtime_accepts_injected_loop_executor():
    """门面支持注入预构建的 loop_executor（测试/高级装配 seam）。"""
    injected = MagicMock()

    runtime = AgentRuntime(
        mtp_executor=MagicMock(),
        alice_config=MagicMock(),
        loop_executor=injected,
    )

    assert runtime._loop_executor is injected
