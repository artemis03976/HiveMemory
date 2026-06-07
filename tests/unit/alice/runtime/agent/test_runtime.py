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
        config=config,
    )

    assert isinstance(runtime._loop_executor, AgentLoopExecutor)
    assert runtime._loop_executor._mtp_executor is mtp_executor
    # 迭代上限由门面从 config.agent_runtime 内部消化
    assert runtime._max_iterations == config.agent_runtime.max_loop_iterations


def test_agent_runtime_accepts_injected_loop_executor():
    """门面支持注入预构建的 loop_executor（测试/高级装配 seam）。"""
    config = HiveMemoryConfig()
    injected = MagicMock()

    runtime = AgentRuntime(
        mtp_executor=MagicMock(),
        config=config,
        loop_executor=injected,
    )

    assert runtime._loop_executor is injected
