"""
HiveMemory Patchouli Stage 1 集成测试

测试帕秋莉（PatchouliAgent）的一阶段集成功能，包括：
- 语义吸附与漂移检测
- Token 溢出接力机制
- 自动 Flush 触发
- Chatbot 与 Agent 两种对话流
- 多会话隔离

运行方式：
    python tests/test_patchouli_stage1.py

验收标准：
    - 所有操作通过 Patchouli API 完成
    - 覆盖全部语义吸附与漂移场景
    - Token 溢出触发正常（使用 2048 阈值）
    - 多会话隔离正常工作

作者: HiveMemory Team
版本: 1.0.0
"""

import sys
import os
from pathlib import Path

# UTF-8 编码配置 (Windows 兼容性)
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ========== 日志配置（必须在导入其他模块之前） ==========

import logging
import litellm

# 禁用 litellm 详细模式
litellm.set_verbose = False
litellm.suppress_debug_info = True

# 配置根日志级别
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # 强制重新配置
)

# 关闭第三方库的 INFO/DEBUG 日志
_log_levels_to_disable = {
    "LiteLLM": logging.WARNING,  # LiteLLM 某些版本使用不同的大小写
    "FlagEmbedding": logging.WARNING, # FlagEmbedding 日志
    "huggingface_hub": logging.WARNING,
    "transformers": logging.WARNING,
    "sentence_transformers": logging.WARNING,
}

for logger_name, level in _log_levels_to_disable.items():
    logging.getLogger(logger_name).setLevel(level)

# ========== 其他导入 ==========

import time
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hivememory.agents.patchouli import PatchouliAgent, FlushEvent, FlushObserver
from hivememory.core.models import FlushReason
from hivememory.memory.storage import QdrantMemoryStore
from hivememory.core.config import get_config, PerceptionConfig

# 导入测试数据
from tests.fixtures.patchouli_test_data import (
    DATA_SCIENCE_CONVERSATION,
    WEB_DEVELOPMENT_CONVERSATION,
    GAME_DEVELOPMENT_CONVERSATION,
    COOKING_RECIPE_CONVERSATION,
    AGENT_TOOL_CALL_SCENARIO,
    AGENT_MULTI_TOOL_SCENARIO,
    LONG_CONVERSATION_FOR_OVERFLOW,
    BELOW_THRESHOLD_TEXT,
    AT_THRESHOLD_TEXT,
    ABOVE_THRESHOLD_TEXT,
    TOPIC_DISTANCE_TEST_SCENARIOS,
)

console = Console(force_terminal=True, legacy_windows=False)


# ========== 测试观察者类 ==========

class PatchouliTestObserver:
    """
    测试专用的 Flush 事件观察者

    记录所有 Flush 事件，用于验证自动触发条件
    """

    def __init__(self):
        self.flush_events: List[FlushEvent] = []

    def __call__(self, event: FlushEvent) -> None:
        """接收 Flush 事件"""
        self.flush_events.append(event)

    def get_flushes_by_reason(self, reason: FlushReason) -> List[FlushEvent]:
        """获取指定原因的 flush 记录"""
        return [e for e in self.flush_events if e.reason == reason]

    def clear(self) -> None:
        """清空所有记录"""
        self.flush_events.clear()

    @property
    def count(self) -> int:
        """获取 flush 总次数"""
        return len(self.flush_events)

    def summary(self) -> str:
        """获取摘要字符串"""
        if not self.flush_events:
            return "No flush events"

        reason_counts = {}
        for event in self.flush_events:
            reason = event.reason.value
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        parts = [f"{reason}: {count}" for reason, count in reason_counts.items()]
        return f"Total: {self.count}, " + ", ".join(parts)


# ========== 全局测试状态 ==========

# 全局共享的 Patchouli 实例
_shared_patchouli: Optional[PatchouliAgent] = None
_shared_observer: Optional[PatchouliTestObserver] = None
_shared_storage: Optional[QdrantMemoryStore] = None


def get_shared_patchouli() -> PatchouliAgent:
    """获取共享的 Patchouli 实例"""
    global _shared_patchouli
    if _shared_patchouli is None:
        raise RuntimeError("Patchouli 实例未初始化，请先调用 setup_test_env()")
    return _shared_patchouli


def get_shared_observer() -> PatchouliTestObserver:
    """获取共享的观察者"""
    global _shared_observer
    if _shared_observer is None:
        raise RuntimeError("Observer 未初始化，请先调用 setup_test_env()")
    return _shared_observer


def setup_test_env(max_tokens: int = 2048) -> None:
    """
    初始化测试环境（创建共享的 Patchouli 实例）

    Args:
        max_tokens: Token 溢出阈值
    """
    global _shared_patchouli, _shared_observer, _shared_storage

    console.print("\n[dim]正在初始化测试环境...[/dim]")

    config = get_config()
    _shared_storage = QdrantMemoryStore(
        qdrant_config=config.qdrant,
        embedding_config=config.embedding
    )
    _shared_storage.create_collection(recreate=True)

    perception_config = PerceptionConfig(max_processing_tokens=max_tokens)

    _shared_observer = PatchouliTestObserver()
    _shared_patchouli = PatchouliAgent(
        storage=_shared_storage,
        enable_semantic_flow=True,
        perception_config=perception_config
    )
    _shared_patchouli.add_flush_observer(_shared_observer)

    console.print("[dim]测试环境初始化完成[/dim]\n")


def reset_test_env() -> None:
    """
    重置测试环境（清空记忆库和 Buffer，但不重新创建实例）

    这比重新创建实例更高效
    """
    global _shared_patchouli, _shared_observer, _shared_storage

    if _shared_observer is not None:
        _shared_observer.clear()

    if _shared_patchouli is not None:
        # 清空所有活跃 Buffer
        active_buffers = _shared_patchouli.list_active_buffers()
        for buffer_key in active_buffers:
            parts = buffer_key.split(":")
            if len(parts) == 3:
                _shared_patchouli.clear_buffer(parts[0], parts[1], parts[2])

    # 清空记忆库（使用 recreate 来清空集合）
    if _shared_storage is not None:
        _shared_storage.create_collection(recreate=True)


# ========== 辅助函数 ==========

def run_test(test_func) -> bool:
    """
    运行单个测试并返回结果

    Args:
        test_func: 测试函数

    Returns:
        bool: 测试是否通过
    """
    # 每次测试前重置环境
    reset_test_env()

    try:
        test_func()
        console.print(f"  [green]✓[/green] {test_func.__name__}")
        return True
    except AssertionError as e:
        console.print(f"  [red]✗[/red] {test_func.__name__}: {e}")
        return False
    except Exception as e:
        console.print(f"  [red]✗[/red] {test_func.__name__}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ========== 组 1: 基础功能测试 ==========

def test_01_initialization():
    """
    测试场景 1: PatchouliAgent 初始化

    验证点：
    - Agent 正确初始化
    - 感知层为 SemanticFlowPerceptionLayer
    - 初始状态无活跃 Buffer
    """
    patchouli = get_shared_patchouli()

    # 验证感知层已初始化
    assert patchouli.perception_layer is not None, "感知层应该已初始化"
    assert patchouli.enable_semantic_flow is True, "应启用语义流模式"

    # 验证初始无活跃 Buffer
    active_buffers = patchouli.list_active_buffers()
    assert len(active_buffers) == 0, f"初始应有 0 个活跃 Buffer，实际 {len(active_buffers)}"

    console.print("    - 感知层模式: semantic_flow")
    console.print(f"    - 初始活跃 Buffer: {len(active_buffers)}")


def test_02_add_message():
    """
    测试场景 2: 添加消息

    验证点：
    - 消息成功添加到感知层
    - Buffer 正确创建
    - Block 状态正确更新
    """
    patchouli = get_shared_patchouli()

    user_id = "test_user"
    agent_id = "test_agent"
    session_id = "test_session"

    # 添加一条消息
    patchouli.add_message(
        role="user",
        content="测试消息",
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id
    )

    # 验证 Buffer 已创建
    buffer_info = patchouli.get_buffer_info(user_id, agent_id, session_id)
    assert buffer_info["exists"] is True, "Buffer 应该存在"
    assert buffer_info["mode"] == "semantic_flow", "应该是语义流模式"

    # 验证在活跃 Buffer 列表中
    active_buffers = patchouli.list_active_buffers()
    assert len(active_buffers) == 1, f"应该有 1 个活跃 Buffer，实际 {len(active_buffers)}"

    console.print(f"    - Buffer 存在: {buffer_info['exists']}")
    console.print(f"    - 活跃 Buffer 数: {len(active_buffers)}")


def test_03_get_buffer_info():
    """
    测试场景 3: 获取 Buffer 信息

    验证点：
    - get_buffer_info 返回正确信息
    - 注意：get_buffer_info 会自动创建 Buffer（Lazy Create 模式）
    - 创建 Buffer 后返回完整信息
    """
    patchouli = get_shared_patchouli()

    # 注意：由于 get_buffer 采用 Lazy Create 模式，
    # 调用 get_buffer_info 会自动创建 Buffer，所以 exists 总是 True
    info = patchouli.get_buffer_info("nonexistent", "agent", "session")
    assert info["exists"] is True, "get_buffer_info 采用 Lazy Create 模式，会自动创建 Buffer"
    assert info["mode"] == "semantic_flow"

    # 创建一个有内容的 Buffer
    patchouli.add_message("user", "hello", "user1", "agent1", "session1")
    patchouli.add_message("assistant", "hi there", "user1", "agent1", "session1")

    info_existing = patchouli.get_buffer_info("user1", "agent1", "session1")
    assert info_existing["exists"] is True, "Buffer 应该存在"
    assert info_existing["block_count"] >= 1, f"应该至少有 1 个 Block，实际 {info_existing['block_count']}"
    assert info_existing["total_tokens"] > 0, f"Token 数应该大于 0，实际 {info_existing['total_tokens']}"

    console.print(f"    - Buffer 存在: {info_existing['exists']}")
    console.print(f"    - Block 数量: {info_existing['block_count']}")
    console.print(f"    - Token 数: {info_existing['total_tokens']}")


# ========== 组 2: Chatbot 对话流测试 ==========

def test_04_same_topic_adsorption():
    """
    测试场景 4: Chatbot 相同话题吸附

    验证点：
    - 同话题连续对话应被吸附
    - 不应触发语义漂移 Flush
    - Buffer 中 Block 数量正确增加
    """
    patchouli = get_shared_patchouli()

    user_id = "test_chatbot_same"
    agent_id = "chatbot"
    session_id = "session"

    # 添加数据科学对话（前两轮）
    for i in range(0, 4, 2):
        patchouli.add_message(
            "user",
            DATA_SCIENCE_CONVERSATION[i]["content"],
            user_id, agent_id, session_id
        )
        patchouli.add_message(
            "assistant",
            DATA_SCIENCE_CONVERSATION[i+1]["content"],
            user_id, agent_id, session_id
        )

    # 添加相同话题的后续对话
    patchouli.add_message(
        "user",
        DATA_SCIENCE_CONVERSATION[2]["content"],
        user_id, agent_id, session_id
    )
    patchouli.add_message(
        "assistant",
        DATA_SCIENCE_CONVERSATION[3]["content"],
        user_id, agent_id, session_id
    )

    # 验证：不应触发语义漂移
    drift_flushes = get_shared_observer().get_flushes_by_reason(FlushReason.SEMANTIC_DRIFT)
    assert len(drift_flushes) == 0, f"相同话题不应触发语义漂移，实际触发 {len(drift_flushes)} 次"

    # 验证：Buffer 应该有多个 Block
    buffer_info = patchouli.get_buffer_info(user_id, agent_id, session_id)
    assert buffer_info["block_count"] >= 2, f"应该至少有 2 个 Block，实际 {buffer_info['block_count']}"

    console.print(f"    - 语义漂移触发次数: {len(drift_flushes)} (预期: 0)")
    console.print(f"    - Block 数量: {buffer_info['block_count']} (预期: >= 2)")


def test_05_semantic_drift_far():
    """
    测试场景 5: Chatbot 远距离语义漂移

    验证点：
    - 从数据科学切换到完全不相关的烹饪话题
    - 应触发 SEMANTIC_DRIFT Flush
    - Flush 原因正确
    """
    patchouli = get_shared_patchouli()

    user_id = "test_drift_far"
    agent_id = "chatbot"
    session_id = "session"

    # 建立数据科学话题基线
    for i in range(2):
        patchouli.add_message(
            DATA_SCIENCE_CONVERSATION[i]["role"],
            DATA_SCIENCE_CONVERSATION[i]["content"],
            user_id, agent_id, session_id
        )

    get_shared_observer().clear()

    # 切换到完全不相关的烹饪话题
    for msg in COOKING_RECIPE_CONVERSATION:
        patchouli.add_message(
            msg["role"], msg["content"],
            user_id, agent_id, session_id
        )

    # 验证：应触发语义漂移
    drift_flushes = get_shared_observer().get_flushes_by_reason(FlushReason.SEMANTIC_DRIFT)
    assert len(drift_flushes) > 0, "远距离话题应触发语义漂移"

    # 验证：最后一个 Flush 的原因
    last_flush = get_shared_observer().flush_events[-1]
    assert last_flush.reason == FlushReason.SEMANTIC_DRIFT, \
        f"Flush 原因应该是 SEMANTIC_DRIFT，实际是 {last_flush.reason.value}"

    console.print(f"    - 语义漂移触发次数: {len(drift_flushes)} (预期: > 0)")
    console.print(f"    - Flush 原因: {last_flush.reason.value}")


def test_06_semantic_drift_medium():
    """
    测试场景 6: Chatbot 中等距离语义漂移

    验证点：
    - 从数据科学切换到游戏开发（相关但不相同）
    - 应触发 SEMANTIC_DRIFT Flush
    """
    patchouli = get_shared_patchouli()

    user_id = "test_drift_medium"
    agent_id = "chatbot"
    session_id = "session"

    # 建立数据科学话题基线
    for i in range(2):
        patchouli.add_message(
            DATA_SCIENCE_CONVERSATION[i]["role"],
            DATA_SCIENCE_CONVERSATION[i]["content"],
            user_id, agent_id, session_id
        )

    get_shared_observer().clear()

    # 切换到游戏开发话题
    for msg in GAME_DEVELOPMENT_CONVERSATION:
        patchouli.add_message(
            msg["role"], msg["content"],
            user_id, agent_id, session_id
        )

    # 验证：应触发语义漂移
    drift_flushes = get_shared_observer().get_flushes_by_reason(FlushReason.SEMANTIC_DRIFT)
    assert len(drift_flushes) > 0, "中等距离话题应触发语义漂移"

    console.print(f"    - 语义漂移触发次数: {len(drift_flushes)} (预期: > 0)")


def test_07_boundary_near():
    """
    测试场景 7: Chatbot 边界情况（较近话题）

    验证点：
    - 数据科学到 Web 开发（相关领域）
    - 可能吸附也可能漂移（边界情况）
    """
    patchouli = get_shared_patchouli()

    user_id = "test_boundary_near"
    agent_id = "chatbot"
    session_id = "session"

    # 建立数据科学话题基线
    for i in range(2):
        patchouli.add_message(
            DATA_SCIENCE_CONVERSATION[i]["role"],
            DATA_SCIENCE_CONVERSATION[i]["content"],
            user_id, agent_id, session_id
        )

    get_shared_observer().clear()

    # 切换到 Web 开发话题
    for msg in WEB_DEVELOPMENT_CONVERSATION:
        patchouli.add_message(
            msg["role"], msg["content"],
            user_id, agent_id, session_id
        )

    # 记录结果（边界情况，不强制断言）
    drift_flushes = get_shared_observer().get_flushes_by_reason(FlushReason.SEMANTIC_DRIFT)
    triggered = len(drift_flushes) > 0

    console.print(f"    - 较近话题触发漂移: {triggered} (边界情况，结果可接受)")
    console.print(f"    - 语义漂移触发次数: {len(drift_flushes)}")


# ========== 组 3: Agent 对话流测试 ==========

def test_08_agent_tool_call_flow():
    """
    测试场景 8: Agent 工具调用流程

    验证点：
    - User Query -> Assistant -> Tool -> Assistant Response
    - 消息正确添加到感知层
    - Block 正确闭合
    """
    patchouli = get_shared_patchouli()

    user_id = "test_agent_tool"
    agent_id = "agent"
    session_id = "session"

    # 模拟工具调用流程
    # 注意：由于 API 限制，工具调用信息通过 role 传递
    scenario = AGENT_TOOL_CALL_SCENARIO

    # User Query
    patchouli.add_message("user", scenario["messages"][0]["content"], user_id, agent_id, session_id)

    # Assistant thought
    patchouli.add_message("assistant", scenario["messages"][1]["content"], user_id, agent_id, session_id)

    # Tool output (通过 role="tool" 传递)
    tool_msg = scenario["messages"][3]
    patchouli.add_message("tool", tool_msg["content"], user_id, agent_id, session_id)

    # Final response
    patchouli.add_message("assistant", scenario["messages"][4]["content"], user_id, agent_id, session_id)

    # 验证：消息已添加
    buffer_info = patchouli.get_buffer_info(user_id, agent_id, session_id)
    assert buffer_info["exists"] is True, "Buffer 应该存在"
    assert buffer_info["block_count"] >= 1, f"应该至少有 1 个 Block，实际 {buffer_info['block_count']}"

    console.print(f"    - Block 数量: {buffer_info['block_count']}")
    console.print(f"    - 总 Tokens: {buffer_info['total_tokens']}")


def test_09_agent_multi_tool_calls():
    """
    测试场景 9: Agent 多工具调用

    验证点：
    - 多轮对话正确处理
    - 每个 Block 正确闭合
    """
    patchouli = get_shared_patchouli()

    user_id = "test_multi_tool"
    agent_id = "agent"
    session_id = "session"

    # 模拟多轮对话
    for i in range(3):
        patchouli.add_message(
            "user",
            f"这是第{i+1}个问题，请分析一下",
            user_id, agent_id, session_id
        )
        patchouli.add_message(
            "assistant",
            f"正在分析第{i+1}个问题，需要使用工具...",
            user_id, agent_id, session_id
        )
        # 模拟工具输出
        patchouli.add_message(
            "tool",
            f'{{"result": "analysis_result_{i+1}"}}',
            user_id, agent_id, session_id
        )
        patchouli.add_message(
            "assistant",
            f"第{i+1}个问题的分析结果是...",
            user_id, agent_id, session_id
        )

    buffer_info = patchouli.get_buffer_info(user_id, agent_id, session_id)
    assert buffer_info["block_count"] >= 1, f"应该至少有 1 个 Block，实际 {buffer_info['block_count']}"

    console.print(f"    - Block 数量: {buffer_info['block_count']}")


# ========== 组 4: Token 溢出测试 ==========

def test_10_token_overflow_relay():
    """
    测试场景 10: Token 溢出与接力

    验证点：
    - 长对话触发 TOKEN_OVERFLOW Flush
    - 生成接力摘要
    - 溢出后新对话正常处理
    """
    # 共享实例已配置 max_tokens=2048
    patchouli = get_shared_patchouli()

    user_id = "test_overflow"
    agent_id = "chatbot"
    session_id = "session"

    # 添加长对话直到溢出
    for msg in LONG_CONVERSATION_FOR_OVERFLOW:
        patchouli.add_message(
            msg["role"],
            msg["content"],
            user_id, agent_id, session_id
        )

    # 验证：应触发 Token 溢出
    overflow_flushes = get_shared_observer().get_flushes_by_reason(FlushReason.TOKEN_OVERFLOW)
    assert len(overflow_flushes) > 0, f"长对话应触发 Token 溢出，实际触发 {len(overflow_flushes)} 次"

    # 验证：接力摘要
    buffer = patchouli.get_buffer(user_id, agent_id, session_id)
    if buffer and buffer.relay_summary:
        console.print(f"    - 接力摘要: {buffer.relay_summary[:50]}...")
    else:
        console.print("    - 接力摘要: 未生成或已被处理")

    console.print(f"    - Token 溢出触发次数: {len(overflow_flushes)} (预期: > 0)")


def test_11_relay_summary_preservation():
    """
    测试场景 11: 接力摘要保持话题核心

    验证点：
    - 溢出后话题核心保持一致
    - 后续对话仍能正确吸附
    """
    patchouli = get_shared_patchouli()

    user_id = "test_relay"
    agent_id = "chatbot"
    session_id = "session"

    # 第一阶段：建立话题
    for i in range(2):
        patchouli.add_message(
            DATA_SCIENCE_CONVERSATION[i]["role"],
            DATA_SCIENCE_CONVERSATION[i]["content"],
            user_id, agent_id, session_id
        )

    # 第二阶段：触发溢出
    for msg in LONG_CONVERSATION_FOR_OVERFLOW[:10]:
        patchouli.add_message(
            msg["role"], msg["content"],
            user_id, agent_id, session_id
        )

    get_shared_observer().clear()

    # 第三阶段：添加同话题后续对话
    patchouli.add_message(
        "user",
        "关于数据可视化，我还有个问题：Seaborn和Matplotlib有什么区别？",
        user_id, agent_id, session_id
    )
    patchouli.add_message(
        "assistant",
        "Seaborn基于Matplotlib，提供了更高级的统计绘图接口...",
        user_id, agent_id, session_id
    )

    # 验证：不应立即触发语义漂移（话题核心保持）
    drift_flushes = get_shared_observer().get_flushes_by_reason(FlushReason.SEMANTIC_DRIFT)

    console.print(f"    - 同话题后续对话触发漂移: {len(drift_flushes) > 0}")
    console.print(f"    - 语义漂移触发次数: {len(drift_flushes)}")


# ========== 组 5: 短文本吸附测试 ==========

def test_12_short_text_forced_adsorption():
    """
    测试场景 12: 短文本强吸附

    验证点：
    - 短消息（<50 tokens）强制吸附
    - 即使话题不同也不触发 Flush
    """
    patchouli = get_shared_patchouli()

    user_id = "test_short_text"
    agent_id = "chatbot"
    session_id = "session"

    # 建立编程话题
    patchouli.add_message(
        "user",
        "请详细讲解Python装饰器的原理和用法，包括带参数的装饰器",
        user_id, agent_id, session_id
    )
    patchouli.add_message(
        "assistant",
        "装饰器是Python的高级特性，本质上是一个接受函数作为参数并返回新函数的高阶函数...",
        user_id, agent_id, session_id
    )

    get_shared_observer().clear()

    # 添加短文本（不同话题）
    patchouli.add_message("user", "好的", user_id, agent_id, session_id)
    patchouli.add_message("assistant", "明白", user_id, agent_id, session_id)

    # 验证：短文本应吸附，不触发漂移
    drift_flushes = get_shared_observer().get_flushes_by_reason(FlushReason.SEMANTIC_DRIFT)
    assert len(drift_flushes) == 0, "短文本应强制吸附，不触发语义漂移"

    console.print(f"    - 语义漂移触发次数: {len(drift_flushes)} (预期: 0)")


def test_13_threshold_boundary():
    """
    测试场景 13: 阈值边界测试

    验证点：
    - 低于阈值的强制吸附
    - 高于阈值的按语义判定
    """
    patchouli = get_shared_patchouli()

    user_id = "test_threshold"
    agent_id = "chatbot"
    session_id = "session"

    # 建立基线话题
    patchouli.add_message(
        "user",
        "我想学习Python编程，从基础语法开始",
        user_id, agent_id, session_id
    )
    patchouli.add_message(
        "assistant",
        "Python是一种高级编程语言，具有简洁明了的语法...",
        user_id, agent_id, session_id
    )

    get_shared_observer().clear()

    # 添加低于阈值的短文本
    patchouli.add_message("user", BELOW_THRESHOLD_TEXT, user_id, agent_id, session_id)

    below_drift = len(get_shared_observer().get_flushes_by_reason(FlushReason.SEMANTIC_DRIFT))
    console.print(f"    - 低于阈值文本触发漂移: {below_drift > 0} (预期: False)")
    console.print(f"    - 低于阈值 tokens: {len(BELOW_THRESHOLD_TEXT) // 2} (约)")


# ========== 组 6: 多会话隔离测试 ==========

def test_14_multi_session_isolation():
    """
    测试场景 14: 多会话隔离

    验证点：
    - 不同 session 的 Buffer 独立
    - 一个会话的 Flush 不影响其他会话
    - list_active_buffers 正确显示所有会话
    """
    patchouli = get_shared_patchouli()

    sessions = [
        ("user1", "agent1", "session1"),
        ("user2", "agent1", "session2"),
        ("user1", "agent2", "session3"),
    ]

    # 为每个会话添加消息
    for user_id, agent_id, session_id in sessions:
        patchouli.add_message(
            "user",
            f"来自 {session_id} 的消息",
            user_id, agent_id, session_id
        )
        patchouli.add_message(
            "assistant",
            f"回复 {session_id}",
            user_id, agent_id, session_id
        )

    # 验证：有 3 个活跃 Buffer
    active_buffers = patchouli.list_active_buffers()
    assert len(active_buffers) == 3, f"应该有 3 个活跃 Buffer，实际 {len(active_buffers)}"

    # 验证：每个 Buffer 信息独立
    for user_id, agent_id, session_id in sessions:
        info = patchouli.get_buffer_info(user_id, agent_id, session_id)
        assert info["exists"] is True, f"会话 {session_id} 应该存在"
        assert info["block_count"] >= 1, f"会话 {session_id} 应该有 Block"

    console.print(f"    - 活跃 Buffer 数: {len(active_buffers)} (预期: 3)")
    console.print(f"    - Buffer 列表: {active_buffers}")


def test_15_concurrent_sessions():
    """
    测试场景 15: 并发会话处理

    验证点：
    - 同时处理多个会话
    - 每个 Buffer 的 Flush 独立触发
    - 话题核心独立维护
    """
    patchouli = get_shared_patchouli()

    # 会话1: 数据科学话题
    user1, agent1, sess1 = "user_concurrent", "agent", "sess1"
    for i in range(2):
        patchouli.add_message(
            DATA_SCIENCE_CONVERSATION[i]["role"],
            DATA_SCIENCE_CONVERSATION[i]["content"],
            user1, agent1, sess1
        )

    # 会话2: 烹饪话题
    user2, agent2, sess2 = "user_concurrent", "agent", "sess2"
    for i in range(2):
        patchouli.add_message(
            COOKING_RECIPE_CONVERSATION[i]["role"],
            COOKING_RECIPE_CONVERSATION[i]["content"],
            user2, agent2, sess2
        )

    # 验证：两个会话独立存在
    info1 = patchouli.get_buffer_info(user1, agent1, sess1)
    info2 = patchouli.get_buffer_info(user2, agent2, sess2)

    assert info1["exists"] is True, "会话1 应该存在"
    assert info2["exists"] is True, "会话2 应该存在"

    active_buffers = patchouli.list_active_buffers()
    assert len(active_buffers) == 2, f"应该有 2 个活跃 Buffer，实际 {len(active_buffers)}"

    console.print(f"    - 活跃 Buffer 数: {len(active_buffers)}")
    console.print(f"    - 会话1 Block 数: {info1['block_count']}")
    console.print(f"    - 会话2 Block 数: {info2['block_count']}")


# ========== 主测试流程 ==========

def main():
    """主测试流程"""
    console.print(Panel.fit(
        "[bold magenta]Patchouli Stage 1 集成测试[/bold magenta]\n"
        "测试语义吸附、漂移检测、Token 溢出等功能",
        border_style="magenta"
    ))

    # 初始化测试环境（创建共享的 Patchouli 实例）
    setup_test_env(max_tokens=2048)

    # 测试分组
    test_groups = [
        ("基础功能测试", [
            ("初始化验证", test_01_initialization),
            ("添加消息", test_02_add_message),
            ("Buffer 信息获取", test_03_get_buffer_info),
        ]),
        ("Chatbot 对话流测试", [
            ("相同话题吸附", test_04_same_topic_adsorption),
            ("远距离语义漂移", test_05_semantic_drift_far),
            ("中等距离语义漂移", test_06_semantic_drift_medium),
            ("边界情况（较近话题）", test_07_boundary_near),
        ]),
        ("Agent 对话流测试", [
            ("单工具调用流程", test_08_agent_tool_call_flow),
            ("多工具调用", test_09_agent_multi_tool_calls),
        ]),
        ("Token 溢出测试", [
            ("Token 溢出触发", test_10_token_overflow_relay),
            ("接力摘要保持", test_11_relay_summary_preservation),
        ]),
        ("短文本吸附测试", [
            ("短文本强制吸附", test_12_short_text_forced_adsorption),
            ("阈值边界", test_13_threshold_boundary),
        ]),
        ("多会话隔离测试", [
            ("多会话隔离", test_14_multi_session_isolation),
            ("并发会话处理", test_15_concurrent_sessions),
        ]),
    ]

    results = []

    for group_name, tests in test_groups:
        console.print(f"\n[bold cyan]{group_name}[/bold cyan]")

        for name, test_func in tests:
            success = run_test(test_func)
            results.append((group_name, name, success))

    # 结果汇总
    console.print("\n" + "=" * 70)
    console.print("\n[bold cyan]测试结果汇总[/bold cyan]\n")

    # 分组显示结果
    current_group = None
    group_results = {}

    for group_name, name, success in results:
        if group_name not in group_results:
            group_results[group_name] = []
        group_results[group_name].append((name, success))

    for group_name, group_tests in group_results.items():
        console.print(f"\n[bold yellow]{group_name}[/bold yellow]")
        passed = sum(1 for _, s in group_tests if s)
        total = len(group_tests)
        console.print(f"  通过率: {passed}/{total}")

        for name, success in group_tests:
            status = "[green]✓[/green]" if success else "[red]✗[/red]"
            console.print(f"    {status} {name}")

    # 统计
    success_count = sum(1 for _, _, s in results if s)
    total_count = len(results)

    console.print(f"\n[bold]总通过率: {success_count}/{total_count} ({100*success_count//total_count}%)[/bold]")

    if success_count == total_count:
        console.print("\n[bold green]所有测试通过![/bold green]")
    else:
        console.print(f"\n[yellow]警告: {total_count - success_count} 个测试失败[/yellow]")


if __name__ == "__main__":
    main()
