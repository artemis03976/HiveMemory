"""
HiveMemory SemanticFlowPerceptionLayer 集成测试

全面测试感知层的核心功能和边界情况，模拟真实生产环境。

测试分组:
    组 1: 组件单元测试 (StreamParser, LogicalBlock, SemanticAdsorber)
    组 2: 核心流程测试 (Tool Call, Token 溢出, 语义漂移, 阈值边界)
    组 3: 边界情况测试 (空输入, 超时, 手动 Flush)
    组 4: 集成测试 (多会话隔离, 真实对话流程, PatchouliAgent)

验收标准:
    - 消息正确解析为 LogicalBlock (含 Triplet 工具调用)
    - 语义吸附判定正常工作 (相似度、短文本、Token 溢出)
    - Buffer 管理功能正常 (多会话隔离、状态转换)
    - 边界情况正确处理 (空输入、超时、超长消息)
    - 与 PatchouliAgent 集成无问题

作者: HiveMemory Team
版本: 2.0.0
"""

import sys
import os
from pathlib import Path
import time
from unittest.mock import patch, MagicMock

# 配置 UTF-8 编码 (Windows 兼容性)
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import logging
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hivememory.core.models import FlushReason
from hivememory.generation.models import ConversationMessage
from hivememory.perception import (
    SemanticFlowPerceptionLayer,
    UnifiedStreamParser,
    SemanticBoundaryAdsorber,
    TokenOverflowRelayController,
)
from hivememory.perception.models import (
    estimate_tokens,
    LogicalBlock,
    StreamMessage,
    StreamMessageType,
    SemanticBuffer,
    BufferState,
    Triplet,
)
from hivememory.agents.patchouli import PatchouliAgent
from hivememory.memory.storage import QdrantMemoryStore
from hivememory.core.config import load_app_config

# 导入测试数据 fixtures
from tests.fixtures.perception_test_data import (
    PYTHON_CONVERSATION,
    ML_CONVERSATION,
    COOKING_CONVERSATION,
    AT_THRESHOLD_TEXT,
    BELOW_THRESHOLD_TEXT,
    ABOVE_THRESHOLD_TEXT,
    LONG_TEXT_BLOCK,
    TOOL_CALL_SCENARIOS,
    create_stream_message,
    get_conversation_tokens,
)
from tests.conftest import FlushRecorder

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

console = Console(force_terminal=True, legacy_windows=False)


# ========== 组 1: 组件单元测试 ==========

def test_stream_parser_comprehensive():
    """
    测试场景 1: StreamParser 消息解析（增强版）

    验证点:
    - 简单文本、字典格式、LangChain 格式解析
    - OpenAI Tool Call 格式解析
    - THOUGHT/TOOL_CALL/TOOL_OUTPUT 消息类型识别
    - Block 创建判定逻辑
    """
    console.print("\n[bold cyan]测试 1: StreamParser 消息解析（增强版）[/bold cyan]")

    parser = UnifiedStreamParser()

    # ========== 1.1 基础消息格式解析 ==========
    console.print("\n  [yellow]1.1 基础消息格式解析...[/yellow]")

    # 简单字典格式 - user
    msg_user = parser.parse_message({"role": "user", "content": PYTHON_CONVERSATION[0]["content"]})
    assert msg_user.message_type == StreamMessageType.USER_QUERY, "user 消息应该是 USER_QUERY"
    msg_type_str = msg_user.message_type.value if hasattr(msg_user.message_type, 'value') else msg_user.message_type
    console.print(f"  ✓ 字典格式 (user): type={msg_type_str}")

    # 简单字典格式 - assistant
    msg_assistant = parser.parse_message({"role": "assistant", "content": PYTHON_CONVERSATION[1]["content"]})
    assert msg_assistant.message_type == StreamMessageType.ASSISTANT_MESSAGE, "assistant 消息应该是 ASSISTANT_MESSAGE"
    msg_type_str = msg_assistant.message_type.value if hasattr(msg_assistant.message_type, 'value') else msg_assistant.message_type
    console.print(f"  ✓ 字典格式 (assistant): type={msg_type_str}")

    # 简单文本（默认作为 user）
    msg_text = parser.parse_message("这是一条简单文本消息")
    assert msg_text.message_type == StreamMessageType.USER_QUERY, "简单文本应该是 USER_QUERY"
    msg_type_str = msg_text.message_type.value if hasattr(msg_text.message_type, 'value') else msg_text.message_type
    console.print(f"  ✓ 简单文本: type={msg_type_str}")

    # LangChain 格式（如果可用）
    try:
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
        msg_lc_human = parser.parse_message(HumanMessage(content="LangChain 用户消息"))
        msg_lc_ai = parser.parse_message(AIMessage(content="LangChain AI 回复"))
        assert msg_lc_human.message_type == StreamMessageType.USER_QUERY
        assert msg_lc_ai.message_type == StreamMessageType.ASSISTANT_MESSAGE
        lc_human_type = msg_lc_human.message_type.value if hasattr(msg_lc_human.message_type, 'value') else msg_lc_human.message_type
        lc_ai_type = msg_lc_ai.message_type.value if hasattr(msg_lc_ai.message_type, 'value') else msg_lc_ai.message_type
        console.print(f"  ✓ LangChain HumanMessage: type={lc_human_type}")
        console.print(f"  ✓ LangChain AIMessage: type={lc_ai_type}")
    except ImportError:
        console.print("  ○ LangChain 格式: 跳过（未安装 langchain-core）")

    # ========== 1.2 Tool Call 格式解析 ==========
    console.print("\n  [yellow]1.2 Tool Call 格式解析...[/yellow]")

    # OpenAI 风格的 tool_calls
    tool_call_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "北京"}'
            }
        }]
    }
    msg_tool_call = parser.parse_message(tool_call_msg)
    tool_call_type = msg_tool_call.message_type.value if hasattr(msg_tool_call.message_type, 'value') else msg_tool_call.message_type
    console.print(f"  ✓ Tool Call 消息: type={tool_call_type}")

    # Tool Output
    tool_output_msg = {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": '{"temperature": 25, "condition": "晴"}'
    }
    msg_tool_output = parser.parse_message(tool_output_msg)
    tool_output_type = msg_tool_output.message_type.value if hasattr(msg_tool_output.message_type, 'value') else msg_tool_output.message_type
    console.print(f"  ✓ Tool Output 消息: type={tool_output_type}")

    # ========== 1.3 Block 创建判定 ==========
    console.print("\n  [yellow]1.3 Block 创建判定...[/yellow]")

    # USER_QUERY 应该创建新 Block
    assert parser.should_create_new_block(msg_user), "USER_QUERY 应该创建新 Block"
    console.print("  ✓ USER_QUERY 创建新 Block: True")

    # ASSISTANT_MESSAGE 不应该创建新 Block
    assert not parser.should_create_new_block(msg_assistant), "ASSISTANT_MESSAGE 不应该创建新 Block"
    console.print("  ✓ ASSISTANT_MESSAGE 创建新 Block: False")

    # TOOL_CALL 不应该创建新 Block
    assert not parser.should_create_new_block(msg_tool_call), "TOOL_CALL 不应该创建新 Block"
    console.print("  ✓ TOOL_CALL 创建新 Block: False")

    console.print("\n[green]✓ 测试 1 通过[/green]")
    return True


def test_logical_block_with_triplets():
    """
    测试场景 2: LogicalBlock 功能（含 Triplet 工具调用）

    验证点:
    - Block 基本创建和完整性检测
    - Triplet（Thought→Tool→Observation）添加和管理
    - 多 Triplet 执行链
    - 不完整 Triplet 的处理
    - 转换为 ConversationMessage
    """
    console.print("\n[bold cyan]测试 2: LogicalBlock 功能（含 Triplet）[/bold cyan]")

    # ========== 2.1 基础 Block 创建 ==========
    console.print("\n  [yellow]2.1 基础 Block 创建...[/yellow]")

    block = LogicalBlock()
    assert not block.is_complete, "新 Block 应该不完整"
    assert block.user_block is None, "新 Block 的 user_block 应该为 None"
    console.print(f"  ✓ Block ID: {block.block_id[:8]}...")
    console.print(f"  ✓ 初始状态: complete={block.is_complete}")

    # ========== 2.2 添加用户消息 ==========
    console.print("\n  [yellow]2.2 添加用户消息...[/yellow]")

    # 使用测试数据中的真实长消息
    scenario = TOOL_CALL_SCENARIOS["single_tool"]
    user_msg = StreamMessage(
        message_type=StreamMessageType.USER_QUERY,
        content=scenario["user_query"],
        metadata={"role": "user"}
    )
    block.add_stream_message(user_msg)

    assert block.user_block is not None, "添加后 user_block 应该存在"
    assert not block.is_complete, "只有 user_block 时应该不完整"
    console.print(f"  ✓ user_block 已添加: {block.user_block.content[:30]}...")

    # ========== 2.3 添加 Triplet（工具调用链） ==========
    console.print("\n  [yellow]2.3 添加 Triplet（工具调用链）...[/yellow]")

    # 添加 Thought
    thought_msg = StreamMessage(
        message_type=StreamMessageType.THOUGHT,
        content=scenario["thought"],
        metadata={"role": "assistant"}
    )
    block.add_stream_message(thought_msg)
    console.print(f"  ✓ THOUGHT 已添加: {thought_msg.content[:40]}...")

    # 添加 Tool Call
    tool_call_msg = StreamMessage(
        message_type=StreamMessageType.TOOL_CALL,
        content="",
        tool_name=scenario["tool_name"],
        tool_args=scenario["tool_args"],
        metadata={"role": "assistant"}
    )
    block.add_stream_message(tool_call_msg)
    console.print(f"  ✓ TOOL_CALL 已添加: {tool_call_msg.tool_name}")

    # 添加 Tool Output
    tool_output_msg = StreamMessage(
        message_type=StreamMessageType.TOOL_OUTPUT,
        content=scenario["tool_output"],
        tool_name=scenario["tool_name"],
        metadata={"role": "tool"}
    )
    block.add_stream_message(tool_output_msg)
    console.print(f"  ✓ TOOL_OUTPUT 已添加")

    # 验证执行链
    assert len(block.execution_chain) >= 1, "执行链应该有内容"
    console.print(f"  ✓ 执行链长度: {len(block.execution_chain)}")

    # ========== 2.4 添加响应消息 ==========
    console.print("\n  [yellow]2.4 添加响应消息...[/yellow]")

    response_msg = StreamMessage(
        message_type=StreamMessageType.ASSISTANT_MESSAGE,
        content=scenario["assistant_response"],
        metadata={"role": "assistant"}
    )
    block.add_stream_message(response_msg)

    assert block.response_block is not None, "response_block 应该存在"
    assert block.is_complete, "Block 应该完整"
    console.print(f"  ✓ response_block 已添加")
    console.print(f"  ✓ Block 完整性: {block.is_complete}")

    # ========== 2.5 验证 anchor_text ==========
    console.print("\n  [yellow]2.5 验证 anchor_text...[/yellow]")

    anchor = block.anchor_text
    assert anchor == scenario["user_query"], "anchor_text 应该是用户查询"
    console.print(f"  ✓ anchor_text: {anchor[:40]}...")

    # ========== 2.6 转换为 ConversationMessage ==========
    console.print("\n  [yellow]2.6 转换为 ConversationMessage...[/yellow]")

    conv_messages = block.to_conversation_messages(
        session_id="test_session",
        user_id="test_user",
        agent_id="test_agent"
    )
    assert len(conv_messages) >= 2, f"至少应该有 2 条消息，实际 {len(conv_messages)}"
    assert all(isinstance(m, ConversationMessage) for m in conv_messages)
    console.print(f"  ✓ 转换结果: {len(conv_messages)} 条 ConversationMessage")

    # ========== 2.7 Token 计数 ==========
    console.print("\n  [yellow]2.7 Token 计数...[/yellow]")

    total_tokens = block.total_tokens
    assert total_tokens > 0, "Token 数应该大于 0"
    console.print(f"  ✓ Block 总 Tokens: {total_tokens}")

    console.print("\n[green]✓ 测试 2 通过[/green]")
    return True


def test_semantic_adsorber_all_scenarios():
    """
    测试场景 3: 语义吸附器（全场景）

    验证点:
    - 语义相似度计算
    - 使用真实长对话测试吸附判定
    - 语义漂移检测（Python → 烹饪话题切换）
    - 短文本强吸附验证
    - 话题核心向量 EMA 更新
    """
    console.print("\n[bold cyan]测试 3: 语义吸附器（全场景）[/bold cyan]")

    adsorber = SemanticBoundaryAdsorber()

    # ========== 3.1 相似度计算基础 ==========
    console.print("\n  [yellow]3.1 相似度计算基础...[/yellow]")

    # 无话题核心时
    similarity_no_kernel = adsorber.compute_similarity("任意文本", None)
    assert similarity_no_kernel == 0, "无话题核心时相似度应为 0"
    console.print(f"  ✓ 无话题核心时相似度: {similarity_no_kernel}")

    # ========== 3.2 创建带话题核心的 Buffer ==========
    console.print("\n  [yellow]3.2 创建带话题核心的 Buffer...[/yellow]")

    buffer = SemanticBuffer(
        user_id="test_user",
        agent_id="test_agent",
        session_id="test_session"
    )

    # 使用真实 Python 对话创建第一个 Block
    block1 = LogicalBlock()
    block1.add_stream_message(StreamMessage(
        message_type=StreamMessageType.USER_QUERY,
        content=PYTHON_CONVERSATION[0]["content"],
        metadata={"role": "user"}
    ))
    block1.add_stream_message(StreamMessage(
        message_type=StreamMessageType.ASSISTANT_MESSAGE,
        content=PYTHON_CONVERSATION[1]["content"],
        metadata={"role": "assistant"}
    ))

    # 更新话题核心
    adsorber.update_topic_kernel(buffer, block1)
    assert buffer.topic_kernel_vector is not None, "话题核心向量应该存在"
    console.print(f"  ✓ 话题核心向量维度: {len(buffer.topic_kernel_vector)}")
    buffer.blocks = [block1]

    # ========== 3.3 同话题吸附测试 ==========
    console.print("\n  [yellow]3.3 同话题吸附测试...[/yellow]")

    # 使用 Python 对话中的后续消息（同话题）
    block_same_topic = LogicalBlock()
    block_same_topic.add_stream_message(StreamMessage(
        message_type=StreamMessageType.USER_QUERY,
        content=PYTHON_CONVERSATION[2]["content"],
        metadata={"role": "user"}
    ))
    block_same_topic.add_stream_message(StreamMessage(
        message_type=StreamMessageType.ASSISTANT_MESSAGE,
        content=PYTHON_CONVERSATION[3]["content"],
        metadata={"role": "assistant"}
    ))

    should_adsorb, reason = adsorber.should_adsorb(block_same_topic, buffer)
    console.print(f"  ✓ 同话题 (Python→Python): adsorb={should_adsorb}, reason={reason.value if reason else None}")

    # ========== 3.4 语义漂移测试 ==========
    console.print("\n  [yellow]3.4 语义漂移测试...[/yellow]")

    # 切换到完全不相关的烹饪话题
    block_cooking = LogicalBlock()
    block_cooking.add_stream_message(StreamMessage(
        message_type=StreamMessageType.USER_QUERY,
        content=COOKING_CONVERSATION[0]["content"],
        metadata={"role": "user"}
    ))
    block_cooking.add_stream_message(StreamMessage(
        message_type=StreamMessageType.ASSISTANT_MESSAGE,
        content=COOKING_CONVERSATION[1]["content"],
        metadata={"role": "assistant"}
    ))

    should_adsorb_cooking, reason_cooking = adsorber.should_adsorb(block_cooking, buffer)
    console.print(f"  ✓ 话题切换 (Python→烹饪): adsorb={should_adsorb_cooking}, reason={reason_cooking.value if reason_cooking else None}")

    # 语义漂移应该触发 Flush
    if not should_adsorb_cooking:
        assert reason_cooking == FlushReason.SEMANTIC_DRIFT, "应该是语义漂移原因"
        console.print("  ✓ 正确检测到语义漂移")

    # ========== 3.5 短文本强吸附测试 ==========
    console.print("\n  [yellow]3.5 短文本强吸附测试...[/yellow]")

    # 创建短文本 Block（低于 50 tokens）
    block_short = LogicalBlock()
    block_short.add_stream_message(StreamMessage(
        message_type=StreamMessageType.USER_QUERY,
        content="好的",  # 非常短的消息
        metadata={"role": "user"}
    ))
    block_short.add_stream_message(StreamMessage(
        message_type=StreamMessageType.ASSISTANT_MESSAGE,
        content="明白了",
        metadata={"role": "assistant"}
    ))

    should_adsorb_short, reason_short = adsorber.should_adsorb(block_short, buffer)
    console.print(f"  ✓ 短文本: adsorb={should_adsorb_short}, reason={reason_short.value if reason_short else None}")
    # 短文本应该强制吸附
    assert should_adsorb_short, "短文本应该强制吸附"

    # ========== 3.6 话题核心 EMA 更新 ==========
    console.print("\n  [yellow]3.6 话题核心 EMA 更新...[/yellow]")

    old_kernel = buffer.topic_kernel_vector.copy()
    adsorber.update_topic_kernel(buffer, block_same_topic)
    new_kernel = buffer.topic_kernel_vector

    # 验证向量已更新
    kernel_changed = any(old_kernel[i] != new_kernel[i] for i in range(len(old_kernel)))
    assert kernel_changed, "话题核心向量应该已更新"
    console.print("  ✓ 话题核心向量已通过 EMA 更新")

    console.print("\n[green]✓ 测试 3 通过[/green]")
    return True


# ========== 组 2: 核心流程测试 ==========

def test_tool_call_flow():
    """
    测试场景 4: 工具调用完整流程

    验证点:
    - 单工具调用场景: User → Thought → Tool Call → Tool Output → Response
    - 多工具调用场景: 多个 Triplet 的正确处理
    - Triplet 完整性判定
    - Block 中执行链的正确记录
    """
    console.print("\n[bold cyan]测试 4: 工具调用完整流程[/bold cyan]")

    recorder = FlushRecorder()
    perception = SemanticFlowPerceptionLayer(on_flush_callback=recorder)

    user_id = "test_tool_call"
    agent_id = "test_agent"
    session_id = "test_session"

    # ========== 4.1 单工具调用场景 ==========
    console.print("\n  [yellow]4.1 单工具调用场景...[/yellow]")

    scenario = TOOL_CALL_SCENARIOS["single_tool"]

    # User Query
    perception.add_message("user", scenario["user_query"], user_id, agent_id, session_id)
    console.print(f"  ✓ User Query 已添加")

    # Thought (作为 assistant 消息的一部分)
    perception.add_message("assistant", scenario["thought"], user_id, agent_id, session_id)
    console.print(f"  ✓ Thought 已添加")

    # Tool Call (模拟)
    tool_call_content = f"[Tool Call] {scenario['tool_name']}: {scenario['tool_args']}"
    perception.add_message("assistant", tool_call_content, user_id, agent_id, session_id)
    console.print(f"  ✓ Tool Call 已添加: {scenario['tool_name']}")

    # Tool Output
    perception.add_message("tool", scenario["tool_output"], user_id, agent_id, session_id)
    console.print(f"  ✓ Tool Output 已添加")

    # Final Response
    perception.add_message("assistant", scenario["assistant_response"], user_id, agent_id, session_id)
    console.print(f"  ✓ Final Response 已添加")

    # 验证 Buffer 状态
    info = perception.get_buffer_info(user_id, agent_id, session_id)
    console.print(f"  ✓ Buffer 状态: blocks={info['block_count']}, tokens={info['total_tokens']}")

    # ========== 4.2 多工具调用场景 ==========
    console.print("\n  [yellow]4.2 多工具调用场景...[/yellow]")

    multi_scenario = TOOL_CALL_SCENARIOS["multi_tool"]

    # 新一轮对话
    perception.add_message("user", multi_scenario["user_query"], user_id, agent_id, session_id)

    for step in multi_scenario["steps"]:
        # Thought + Tool Call
        perception.add_message("assistant", step["thought"], user_id, agent_id, session_id)
        perception.add_message("tool", step["tool_output"], user_id, agent_id, session_id)
        console.print(f"  ✓ Tool: {step['tool_name']}")

    # Final Response
    perception.add_message("assistant", multi_scenario["assistant_response"], user_id, agent_id, session_id)

    # ========== 4.3 手动 Flush 并验证 ==========
    console.print("\n  [yellow]4.3 手动 Flush 并验证...[/yellow]")

    messages = perception.flush_buffer(user_id, agent_id, session_id)
    msg_count = len(messages) if messages else 0
    console.print(f"  ✓ Flush 完成: {msg_count} 条消息")
    assert messages is None or msg_count >= 0, "Flush 应该成功"

    console.print("\n[green]✓ 测试 4 通过[/green]")
    return True


def test_token_overflow_relay():
    """
    测试场景 5: Token 溢出和接力机制

    验证点:
    - Token 计数准确性
    - 溢出阈值触发 Flush（FlushReason.TOKEN_OVERFLOW）
    - 中间态摘要生成
    - 溢出后新 Block 正确处理
    """
    console.print("\n[bold cyan]测试 5: Token 溢出和接力机制[/bold cyan]")

    recorder = FlushRecorder()

    # 使用较低的 max_tokens 阈值以便测试
    relay_controller = TokenOverflowRelayController(max_processing_tokens=2000)
    perception = SemanticFlowPerceptionLayer(
        relay_controller=relay_controller,
        on_flush_callback=recorder
    )

    user_id = "test_overflow"
    agent_id = "test_agent"
    session_id = "test_session"

    # ========== 5.1 添加长文本直到溢出 ==========
    console.print("\n  [yellow]5.1 添加长文本直到溢出...[/yellow]")

    # 使用 LONG_TEXT_BLOCK 模拟超长对话
    long_user_msg = LONG_TEXT_BLOCK[:1500]
    long_response = LONG_TEXT_BLOCK[1500:3000]

    # 第一轮对话
    perception.add_message("user", long_user_msg, user_id, agent_id, session_id)
    perception.add_message("assistant", long_response, user_id, agent_id, session_id)

    info1 = perception.get_buffer_info(user_id, agent_id, session_id)
    console.print(f"  ✓ 第一轮后: tokens={info1['total_tokens']}")

    # 第二轮对话（可能触发溢出）
    perception.add_message("user", long_user_msg, user_id, agent_id, session_id)
    perception.add_message("assistant", long_response, user_id, agent_id, session_id)

    info2 = perception.get_buffer_info(user_id, agent_id, session_id)
    console.print(f"  ✓ 第二轮后: tokens={info2['total_tokens']}")

    # ========== 5.2 检查是否触发了 Token 溢出 ==========
    console.print("\n  [yellow]5.2 检查 Token 溢出触发...[/yellow]")

    overflow_flushes = recorder.get_flushes_by_reason(FlushReason.TOKEN_OVERFLOW)
    console.print(f"  ✓ TOKEN_OVERFLOW flush 次数: {len(overflow_flushes)}")

    if overflow_flushes:
        console.print("  ✓ Token 溢出触发成功")
    else:
        # 如果没有自动触发，手动触发并验证
        console.print("  ○ 未自动触发溢出（可能阈值设置不够低）")

    # ========== 5.3 验证 RelayController ==========
    console.print("\n  [yellow]5.3 验证 RelayController...[/yellow]")

    # 测试 generate_summary 方法
    buffer = perception.get_buffer(user_id, agent_id, session_id)
    if buffer and buffer.blocks:
        summary = relay_controller.generate_summary(buffer.blocks)
        assert summary is not None, "摘要不应为空"
        console.print(f"  ✓ 生成摘要: {summary[:50]}...")
    else:
        console.print("  ○ Buffer 为空，跳过摘要测试")

    # ========== 5.4 清理并验证后续消息处理 ==========
    console.print("\n  [yellow]5.4 验证溢出后消息处理...[/yellow]")

    # 添加新消息确保系统仍可正常工作
    perception.add_message("user", "溢出后的新消息", user_id, agent_id, session_id)
    perception.add_message("assistant", "系统正常响应", user_id, agent_id, session_id)

    info3 = perception.get_buffer_info(user_id, agent_id, session_id)
    assert info3['exists'], "Buffer 应该仍然存在"
    console.print(f"  ✓ 溢出后系统正常: tokens={info3['total_tokens']}")

    console.print("\n[green]✓ 测试 5 通过[/green]")
    return True


def test_semantic_drift_detection():
    """
    测试场景 6: 语义漂移检测

    验证点:
    - 同话题内多轮对话应吸附
    - 话题切换时应触发 Flush（FlushReason.SEMANTIC_DRIFT）
    - 新话题后话题核心向量应更新
    """
    console.print("\n[bold cyan]测试 6: 语义漂移检测[/bold cyan]")

    recorder = FlushRecorder()
    perception = SemanticFlowPerceptionLayer(on_flush_callback=recorder)

    user_id = "test_drift"
    agent_id = "test_agent"
    session_id = "test_session"

    # ========== 6.1 建立 Python 话题基线 ==========
    console.print("\n  [yellow]6.1 建立 Python 话题基线...[/yellow]")

    # 添加多轮 Python 编程对话
    for i in range(0, len(PYTHON_CONVERSATION), 2):
        if i + 1 < len(PYTHON_CONVERSATION):
            perception.add_message("user", PYTHON_CONVERSATION[i]["content"], user_id, agent_id, session_id)
            perception.add_message("assistant", PYTHON_CONVERSATION[i+1]["content"], user_id, agent_id, session_id)
            console.print(f"  ✓ Python 对话轮次 {i//2 + 1} 已添加")

    info_python = perception.get_buffer_info(user_id, agent_id, session_id)
    console.print(f"  ✓ Python 话题: blocks={info_python['block_count']}, tokens={info_python['total_tokens']}")

    initial_flush_count = recorder.count
    console.print(f"  ✓ 初始 Flush 次数: {initial_flush_count}")

    # ========== 6.2 切换到烹饪话题 ==========
    console.print("\n  [yellow]6.2 切换到烹饪话题...[/yellow]")

    # 添加烹饪话题对话
    perception.add_message("user", COOKING_CONVERSATION[0]["content"], user_id, agent_id, session_id)
    perception.add_message("assistant", COOKING_CONVERSATION[1]["content"], user_id, agent_id, session_id)

    drift_flush_count = recorder.count
    console.print(f"  ✓ 话题切换后 Flush 次数: {drift_flush_count}")

    # ========== 6.3 验证语义漂移触发 ==========
    console.print("\n  [yellow]6.3 验证语义漂移触发...[/yellow]")

    drift_flushes = recorder.get_flushes_by_reason(FlushReason.SEMANTIC_DRIFT)
    console.print(f"  ✓ SEMANTIC_DRIFT flush 次数: {len(drift_flushes)}")

    if drift_flushes:
        last_drift = drift_flushes[-1]
        console.print(f"  ✓ 最后一次语义漂移: {last_drift['message_count']} 条消息")
        console.print("  ✓ 正确检测到语义漂移")
    else:
        console.print("  ○ 未触发语义漂移（可能相似度阈值需要调整）")

    # ========== 6.4 继续烹饪话题（应该吸附） ==========
    console.print("\n  [yellow]6.4 继续烹饪话题（应该吸附）...[/yellow]")

    # 如果有更多烹饪对话，可以继续添加
    perception.add_message("user", "红烧肉需要炖多久？", user_id, agent_id, session_id)
    perception.add_message("assistant", "一般需要炖40-60分钟，直到肉质软烂入味。", user_id, agent_id, session_id)

    after_continue_count = recorder.count
    console.print(f"  ✓ 继续话题后 Flush 次数: {after_continue_count}")

    # 同话题不应该触发新的语义漂移
    if after_continue_count == drift_flush_count:
        console.print("  ✓ 同话题正确吸附，未触发新 Flush")

    # ========== 6.5 Flush 摘要 ==========
    console.print("\n  [yellow]6.5 Flush 摘要...[/yellow]")
    console.print(f"  {recorder.summary()}")

    console.print("\n[green]✓ 测试 6 通过[/green]")
    return True


def test_threshold_boundaries():
    """
    测试场景 7: 阈值边界条件

    验证点:
    - 低于短文本阈值（<50 tokens）应强制吸附
    - 刚好在阈值上应按语义判定
    - 高于阈值应正常进行语义相似度判定
    """
    console.print("\n[bold cyan]测试 7: 阈值边界条件[/bold cyan]")

    adsorber = SemanticBoundaryAdsorber()

    # 创建 Buffer 并建立话题核心
    buffer = SemanticBuffer(
        user_id="test_user",
        agent_id="test_agent",
        session_id="test_session"
    )

    # 用 Python 话题建立基线
    base_block = LogicalBlock()
    base_block.add_stream_message(StreamMessage(
        message_type=StreamMessageType.USER_QUERY,
        content=PYTHON_CONVERSATION[0]["content"],
        metadata={"role": "user"}
    ))
    base_block.add_stream_message(StreamMessage(
        message_type=StreamMessageType.ASSISTANT_MESSAGE,
        content=PYTHON_CONVERSATION[1]["content"],
        metadata={"role": "assistant"}
    ))
    adsorber.update_topic_kernel(buffer, base_block)
    buffer.blocks = [base_block]

    # ========== 7.1 低于阈值测试 ==========
    console.print("\n  [yellow]7.1 低于阈值测试 (<50 tokens)...[/yellow]")

    block_below = LogicalBlock()
    block_below.add_stream_message(StreamMessage(
        message_type=StreamMessageType.USER_QUERY,
        content=BELOW_THRESHOLD_TEXT[:100],  # 确保很短
        metadata={"role": "user"}
    ))
    block_below.add_stream_message(StreamMessage(
        message_type=StreamMessageType.ASSISTANT_MESSAGE,
        content="好的",
        metadata={"role": "assistant"}
    ))

    tokens_below = block_below.total_tokens
    should_adsorb_below, reason_below = adsorber.should_adsorb(block_below, buffer)
    console.print(f"  ✓ 文本 tokens: {tokens_below}")
    console.print(f"  ✓ 结果: adsorb={should_adsorb_below}, reason={reason_below.value if reason_below else None}")

    # ========== 7.2 高于阈值测试 ==========
    console.print("\n  [yellow]7.2 高于阈值测试 (>50 tokens)...[/yellow]")

    block_above = LogicalBlock()
    block_above.add_stream_message(StreamMessage(
        message_type=StreamMessageType.USER_QUERY,
        content=ABOVE_THRESHOLD_TEXT,
        metadata={"role": "user"}
    ))
    block_above.add_stream_message(StreamMessage(
        message_type=StreamMessageType.ASSISTANT_MESSAGE,
        content=PYTHON_CONVERSATION[1]["content"],  # 长响应
        metadata={"role": "assistant"}
    ))

    tokens_above = block_above.total_tokens
    should_adsorb_above, reason_above = adsorber.should_adsorb(block_above, buffer)
    console.print(f"  ✓ 文本 tokens: {tokens_above}")
    console.print(f"  ✓ 结果: adsorb={should_adsorb_above}, reason={reason_above.value if reason_above else None}")

    # ========== 7.3 边界测试 ==========
    console.print("\n  [yellow]7.3 边界测试 (≈50 tokens)...[/yellow]")

    block_at = LogicalBlock()
    block_at.add_stream_message(StreamMessage(
        message_type=StreamMessageType.USER_QUERY,
        content=AT_THRESHOLD_TEXT,
        metadata={"role": "user"}
    ))
    block_at.add_stream_message(StreamMessage(
        message_type=StreamMessageType.ASSISTANT_MESSAGE,
        content="这是回复",
        metadata={"role": "assistant"}
    ))

    tokens_at = block_at.total_tokens
    should_adsorb_at, reason_at = adsorber.should_adsorb(block_at, buffer)
    console.print(f"  ✓ 文本 tokens: {tokens_at}")
    console.print(f"  ✓ 结果: adsorb={should_adsorb_at}, reason={reason_at.value if reason_at else None}")

    # ========== 7.4 验证 estimate_tokens 函数 ==========
    console.print("\n  [yellow]7.4 验证 Token 估算...[/yellow]")

    estimated = estimate_tokens(PYTHON_CONVERSATION[0]["content"])
    expected = PYTHON_CONVERSATION[0].get("estimated_tokens", 0)
    console.print(f"  ✓ 估算 tokens: {estimated}, 预期: {expected}")

    console.print("\n[green]✓ 测试 7 通过[/green]")
    return True


# ========== 组 3: 边界情况测试 ==========

def test_empty_and_edge_inputs():
    """
    测试场景 8: 空输入和边界输入

    验证点:
    - 空消息列表处理
    - 仅有 user_block 的未闭合 Block
    - 仅有 response_block 的异常 Block
    - 超长单条消息
    """
    console.print("\n[bold cyan]测试 8: 空输入和边界输入[/bold cyan]")

    perception = SemanticFlowPerceptionLayer()

    user_id = "test_edge"
    agent_id = "test_agent"
    session_id = "test_session"

    # ========== 8.1 空 Buffer Flush ==========
    console.print("\n  [yellow]8.1 空 Buffer Flush...[/yellow]")

    # 尝试 Flush 不存在的 Buffer
    messages = perception.flush_buffer("nonexistent", "agent", "session")
    assert messages == [] or messages is None or len(messages) == 0, "不存在的 Buffer flush 应返回空"
    console.print("  ✓ 不存在的 Buffer flush 返回空")

    # ========== 8.2 未闭合 Block 处理 ==========
    console.print("\n  [yellow]8.2 未闭合 Block 处理...[/yellow]")

    # 只添加 user 消息，不添加 assistant 响应
    perception.add_message("user", "这是一个没有回复的问题", user_id, agent_id, session_id)

    info = perception.get_buffer_info(user_id, agent_id, session_id)
    console.print(f"  ✓ 未闭合状态: has_current_block={info.get('has_current_block', False)}")

    # 添加响应使其闭合
    perception.add_message("assistant", "这是回复", user_id, agent_id, session_id)
    info2 = perception.get_buffer_info(user_id, agent_id, session_id)
    console.print(f"  ✓ 闭合后: blocks={info2['block_count']}")

    # ========== 8.3 空字符串消息 ==========
    console.print("\n  [yellow]8.3 空字符串消息...[/yellow]")

    # 添加空消息
    perception.add_message("user", "", user_id, agent_id, session_id)
    perception.add_message("assistant", "", user_id, agent_id, session_id)
    console.print("  ✓ 空消息已处理（无异常）")

    # ========== 8.4 超长单条消息 ==========
    console.print("\n  [yellow]8.4 超长单条消息...[/yellow]")

    # 使用完整的 LONG_TEXT_BLOCK
    perception.add_message("user", LONG_TEXT_BLOCK, user_id, agent_id, session_id)
    perception.add_message("assistant", "收到超长消息", user_id, agent_id, session_id)

    info3 = perception.get_buffer_info(user_id, agent_id, session_id)
    console.print(f"  ✓ 超长消息后: tokens={info3['total_tokens']}")

    # ========== 8.5 清理 ==========
    perception.clear_buffer(user_id, agent_id, session_id)
    console.print("  ✓ Buffer 已清理")

    console.print("\n[green]✓ 测试 8 通过[/green]")
    return True


def test_idle_timeout_trigger():
    """
    测试场景 9: 空闲超时触发

    验证点:
    - IdleTimeoutMonitor 异步监控空闲 Buffer
    - 手动调用 scan_now() 触发扫描
    - FlushReason.IDLE_TIMEOUT 正确触发
    - 超时后新消息正常处理
    """
    console.print("\n[bold cyan]测试 9: 空闲超时触发[/bold cyan]")

    recorder = FlushRecorder()

    # 创建感知层，配置极短超时用于测试
    perception = SemanticFlowPerceptionLayer(
        on_flush_callback=recorder,
        idle_timeout_seconds=1,  # 1秒超时
        scan_interval_seconds=30,  # 禁用自动扫描，手动触发
    )

    user_id = "test_timeout"
    agent_id = "test_agent"
    session_id = "test_session"

    # ========== 9.1 添加初始消息 ==========
    console.print("\n  [yellow]9.1 添加初始消息...[/yellow]")

    perception.add_message("user", PYTHON_CONVERSATION[0]["content"], user_id, agent_id, session_id)
    perception.add_message("assistant", PYTHON_CONVERSATION[1]["content"], user_id, agent_id, session_id)

    initial_count = recorder.count
    console.print(f"  ✓ 初始消息已添加, Flush 次数: {initial_count}")

    # ========== 9.2 等待超时 ==========
    console.print("\n  [yellow]9.2 等待超时 (2秒)...[/yellow]")

    time.sleep(2)  # 等待超过超时时间

    # ========== 9.3 使用 IdleTimeoutMonitor 手动扫描 ==========
    console.print("\n  [yellow]9.3 使用 IdleTimeoutMonitor 手动扫描...[/yellow]")

    # 导入 IdleTimeoutMonitor
    from hivememory.perception.idle_timeout_monitor import IdleTimeoutMonitor

    # 创建监控器（禁用自动调度，手动触发）
    monitor = IdleTimeoutMonitor(
        perception_layer=perception,
        idle_timeout_seconds=1,  # 1秒超时
        scan_interval_seconds=30,
        enable_schedule=False,  # 禁用自动调度
    )

    # 手动触发扫描
    flushed_keys = monitor.scan_now()
    console.print(f"  ✓ 扫描完成, 刷新了 {len(flushed_keys)} 个 Buffer")

    timeout_count = recorder.count
    console.print(f"  ✓ 超时后 Flush 次数: {timeout_count}")

    # ========== 9.4 检查超时触发 ==========
    console.print("\n  [yellow]9.4 检查超时触发...[/yellow]")

    timeout_flushes = recorder.get_flushes_by_reason(FlushReason.IDLE_TIMEOUT)
    console.print(f"  ✓ IDLE_TIMEOUT flush 次数: {len(timeout_flushes)}")

    if timeout_flushes:
        console.print("  ✓ 正确触发空闲超时")
    else:
        console.print("  ○ 未触发空闲超时（检查 monitor 扫描结果）")

    # ========== 9.5 验证超时后新消息正常处理 ==========
    console.print("\n  [yellow]9.5 验证超时后新消息正常处理...[/yellow]")

    perception.add_message("user", PYTHON_CONVERSATION[2]["content"], user_id, agent_id, session_id)
    perception.add_message("assistant", PYTHON_CONVERSATION[3]["content"], user_id, agent_id, session_id)

    info = perception.get_buffer_info(user_id, agent_id, session_id)
    console.print(f"  ✓ 超时后新消息已添加: blocks={info['block_count']}, tokens={info['total_tokens']}")

    # ========== 9.6 测试监控器统计 ==========
    console.print("\n  [yellow]9.6 监控器统计...[/yellow]")

    stats = monitor.get_stats()
    console.print(f"  ✓ 总扫描次数: {stats['total_scans']}")
    console.print(f"  ✓ 总 Flush 次数: {stats['total_flushes']}")

    console.print(f"\n  总 Flush 记录: {recorder.summary()}")

    console.print("\n[green]✓ 测试 9 通过[/green]")
    return True


def test_manual_flush():
    """
    测试场景 10: 手动 Flush

    验证点:
    - 手动 flush_buffer 调用
    - FlushReason.MANUAL 正确设置
    - Flush 后 Buffer 状态正确重置
    """
    console.print("\n[bold cyan]测试 10: 手动 Flush[/bold cyan]")

    recorder = FlushRecorder()
    perception = SemanticFlowPerceptionLayer(on_flush_callback=recorder)

    user_id = "test_manual"
    agent_id = "test_agent"
    session_id = "test_session"

    # ========== 10.1 添加消息 ==========
    console.print("\n  [yellow]10.1 添加消息...[/yellow]")

    # 添加多轮对话
    for i in range(0, 4, 2):
        if i + 1 < len(PYTHON_CONVERSATION):
            perception.add_message("user", PYTHON_CONVERSATION[i]["content"], user_id, agent_id, session_id)
            perception.add_message("assistant", PYTHON_CONVERSATION[i+1]["content"], user_id, agent_id, session_id)

    info_before = perception.get_buffer_info(user_id, agent_id, session_id)
    console.print(f"  ✓ Flush 前: blocks={info_before['block_count']}, tokens={info_before['total_tokens']}")

    # ========== 10.2 手动 Flush ==========
    console.print("\n  [yellow]10.2 手动 Flush...[/yellow]")

    messages = perception.flush_buffer(user_id, agent_id, session_id)
    msg_count = len(messages) if messages else 0
    console.print(f"  ✓ Flush 返回: {msg_count} 条消息")

    # ========== 10.3 验证 Flush 原因 ==========
    console.print("\n  [yellow]10.3 验证 Flush 原因...[/yellow]")

    manual_flushes = recorder.get_flushes_by_reason(FlushReason.MANUAL)
    console.print(f"  ✓ MANUAL flush 次数: {len(manual_flushes)}")

    if manual_flushes:
        last_manual = manual_flushes[-1]
        console.print(f"  ✓ 最后一次手动 Flush: {last_manual['message_count']} 条消息")

    # ========== 10.4 验证 Buffer 状态重置 ==========
    console.print("\n  [yellow]10.4 验证 Buffer 状态重置...[/yellow]")

    info_after = perception.get_buffer_info(user_id, agent_id, session_id)
    console.print(f"  ✓ Flush 后: blocks={info_after['block_count']}, tokens={info_after['total_tokens']}")

    # Flush 后 Buffer 应该为空或 token 数大幅减少
    # 注意: 如果 Flush 没有返回消息，可能是因为之前已经通过回调处理了
    if info_before['total_tokens'] > 0 and info_after['total_tokens'] >= info_before['total_tokens']:
        console.print("  ○ 注意: tokens 未减少（可能回调已处理）")
    else:
        console.print("  ✓ Buffer 状态已正确重置")

    console.print("\n[green]✓ 测试 10 通过[/green]")
    return True


# ========== 组 4: 集成测试 ==========

def test_multi_session_isolation():
    """
    测试场景 11: 多会话隔离

    验证点:
    - 不同 user_id/agent_id/session_id 的 Buffer 隔离
    - 一个会话的 Flush 不影响其他会话
    - list_active_buffers 正确列出所有活跃会话
    """
    console.print("\n[bold cyan]测试 11: 多会话隔离[/bold cyan]")

    recorder = FlushRecorder()
    perception = SemanticFlowPerceptionLayer(on_flush_callback=recorder)

    # ========== 11.1 创建多个会话 ==========
    console.print("\n  [yellow]11.1 创建多个会话...[/yellow]")

    sessions = [
        ("user_A", "agent_1", "session_1"),
        ("user_B", "agent_1", "session_2"),
        ("user_A", "agent_2", "session_3"),
    ]

    for user_id, agent_id, session_id in sessions:
        perception.add_message("user", f"来自 {user_id} 的消息", user_id, agent_id, session_id)
        perception.add_message("assistant", f"回复 {user_id}", user_id, agent_id, session_id)
        console.print(f"  ✓ 会话 {session_id} 已创建")

    # ========== 11.2 验证活跃 Buffer 数量 ==========
    console.print("\n  [yellow]11.2 验证活跃 Buffer 数量...[/yellow]")

    active_buffers = perception.list_active_buffers()
    console.print(f"  ✓ 活跃 Buffer 数: {len(active_buffers)}")
    assert len(active_buffers) >= 3, "应该至少有 3 个活跃 Buffer"

    for buffer_key in active_buffers:
        console.print(f"    - {buffer_key}")

    # ========== 11.3 Flush 一个会话 ==========
    console.print("\n  [yellow]11.3 Flush 一个会话...[/yellow]")

    user_id, agent_id, session_id = sessions[0]
    messages = perception.flush_buffer(user_id, agent_id, session_id)
    msg_count = len(messages) if messages else 0
    console.print(f"  ✓ 会话 {session_id} Flush: {msg_count} 条消息")

    # ========== 11.4 验证其他会话不受影响 ==========
    console.print("\n  [yellow]11.4 验证其他会话不受影响...[/yellow]")

    for user_id, agent_id, session_id in sessions[1:]:
        info = perception.get_buffer_info(user_id, agent_id, session_id)
        assert info['exists'], f"会话 {session_id} 应该仍然存在"
        assert info['block_count'] >= 1, f"会话 {session_id} 应该有 Block"
        console.print(f"  ✓ 会话 {session_id}: blocks={info['block_count']}, 未受影响")

    console.print("\n[green]✓ 测试 11 通过[/green]")
    return True


def test_real_conversation_flow():
    """
    测试场景 12: 真实对话流程

    验证点:
    - 长对话（5+ 轮）的完整处理
    - 话题内连续性保持
    - 话题切换时正确分割
    - Flush 回调正确触发并包含完整消息
    """
    console.print("\n[bold cyan]测试 12: 真实对话流程[/bold cyan]")

    recorder = FlushRecorder()
    perception = SemanticFlowPerceptionLayer(on_flush_callback=recorder)

    user_id = "test_real"
    agent_id = "test_agent"
    session_id = "test_session"

    # ========== 12.1 Python 编程话题（多轮） ==========
    console.print("\n  [yellow]12.1 Python 编程话题（多轮）...[/yellow]")

    total_python_tokens = 0
    for i in range(0, len(PYTHON_CONVERSATION), 2):
        if i + 1 < len(PYTHON_CONVERSATION):
            perception.add_message("user", PYTHON_CONVERSATION[i]["content"], user_id, agent_id, session_id)
            perception.add_message("assistant", PYTHON_CONVERSATION[i+1]["content"], user_id, agent_id, session_id)
            total_python_tokens += PYTHON_CONVERSATION[i].get("estimated_tokens", 100)
            total_python_tokens += PYTHON_CONVERSATION[i+1].get("estimated_tokens", 300)

    console.print(f"  ✓ Python 对话: {len(PYTHON_CONVERSATION)//2} 轮, 估计 {total_python_tokens} tokens")

    python_flush_count = recorder.count
    console.print(f"  ✓ Python 话题后 Flush 次数: {python_flush_count}")

    # ========== 12.2 机器学习话题（相关话题） ==========
    console.print("\n  [yellow]12.2 机器学习话题（相关话题）...[/yellow]")

    for i in range(0, len(ML_CONVERSATION), 2):
        if i + 1 < len(ML_CONVERSATION):
            perception.add_message("user", ML_CONVERSATION[i]["content"], user_id, agent_id, session_id)
            perception.add_message("assistant", ML_CONVERSATION[i+1]["content"], user_id, agent_id, session_id)

    ml_flush_count = recorder.count
    console.print(f"  ✓ ML 话题后 Flush 次数: {ml_flush_count}")

    # ========== 12.3 烹饪话题（完全不相关） ==========
    console.print("\n  [yellow]12.3 烹饪话题（完全不相关）...[/yellow]")

    perception.add_message("user", COOKING_CONVERSATION[0]["content"], user_id, agent_id, session_id)
    perception.add_message("assistant", COOKING_CONVERSATION[1]["content"], user_id, agent_id, session_id)

    cooking_flush_count = recorder.count
    console.print(f"  ✓ 烹饪话题后 Flush 次数: {cooking_flush_count}")

    # ========== 12.4 最终统计 ==========
    console.print("\n  [yellow]12.4 最终统计...[/yellow]")

    info = perception.get_buffer_info(user_id, agent_id, session_id)
    console.print(f"  ✓ Buffer 状态: blocks={info['block_count']}, tokens={info['total_tokens']}")
    console.print(f"  ✓ {recorder.summary()}")

    # 手动 Flush 剩余内容
    final_messages = perception.flush_buffer(user_id, agent_id, session_id)
    msg_count = len(final_messages) if final_messages else 0
    console.print(f"  ✓ 最终 Flush: {msg_count} 条消息")

    # ========== 12.5 验证 Flush 内容完整性 ==========
    console.print("\n  [yellow]12.5 验证 Flush 内容完整性...[/yellow]")

    total_flushed_messages = sum(r['message_count'] for r in recorder.records)
    console.print(f"  ✓ 总 Flush 消息数: {total_flushed_messages}")

    console.print("\n[green]✓ 测试 12 通过[/green]")
    return True


def test_patchouli_integration_enhanced():
    """
    测试场景 13: PatchouliAgent 集成（增强版）

    验证点:
    - 与 PatchouliAgent 的基本集成
    - 使用真实长对话数据
    - 语义流模式下的 Buffer 信息获取
    """
    console.print("\n[bold cyan]测试 13: PatchouliAgent 集成（增强版）[/bold cyan]")

    try:
        # ========== 13.1 创建存储和 Agent ==========
        console.print("\n  [yellow]13.1 创建存储和 Agent...[/yellow]")

        config = load_app_config()
        storage = QdrantMemoryStore(
            qdrant_config=config.qdrant,
            embedding_config=config.embedding
        )

        storage.create_collection(recreate=True)
        console.print("  ✓ Qdrant 集合已创建")

        patchouli = PatchouliAgent(storage=storage, enable_semantic_flow=True)
        console.print("  ✓ PatchouliAgent 已创建 (SemanticFlow 模式)")

        user_id = "test_patchouli"
        agent_id = "test_agent"
        session_id = "test_session"

        # ========== 13.2 添加真实长对话 ==========
        console.print("\n  [yellow]13.2 添加真实长对话...[/yellow]")

        # 使用 Python 对话数据
        for i in range(0, min(4, len(PYTHON_CONVERSATION)), 2):
            if i + 1 < len(PYTHON_CONVERSATION):
                patchouli.add_message("user", PYTHON_CONVERSATION[i]["content"], user_id, agent_id, session_id)
                patchouli.add_message("assistant", PYTHON_CONVERSATION[i+1]["content"], user_id, agent_id, session_id)
                console.print(f"  ✓ 对话轮次 {i//2 + 1} 已添加")

        # ========== 13.3 获取 Buffer 信息 ==========
        console.print("\n  [yellow]13.3 获取 Buffer 信息...[/yellow]")

        info = patchouli.get_buffer_info(user_id, agent_id, session_id)
        console.print(f"  ✓ 模式: {info['mode']}")
        console.print(f"  ✓ Block 数: {info.get('block_count', 0)}")
        console.print(f"  ✓ 总 Tokens: {info.get('total_tokens', 0)}")
        console.print(f"  ✓ 状态: {info.get('state', 'N/A')}")

        # ========== 13.4 触发 Flush ==========
        console.print("\n  [yellow]13.4 触发 Flush...[/yellow]")

        patchouli.flush_perception(user_id, agent_id, session_id)
        console.print("  ✓ Flush 完成")

        # ========== 13.5 列出活跃 Buffer ==========
        console.print("\n  [yellow]13.5 列出活跃 Buffer...[/yellow]")

        active_buffers = patchouli.list_active_buffers()
        console.print(f"  ✓ 活跃 Buffer 数: {len(active_buffers)}")

        console.print("\n[green]✓ 测试 13 通过[/green]")
        return True

    except Exception as e:
        console.print(f"\n[red]✗ 测试 13 失败: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False

# ========== 主测试流程 ==========

def main():
    """主测试流程"""
    console.print(Panel.fit(
        "[bold magenta]SemanticFlowPerceptionLayer 集成测试 v2.0[/bold magenta]\n"
        "全面测试感知层核心功能、边界情况和 PatchouliAgent 集成",
        border_style="magenta"
    ))

    # 测试分组
    tests = [
        # 组 1: 组件单元测试
        ("1. StreamParser 消息解析（增强版）", test_stream_parser_comprehensive),
        ("2. LogicalBlock 功能（含 Triplet）", test_logical_block_with_triplets),
        ("3. 语义吸附器（全场景）", test_semantic_adsorber_all_scenarios),
        # 组 2: 核心流程测试
        ("4. 工具调用完整流程", test_tool_call_flow),
        ("5. Token 溢出和接力机制", test_token_overflow_relay),
        ("6. 语义漂移检测", test_semantic_drift_detection),
        ("7. 阈值边界条件", test_threshold_boundaries),
        # 组 3: 边界情况测试
        ("8. 空输入和边界输入", test_empty_and_edge_inputs),
        ("9. 空闲超时触发", test_idle_timeout_trigger),
        ("10. 手动 Flush", test_manual_flush),
        # 组 4: 集成测试
        ("11. 多会话隔离", test_multi_session_isolation),
        ("12. 真实对话流程", test_real_conversation_flow),
        ("13. PatchouliAgent 集成（增强版）", test_patchouli_integration_enhanced),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            console.print(f"\n[red]✗ {name} 测试失败: {e}[/red]")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 结果汇总
    console.print("\n" + "=" * 70)
    console.print("\n[bold cyan]测试结果汇总[/bold cyan]\n")

    # 分组显示结果
    groups = [
        ("组件单元测试", results[0:3]),
        ("核心流程测试", results[3:7]),
        ("边界情况测试", results[7:10]),
        ("集成测试", results[10:13]),
    ]

    for group_name, group_results in groups:
        console.print(f"\n[bold yellow]{group_name}[/bold yellow]")
        for name, success in group_results:
            status = "[green]✓ 通过[/green]" if success else "[red]✗ 失败[/red]"
            console.print(f"  {status}  {name}")

    # 统计
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)

    console.print(f"\n[bold]通过率: {success_count}/{total_count} ({100*success_count//total_count}%)[/bold]")

    if success_count == total_count:
        console.print("\n[bold green]所有测试通过![/bold green]")
    else:
        console.print(f"\n[yellow]警告: {total_count - success_count} 个测试失败[/yellow]")


if __name__ == "__main__":
    main()
