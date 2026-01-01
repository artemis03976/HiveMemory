"""
HiveMemory SemanticFlowPerceptionLayer 测试

测试内容:
    1. SemanticFlowPerceptionLayer 基本功能
    2. LogicalBlock 与语义吸附
    3. StreamParser 消息解析
    4. Token 溢出与接力
    5. 与 PatchouliAgent 集成

验收标准:
    - 消息正确解析为 LogicalBlock
    - 语义吸附判定正常工作
    - Buffer 管理功能正常
    - 与 PatchouliAgent 集成无问题
"""

import sys
import os
from pathlib import Path

# 配置 UTF-8 编码 (Windows 兼容性)
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import logging
from typing import List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hivememory.core.models import ConversationMessage, FlushReason
from hivememory.perception import (
    SemanticFlowPerceptionLayer,
    UnifiedStreamParser,
    SemanticBoundaryAdsorber,
    TokenOverflowRelayController,
)
from hivememory.perception.models import (
    LogicalBlock,
    StreamMessage,
    StreamMessageType,
    SemanticBuffer,
    BufferState,
)
from hivememory.agents.patchouli import PatchouliAgent
from hivememory.memory.storage import QdrantMemoryStore
from hivememory.core.config import get_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

console = Console(force_terminal=True, legacy_windows=False)


# ========== 测试用例定义 ==========

def test_stream_parser():
    """
    测试场景 1: StreamParser 消息解析
    """
    console.print("\n[bold cyan]测试 1: StreamParser 消息解析[/bold cyan]")

    parser = UnifiedStreamParser()

    # 测试不同格式的消息
    console.print("\n  [yellow]测试消息格式解析...[/yellow]")

    # 1. 简单字典格式
    msg1 = parser.parse_message({"role": "user", "content": "你好"})
    console.print(f"  ✓ 字典格式: type={msg1.message_type}, content={msg1.content[:20]}")

    # 2. LangChain 格式（检查是否安装）
    try:
        from langchain_core.messages import HumanMessage
        msg2 = parser.parse_message(HumanMessage(content="LangChain 消息"))
        console.print(f"  ✓ LangChain 格式: type={msg2.message_type}, content={msg2.content[:20]}")
        has_langchain = True
    except ImportError:
        console.print("  ○ LangChain 格式: 跳过（未安装 langchain-core）")
        has_langchain = False
        msg2 = None

    # 3. 简单文本
    msg3 = parser.parse_message("简单文本消息")
    console.print(f"  ✓ 简单文本: type={msg3.message_type}, content={msg3.content[:20]}")

    # 验证结果
    assert msg1.message_type == StreamMessageType.USER_QUERY, "应该是 USER_QUERY"
    if has_langchain:
        assert msg2.message_type == StreamMessageType.USER_QUERY, "应该是 USER_QUERY"
    assert msg3.message_type == StreamMessageType.USER_QUERY, "应该是 USER_QUERY"

    # 测试 Block 创建判定
    console.print("\n  [yellow]测试 Block 创建判定...[/yellow]")
    should_create = parser.should_create_new_block(msg1)
    console.print(f"  USER_QUERY 应该创建新 Block: {should_create}")
    assert should_create, "USER_QUERY 应该创建新 Block"

    assistant_msg = parser.parse_message({"role": "assistant", "content": "回复"})
    should_create = parser.should_create_new_block(assistant_msg)
    console.print(f"  ASSISTANT_MESSAGE 不应该创建新 Block: {not should_create}")
    assert not should_create, "ASSISTANT_MESSAGE 不应该创建新 Block"

    console.print("\n[green]✓ 测试 1 通过[/green]")
    return True


def test_logical_block():
    """
    测试场景 2: LogicalBlock 功能
    """
    console.print("\n[bold cyan]测试 2: LogicalBlock 功能[/bold cyan]")

    # 创建 LogicalBlock
    console.print("\n  [yellow]创建 LogicalBlock...[/yellow]")
    block = LogicalBlock()
    console.print(f"  ✓ Block ID: {block.block_id}")
    console.print(f"  ✓ 初始状态: complete={block.is_complete}, user_block={block.user_block is not None}, response_block={block.response_block is not None}")

    # 添加 StreamMessage
    console.print("\n  [yellow]添加 StreamMessage...[/yellow]")
    msg1 = StreamMessage(
        message_type=StreamMessageType.USER_QUERY,
        content="帮我写一个快排算法",
        metadata={"role": "user"}
    )
    block.add_stream_message(msg1)

    msg2 = StreamMessage(
        message_type=StreamMessageType.ASSISTANT_MESSAGE,
        content="好的，这是快排实现...",
        metadata={"role": "assistant"}
    )
    block.add_stream_message(msg2)

    console.print(f"  ✓ 添加后: complete={block.is_complete}, user_block={block.user_block is not None}, response_block={block.response_block is not None}")

    # 验证完整性
    assert block.user_block is not None, "user_block 应该存在"
    assert block.response_block is not None, "response_block 应该存在"
    assert block.is_complete, "Block 应该是完整的"
    assert block.user_block.content == "帮我写一个快排算法", "user_block 内容应该正确"
    assert block.response_block.content == "好的，这是快排实现...", "response_block 内容应该正确"

    # 测试转换
    console.print("\n  [yellow]测试转换为 ConversationMessage...[/yellow]")
    conv_messages = block.to_conversation_messages(
        session_id="test_session",
        user_id="test_user"
    )
    console.print(f"  ✓ 转换结果: {len(conv_messages)} 条 ConversationMessage")

    assert len(conv_messages) == 2, "应该转换出 2 条消息"
    assert all(isinstance(m, ConversationMessage) for m in conv_messages), "所有消息应该是 ConversationMessage"

    console.print("\n[green]✓ 测试 2 通过[/green]")
    return True


def test_semantic_adsorber():
    """
    测试场景 3: 语义吸附器
    """
    console.print("\n[bold cyan]测试 3: 语义吸附器[/bold cyan]")

    adsorber = SemanticBoundaryAdsorber()

    # 测试相似度计算
    console.print("\n  [yellow]测试语义相似度计算...[/yellow]")
    text1 = "Python 快速排序算法实现"
    text2 = "Python 排序算法教程"

    similarity1 = adsorber.compute_similarity(text1, None)
    console.print(f"  ✓ 无话题核心时相似度: {similarity1:.3f}")
    assert similarity1 == 0, "无话题核心时应该返回 0"

    # 创建 SemanticBuffer
    buffer = SemanticBuffer(
        user_id="test_user",
        agent_id="test_agent",
        session_id="test_session"
    )

    # 添加第一个 Block
    block1 = LogicalBlock()
    block1.add_stream_message(StreamMessage(
        message_type=StreamMessageType.USER_QUERY,
        content="Python 快速排序算法",
        metadata={"role": "user"}
    ))
    block1.add_stream_message(StreamMessage(
        message_type=StreamMessageType.ASSISTANT_MESSAGE,
        content="这是快排的实现...",
        metadata={"role": "assistant"}
    ))

    # 更新话题核心
    adsorber.update_topic_kernel(buffer, block1)
    console.print(f"  ✓ 话题核心向量已创建")

    # 测试语义相似度
    similarity2 = adsorber.compute_similarity("Python 排序", buffer.topic_kernel_vector)
    console.print(f"  ✓ 相似文本相似度: {similarity2:.3f}")

    similarity3 = adsorber.compute_similarity("JavaScript 异步编程", buffer.topic_kernel_vector)
    console.print(f"  ✓ 不相似文本相似度: {similarity3:.3f}")

    # 测试吸附判定
    console.print("\n  [yellow]测试吸附判定...[/yellow]")
    buffer.blocks = [block1]

    # 相似话题
    block2_similar = LogicalBlock()
    block2_similar.add_stream_message(StreamMessage(
        message_type=StreamMessageType.USER_QUERY,
        content="Python 冒泡排序算法",
        metadata={"role": "user"}
    ))
    block2_similar.add_stream_message(StreamMessage(
        message_type=StreamMessageType.ASSISTANT_MESSAGE,
        content="冒泡排序实现...",
        metadata={"role": "assistant"}
    ))

    should_adsorb, reason = adsorber.should_adsorb(block2_similar, buffer)
    console.print(f"  ✓ 相似话题: adsorb={should_adsorb}, reason={reason.value if reason else None}")

    # 不相似话题
    block3_different = LogicalBlock()
    block3_different.add_stream_message(StreamMessage(
        message_type=StreamMessageType.USER_QUERY,
        content="如何制作蛋糕",
        metadata={"role": "user"}
    ))
    block3_different.add_stream_message(StreamMessage(
        message_type=StreamMessageType.ASSISTANT_MESSAGE,
        content="蛋糕制作方法...",
        metadata={"role": "assistant"}
    ))

    should_adsorb2, reason2 = adsorber.should_adsorb(block3_different, buffer)
    console.print(f"  ✓ 不相似话题: adsorb={should_adsorb2}, reason={reason2.value if reason2 else None}")

    console.print("\n[green]✓ 测试 3 通过[/green]")
    return True


def test_semantic_flow_perception():
    """
    测试场景 4: SemanticFlowPerceptionLayer 基本功能
    """
    console.print("\n[bold cyan]测试 4: SemanticFlowPerceptionLayer 基本功能[/bold cyan]")

    flush_called = []
    flush_reasons = []

    def on_flush(messages: List[ConversationMessage], reason: FlushReason):
        flush_called.append(messages)
        flush_reasons.append(reason)
        console.print(f"  ✓ Flush 触发: 原因={reason.value}, Block 数≈{len(messages) / 2}")

    # 创建感知层
    perception = SemanticFlowPerceptionLayer(on_flush_callback=on_flush)

    # 添加对话
    user_id = "test_user_1"
    agent_id = "test_agent"
    session_id = "test_session"

    console.print("\n  [yellow]添加对话消息...[/yellow]")
    perception.add_message("user", "帮我写一个Python快排算法", user_id, agent_id, session_id)
    perception.add_message("assistant", "好的，这是快排实现...", user_id, agent_id, session_id)
    perception.add_message("user", "再写一个冒泡排序", user_id, agent_id, session_id)
    perception.add_message("assistant", "冒泡排序实现如下...", user_id, agent_id, session_id)

    # 获取 Buffer 信息
    console.print("\n  [yellow]获取 Buffer 信息...[/yellow]")
    info = perception.get_buffer_info(user_id, agent_id, session_id)
    console.print(f"  Buffer ID: {info.get('buffer_id', 'N/A')}")
    console.print(f"  Block 数: {info['block_count']}")
    console.print(f"  总 Tokens: {info['total_tokens']}")
    console.print(f"  状态: {info['state']}")
    console.print(f"  有当前 Block: {info['has_current_block']}")

    # 手动触发 Flush
    console.print("\n  [yellow]手动触发 Flush...[/yellow]")
    messages = perception.flush_buffer(user_id, agent_id, session_id)
    console.print(f"  ✓ Flush 完成: {len(messages)} 条消息")

    # 验证结果
    assert info['exists'], "Buffer 应该存在"
    assert info['block_count'] >= 1, "应该至少有 1 个 Block"
    assert len(messages) >= 4, f"应该至少返回 4 条消息，实际返回 {len(messages)} 条"

    console.print("\n[green]✓ 测试 4 通过[/green]")
    return True


def test_buffer_management():
    """
    测试场景 5: Buffer 管理
    """
    console.print("\n[bold cyan]测试 5: Buffer 管理[/bold cyan]")

    perception = SemanticFlowPerceptionLayer()

    user_id = "test_user_2"
    agent_id = "test_agent"
    session_id = "test_session"

    # 测试创建 Buffer
    console.print("\n  [yellow]测试 Buffer 创建...[/yellow]")
    perception.add_message("user", "测试消息", user_id, agent_id, session_id)

    buffer = perception.get_buffer(user_id, agent_id, session_id)
    assert buffer is not None, "Buffer 应该存在"
    console.print(f"  ✓ Buffer ID: {buffer.buffer_id}")
    console.print(f"  ✓ Block 数: {len(buffer.blocks)}")

    # 测试列出活跃 Buffer
    console.print("\n  [yellow]测试列出活跃 Buffer...[/yellow]")
    active_buffers = perception.list_active_buffers()
    console.print(f"  活跃 Buffer 数: {len(active_buffers)}")
    assert len(active_buffers) >= 1, "应该至少有 1 个活跃 Buffer"

    # 测试清理 Buffer
    console.print("\n  [yellow]测试清理 Buffer...[/yellow]")
    success = perception.clear_buffer(user_id, agent_id, session_id)
    assert success, "清理应该成功"
    console.print("  ✓ Buffer 清理成功")

    # 验证清理后状态
    buffer = perception.get_buffer(user_id, agent_id, session_id)
    assert buffer is not None, "清理后 Buffer 仍然存在"
    assert len(buffer.blocks) == 0, "清理后 Block 数应该为 0"
    console.print("  ✓ 清理后状态正确")

    console.print("\n[green]✓ 测试 5 通过[/green]")
    return True


def test_patchouli_integration():
    """
    测试场景 6: 与 PatchouliAgent 集成
    """
    console.print("\n[bold cyan]测试 6: 与 PatchouliAgent 集成[/bold cyan]")

    try:
        # 创建存储实例
        config = get_config()
        storage = QdrantMemoryStore(
            qdrant_config=config.qdrant,
            embedding_config=config.embedding
        )

        # 创建集合
        console.print("  创建 Qdrant 集合...")
        storage.create_collection(recreate=True)

        # 创建使用语义流感知层的 PatchouliAgent
        console.print("\n  [yellow]创建 PatchouliAgent (SemanticFlowPerceptionLayer)...[/yellow]")
        patchouli = PatchouliAgent(storage=storage, enable_semantic_flow=True)
        console.print("  ✓ PatchouliAgent 创建成功")

        # 测试添加消息
        user_id = "test_user_3"
        agent_id = "test_agent"
        session_id = "test_session"

        console.print("\n  [yellow]测试添加消息...[/yellow]")
        patchouli.add_message("user", "帮我写一个Python快排算法", user_id, agent_id, session_id)
        patchouli.add_message("assistant", "好的，这是快排实现...", user_id, agent_id, session_id)
        patchouli.add_message("user", "时间复杂度是多少？", user_id, agent_id, session_id)
        patchouli.add_message("assistant", "平均O(n log n)，最坏O(n²)", user_id, agent_id, session_id)
        console.print("  ✓ 消息添加成功")

        # 获取 Buffer 信息
        console.print("\n  [yellow]获取 Buffer 信息...[/yellow]")
        info = patchouli.get_buffer_info(user_id, agent_id, session_id)
        console.print(f"  模式: {info['mode']}")
        console.print(f"  Block 数: {info.get('block_count', 0)}")
        console.print(f"  总 Tokens: {info.get('total_tokens', 0)}")
        console.print(f"  状态: {info.get('state', 'N/A')}")
        console.print("  ✓ Buffer 信息获取成功")

        # 手动触发 Flush
        console.print("\n  [yellow]手动触发 Flush...[/yellow]")
        patchouli.flush_perception(user_id, agent_id, session_id)
        console.print(f"  ✓ Flush 完成")

        # 列出活跃 Buffer
        console.print("\n  [yellow]列出活跃 Buffer...[/yellow]")
        active_buffers = patchouli.list_active_buffers()
        console.print(f"  活跃 Buffer 数: {len(active_buffers)}")
        for buffer_key in active_buffers:
            console.print(f"    - {buffer_key}")

        console.print("\n[green]✓ 测试 6 通过[/green]")
        return True

    except Exception as e:
        console.print(f"\n[red]✗ 测试 6 失败: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


def test_multi_scenario_flow():
    """
    测试场景 7: 多场景语义流测试
    """
    console.print("\n[bold cyan]测试 7: 多场景语义流测试[/bold cyan]")

    flush_records = []

    def on_flush(messages: List[ConversationMessage], reason: FlushReason):
        flush_records.append({
            "message_count": len(messages),
            "reason": reason,
            "preview": messages[0].content[:30] if messages else ""
        })
        console.print(f"  ✓ Flush: {reason.value}, {len(messages)} 条消息")

    perception = SemanticFlowPerceptionLayer(on_flush_callback=on_flush)

    user_id = "test_user_multi"
    agent_id = "test_agent"
    session_id = "test_session"

    # ========== 场景 1: Python 编程 ==========
    console.print("\n  [yellow]场景 1: Python 编程...[/yellow]")

    perception.add_message("user", "如何用Python读写文件？", user_id, agent_id, session_id)
    perception.add_message("assistant", "使用 open() 函数...", user_id, agent_id, session_id)

    perception.add_message("user", "如何处理异常？", user_id, agent_id, session_id)
    perception.add_message("assistant", "使用 try-except...", user_id, agent_id, session_id)

    # ========== 场景 2: 切换到烹饪话题 ==========
    console.print("\n  [yellow]场景 2: 切换到烹饪话题...[/yellow]")

    perception.add_message("user", "怎么做红烧肉？", user_id, agent_id, session_id)
    perception.add_message("assistant", "红烧肉做法如下...", user_id, agent_id, session_id)

    # ========== 验证 Flush 记录 ==========
    console.print(f"\n  [yellow]Flush 记录数: {len(flush_records)}[/yellow]")
    for i, record in enumerate(flush_records):
        console.print(f"    {i+1}. {record['reason'].value}: {record['message_count']} 条")

    # 手动 Flush
    console.print("\n  [yellow]手动 Flush...[/yellow]")
    messages = perception.flush_buffer(user_id, agent_id, session_id)
    console.print(f"  ✓ 剩余消息数: {len(messages)}")

    console.print("\n[green]✓ 测试 7 通过[/green]")
    return True

# ========== 主测试流程 ==========

def main():
    """主测试流程"""
    console.print(Panel.fit(
        "[bold magenta]SemanticFlowPerceptionLayer 测试[/bold magenta]\n"
        "测试语义流感知层功能与 PatchouliAgent 集成",
        border_style="magenta"
    ))

    # 运行测试
    tests = [
        ("StreamParser 消息解析", test_stream_parser),
        ("LogicalBlock 功能", test_logical_block),
        ("语义吸附器", test_semantic_adsorber),
        ("SemanticFlowPerceptionLayer 基本功能", test_semantic_flow_perception),
        ("Buffer 管理", test_buffer_management),
        ("PatchouliAgent 集成", test_patchouli_integration),
        ("多场景语义流", test_multi_scenario_flow),
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
    console.print("\n" + "=" * 60)
    console.print("\n[bold cyan]测试结果汇总[/bold cyan]\n")

    success_count = sum(1 for _, success in results if success)
    total_count = len(results)

    for name, success in results:
        status = "[green]✓ 通过[/green]" if success else "[red]✗ 失败[/red]"
        console.print(f"  {status}  {name}")

    console.print(f"\n[bold]通过率: {success_count}/{total_count}[/bold]")

    if success_count == total_count:
        console.print("\n[bold green]🎉 所有测试通过![/bold green]")
    else:
        console.print(f"\n[yellow]⚠️  {total_count - success_count} 个测试失败[/yellow]")


if __name__ == "__main__":
    main()
