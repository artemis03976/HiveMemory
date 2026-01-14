"""
HiveMemory SimplePerceptionLayer 测试

测试内容:
    1. SimplePerceptionLayer 基本功能
    2. 三重触发机制
    3. Buffer 管理
    4. 与 PatchouliAgent 集成

验收标准:
    - 消息正确添加到缓冲区
    - 触发机制正常工作
    - Buffer 管理功能正常
    - 与 PatchouliAgent 集成无问题
"""

import sys
import os
from pathlib import Path

# 设置 UTF-8 编码 (Windows 兼容性)
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import time
import logging
from typing import List
from unittest.mock import MagicMock, patch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hivememory.core.models import FlushReason
from hivememory.generation.models import ConversationMessage
from hivememory.perception import SimplePerceptionLayer
from hivememory.perception.trigger_strategies import (
    TriggerManager,
    MessageCountTrigger,
    IdleTimeoutTrigger,
    SemanticBoundaryTrigger,
)
from hivememory.agents.patchouli import PatchouliAgent
from hivememory.memory.storage import QdrantMemoryStore

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

console = Console(force_terminal=True, legacy_windows=False)


# ========== 测试用例定义 ==========

def test_simple_perception_basic():
    """
    测试场景 1: SimplePerceptionLayer 基本功能
    """
    console.print("\n[bold cyan]测试 1: SimplePerceptionLayer 基本功能[/bold cyan]")

    flush_called = []
    flush_reasons = []

    def on_flush(messages: List[ConversationMessage], reason: FlushReason):
        flush_called.append(messages)
        flush_reasons.append(reason)
        console.print(f"  ✓ Flush 触发: 原因={reason.value}, 消息数={len(messages)}")

    # 创建感知层
    perception = SimplePerceptionLayer(on_flush_callback=on_flush)

    # 添加消息
    user_id = "test_user_1"
    agent_id = "test_agent"
    session_id = "test_session"

    perception.add_message("user", "你好", user_id, agent_id, session_id)
    perception.add_message("assistant", "你好！有什么可以帮助你的吗？", user_id, agent_id, session_id)
    perception.add_message("user", "帮我写一个Python函数", user_id, agent_id, session_id)

    # 验证 Buffer 信息（添加消息后）
    info = perception.get_buffer_info(user_id, agent_id, session_id)
    console.print(f"  Buffer 信息: 消息数={info['message_count']}, 状态={'存在' if info['exists'] else '不存在'}")
    assert info['exists'], "Buffer 应该存在"
    assert info['message_count'] == 3, f"添加后应该有 3 条消息，实际有 {info['message_count']} 条"

    # 手动触发 Flush
    messages = perception.flush_buffer(user_id, agent_id, session_id)
    console.print(f"  ✓ 手动 Flush: 返回 {len(messages)} 条消息")

    # Flush 后再次获取 Buffer 信息验证清空
    info_after_flush = perception.get_buffer_info(user_id, agent_id, session_id)
    console.print(f"  Flush 后 Buffer 信息: 消息数={info_after_flush['message_count']}, 状态={'存在' if info_after_flush['exists'] else '不存在'}")

    # 验证结果
    assert info_after_flush['exists'], "Flush 后 Buffer 仍然应该存在"
    assert info_after_flush['message_count'] == 0, f"Flush 后消息数应该为 0，实际为 {info_after_flush['message_count']}"
    assert len(messages) == 3, f"应该返回 3 条消息，实际返回 {len(messages)} 条"
    assert len(flush_called) >= 1, "Flush 回调应该被调用"

    console.print("[green]✓ 测试 1 通过[/green]")
    return True


def test_trigger_mechanisms():
    """
    测试场景 2: 三重触发机制
    """
    console.print("\n[bold cyan]测试 2: 三重触发机制[/bold cyan]")

    flush_count = []
    flush_triggers = []

    def on_flush(messages: List[ConversationMessage], reason: FlushReason):
        flush_count.append(len(messages))
        flush_triggers.append(reason)
        console.print(f"  ✓ Flush 触发: 原因={reason.value}, 消息数={len(messages)}")

    # 测试消息数触发
    console.print("\n  [yellow]测试消息数触发...[/yellow]")
    trigger_manager = TriggerManager(strategies=[
        MessageCountTrigger(threshold=3)
    ])
    perception = SimplePerceptionLayer(
        trigger_manager=trigger_manager,
        on_flush_callback=on_flush
    )

    perception.add_message("user", "消息1", "user2", "agent", "sess")
    perception.add_message("assistant", "回复1", "user2", "agent", "sess")
    perception.add_message("user", "消息2", "user2", "agent", "sess")
    # 应该触发 Flush

    assert len(flush_count) >= 1, "消息数触发应该生效"
    assert FlushReason.MESSAGE_COUNT in flush_triggers, "应该触发 MESSAGE_COUNT"
    console.print("  [green]✓ 消息数触发正常[/green]")

    # 测试空闲超时触发
    console.print("\n  [yellow]测试空闲超时触发...[/yellow]")
    flush_count.clear()
    flush_triggers.clear()

    trigger_manager = TriggerManager(strategies=[
        IdleTimeoutTrigger(timeout=5)
    ])
    perception = SimplePerceptionLayer(
        trigger_manager=trigger_manager,
        on_flush_callback=on_flush
    )

    perception.add_message("user", "消息1", "user3", "agent", "sess")
    console.print("  等待超时...")
    time.sleep(6)

    perception.add_message("user", "消息2", "user3", "agent", "sess")
    # 应该触发 Flush

    assert len(flush_count) >= 1, "空闲超时触发应该生效"
    assert FlushReason.IDLE_TIMEOUT in flush_triggers, "应该触发 IDLE_TIMEOUT"
    console.print("  [green]✓ 空闲超时触发正常[/green]")

    # 测试语义边界触发
    console.print("\n  [yellow]测试语义边界触发...[/yellow]")
    flush_count.clear()
    flush_triggers.clear()

    trigger_manager = TriggerManager(strategies=[
        SemanticBoundaryTrigger()
    ])
    perception = SimplePerceptionLayer(
        trigger_manager=trigger_manager,
        on_flush_callback=on_flush
    )

    # 添加包含结束语的对话
    perception.add_message("user", "如何使用Python？", "user4", "agent", "sess")
    perception.add_message("assistant", "Python是一种编程语言，希望这对您有帮助！", "user4", "agent", "sess")
    # 应该触发 Flush

    assert len(flush_count) >= 1, "语义边界触发应该生效"
    assert FlushReason.SEMANTIC_DRIFT in flush_triggers, "应该触发 SEMANTIC_DRIFT"
    console.print("  [green]✓ 语义边界触发正常[/green]")

    console.print("\n[green]✓ 测试 2 通过[/green]")
    return True


def test_buffer_management():
    """
    测试场景 3: Buffer 管理
    """
    console.print("\n[bold cyan]测试 3: Buffer 管理[/bold cyan]")

    perception = SimplePerceptionLayer()

    user_id = "test_user_4"
    agent_id = "test_agent"
    session_id = "test_session"

    # 测试创建 Buffer
    console.print("\n  [yellow]测试 Buffer 创建...[/yellow]")
    perception.add_message("user", "测试消息", user_id, agent_id, session_id)

    buffer = perception.get_buffer(user_id, agent_id, session_id)
    assert buffer is not None, "Buffer 应该存在"
    assert buffer.message_count == 1, "应该有 1 条消息"
    console.print("  ✓ Buffer 创建成功")

    # 测试 Buffer 信息
    console.print("\n  [yellow]测试 Buffer 信息查询...[/yellow]")
    info = perception.get_buffer_info(user_id, agent_id, session_id)
    console.print(f"  Buffer ID: {info.get('buffer_id', 'N/A')}")
    console.print(f"  消息数: {info['message_count']}")
    console.print(f"  用户 ID: {info['user_id']}")
    console.print(f"  Agent ID: {info['agent_id']}")
    console.print(f"  会话 ID: {info['session_id']}")
    console.print("  ✓ Buffer 信息查询正常")

    # 测试列出活跃 Buffer
    console.print("\n  [yellow]测试列出活跃 Buffer...[/yellow]")
    active_buffers = perception.list_active_buffers()
    console.print(f"  活跃 Buffer 数: {len(active_buffers)}")
    assert len(active_buffers) >= 1, "应该至少有 1 个活跃 Buffer"
    console.print("  ✓ 列出活跃 Buffer 正常")

    # 测试清理 Buffer
    console.print("\n  [yellow]测试清理 Buffer...[/yellow]")
    success = perception.clear_buffer(user_id, agent_id, session_id)
    assert success, "清理应该成功"
    console.print("  ✓ Buffer 清理成功")

    # 验证清理后状态
    info = perception.get_buffer_info(user_id, agent_id, session_id)
    assert info['message_count'] == 0, "清理后消息数应该为 0"
    console.print("  ✓ 清理后状态正确")

    console.print("\n[green]✓ 测试 3 通过[/green]")
    return True


def test_patchouli_integration():
    """
    测试场景 4: 与 PatchouliAgent 集成
    """
    console.print("\n[bold cyan]测试 4: 与 PatchouliAgent 集成[/bold cyan]")

    try:
        # 使用 Mock 存储，避免连接真实 Qdrant
        storage = MagicMock(spec=QdrantMemoryStore)

        # 创建使用简单感知层的 PatchouliAgent
        # 使用 MemoryPerceptionConfig 替代过时的 enable_semantic_flow 参数
        from hivememory.core.config import MemoryPerceptionConfig
        perception_config = MemoryPerceptionConfig(layer_type="simple")

        console.print("\n  [yellow]创建 PatchouliAgent (SimplePerceptionLayer)...[/yellow]")
        patchouli = PatchouliAgent(
            storage=storage,
            perception_config=perception_config
        )
        console.print("  ✓ PatchouliAgent 创建成功")

        # 测试添加消息
        user_id = "test_user_5"
        agent_id = "test_agent"
        session_id = "test_session"

        console.print("\n  [yellow]测试添加消息...[/yellow]")
        patchouli.add_message("user", "你好", user_id, agent_id, session_id)
        patchouli.add_message("assistant", "你好！有什么可以帮助你的吗？", user_id, agent_id, session_id)
        patchouli.add_message("user", "帮我写一个Python快排算法", user_id, agent_id, session_id)
        console.print("  ✓ 消息添加成功")

        # 获取 Buffer 信息
        console.print("\n  [yellow]获取 Buffer 信息...[/yellow]")
        info = patchouli.get_buffer_info(user_id, agent_id, session_id)
        console.print(f"  模式: {info['mode']}")
        console.print(f"  消息数: {info['message_count']}")
        console.print("  ✓ Buffer 信息获取成功")

        # Mock 生成编排器，避免调用真实 LLM
        patchouli.generation_orchestrator = MagicMock()
        # 模拟返回一条记忆
        mock_memory = MagicMock()
        mock_memory.content = "Mock Memory"
        patchouli.generation_orchestrator.process.return_value = [mock_memory]

        # 手动触发 Flush
        console.print("\n  [yellow]手动触发 Flush...[/yellow]")
        
        # 注册观察者捕获结果
        results = []
        def observer(event):
            results.extend(event.memories)
        patchouli.add_flush_observer(observer)

        # 使用 flush_perception
        patchouli.flush_perception(user_id, agent_id, session_id)
        
        memories = results
        console.print(f"  ✓ Flush 完成, 提取了 {len(memories)} 条记忆")

        # 列出活跃 Buffer
        console.print("\n  [yellow]列出活跃 Buffer...[/yellow]")
        active_buffers = patchouli.list_active_buffers()
        console.print(f"  活跃 Buffer 数: {len(active_buffers)}")
        for buffer_key in active_buffers:
            console.print(f"    - {buffer_key}")

        console.print("\n[green]✓ 测试 4 通过[/green]")
        return True

    except Exception as e:
        console.print(f"\n[red]✗ 测试 4 失败: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


def test_concurrent_access():
    """
    测试场景 5: 并发访问测试
    """
    console.print("\n[bold cyan]测试 5: 并发访问测试[/bold cyan]")

    import threading

    perception = SimplePerceptionLayer()
    errors = []

    def add_messages(worker_id: int):
        try:
            for i in range(10):
                perception.add_message(
                    "user",
                    f"Worker {worker_id} - Message {i}",
                    f"user_{worker_id}",
                    f"agent_{worker_id}",
                    f"sess_{worker_id}"
                )
        except Exception as e:
            errors.append(e)

    # 创建多个线程
    console.print("\n  [yellow]启动 5 个并发线程...[/yellow]")
    threads = []
    for i in range(5):
        t = threading.Thread(target=add_messages, args=(i,))
        threads.append(t)
        t.start()

    # 等待所有线程完成
    for t in threads:
        t.join()

    # 验证结果
    console.print(f"  活跃 Buffer 数: {len(perception.list_active_buffers())}")
    console.print(f"  错误数: {len(errors)}")

    assert len(errors) == 0, f"并发访问应该无错误，实际发生 {len(errors)} 个错误"
    assert len(perception.list_active_buffers()) == 5, "应该有 5 个活跃 Buffer"

    console.print("\n[green]✓ 测试 5 通过[/green]")
    return True


# ========== 主测试流程 ==========

def main():
    """主测试流程"""
    console.print(Panel.fit(
        "[bold magenta]SimplePerceptionLayer 测试[/bold magenta]\n"
        "测试简单感知层功能与 PatchouliAgent 集成",
        border_style="magenta"
    ))

    # 运行测试
    tests = [
        ("基本功能", test_simple_perception_basic),
        ("三重触发机制", test_trigger_mechanisms),
        ("Buffer 管理", test_buffer_management),
        ("PatchouliAgent 集成", test_patchouli_integration),
        ("并发访问", test_concurrent_access),
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
