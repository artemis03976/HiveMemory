"""
HiveMemory 测试共享 Fixtures

提供跨测试文件的共享 fixtures 和辅助函数。

作者: HiveMemory Team
版本: 2.0.0
"""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import os

import pytest
from rich.console import Console
from rich.table import Table

# 添加项目根目录到路径
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from hivememory.core.models import StreamMessage, MemoryAtom, Identity
from hivememory.engines.perception.models import FlushReason, FlushEvent, LogicalBlock, SemanticBuffer
from hivememory.system.config import HiveMemoryConfig
from unittest.mock import patch


# ========== FlushRecorder 类 ==========

class FlushRecorder:
    """
    Flush 事件记录器

    用于在测试中记录感知层的 flush 事件，方便验证触发条件和原因。
    """

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def __call__(
        self,
        messages: List[StreamMessage],
        reason: FlushReason
    ) -> None:
        """
        记录 flush 事件

        Args:
            messages: 被 flush 的消息列表
            reason: Flush 原因
        """
        self.records.append({
            "message_count": len(messages),
            "reason": reason,
            "messages": messages,
            "preview": messages[0].content[:50] if messages else "",
            "timestamp": datetime.now().timestamp(),
        })

    def get_flushes_by_reason(self, reason: FlushReason) -> List[Dict[str, Any]]:
        """
        获取指定原因的 flush 记录

        Args:
            reason: Flush 原因

        Returns:
            List[Dict]: 匹配的记录列表
        """
        return [r for r in self.records if r['reason'] == reason]

    def get_last_flush(self) -> Optional[Dict[str, Any]]:
        """获取最后一次 flush 记录"""
        return self.records[-1] if self.records else None

    def clear(self) -> None:
        """清空所有记录"""
        self.records.clear()

    @property
    def count(self) -> int:
        """获取 flush 总次数"""
        return len(self.records)

    def summary(self) -> str:
        """获取摘要字符串"""
        if not self.records:
            return "No flush records"

        reason_counts = {}
        for record in self.records:
            reason = record['reason'].value
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        parts = [f"{reason}: {count}" for reason, count in reason_counts.items()]
        return f"Total: {self.count}, " + ", ".join(parts)


# ========== Pytest Fixtures ==========

@pytest.fixture
def mock_env():
    """
    提供一个干净的环境变量上下文
    
    使用 patch.dict 确保测试期间的环境变量更改不会影响其他测试或系统。
    """
    with patch.dict(os.environ):
        yield os.environ


@pytest.fixture
def test_config(mock_env):
    """
    提供测试用的 HiveMemoryConfig 实例
    
    强制忽略本地配置文件，使用默认值。
    """
    # 指向不存在的配置文件路径，确保只使用默认值和环境变量
    mock_env["HIVEMEMORY_CONFIG_PATH"] = "non_existent_config_for_test.yaml"
    return HiveMemoryConfig()


@pytest.fixture
def flush_recorder() -> FlushRecorder:
    """
    提供 FlushRecorder 实例

    Usage:
        def test_something(flush_recorder):
            recorder = flush_recorder
            perception = SemanticFlowPerceptionLayer(
                on_flush_callback=recorder
            )
            # ... test code ...
            assert recorder.count > 0
    """
    return FlushRecorder()


@pytest.fixture
def console() -> Console:
    """
    提供 Rich Console 实例用于测试输出

    Usage:
        def test_something(console):
            console.print("[green]Test passed[/green]")
    """
    return Console(force_terminal=True, legacy_windows=False)


# ========== 辅助函数 ==========

def print_flush_summary(
    console: Console,
    flush_records: List[Dict[str, Any]],
    title: str = "Flush Events Summary"
) -> None:
    """
    打印格式化的 flush 摘要表格

    Args:
        console: Rich Console 实例
        flush_records: flush 记录列表
        title: 表格标题
    """
    if not flush_records:
        console.print(f"[dim]No flush records to display[/dim]")
        return

    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Reason", style="yellow", width=20)
    table.add_column("Messages", justify="right", width=8)
    table.add_column("Preview", style="dim", width=40)

    for i, record in enumerate(flush_records):
        table.add_row(
            str(i + 1),
            record['reason'].value,
            str(record['message_count']),
            record['preview']
        )

    console.print(table)


def print_buffer_comparison(
    console: Console,
    before: Dict[str, Any],
    after: Dict[str, Any],
    title: str = "Buffer State Change"
) -> None:
    """
    打印 buffer 状态对比表格

    Args:
        console: Rich Console 实例
        before: 之前的状态
        after: 之后的状态
        title: 表格标题
    """
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", width=15)
    table.add_column("Before", justify="right", width=10)
    table.add_column("After", justify="right", width=10)
    table.add_column("Delta", justify="right", width=10)

    for key in ['block_count', 'total_tokens', 'message_count']:
        if key in before and key in after:
            delta = after[key] - before[key]
            if delta > 0:
                delta_str = f"[green]+{delta}[/green]"
            elif delta < 0:
                delta_str = f"[red]{delta}[/red]"
            else:
                delta_str = "[dim]0[/dim]"
            table.add_row(
                key,
                str(before[key]),
                str(after[key]),
                delta_str
            )

    console.print(table)


def print_test_header(console: Console, test_name: str) -> None:
    """
    打印测试标题

    Args:
        console: Rich Console 实例
        test_name: 测试名称
    """
    console.print(f"\n{'='*60}")
    console.print(f"[bold cyan]{test_name}[/bold cyan]")
    console.print(f"{'='*60}")


def print_test_result(console: Console, test_name: str, success: bool, error: Optional[str] = None) -> None:
    """
    打印测试结果

    Args:
        console: Rich Console 实例
        test_name: 测试名称
        success: 是否成功
        error: 错误信息（如果失败）
    """
    if success:
        console.print(f"[green]✓[/green] {test_name}")
    else:
        console.print(f"[red]✗[/red] {test_name}")
        if error:
            console.print(f"    [red]{error}[/red]")


# ========== Mock 类 (用于冷链路测试) ==========

class MockGenerationEngine:
    """
    Mock 记忆生成引擎

    记录 process 调用，验证生成触发时机，不实际生成记忆。

    Usage:
        >>> mock_engine = MockGenerationEngine()
        >>> # ... 测试代码 ...
        >>> assert mock_engine.call_count > 0
        >>> assert mock_engine.last_call["message_count"] == 5
    """

    def __init__(self):
        self.process_calls: List[Dict[str, Any]] = []

    def process(self, messages: List[StreamMessage]) -> List[MemoryAtom]:
        """
        记录 process 调用

        Args:
            messages: 消息列表

        Returns:
            空列表（不实际生成记忆）
        """
        self.process_calls.append({
            "message_count": len(messages),
            "messages": messages,
            "timestamp": datetime.now(),
            "first_content": messages[0].content[:50] if messages else "",
        })
        return []  # 不实际生成记忆

    @property
    def call_count(self) -> int:
        """获取调用次数"""
        return len(self.process_calls)

    @property
    def last_call(self) -> Optional[Dict[str, Any]]:
        """获取最后一次调用"""
        return self.process_calls[-1] if self.process_calls else None

    def clear(self) -> None:
        """清空调用记录"""
        self.process_calls.clear()


class MockLifecycleEngine:
    """
    Mock 生命周期引擎

    空操作实现，用于测试时屏蔽生命周期管理。
    """

    def __init__(self):
        pass

    def refresh_vitality(self, memory, *, persist: bool = False) -> float:
        memory.meta.vitality_score = 100.0
        return 100.0

    def refresh_vitality_batch(self, memories, *, persist: bool = False):
        return [(memory.id, self.refresh_vitality(memory, persist=persist)) for memory in memories]

    def record_event(self, event) -> None:
        pass

    def run_garbage_collection(self, force: bool = False) -> int:
        return 0


class MockRetrievalFamiliar:
    """
    Mock 检索使魔

    返回空结果，用于测试时屏蔽检索功能。
    """

    def __init__(self):
        self.search_calls: List[Dict[str, Any]] = []

    def search(self, query: str, **kwargs) -> List:
        """记录搜索调用并返回空结果"""
        self.search_calls.append({
            "query": query,
            "kwargs": kwargs,
            "timestamp": datetime.now(),
        })
        return []

    @property
    def call_count(self) -> int:
        return len(self.search_calls)

    def clear(self) -> None:
        self.search_calls.clear()


class FlushEventRecorder:
    """
    Flush 事件记录器 (基于 FlushEvent 对象)

    记录 FlushEvent 对象，用于验证 Flush 触发条件。
    与 FlushRecorder 不同，此类直接接收 FlushEvent 对象。

    Usage:
        >>> recorder = FlushEventRecorder()
        >>> librarian_core.add_flush_observer(recorder)
        >>> # ... 测试代码 ...
        >>> assert recorder.count > 0
        >>> drift_events = recorder.get_events_by_reason(FlushReason.SEMANTIC_DRIFT)
    """

    def __init__(self):
        self.events: List[FlushEvent] = []

    def __call__(self, event: FlushEvent) -> None:
        """接收 FlushEvent"""
        self.events.append(event)

    def get_events_by_reason(self, reason: FlushReason) -> List[FlushEvent]:
        """获取指定原因的事件"""
        return [e for e in self.events if e.flush_reason == reason]

    @property
    def count(self) -> int:
        """获取事件总数"""
        return len(self.events)

    @property
    def last_event(self) -> Optional[FlushEvent]:
        """获取最后一个事件"""
        return self.events[-1] if self.events else None

    def clear(self) -> None:
        """清空事件记录"""
        self.events.clear()

    def summary(self) -> str:
        """获取摘要"""
        if not self.events:
            return "No flush events"

        reason_counts = {}
        for event in self.events:
            reason = event.flush_reason.value
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        parts = [f"{reason}: {count}" for reason, count in reason_counts.items()]
        return f"Total: {self.count}, " + ", ".join(parts)


# ========== Pytest Fixtures (Mock 相关) ==========

@pytest.fixture
def mock_generation_engine() -> MockGenerationEngine:
    """提供 MockGenerationEngine 实例"""
    return MockGenerationEngine()


@pytest.fixture
def mock_lifecycle_engine() -> MockLifecycleEngine:
    """提供 MockLifecycleEngine 实例"""
    return MockLifecycleEngine()


@pytest.fixture
def mock_retrieval_familiar() -> MockRetrievalFamiliar:
    """提供 MockRetrievalFamiliar 实例"""
    return MockRetrievalFamiliar()


@pytest.fixture
def flush_event_recorder() -> FlushEventRecorder:
    """提供 FlushEventRecorder 实例"""
    return FlushEventRecorder()


# ========== 导出 ==========

__all__ = [
    # 记录器
    "FlushRecorder",
    "FlushEventRecorder",
    # Mock 类
    "MockGenerationEngine",
    "MockLifecycleEngine",
    "MockRetrievalFamiliar",
    # 辅助函数
    "print_flush_summary",
    "print_buffer_comparison",
    "print_test_header",
    "print_test_result",
]
