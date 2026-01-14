"""
HiveMemory 测试共享 Fixtures

提供跨测试文件的共享 fixtures 和辅助函数。

作者: HiveMemory Team
版本: 1.0.0
"""

from typing import List, Dict, Any, Optional
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

from hivememory.core.models import FlushReason
from hivememory.generation.models import ConversationMessage
from hivememory.core.config import HiveMemoryConfig
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
        messages: List[ConversationMessage],
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


# ========== 导出 ==========

__all__ = [
    "FlushRecorder",
    "print_flush_summary",
    "print_buffer_comparison",
    "print_test_header",
    "print_test_result",
]
