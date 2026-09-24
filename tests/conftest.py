"""
HiveMemory 测试共享 Fixtures

提供跨测试文件的共享 fixtures 和辅助函数。

作者: HiveMemory Team
版本: 2.0.0
"""

import os

# 添加项目根目录到路径
import sys
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from rich.table import Table

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from unittest.mock import patch

from hivememory.i18n import set_default_language
from hivememory.system.config import HiveMemoryConfig

# ========== Pytest Fixtures ==========


@pytest.fixture(autouse=True)
def reset_i18n_default_language_between_tests():
    """防止进程级 i18n 状态在测试之间泄漏。"""
    set_default_language("zh")
    yield
    set_default_language("zh")


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
def console() -> Console:
    """
    提供 Rich Console 实例用于测试输出

    Usage:
        def test_something(console):
            console.print("[green]Test passed[/green]")
    """
    return Console(force_terminal=True, legacy_windows=False)


# ========== 辅助函数 ==========


def print_buffer_comparison(
    console: Console,
    before: dict[str, Any],
    after: dict[str, Any],
    title: str = "Buffer State Change",
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

    for key in ["block_count", "total_tokens", "message_count"]:
        if key in before and key in after:
            delta = after[key] - before[key]
            if delta > 0:
                delta_str = f"[green]+{delta}[/green]"
            elif delta < 0:
                delta_str = f"[red]{delta}[/red]"
            else:
                delta_str = "[dim]0[/dim]"
            table.add_row(key, str(before[key]), str(after[key]), delta_str)

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


def print_test_result(
    console: Console, test_name: str, success: bool, error: str | None = None
) -> None:
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
    "print_buffer_comparison",
    "print_test_header",
    "print_test_result",
]
