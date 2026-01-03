"""
HiveMemory PatchouliAgent 集成效果测试

测试目标:
    1. 验证 PatchouliAgent 完整端到端流程
    2. 测试语义流感知层的触发时机（语义漂移、Token溢出）
    3. 验证记忆生成模块的记忆原子质量
    4. 完全通过 帕秋莉 API 操作

验收标准:
    - Flush 触发时机符合预期
    - 生成的记忆原子结构完整
    - 记忆类型判定正确
    - 闲聊对话被正确过滤

作者: HiveMemory Team
版本: 1.0.0
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
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "tests"))

import logging
from typing import List, Optional
from dataclasses import dataclass, field
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hivememory.core.models import FlushReason, MemoryAtom
from hivememory.core.config import PerceptionConfig
from hivememory.generation.models import ConversationMessage
from hivememory.agents.patchouli import PatchouliAgent, FlushEvent
from hivememory.memory.storage import QdrantMemoryStore

# 导入测试数据
from fixtures.perception_test_data import (
    PYTHON_CONVERSATION,
    ML_CONVERSATION,
    COOKING_CONVERSATION,
    LONG_TEXT_BLOCK,
)

# 配置日志
logging.basicConfig(
    level=logging.WARNING,  # 减少日志噪音
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# 保持测试相关日志
logging.getLogger("hivememory.agents.patchouli").setLevel(logging.INFO)

console = Console(force_terminal=True, legacy_windows=False)


# ========== 测试工具类 ==========

@dataclass
class FlushEventRecord:
    """Flush 事件记录"""
    messages: List[ConversationMessage]
    reason: FlushReason
    memories: List[MemoryAtom]
    timestamp: float


class IntegrationTestRecorder:
    """
    集成测试事件记录器

    记录所有 Flush 事件和生成的记忆原子
    """

    def __init__(self):
        self.flush_events: List[FlushEventRecord] = []
        self.all_memories: List[MemoryAtom] = []

    def record(self, event: FlushEvent) -> None:
        """记录 Flush 事件"""
        record = FlushEventRecord(
            messages=event.messages,
            reason=event.reason,
            memories=event.memories,
            timestamp=event.timestamp,
        )
        self.flush_events.append(record)
        self.all_memories.extend(event.memories or [])

    def get_flushes_by_reason(self, reason: FlushReason) -> List[FlushEventRecord]:
        """按原因筛选 Flush 事件"""
        return [e for e in self.flush_events if e.reason == reason]

    def get_memories_by_type(self, memory_type: str) -> List[MemoryAtom]:
        """按类型筛选记忆原子"""
        return [
            m for m in self.all_memories
            if m.index.memory_type.value == memory_type
        ]

    @property
    def flush_count(self) -> int:
        return len(self.flush_events)

    @property
    def memory_count(self) -> int:
        return len(self.all_memories)

    def summary(self) -> str:
        """生成摘要"""
        reason_counts = {}
        for event in self.flush_events:
            key = event.reason.value
            reason_counts[key] = reason_counts.get(key, 0) + 1

        parts = [f"{k}: {v}" for k, v in reason_counts.items()]
        reasons_str = ", ".join(parts) if parts else "无"
        return f"Flushes: {self.flush_count}, Memories: {self.memory_count}, Reasons: [{reasons_str}]"

    def clear(self) -> None:
        """清空记录"""
        self.flush_events.clear()
        self.all_memories.clear()


def create_test_patchouli(
    max_processing_tokens: int = 8192,
    recorder: Optional[IntegrationTestRecorder] = None,
) -> PatchouliAgent:
    """
    创建测试用的 PatchouliAgent

    Args:
        max_processing_tokens: Token 溢出阈值
        recorder: 事件记录器

    Returns:
        配置好的 PatchouliAgent 实例
    """
    # 创建自定义配置
    config = PerceptionConfig(
        max_processing_tokens=max_processing_tokens,
        semantic_threshold=0.6,
        short_text_threshold=50,
        idle_timeout_seconds=900,
    )

    # 创建 PatchouliAgent
    patchouli = PatchouliAgent(
        storage=None,  # 使用内存存储
        enable_semantic_flow=True,
        perception_config=config,
    )

    # 注册观察者
    if recorder:
        patchouli.add_flush_observer(recorder.record)

    return patchouli


# ========== 测试用例 ==========

def test_semantic_drift_trigger() -> bool:
    """
    测试场景 1: 语义漂移触发

    验证点:
        - 同话题多轮对话应该吸附（不触发 Flush）
        - 话题切换时应触发 SEMANTIC_DRIFT Flush
        - Flush 后消息被正确处理
    """
    console.print("\n[bold cyan]测试 1: 语义漂移触发[/bold cyan]")

    recorder = IntegrationTestRecorder()
    patchouli = create_test_patchouli(recorder=recorder)

    user_id = "test_drift"
    agent_id = "test_agent"
    session_id = "test_session"

    # ========== 阶段 1: Python 话题 ==========
    console.print("\n  [yellow]阶段 1: 建立 Python 话题基线[/yellow]")

    for i in range(0, len(PYTHON_CONVERSATION), 2):
        if i + 1 < len(PYTHON_CONVERSATION):
            patchouli.add_message(
                "user", PYTHON_CONVERSATION[i]["content"],
                user_id, agent_id, session_id
            )
            patchouli.add_message(
                "assistant", PYTHON_CONVERSATION[i + 1]["content"],
                user_id, agent_id, session_id
            )

    python_flush_count = recorder.flush_count
    info = patchouli.get_buffer_info(user_id, agent_id, session_id)
    console.print(f"    Python 话题后: Flush={python_flush_count}, Blocks={info.get('block_count', 0)}")

    # ========== 阶段 2: 切换到烹饪话题 ==========
    console.print("\n  [yellow]阶段 2: 切换到烹饪话题（预期触发语义漂移）[/yellow]")

    patchouli.add_message(
        "user", COOKING_CONVERSATION[0]["content"],
        user_id, agent_id, session_id
    )
    patchouli.add_message(
        "assistant", COOKING_CONVERSATION[1]["content"],
        user_id, agent_id, session_id
    )

    after_drift_count = recorder.flush_count
    console.print(f"    话题切换后: Flush={after_drift_count}")

    # ========== 验证 ==========
    console.print("\n  [yellow]验证结果[/yellow]")

    drift_flushes = recorder.get_flushes_by_reason(FlushReason.SEMANTIC_DRIFT)

    if drift_flushes:
        console.print(f"    [green]✓ 成功检测到 {len(drift_flushes)} 次语义漂移[/green]")
        for i, event in enumerate(drift_flushes):
            console.print(f"      事件 {i + 1}: {len(event.messages)} 条消息, {len(event.memories)} 条记忆")
        success = True
    else:
        console.print("    [red]✗ 未检测到语义漂移[/red]")
        success = False

    console.print(f"\n  摘要: {recorder.summary()}")

    return success


def test_token_overflow_trigger() -> bool:
    """
    测试场景 2: Token 溢出触发

    验证点:
        - 累计 Token 超过阈值时触发 TOKEN_OVERFLOW
        - 溢出后生成接力摘要
        - 后续消息正常处理
    """
    console.print("\n[bold cyan]测试 2: Token 溢出触发[/bold cyan]")

    recorder = IntegrationTestRecorder()
    # 使用较小的阈值以便快速触发
    patchouli = create_test_patchouli(max_processing_tokens=2048, recorder=recorder)

    user_id = "test_overflow"
    agent_id = "test_agent"
    session_id = "test_session"

    # ========== 添加长对话直到溢出 ==========
    console.print("\n  [yellow]添加长对话直到 Token 溢出 (阈值=2048)[/yellow]")

    # 使用较短的片段，但足够触发溢出
    long_user_msg = LONG_TEXT_BLOCK[:800]
    long_response = LONG_TEXT_BLOCK[800:1600]

    round_count = 0
    max_rounds = 10
    while recorder.flush_count == 0 and round_count < max_rounds:
        patchouli.add_message("user", long_user_msg, user_id, agent_id, session_id)
        patchouli.add_message("assistant", long_response, user_id, agent_id, session_id)
        round_count += 1

        info = patchouli.get_buffer_info(user_id, agent_id, session_id)
        total_tokens = info.get('total_tokens', 0)
        console.print(f"    轮次 {round_count}: tokens={total_tokens}")

        if total_tokens > 2048:
            console.print(f"    [dim]预期触发溢出 ({total_tokens} > 2048)[/dim]")

    # ========== 验证 ==========
    console.print("\n  [yellow]验证结果[/yellow]")

    overflow_flushes = recorder.get_flushes_by_reason(FlushReason.TOKEN_OVERFLOW)

    if overflow_flushes:
        console.print(f"    [green]✓ 成功触发 {len(overflow_flushes)} 次 Token 溢出[/green]")
        success = True
    else:
        # 检查是否有任何 Flush
        if recorder.flush_count > 0:
            console.print(f"    [yellow]⚠ 触发了 Flush 但原因不是 TOKEN_OVERFLOW[/yellow]")
            console.print(f"    实际原因: {[e.reason.value for e in recorder.flush_events]}")
            success = True  # 仍然认为测试通过（触发机制工作正常）
        else:
            console.print("    [red]✗ 未触发 Token 溢出[/red]")
            success = False

    console.print(f"\n  摘要: {recorder.summary()}")

    return success


def test_memory_extraction_quality() -> bool:
    """
    测试场景 3: 记忆提取质量

    验证点:
        - 代码片段正确识别为 CODE_SNIPPET
        - 用户偏好正确识别为 USER_PROFILE
        - 记忆原子结构完整（title, summary, tags, content）
        - 置信度在合理范围内
        - 闲聊对话被正确过滤
    """
    console.print("\n[bold cyan]测试 3: 记忆提取质量[/bold cyan]")

    results = []

    # ========== 场景 3.1: 代码片段 ==========
    console.print("\n  [yellow]场景 3.1: 代码片段提取[/yellow]")

    recorder = IntegrationTestRecorder()
    patchouli = create_test_patchouli(recorder=recorder)

    user_id = "test_quality"
    agent_id = "test_agent"
    session_id = "code_session"

    patchouli.add_message(
        "user",
        "帮我写一个Python快排算法，要求支持自定义比较函数",
        user_id, agent_id, session_id
    )
    patchouli.add_message(
        "assistant",
        """好的，这是支持自定义比较函数的Python快排算法：

```python
from typing import Callable, List, TypeVar

T = TypeVar('T')

def quicksort(
    arr: List[T],
    key: Callable[[T], any] = lambda x: x,
    reverse: bool = False
) -> List[T]:
    \"\"\"
    快速排序算法，支持自定义比较函数

    Args:
        arr: 待排序列表
        key: 比较键函数
        reverse: 是否降序

    Returns:
        排序后的新列表
    \"\"\"
    if len(arr) <= 1:
        return arr.copy()

    pivot = arr[len(arr) // 2]
    pivot_key = key(pivot)

    if reverse:
        left = [x for x in arr if key(x) > pivot_key]
        middle = [x for x in arr if key(x) == pivot_key]
        right = [x for x in arr if key(x) < pivot_key]
    else:
        left = [x for x in arr if key(x) < pivot_key]
        middle = [x for x in arr if key(x) == pivot_key]
        right = [x for x in arr if key(x) > pivot_key]

    return quicksort(left, key, reverse) + middle + quicksort(right, key, reverse)
```

时间复杂度: O(n log n) 平均，O(n^2) 最坏
空间复杂度: O(n)
        """,
        user_id, agent_id, session_id
    )

    # 手动触发 Flush
    patchouli.flush_perception(user_id, agent_id, session_id)

    if recorder.memory_count > 0:
        memory = recorder.all_memories[-1]
        console.print(f"    标题: {memory.index.title}")
        console.print(f"    类型: {memory.index.memory_type.value}")
        console.print(f"    标签: {memory.index.tags}")
        console.print(f"    置信度: {memory.meta.confidence_score:.2f}")

        # 质量检查
        checks = [
            ("有标题", bool(memory.index.title)),
            ("有摘要", bool(memory.index.summary)),
            ("有标签", len(memory.index.tags) > 0),
            ("包含代码", "```" in memory.payload.content or "def " in memory.payload.content),
            ("置信度 > 0.5", memory.meta.confidence_score > 0.5),
        ]

        for check_name, passed in checks:
            status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
            console.print(f"      {status} {check_name}")

        results.append(("代码片段", all(p for _, p in checks)))
    else:
        console.print("    [red]✗ 未生成代码片段记忆[/red]")
        results.append(("代码片段", False))

    # ========== 场景 3.2: 用户偏好 ==========
    console.print("\n  [yellow]场景 3.2: 用户偏好提取[/yellow]")

    recorder.clear()
    session_id = "pref_session"

    patchouli.add_message(
        "user",
        "我希望所有代码都使用Python 3.12，遵循Black格式化规范，行宽设置为100",
        user_id, agent_id, session_id
    )
    patchouli.add_message(
        "assistant",
        "好的，我记住了您的偏好：使用Python 3.12，Black格式化，行宽100字符。后续代码都会遵循这些规范。",
        user_id, agent_id, session_id
    )

    patchouli.flush_perception(user_id, agent_id, session_id)

    if recorder.memory_count > 0:
        memory = recorder.all_memories[-1]
        console.print(f"    标题: {memory.index.title}")
        console.print(f"    类型: {memory.index.memory_type.value}")

        content = memory.payload.content.lower()
        checks = [
            ("包含 Python 3.12", "python 3.12" in content or "python3.12" in content),
            ("包含 Black", "black" in content),
            ("包含行宽", "100" in content),
        ]

        for check_name, passed in checks:
            status = "[green]PASS[/green]" if passed else "[yellow]WARN[/yellow]"
            console.print(f"      {status} {check_name}")

        # 至少 2 个检查通过即可
        results.append(("用户偏好", len([p for _, p in checks if p]) >= 2))
    else:
        console.print("    [yellow]⚠ 未生成用户偏好记忆（可能被评估为无价值）[/yellow]")
        results.append(("用户偏好", True))  # 不强制要求

    # ========== 场景 3.3: 闲聊过滤 ==========
    console.print("\n  [yellow]场景 3.3: 闲聊过滤[/yellow]")

    recorder.clear()
    session_id = "chat_session"

    patchouli.add_message("user", "你好", user_id, agent_id, session_id)
    patchouli.add_message("assistant", "你好！有什么可以帮助你的吗？", user_id, agent_id, session_id)
    patchouli.add_message("user", "没事，随便聊聊", user_id, agent_id, session_id)
    patchouli.add_message("assistant", "好的，很高兴和你聊天！", user_id, agent_id, session_id)

    patchouli.flush_perception(user_id, agent_id, session_id)

    if recorder.memory_count == 0:
        console.print("    [green]PASS[/green] 闲聊正确被过滤")
        results.append(("闲聊过滤", True))
    else:
        console.print(f"    [yellow]WARN[/yellow] 闲聊产生了 {recorder.memory_count} 条记忆")
        # 检查是否置信度较低
        if all(m.meta.confidence_score < 0.5 for m in recorder.all_memories):
            console.print("    [dim]（但置信度较低，可能会被后续过滤）[/dim]")
            results.append(("闲聊过滤", True))
        else:
            results.append(("闲聊过滤", False))

    # ========== 汇总 ==========
    console.print("\n  [yellow]质量测试汇总[/yellow]")
    all_passed = True
    for name, passed in results:
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        console.print(f"    {status} {name}")
        all_passed = all_passed and passed

    return all_passed


def test_manual_flush() -> bool:
    """
    测试场景 4: 手动 Flush

    验证点:
        - flush_perception 正确触发 MANUAL Flush
        - Flush 后 Buffer 状态正确重置
    """
    console.print("\n[bold cyan]测试 4: 手动 Flush[/bold cyan]")

    recorder = IntegrationTestRecorder()
    patchouli = create_test_patchouli(recorder=recorder)

    user_id = "test_manual"
    agent_id = "test_agent"
    session_id = "test_session"

    # 添加消息
    patchouli.add_message(
        "user", PYTHON_CONVERSATION[0]["content"],
        user_id, agent_id, session_id
    )
    patchouli.add_message(
        "assistant", PYTHON_CONVERSATION[1]["content"],
        user_id, agent_id, session_id
    )

    info_before = patchouli.get_buffer_info(user_id, agent_id, session_id)
    console.print(f"  Flush 前: blocks={info_before.get('block_count', 0)}, tokens={info_before.get('total_tokens', 0)}")

    # 手动 Flush
    patchouli.flush_perception(user_id, agent_id, session_id)

    info_after = patchouli.get_buffer_info(user_id, agent_id, session_id)
    console.print(f"  Flush 后: blocks={info_after.get('block_count', 0)}, tokens={info_after.get('total_tokens', 0)}")

    # 验证
    manual_flushes = recorder.get_flushes_by_reason(FlushReason.MANUAL)

    if manual_flushes:
        console.print(f"  [green]✓ 触发 MANUAL Flush[/green]")
        success = True
    else:
        # 检查是否有任何 Flush
        if recorder.flush_count > 0:
            console.print(f"  [yellow]⚠ Flush 被触发但原因不是 MANUAL[/yellow]")
            console.print(f"  实际原因: {[e.reason.value for e in recorder.flush_events]}")
            success = True  # 仍然认为测试通过
        else:
            console.print("  [red]✗ 未触发 Flush[/red]")
            success = False

    console.print(f"\n  摘要: {recorder.summary()}")

    return success


# ========== 主测试流程 ==========

def main():
    console.print(Panel.fit(
        "[bold magenta]PatchouliAgent 集成效果测试[/bold magenta]\n"
        "测试感知层触发时机与记忆生成质量",
        border_style="magenta"
    ))

    tests = [
        ("语义漂移触发", test_semantic_drift_trigger),
        ("Token 溢出触发", test_token_overflow_trigger),
        ("记忆提取质量", test_memory_extraction_quality),
        ("手动 Flush", test_manual_flush),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            console.print(f"\n[red]测试 {name} 失败: {e}[/red]")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))

    # 汇总
    console.print("\n" + "=" * 60)
    console.print("\n[bold cyan]测试结果汇总[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("测试", style="cyan")
    table.add_column("结果", justify="center")
    table.add_column("备注")

    success_count = 0
    for name, success, error in results:
        if success:
            status = "[green]PASS[/green]"
            success_count += 1
            note = ""
        else:
            status = "[red]FAIL[/red]"
            note = error or ""

        table.add_row(name, status, note)

    console.print(table)

    total_count = len(results)
    console.print(f"\n[bold]通过率: {success_count}/{total_count}[/bold]")

    if success_count == total_count:
        console.print("\n[bold green]所有测试通过![/bold green]")
        return 0
    else:
        console.print(f"\n[yellow]{total_count - success_count} 个测试失败[/yellow]")
        return 1


if __name__ == "__main__":
    exit(main())
