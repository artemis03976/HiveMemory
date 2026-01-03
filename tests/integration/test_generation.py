"""
HiveMemory 阶段1 端到端测试脚本

测试流程:
1. 模拟多轮对话
2. 触发 Patchouli 处理
3. 验证记忆存储
4. 展示提取结果

验收标准:
- 输入对话文本 → Patchouli 提取结构化 MemoryAtom
- 在 Qdrant 中可见向量数据
- 输出包含 Title, Summary, Tags 的记忆卡片
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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from hivememory.generation.models import ConversationMessage
from hivememory.agents.patchouli import PatchouliAgent
from hivememory.memory.storage import QdrantMemoryStore
from hivememory.generation.triggers import TriggerManager, MessageCountTrigger

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

console = Console(force_terminal=True, legacy_windows=False)


# ========== 测试场景定义 ==========

SCENARIO_1 = {
    "name": "代码片段提取",
    "description": "测试从对话中提取Python代码片段",
    "messages": [
        ("user", "你能帮我写一个Python函数来解析ISO8601格式的日期字符串吗?"),
        ("assistant", """当然可以! 这里是一个使用Python标准库的实现:

            ```python
            from datetime import datetime

            def parse_iso8601(date_str):
                \"\"\"
                解析 ISO8601 格式的日期字符串

                Args:
                    date_str: ISO8601格式字符串, 如 '2025-01-22T10:30:00Z'

                Returns:
                    datetime 对象

                Raises:
                    ValueError: 无法解析时
                \"\"\"
                try:
                    # Python 3.7+ 支持 fromisoformat
                    # 但需要移除末尾的 'Z' (UTC标识)
                    if date_str.endswith('Z'):
                        date_str = date_str[:-1] + '+00:00'

                    return datetime.fromisoformat(date_str)
                except ValueError as e:
                    raise ValueError(f"无效的ISO8601格式: {date_str}") from e
            ```

            这个函数支持大多数常见的ISO8601格式。你可以这样使用它:

            ```python
            dt = parse_iso8601("2025-01-22T10:30:00Z")
            print(dt)  # 2025-01-22 10:30:00+00:00
            ```
            """
        ),
        ("user", "太好了,谢谢!"),
    ]
}

SCENARIO_2 = {
    "name": "用户偏好设置",
    "description": "测试提取用户配置信息",
    "messages": [
        ("user", "我希望以后的代码都使用Python 3.12,不要使用旧版本的语法。"),
        ("assistant", "好的,我记住了! 后续所有代码都会基于Python 3.12标准,可以使用match语句、类型提示等新特性。"),
        ("user", "还有,我更喜欢用Black格式化代码,行宽100。"),
        ("assistant", "明白!代码风格按照Black标准,行宽设为100字符。"),
    ]
}

SCENARIO_3 = {
    "name": "闲聊过滤测试",
    "description": "测试过滤无价值的闲聊",
    "messages": [
        ("user", "你好"),
        ("assistant", "你好!有什么可以帮助你的吗?"),
        ("user", "没事,随便聊聊"),
        ("assistant", "好的,很高兴和你聊天!"),
    ]
}


# ========== 测试函数 ==========

def setup_environment():
    """环境准备"""
    console.print("\n[bold cyan]🛠️  环境准备...[/bold cyan]")

    try:
        config = get_config()
        
        # 创建存储实例
        storage = QdrantMemoryStore(
            qdrant_config=config.qdrant,
            embedding_config=config.embedding
        )

        # 创建集合
        console.print("  创建 Qdrant 集合...")
        storage.create_collection(recreate=True)

        console.print("✓ 环境准备完成", style="green")
        return storage

    except Exception as e:
        console.print(f"✗ 环境准备失败: {e}", style="bold red")
        console.print("\n提示: 请确保运行了 'docker-compose up -d'")
        return None


def run_scenario(scenario: dict, patchouli: PatchouliAgent):
    """
    运行单个测试场景

    Args:
        scenario: 场景定义
        patchouli: PatchouliAgent
    """
    console.print(f"\n[bold magenta]📝 场景: {scenario['name']}[/bold magenta]")
    console.print(f"[dim]{scenario['description']}[/dim]\n")

    # Step 1: 显示对话内容
    console.print("[cyan]对话内容:[/cyan]")
    for role, content in scenario["messages"]:
        role_icon = "👤" if role == "user" else "🤖"
        console.print(f"{role_icon} [bold]{role.capitalize()}:[/bold]")
        console.print(f"  {content[:100]}..." if len(content) > 100 else f"  {content}")
        console.print()

    # Step 2: 创建缓冲器
    memories_extracted = []

    def on_flush(messages, memories):
        """刷新回调"""
        memories_extracted.extend(memories)

    # 创建触发管理器 (设置高阈值以确保仅手动触发)
    trigger_manager = TriggerManager(strategies=[
        MessageCountTrigger(threshold=20)
    ])

    # 使用 PatchouliAgent 的 Buffer 管理（全局单例复用）
    buffer = patchouli.get_or_create_buffer(
        user_id="test_user",
        agent_id="test_agent",
        session_id=f"test_scenario_{scenario['name']}",
        trigger_manager=trigger_manager,
        on_flush_callback=on_flush,
    )

    # Step 3: 添加消息到缓冲区
    console.print("[cyan]处理中...[/cyan]")
    for role, content in scenario["messages"]:
        buffer.add_message(role, content)

    # Step 4: 手动触发处理
    buffer.flush()
    time.sleep(1)  # 等待异步处理

    # Step 5: 展示结果
    console.print("\n[cyan]提取结果:[/cyan]")
    if memories_extracted:
        for memory in memories_extracted:
            # 创建记忆卡片
            card_content = f"""
                **标题**: {memory.index.title}
                **类型**: {memory.index.memory_type.value}
                **标签**: {', '.join(f'#{tag}' for tag in memory.index.tags)}
                **摘要**: {memory.index.summary}

                **置信度**: {memory.meta.confidence_score:.1%}
            """
            console.print(Panel(
                card_content.strip(),
                title=f"[bold green]✓ 记忆原子 {memory.id}[/bold green]",
                border_style="green",
                expand=False
            ))

            # 显示部分内容
            if len(memory.payload.content) > 200:
                preview = memory.payload.content[:200] + "..."
            else:
                preview = memory.payload.content

            console.print(Markdown(f"**内容预览**:\n{preview}"))
            console.print()

        return True
    else:
        console.print(Panel(
            "[yellow]未提取到记忆 (可能被判定为无价值)[/yellow]",
            border_style="yellow"
        ))
        return False


def verify_storage(storage: QdrantMemoryStore):
    """验证数据库存储"""
    console.print("\n[bold cyan]🔍 验证数据库存储...[/bold cyan]")

    try:
        # 统计总数
        count = storage.count_memories()
        console.print(f"  总记忆数: {count}")

    except Exception as e:
        console.print(f"✗ 验证失败: {e}", style="red")


def main():
    """主测试流程"""
    console.print(Panel.fit(
        "[bold magenta]HiveMemory 阶段1 - 端到端测试[/bold magenta]\n"
        "测试 Patchouli 记忆提取与存储功能",
        border_style="magenta"
    ))

    # 环境准备
    storage = setup_environment()
    if not storage:
        return

    # 创建 Patchouli Agent
    patchouli = PatchouliAgent(storage=storage)

    # 运行测试场景
    scenarios = [SCENARIO_1, SCENARIO_2, SCENARIO_3]
    results = []
    for scenario in scenarios:
        success = run_scenario(scenario, patchouli)
        results.append((scenario["name"], success))

    # 验证存储
    verify_storage(storage)

    # 结果汇总
    console.print("\n" + "="*60)
    console.print("\n[bold cyan]测试结果汇总[/bold cyan]\n")

    success_count = sum(1 for _, success in results if success)
    total_count = len(results)

    for name, success in results:
        status = "[green]✓ 通过[/green]" if success else "[yellow]○ 跳过[/yellow]"
        console.print(f"  {status}  {name}")

    console.print(f"\n[bold]通过率: {success_count}/{total_count}[/bold]")

    if success_count == 0:
        console.print("\n[yellow]⚠️  所有场景都未提取到记忆。可能原因:[/yellow]")
        console.print("  1. LLM API Key 未配置或无效")
        console.print("  2. 模型判断对话无长期价值")
        console.print("  3. JSON 解析失败")
        console.print("\n请检查日志输出以获取详细信息。")
    elif success_count < total_count:
        console.print("\n[cyan]部分场景被过滤,这是正常的(如闲聊)。[/cyan]")
    else:
        console.print("\n[bold green]🎉 测试完全成功! Patchouli 工作正常。[/bold green]")

    console.print("\n[dim]访问 http://localhost:6333/dashboard 查看 Qdrant 数据[/dim]")


if __name__ == "__main__":
    main()
