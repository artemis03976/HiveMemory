"""
SessionManager 核心功能测试

测试内容:
1. 会话创建与存在性检查
2. 消息存储与检索
3. 消息顺序验证
4. 会话清空
5. 多会话隔离

运行方式:
    python tests/test_session_manager.py
"""

import sys
import os
from pathlib import Path
from time import sleep
from uuid import uuid4

# 设置 UTF-8 编码 (Windows 兼容性)
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import redis
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from hivememory.core.config import get_config
from hivememory.agents.session_manager import SessionManager

console = Console(force_terminal=True, legacy_windows=False)


def setup_session_manager():
    """初始化 SessionManager"""
    console.print("\n[bold cyan]📦 初始化 SessionManager...[/bold cyan]")

    try:
        # 加载配置
        config = get_config()

        # 初始化 Redis
        redis_client = redis.Redis(
            **config.redis.model_dump(),
            socket_connect_timeout=5
        )
        redis_client.ping()
        console.print(f"  ✓ Redis 连接成功 ({config.redis.host}:{config.redis.port})")

        # 初始化 SessionManager
        session_manager = SessionManager(
            redis_client=redis_client,
            key_prefix="hivememory:test:session_mgr",
            ttl_days=7
        )
        console.print("  ✓ SessionManager 初始化成功")

        return session_manager

    except Exception as e:
        console.print(f"  ✗ 初始化失败: {e}", style="bold red")
        return None


def test_session_create_and_exists(session_manager):
    """测试1: 会话创建与存在性检查"""
    console.print("\n[bold magenta]🧪 测试1: 会话创建与存在性检查[/bold magenta]")

    session_id = f"test_session_{uuid4().hex[:8]}"

    try:
        # 1.1 检查不存在的会话
        exists_before = session_manager.session_exists(session_id)
        console.print(f"\n  1️⃣  检查不存在的会话: {exists_before}")
        assert not exists_before, "新会话不应存在"

        # 1.2 自动创建会话（通过添加消息）
        session_manager.add_message(session_id, "user", "第一条消息，自动创建会话")
        console.print(f"  2️⃣  通过添加消息自动创建会话: {session_id}")

        # 1.3 检查会话是否存在
        exists_after = session_manager.session_exists(session_id)
        console.print(f"  3️⃣  验证会话存在: {exists_after}")
        assert exists_after, "会话应该存在"

        # 1.4 检查不存在的会话（负面测试）
        fake_session_id = f"fake_session_{uuid4().hex[:8]}"
        exists_fake = session_manager.session_exists(fake_session_id)
        console.print(f"  4️⃣  检查假的会话ID: {exists_fake}")
        assert not exists_fake, "不存在的会话应返回 False"

        console.print("\n[green]✓ 测试1通过[/green]")
        return True

    except AssertionError as e:
        console.print(f"\n[red]✗ 测试1失败: {e}[/red]")
        return False
    except Exception as e:
        console.print(f"\n[red]✗ 测试1出错: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return False


def test_message_storage_and_retrieval(session_manager):
    """测试2: 消息存储与检索"""
    console.print("\n[bold magenta]🧪 测试2: 消息存储与检索[/bold magenta]")

    session_id = f"test_session_{uuid4().hex[:8]}"

    try:
        # 2.1 创建会话并添加第一条消息
        console.print(f"\n  1️⃣  创建会话: {session_id}")

        # 2.2 添加测试消息
        test_messages = [
            ("user", "你好，我是小明"),
            ("assistant", "你好小明！很高兴认识你"),
            ("user", "我的工作是什么？"),
            ("assistant", "你是一名软件工程师"),
            ("user", "谢谢你的回答"),
        ]

        console.print(f"\n  2️⃣  添加 {len(test_messages)} 条消息")
        for role, content in test_messages:
            session_manager.add_message(session_id, role, content)

        # 2.3 获取历史记录
        console.print(f"\n  3️⃣  获取历史记录")
        history = session_manager.get_history(session_id, limit=10)

        console.print(f"\n     检索到 {len(history)} 条消息")
        assert len(history) == len(test_messages), f"消息数量不匹配: {len(history)} != {len(test_messages)}"

        # 2.4 验证消息内容（使用属性访问）
        console.print(f"\n  4️⃣  验证消息内容")
        for i, (actual_msg, (expected_role, expected_content)) in enumerate(zip(history, test_messages)):
            assert actual_msg.role == expected_role, f"消息{i+1}角色不匹配"
            assert actual_msg.content == expected_content, f"消息{i+1}内容不匹配"
            console.print(f"     ✓ 消息{i+1}: [{actual_msg.role}] {actual_msg.content[:20]}...")

        # 2.5 获取消息计数
        console.print(f"\n  5️⃣  验证消息计数")
        count = session_manager.get_message_count(session_id)
        console.print(f"     消息总数: {count}")
        assert count == len(test_messages), f"计数不匹配: {count} != {len(test_messages)}"

        console.print("\n[green]✓ 测试2通过[/green]")
        return True

    except AssertionError as e:
        console.print(f"\n[red]✗ 测试2失败: {e}[/red]")
        return False
    except Exception as e:
        console.print(f"\n[red]✗ 测试2出错: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return False


def test_message_ordering(session_manager):
    """测试3: 消息顺序验证"""
    console.print("\n[bold magenta]🧪 测试3: 消息顺序验证[/bold magenta]")

    session_id = f"test_session_{uuid4().hex[:8]}"

    try:
        # 3.1 创建会话并添加消息
        console.print(f"\n  1️⃣  创建会话并添加消息")

        messages_to_add = [
            ("user", "第一条消息"),
            ("assistant", "回复第一条"),
            ("user", "第二条消息"),
            ("assistant", "回复第二条"),
            ("user", "第三条消息"),
        ]

        for role, content in messages_to_add:
            session_manager.add_message(session_id, role, content)
            console.print(f"     添加: [{role}] {content}")
            sleep(0.01)  # 确保时间戳不同

        # 3.2 获取历史记录并验证顺序
        console.print(f"\n  2️⃣  验证消息顺序")
        history = session_manager.get_history(session_id, limit=10)

        assert len(history) == len(messages_to_add), "消息数量不匹配"

        for i, msg in enumerate(history):
            expected_role, expected_content = messages_to_add[i]
            assert msg.role == expected_role, f"消息{i+1}角色顺序错误"
            assert msg.content == expected_content, f"消息{i+1}内容顺序错误"
            console.print(f"     ✓ 位置{i+1}: [{msg.role}] {msg.content}")

        # 3.3 验证时间戳递增
        console.print(f"\n  3️⃣  验证时间戳递增")
        timestamps = [msg.timestamp for msg in history]
        assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)), "时间戳未递增"
        console.print(f"     ✓ 时间戳正确递增")

        # 3.4 验证消息数量正确（不需要 message_id，ChatMessage没有这个字段）
        console.print(f"\n  4️⃣  验证消息对象完整性")
        assert all(hasattr(msg, 'role') for msg in history), "所有消息应有role属性"
        assert all(hasattr(msg, 'content') for msg in history), "所有消息应有content属性"
        assert all(hasattr(msg, 'timestamp') for msg in history), "所有消息应有timestamp属性"
        console.print(f"     ✓ 所有消息对象完整")

        console.print("\n[green]✓ 测试3通过[/green]")
        return True

    except AssertionError as e:
        console.print(f"\n[red]✗ 测试3失败: {e}[/red]")
        return False
    except Exception as e:
        console.print(f"\n[red]✗ 测试3出错: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return False


def test_session_clear(session_manager):
    """测试4: 会话清空"""
    console.print("\n[bold magenta]🧪 测试4: 会话清空[/bold magenta]")

    session_id = f"test_session_{uuid4().hex[:8]}"

    try:
        # 4.1 创建会话并添加消息
        console.print(f"\n  1️⃣  创建会话并添加消息")

        for i in range(5):
            session_manager.add_message(
                session_id,
                "user" if i % 2 == 0 else "assistant",
                f"消息{i+1}"
            )

        count_before = session_manager.get_message_count(session_id)
        console.print(f"     清空前消息数: {count_before}")
        assert count_before == 5, "清空前消息数应为5"

        # 4.2 清空会话
        console.print(f"\n  2️⃣  清空会话")
        session_manager.clear_session(session_id)

        # 4.3 验证消息被清空
        console.print(f"\n  3️⃣  验证消息已清空")
        count_after = session_manager.get_message_count(session_id)
        console.print(f"     清空后消息数: {count_after}")
        assert count_after == 0, "清空后消息数应为0"

        # 4.4 验证会话被删除（Redis行为：clear_session删除key）
        console.print(f"\n  4️⃣  验证会话被删除")
        exists = session_manager.session_exists(session_id)
        console.print(f"     会话存在: {exists}")
        assert not exists, "清空后会话不应存在（Redis key被删除）"

        # 4.5 可以继续添加消息（会自动重新创建会话）
        console.print(f"\n  5️⃣  验证可以继续添加消息（自动重新创建）")
        session_manager.add_message(session_id, "user", "清空后的新消息")
        count_new = session_manager.get_message_count(session_id)
        console.print(f"     新消息数: {count_new}")
        assert count_new == 1, "添加新消息后应自动重新创建会话"

        console.print("\n[green]✓ 测试4通过[/green]")
        return True

    except AssertionError as e:
        console.print(f"\n[red]✗ 测试4失败: {e}[/red]")
        return False
    except Exception as e:
        console.print(f"\n[red]✗ 测试4出错: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return False


def test_multiple_sessions_isolation(session_manager):
    """测试5: 多会话隔离"""
    console.print("\n[bold magenta]🧪 测试5: 多会话隔离[/bold magenta]")

    session_a = f"test_session_a_{uuid4().hex[:8]}"
    session_b = f"test_session_b_{uuid4().hex[:8]}"

    try:
        # 5.1 创建两个会话
        console.print(f"\n  1️⃣  创建两个独立会话")
        console.print(f"     会话A: {session_a}")
        console.print(f"     会话B: {session_b}")

        # 5.2 为会话A添加消息
        console.print(f"\n  2️⃣  为会话A添加消息")
        messages_a = [
            ("user", "我是用户A"),
            ("assistant", "你好用户A"),
        ]
        for role, content in messages_a:
            session_manager.add_message(session_a, role, content)
            console.print(f"     [A] [{role}] {content}")

        # 5.3 为会话B添加不同的消息
        console.print(f"\n  3️⃣  为会话B添加消息")
        messages_b = [
            ("user", "我是用户B"),
            ("assistant", "你好用户B"),
        ]
        for role, content in messages_b:
            session_manager.add_message(session_b, role, content)
            console.print(f"     [B] [{role}] {content}")

        # 5.4 验证会话A的数据
        console.print(f"\n  4️⃣  验证会话隔离")
        history_a = session_manager.get_history(session_a)
        console.print(f"\n     会话A历史 ({len(history_a)} 条):")
        for msg in history_a:
            console.print(f"       [{msg.role}] {msg.content}")

        assert len(history_a) == len(messages_a), "会话A消息数不匹配"
        assert all("用户A" in msg.content for msg in history_a), "会话A包含会话B的数据"

        # 5.5 验证会话B的数据
        history_b = session_manager.get_history(session_b)
        console.print(f"\n     会话B历史 ({len(history_b)} 条):")
        for msg in history_b:
            console.print(f"       [{msg.role}] {msg.content}")

        assert len(history_b) == len(messages_b), "会话B消息数不匹配"
        assert all("用户B" in msg.content for msg in history_b), "会话B包含会话A的数据"

        # 5.6 清空会话A不影响会话B
        console.print(f"\n  5️⃣  清空会话A并验证会话B不受影响")
        session_manager.clear_session(session_a)
        count_a = session_manager.get_message_count(session_a)
        count_b = session_manager.get_message_count(session_b)
        console.print(f"     会话A消息数: {count_a}")
        console.print(f"     会话B消息数: {count_b}")
        assert count_a == 0, "会话A应被清空"
        assert count_b == len(messages_b), "会话B不应受影响"

        console.print("\n[green]✓ 测试5通过[/green]")
        return True

    except AssertionError as e:
        console.print(f"\n[red]✗ 测试5失败: {e}[/red]")
        return False
    except Exception as e:
        console.print(f"\n[red]✗ 测试5出错: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return False


def main():
    """主测试流程"""
    console.print(Panel.fit(
        "[bold magenta]SessionManager 核心功能测试[/bold magenta]\n"
        "测试会话管理、消息存储、顺序、清空与隔离",
        border_style="magenta"
    ))

    # 1. 初始化 SessionManager
    session_manager = setup_session_manager()
    if not session_manager:
        console.print("\n[red]✗ SessionManager 初始化失败，测试终止[/red]")
        sys.exit(1)

    # 2. 执行测试套件
    console.print("\n" + "="*60)
    console.print("[bold cyan]🧪 开始执行测试套件[/bold cyan]\n")

    test_results = {}

    # 测试1: 会话创建与存在性
    test_results["test1"] = test_session_create_and_exists(session_manager)

    # 测试2: 消息存储与检索
    test_results["test2"] = test_message_storage_and_retrieval(session_manager)

    # 测试3: 消息顺序
    test_results["test3"] = test_message_ordering(session_manager)

    # 测试4: 会话清空
    test_results["test4"] = test_session_clear(session_manager)

    # 测试5: 多会话隔离
    test_results["test5"] = test_multiple_sessions_isolation(session_manager)

    # 3. 汇总测试结果
    console.print("\n" + "="*60)
    console.print("[bold cyan]📊 测试结果汇总[/bold cyan]\n")

    # 创建结果表格
    table = Table(title="SessionManager 测试结果", show_header=True, header_style="bold magenta")
    table.add_column("测试用例", style="cyan", width=35)
    table.add_column("状态", justify="center", width=10)
    table.add_column("说明", style="dim")

    test_names = {
        "test1": "会话创建与存在性检查",
        "test2": "消息存储与检索",
        "test3": "消息顺序验证",
        "test4": "会话清空",
        "test5": "多会话隔离"
    }

    all_passed = True
    for test_id, passed in test_results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        status_style = "green" if passed else "red"
        table.add_row(test_names[test_id], f"[{status_style}]{status}[/{status_style}]", "")
        if not passed:
            all_passed = False

    console.print(table)

    # 4. 最终结果
    console.print("\n" + "="*60)
    if all_passed:
        console.print(Panel(
            "[bold green]✅ 全部测试通过！[/bold green]\n\n"
            f"共执行 {len(test_results)} 个测试用例，全部成功。\n"
            "SessionManager 核心功能正常。",
            border_style="green"
        ))
    else:
        failed_count = sum(1 for passed in test_results.values() if not passed)
        console.print(Panel(
            f"[bold red]❌ 有 {failed_count} 个测试失败[/bold red]\n\n"
            f"共执行 {len(test_results)} 个测试用例，{len(test_results) - failed_count} 个成功，{failed_count} 个失败。\n"
            "请查看上方详细输出排查问题。",
            border_style="red"
        ))
        sys.exit(1)


if __name__ == "__main__":
    main()
