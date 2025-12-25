"""
ChatBot Agent 端到端测试

测试内容:
1. ChatBot 能否正常调用 LLM 并回复
2. 对话能否正确保存到 SessionManager
3. 对话能否推送到 ConversationBuffer（触发帕秋莉）
4. 帕秋莉能否自动提取并写入记忆到 Qdrant

运行方式:
    python tests/test_chatbot.py
"""

import sys
import os
from pathlib import Path

# 设置 UTF-8 编码 (Windows 兼容性)
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import time
import redis
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from hivememory.core.config import get_config
from hivememory.memory.storage import QdrantMemoryStore
from hivememory.agents.patchouli import PatchouliAgent
from hivememory.agents.chatbot import ChatBotAgent
from hivememory.agents.session_manager import SessionManager


console = Console(force_terminal=True, legacy_windows=False)


def setup_system():
    """初始化系统组件"""
    console.print("\n[bold cyan]📦 初始化系统组件...[/bold cyan]")

    try:
        # 加载配置
        config = get_config()
        console.print("  ✓ 配置加载成功")

        # 初始化 Redis
        redis_client = redis.Redis(
            **config.redis.model_dump(),
            socket_connect_timeout=5
        )
        # 测试连接
        redis_client.ping()
        console.print(f"  ✓ Redis 连接成功 ({config.redis.host}:{config.redis.port})")

        # 初始化 Qdrant Storage
        storage = QdrantMemoryStore(
            qdrant_config=config.qdrant,
            embedding_config=config.embedding
        )
        console.print(f"  ✓ Qdrant 连接成功 ({config.qdrant.host}:{config.qdrant.port})")

        # 初始化 Patchouli Agent（图书管理员）
        patchouli = PatchouliAgent(storage=storage)
        console.print("  ✓ PatchouliAgent 初始化成功")

        # 初始化 Session Manager
        session_manager = SessionManager(
            redis_client=redis_client,
            key_prefix="hivememory:test",
            ttl_days=7
        )
        console.print("  ✓ SessionManager 初始化成功")

        return config, patchouli, session_manager, storage

    except Exception as e:
        console.print(f"  ✗ 初始化失败: {e}", style="bold red")
        console.print("\n[yellow]提示: 请确保运行了 'docker-compose up -d'[/yellow]")
        return None


def create_chatbot(config, patchouli, session_manager):
    """创建 ChatBot Agent"""
    console.print("\n[bold cyan]🤖 创建 ChatBot Agent...[/bold cyan]")

    try:
        worker_llm_config = config.get_worker_llm_config()

        chatbot = ChatBotAgent(
            patchouli=patchouli,
            session_manager=session_manager,
            user_id="test_user",
            agent_id="test_chatbot",
            llm_config={
                "model": worker_llm_config.model,
                "api_key": worker_llm_config.api_key,
                "api_base": worker_llm_config.api_base,
                "temperature": worker_llm_config.temperature,
                "max_tokens": worker_llm_config.max_tokens
            }
        )

        console.print(f"  ✓ ChatBot 创建成功")
        console.print(f"  模型: {worker_llm_config.model}")
        console.print(f"  温度: {worker_llm_config.temperature}")
        console.print(f"  最大 Tokens: {worker_llm_config.max_tokens}")

        return chatbot

    except Exception as e:
        console.print(f"  ✗ 创建失败: {e}", style="bold red")
        raise


def test_conversation(chatbot, session_id):
    """测试对话流程"""
    console.print(f"\n[bold magenta]💬 测试对话流程[/bold magenta]")
    console.print(f"[dim]Session ID: {session_id}[/dim]")

    # 清空会话历史（确保测试独立性）
    console.print("[yellow]清空历史会话记录...[/yellow]")
    chatbot.clear_session(session_id)
    console.print("[green]✓ 会话已清空[/green]\n")

    # 测试对话（包含个人信息，便于帕秋莉提取记忆）
    test_messages = [
        "你好！",
        "我叫张三，是一名软件工程师",
        "我在北京工作，最喜欢的编程语言是 Python",
        "我目前在做一个叫 HiveMemory 的项目，这是一个记忆管理系统",
        "我的爱好是阅读科幻小说和打篮球",
    ]

    success_count = 0

    for i, user_msg in enumerate(test_messages, 1):
        console.print(f"[cyan]轮次 {i}/{len(test_messages)}[/cyan]")
        console.print(f"👤 [bold]User:[/bold] {user_msg}")

        try:
            # 创建进度指示器
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            )
            progress.start()
            task = progress.add_task("思考中...", total=1)

            # 调用 ChatBot
            response = chatbot.chat(
                session_id=session_id,
                user_message=user_msg,
                record_to_patchouli=True
            )

            # 完成进度并停止
            progress.update(task, advance=1)
            progress.stop()

            # 显示回复（截断过长内容）
            if len(response) > 100:
                display_response = response[:100] + "..."
            else:
                display_response = response

            console.print(f"🤖 [bold]Bot:[/bold] {display_response}")
            console.print()

            success_count += 1
            time.sleep(0.5)  # 短暂延时

        except Exception as e:
            console.print(f"  ✗ 对话失败: {e}", style="bold red")
            raise

    # 显示结果
    if success_count == len(test_messages):
        console.print(f"[green]✓ 对话测试完成！成功 {success_count}/{len(test_messages)} 轮[/green]")

        # 手动触发 Buffer 刷新，处理剩余消息（如果有）
        console.print("\n[cyan]手动触发 Buffer 刷新，处理剩余消息...[/cyan]")
        try:
            remaining_memories = chatbot.patchouli.flush_buffer(
                user_id=chatbot.user_id,
                agent_id=chatbot.agent_id,
                session_id=session_id
            )
            if remaining_memories:
                console.print(f"[green]✓ 从剩余消息中提取了 {len(remaining_memories)} 条记忆[/green]")
            else:
                console.print("[dim]  没有剩余消息需要处理[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠ Buffer 刷新失败: {e}[/yellow]")

        return True
    else:
        console.print(f"[yellow]⚠ 对话部分成功 {success_count}/{len(test_messages)} 轮[/yellow]")
        return False


def verify_session_storage(chatbot, session_id):
    """验证会话存储"""
    console.print(f"\n[bold cyan]📊 验证会话存储...[/bold cyan]")

    try:
        session_info = chatbot.get_session_info(session_id)

        # 创建信息表格
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("属性", style="cyan")
        table.add_column("值", style="white")

        table.add_row("Session ID", session_info['session_id'])
        table.add_row("消息数量", str(session_info['message_count']))
        table.add_row("存在状态", "✓ 存在" if session_info['exists'] else "✗ 不存在")

        console.print(table)

        # 验证
        assert session_info['exists'], "会话应该存在！"
        assert session_info['message_count'] >= 10, "消息数量不足（应包含 user + assistant）"

        console.print("[green]✓ 会话存储验证通过[/green]")
        return True

    except AssertionError as e:
        console.print(f"[red]✗ 验证失败: {e}[/red]")
        return False
    except Exception as e:
        console.print(f"[red]✗ 验证出错: {e}[/red]")
        return False


def verify_memory_generation(storage, user_id):
    """验证记忆生成（等待帕秋莉处理）"""
    console.print(f"\n[bold cyan]📚 验证记忆生成...[/bold cyan]")

    # 等待 Buffer 触发
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("等待帕秋莉处理对话（Buffer 触发条件: 6条消息）...", total=None)
        time.sleep(3)
        progress.update(task, completed=True)

    # 查询 Qdrant 中的记忆
    try:
        memories = storage.search_memories(
            query_text="张三的个人信息",
            top_k=10,
            score_threshold=0.5,
            filters={"meta.user_id": user_id}  # 将 user_id 作为过滤条件
        )

        console.print(f"\n  找到 [bold]{len(memories)}[/bold] 条相关记忆\n")

        if len(memories) > 0:
            # 显示前3条记忆
            for i, result in enumerate(memories[:3], 1):
                mem = result["memory"]  # 从结果字典中提取 MemoryAtom 对象
                panel_content = f"""[bold cyan]标题:[/bold cyan] {mem.index.title}
[bold cyan]类型:[/bold cyan] {mem.index.memory_type}
[bold cyan]置信度:[/bold cyan] {mem.meta.confidence_score:.2f}
[bold cyan]摘要:[/bold cyan] {mem.index.summary[:100]}{'...' if len(mem.index.summary) > 100 else ''}"""

                console.print(Panel(
                    panel_content,
                    title=f"记忆 {i}",
                    border_style="green",
                    expand=False
                ))

            console.print("\n[green]✓ 记忆生成验证通过！帕秋莉已成功提取并存储记忆[/green]")
            return True
        else:
            console.print("[yellow]⚠️  未找到记忆，可能需要等待 Buffer 触发（15分钟空闲）或手动调用 buffer.flush()[/yellow]")
            return False

    except Exception as e:
        console.print(f"[red]✗ 查询记忆失败: {e}[/red]")
        return False


def main():
    """主测试流程"""
    console.print(Panel.fit(
        "[bold magenta]HiveMemory ChatBot 端到端测试[/bold magenta]\n"
        "测试 ChatBot 对话、会话管理与记忆生成",
        border_style="magenta"
    ))

    results = []

    # 1. 初始化系统
    system_components = setup_system()
    if not system_components:
        console.print("\n[red]✗ 系统初始化失败，测试终止[/red]")
        sys.exit(1)

    config, patchouli, session_manager, storage = system_components

    try:
        # 2. 创建 ChatBot
        chatbot = create_chatbot(config, patchouli, session_manager)

        # 3. 测试对话
        session_id = "test_session_001"
        conversation_success = test_conversation(chatbot, session_id)
        results.append(("对话测试", conversation_success))

        # 4. 验证会话存储
        session_success = verify_session_storage(chatbot, session_id)
        results.append(("会话存储", session_success))

        # 5. 验证记忆生成
        memory_success = verify_memory_generation(storage, user_id="test_user")
        results.append(("记忆生成", memory_success))

    except Exception as e:
        console.print(f"\n[red]✗ 测试执行失败: {e}[/red]")
        import traceback
        console.print("\n[dim]详细错误信息:[/dim]")
        console.print(traceback.format_exc())
        sys.exit(1)

    # 结果汇总
    console.print("\n" + "="*60)
    console.print("\n[bold cyan]📋 测试结果汇总[/bold cyan]\n")

    # 创建结果表格
    result_table = Table(show_header=True, header_style="bold cyan")
    result_table.add_column("测试项", style="white", width=20)
    result_table.add_column("状态", justify="center", width=10)

    for name, success in results:
        status = "[green]✓ 通过[/green]" if success else "[red]✗ 失败[/red]"
        result_table.add_row(name, status)

    console.print(result_table)

    # 统计
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)

    console.print()
    if success_count == total_count:
        console.print(Panel(
            f"[bold green]✅ 所有测试通过！({success_count}/{total_count})[/bold green]",
            border_style="green"
        ))

        console.print("\n[bold cyan]📝 下一步:[/bold cyan]")
        console.print("  1. 启动 Streamlit UI: [yellow]streamlit run examples/chatbot_ui.py[/yellow]")
        console.print("  2. 在 UI 中进行更多对话测试")
        console.print("  3. 查看 Qdrant Dashboard: [yellow]http://localhost:6333/dashboard[/yellow]")
    else:
        console.print(Panel(
            f"[bold yellow]⚠ 部分测试失败 ({success_count}/{total_count})[/bold yellow]",
            border_style="yellow"
        ))
        sys.exit(1)


if __name__ == "__main__":
    main()
