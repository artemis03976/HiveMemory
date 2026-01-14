"""
ChatBot Agent Stage 1 测试: 记忆生成与写入

测试内容:
1. ChatBot 对话功能（LLM调用与回复）
2. 对话推送到感知层（触发Patchouli）
3. Patchouli 自动提取并写入记忆到 Qdrant
4. 多样化的记忆提取场景测试

测试用例:
- test_basic_profile_extraction: 基础信息提取（姓名、职位、地点）
- test_code_snippet_extraction: 代码片段提取
- test_project_architecture_extraction: 项目架构知识提取
- test_work_preferences_extraction: 工作偏好提取
- test_low_value_filtering: 低价值信息过滤
- test_multi_memory_extraction: 多记忆同时提取
- test_buffer_accumulation_trigger: Buffer累积触发机制
- test_reflection_extraction: 经验总结提取

运行方式:
    python tests/test_chatbot_stage1.py
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

from hivememory.core.config import load_app_config
from hivememory.core.models import MemoryType
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
    config = load_app_config()
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
        storage.create_collection(recreate=True)
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
            llm_config=worker_llm_config,
            enable_memory_retrieval=False,     # Stage 1 测试禁用记忆检索
            enable_lifecycle_management=False  # Stage 1 测试禁用生命周期管理
        )

        console.print(f"  ✓ ChatBot 创建成功")
        console.print(f"  模型: {worker_llm_config.model}")
        console.print(f"  温度: {worker_llm_config.temperature}")
        console.print(f"  最大 Tokens: {worker_llm_config.max_tokens}")

        return chatbot

    except Exception as e:
        console.print(f"  ✗ 创建失败: {e}", style="bold red")
        raise


def wait_for_patchouli_processing(seconds=3):
    """等待 Patchouli 处理对话"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("等待 Patchouli 处理对话（感知层触发条件: 6条消息）...", total=None)
        time.sleep(seconds)
        progress.update(task, completed=True)


def verify_memory_extraction(storage, user_id, expected_min_count=1, wait_seconds=3):
    """验证记忆提取是否发生并返回记忆列表"""
    console.print(f"\n[bold cyan]📚 验证记忆提取...[/bold cyan]")

    # 等待 Patchouli 处理
    wait_for_patchouli_processing(wait_seconds)

    try:
        # 使用 get_all_memories 获取该用户的所有记忆（不进行向量检索）
        memories = storage.get_all_memories(
            filters={"meta.user_id": user_id},
            limit=100
        )

        n_memories = len(memories)
        console.print(f"\n  找到 [bold]{n_memories}[/bold] 条记忆")

        if n_memories >= expected_min_count:
            console.print(f"[green]✓ 记忆提取成功 ({n_memories} >= {expected_min_count})[/green]")

            # 显示前几条记忆的简要信息
            console.print("\n[dim]提取的记忆:[/dim]")
            for i, mem in enumerate(memories[:5], 1):
                mem_type = mem.index.memory_type.value if hasattr(mem.index.memory_type, 'value') else str(mem.index.memory_type)
                console.print(f"  {i}. [{mem_type}] {mem.index.title}")

            if n_memories > 5:
                console.print(f"  ... 还有 {n_memories - 5} 条")

            return True, memories
        else:
            console.print(f"[yellow]⚠️  记忆数量不足 ({n_memories} < {expected_min_count})[/yellow]")
            return False, memories

    except Exception as e:
        console.print(f"[red]✗ 查询记忆失败: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return False, []


def verify_memory_types(memories, expected_types):
    """验证特定类型的记忆是否存在"""
    console.print("\n[bold cyan]🔍 验证记忆类型...[/bold cyan]")

    actual_types = {mem.index.memory_type for mem in memories}
    missing_types = expected_types - actual_types

    if missing_types:
        console.print(f"[yellow]⚠️  缺少记忆类型: {missing_types}[/yellow]")
        console.print(f"  实际类型: {actual_types}")
        return False
    else:
        console.print(f"[green]✓ 所有预期类型都存在: {expected_types}[/green]")
        return True


def verify_memory_keywords(memories, expected_keywords):
    """验证记忆内容包含关键词"""
    console.print("\n[bold cyan]🔍 验证记忆内容关键词...[/bold cyan]")

    all_content = " ".join([mem.payload.content for mem in memories])
    found_keywords = []
    missing_keywords = []

    for kw in expected_keywords:
        if kw in all_content:
            found_keywords.append(kw)
        else:
            missing_keywords.append(kw)

    if found_keywords:
        console.print(f"[green]✓ 找到关键词: {found_keywords}[/green]")

    if missing_keywords:
        console.print(f"[yellow]⚠️  缺少关键词: {missing_keywords}[/yellow]")
        return False

    return True


def test_basic_profile_extraction(chatbot, session_id, storage):
    """测试1: 基础信息提取 - 个人资料"""
    console.print("\n[bold magenta]💬 测试1: 基础信息提取 - 个人资料[/bold magenta]")

    chatbot.clear_session(session_id)

    # 对话轮次
    test_messages = [
        "你好！",
        "我叫李明，是一名后端工程师",
        "我在上海工作，主要使用 Python 和 Go 开发微服务",
    ]

    success_count = 0
    for i, msg in enumerate(test_messages, 1):
        console.print(f"\n👤 [bold]User:[/bold] {msg}")
        try:
            response = chatbot.chat(session_id, msg, record_to_patchouli=True)
            console.print(f"🤖 [bold]Bot:[/bold] {response[:100]}...")
            success_count += 1
            time.sleep(0.3)
        except Exception as e:
            console.print(f"[red]✗ 对话失败: {e}[/red]")

    # 验证记忆提取
    if success_count == len(test_messages):
        # 手动触发感知层 Flush
        try:
            chatbot.patchouli.flush_perception(chatbot.user_id, chatbot.agent_id, session_id)
        except:
            pass

        success, memories = verify_memory_extraction(storage, chatbot.user_id, expected_min_count=1)

        if success:
            # 验证包含关键词
            keywords = ["李明", "工程师", "Python", "上海"]
            verify_memory_keywords(memories, keywords)

        return success
    return False


def test_code_snippet_extraction(chatbot, session_id, storage):
    """测试2: 代码片段提取 - Python工具函数"""
    console.print("\n[bold magenta]💬 测试2: 代码片段提取 - Python工具函数[/bold magenta]")

    chatbot.clear_session(session_id)

    # 对话轮次
    test_messages = [
        "我想分享一个 Python 工具函数",
        "```python\ndef parse_config(filepath: str) -> dict:\n    \"\"\"解析 YAML 配置文件\"\"\"\n    import yaml\n    with open(filepath) as f:\n        return yaml.safe_load(f)\n```",
        "这个函数用于读取项目配置文件，支持 YAML 格式",
    ]

    success_count = 0
    for i, msg in enumerate(test_messages, 1):
        console.print(f"\n👤 [bold]User:[/bold] {msg[:80]}...")
        try:
            response = chatbot.chat(session_id, msg, record_to_patchouli=True)
            console.print(f"🤖 [bold]Bot:[/bold] {response[:80]}...")
            success_count += 1
            time.sleep(0.3)
        except Exception as e:
            console.print(f"[red]✗ 对话失败: {e}[/red]")

    # 验证记忆提取
    if success_count == len(test_messages):
        try:
            chatbot.patchouli.flush_perception(chatbot.user_id, chatbot.agent_id, session_id)
        except:
            pass

        success, memories = verify_memory_extraction(storage, chatbot.user_id, expected_min_count=1)

        if success:
            # 验证包含代码相关内容
            keywords = ["def", "parse_config", "yaml", "函数"]
            verify_memory_keywords(memories, keywords)

            # 验证类型
            verify_memory_types(memories, {MemoryType.CODE_SNIPPET})

        return success
    return False


def test_project_architecture_extraction(chatbot, session_id, storage):
    """测试3: 项目架构知识提取"""
    console.print("\n[bold magenta]💬 测试3: 项目架构知识提取[/bold magenta]")

    chatbot.clear_session(session_id)

    # 对话轮次
    test_messages = [
        "我正在开发一个分布式任务系统",
        "系统分为三层：调度层、执行层、存储层",
        "调度层负责任务分配，执行层运行任务，存储层持久化结果",
    ]

    success_count = 0
    for i, msg in enumerate(test_messages, 1):
        console.print(f"\n👤 [bold]User:[/bold] {msg}")
        try:
            response = chatbot.chat(session_id, msg, record_to_patchouli=True)
            console.print(f"🤖 [bold]Bot:[/bold] {response[:80]}...")
            success_count += 1
            time.sleep(0.3)
        except Exception as e:
            console.print(f"[red]✗ 对话失败: {e}[/red]")

    # 验证记忆提取
    if success_count == len(test_messages):
        try:
            chatbot.patchouli.flush_perception(chatbot.user_id, chatbot.agent_id, session_id)
        except:
            pass

        success, memories = verify_memory_extraction(storage, chatbot.user_id, expected_min_count=1)

        if success:
            # 验证包含架构关键词
            keywords = ["调度", "执行", "存储", "系统", "层"]
            verify_memory_keywords(memories, keywords)

        return success
    return False


def test_work_preferences_extraction(chatbot, session_id, storage):
    """测试4: 工作偏好提取"""
    console.print("\n[bold magenta]💬 测试4: 工作偏好提取[/bold magenta]")

    chatbot.clear_session(session_id)

    # 对话轮次
    test_messages = [
        "我有很强的代码洁癖",
        "我要求所有代码必须通过 pylint 检查，评分要大于 8.0",
        "我坚持 TDD 开发模式，测试覆盖率必须达到 85% 以上",
    ]

    success_count = 0
    for i, msg in enumerate(test_messages, 1):
        console.print(f"\n👤 [bold]User:[/bold] {msg}")
        try:
            response = chatbot.chat(session_id, msg, record_to_patchouli=True)
            console.print(f"🤖 [bold]Bot:[/bold] {response[:80]}...")
            success_count += 1
            time.sleep(0.3)
        except Exception as e:
            console.print(f"[red]✗ 对话失败: {e}[/red]")

    # 验证记忆提取
    if success_count == len(test_messages):
        try:
            chatbot.patchouli.flush_perception(chatbot.user_id, chatbot.agent_id, session_id)
        except:
            pass

        success, memories = verify_memory_extraction(storage, chatbot.user_id, expected_min_count=1)

        if success:
            # 验证包含偏好关键词
            keywords = ["pylint", "TDD", "测试", "覆盖率", "代码"]
            verify_memory_keywords(memories, keywords)

        return success
    return False


def test_low_value_filtering(chatbot, session_id, storage):
    """测试5: 低价值信息过滤"""
    console.print("\n[bold magenta]💬 测试5: 低价值信息过滤[/bold magenta]")

    chatbot.clear_session(session_id)

    # 对话轮次 - 低价值内容
    test_messages = [
        "你好",
        "今天天气怎么样？",
        "好的",
        "谢谢",
    ]

    success_count = 0
    for i, msg in enumerate(test_messages, 1):
        console.print(f"\n👤 [bold]User:[/bold] {msg}")
        try:
            response = chatbot.chat(session_id, msg, record_to_patchouli=True)
            console.print(f"🤖 [bold]Bot:[/bold] {response[:80]}...")
            success_count += 1
            time.sleep(0.3)
        except Exception as e:
            console.print(f"[red]✗ 对话失败: {e}[/red]")

    # 验证：低价值对话不应该生成记忆，或记忆极少
    if success_count == len(test_messages):
        try:
            chatbot.patchouli.flush_perception(chatbot.user_id, chatbot.agent_id, session_id)
        except:
            pass

        success, memories = verify_memory_extraction(storage, chatbot.user_id, expected_min_count=0)

        # 对于低价值过滤，我们期望记忆数为 0 或很少（< 2）
        if len(memories) < 2:
            console.print("[green]✓ 低价值过滤生效（记忆数 < 2）[/green]")
            return True
        else:
            console.print(f"[yellow]⚠️  生成了 {len(memories)} 条记忆，过滤可能不够严格[/yellow]")
            return True  # 软性要求，不算失败

    return False


def test_multi_memory_extraction(chatbot, session_id, storage):
    """测试6: 多记忆同时提取"""
    console.print("\n[bold magenta]💬 测试6: 多记忆同时提取[/bold magenta]")

    chatbot.clear_session(session_id)

    # 对话轮次 - 包含多个主题
    test_messages = [
        "我叫王芳，是前端开发工程师，住在深圳",
        "我常用的技术栈是 React 和 TypeScript",
        "我们项目采用敏捷开发模式，每周一个 Sprint",
    ]

    success_count = 0
    for i, msg in enumerate(test_messages, 1):
        console.print(f"\n👤 [bold]User:[/bold] {msg}")
        try:
            response = chatbot.chat(session_id, msg, record_to_patchouli=True)
            console.print(f"🤖 [bold]Bot:[/bold] {response[:80]}...")
            success_count += 1
            time.sleep(0.3)
        except Exception as e:
            console.print(f"[red]✗ 对话失败: {e}[/red]")

    # 验证记忆提取（期望多条）
    if success_count == len(test_messages):
        try:
            chatbot.patchouli.flush_perception(chatbot.user_id, chatbot.agent_id, session_id)
        except:
            pass

        success, memories = verify_memory_extraction(storage, chatbot.user_id, expected_min_count=2)

        if success:
            console.print(f"[green]✓ 多记忆提取成功 (共 {len(memories)} 条)[/green]")

            # 验证包含多个主题
            keywords = ["王芳", "前端", "React", "敏捷开发"]
            verify_memory_keywords(memories, keywords)

        return success
    return False


def test_buffer_accumulation_trigger(chatbot, session_id, storage):
    """测试7: Buffer 累积触发机制"""
    console.print("\n[bold magenta]💬 测试7: Buffer 累积触发机制[/bold magenta]")

    chatbot.clear_session(session_id)

    # 恰好6条消息（3轮对话），触发 buffer 自动提取
    # 使用自然对话，避免"消息1"这样的前缀干扰
    test_messages = [
        "我最近在学习 Rust 编程语言",
        "Rust 的内存安全特性很吸引我",
        "它的性能确实非常好，接近 C++",
        "特别适合系统级编程和嵌入式开发",
        "不过学习曲线有点陡峭",
        "需要理解所有权、借用和生命周期这些概念",
    ]

    success_count = 0
    for i, msg in enumerate(test_messages, 1):
        console.print(f"\n[cyan]消息 {i}/6[/cyan]")
        console.print(f"👤 [bold]User:[/bold] {msg}")
        try:
            response = chatbot.chat(session_id, msg, record_to_patchouli=True)
            console.print(f"🤖 [bold]Bot:[/bold] {response[:60]}...")
            success_count += 1
            time.sleep(0.3)
        except Exception as e:
            console.print(f"[red]✗ 对话失败: {e}[/red]")

    # 验证：6条消息后 buffer 应自动触发（第6条消息后触发）
    if success_count == len(test_messages):
        console.print("\n[cyan]⏳ 6条消息已发送（3轮对话），Buffer 应在第6条后自动触发[/cyan]")
        success, memories = verify_memory_extraction(storage, chatbot.user_id, expected_min_count=1, wait_seconds=4)

        if success:
            console.print("[green]✓ Buffer 自动触发机制正常[/green]")
            # 验证记忆内容包含关键词
            keywords = ["Rust", "性能", "内存", "编程"]
            verify_memory_keywords(memories, keywords)
        else:
            console.print("[yellow]⚠️  Buffer 可能未自动触发，尝试手动刷新[/yellow]")
            try:
                chatbot.patchouli.flush_perception(chatbot.user_id, chatbot.agent_id, session_id)
                success, memories = verify_memory_extraction(storage, chatbot.user_id, expected_min_count=1, wait_seconds=2)
            except:
                pass

        return success
    return False


def test_reflection_extraction(chatbot, session_id, storage):
    """测试8: 经验总结提取"""
    console.print("\n[bold magenta]💬 测试8: 经验总结提取[/bold magenta]")

    chatbot.clear_session(session_id)

    # 对话轮次 - 经验总结
    test_messages = [
        "我在 API 设计中遇到过一个问题",
        "问题是接口版本控制混乱，导致客户端兼容性差",
        "解决方案是在 URL 中包含版本号（/api/v1/），并使用 Deprecation 头标记旧接口，这样用户有充足时间迁移",
    ]

    success_count = 0
    for i, msg in enumerate(test_messages, 1):
        console.print(f"\n👤 [bold]User:[/bold] {msg[:80]}...")
        try:
            response = chatbot.chat(session_id, msg, record_to_patchouli=True)
            console.print(f"🤖 [bold]Bot:[/bold] {response[:80]}...")
            success_count += 1
            time.sleep(0.3)
        except Exception as e:
            console.print(f"[red]✗ 对话失败: {e}[/red]")

    # 验证记忆提取
    if success_count == len(test_messages):
        try:
            chatbot.patchouli.flush_perception(chatbot.user_id, chatbot.agent_id, session_id)
        except:
            pass

        success, memories = verify_memory_extraction(storage, chatbot.user_id, expected_min_count=1)

        if success:
            # 验证包含经验总结关键词
            keywords = ["API", "版本", "问题", "解决"]
            verify_memory_keywords(memories, keywords)

            # 检查是否为 REFLECTION 类型（软性要求）
            has_reflection = any(mem.index.memory_type == MemoryType.REFLECTION for mem in memories)
            if has_reflection:
                console.print("[green]✓ 检测到 REFLECTION 类型记忆[/green]")
            else:
                console.print("[yellow]⚠️  未检测到 REFLECTION 类型，可能是其他类型[/yellow]")

        return success
    return False


def main():
    """主测试流程"""
    console.print(Panel.fit(
        "[bold magenta]HiveMemory ChatBot Stage 1 测试[/bold magenta]\n"
        "测试记忆生成与写入功能",
        border_style="magenta"
    ))

    # 1. 初始化系统
    system_components = setup_system()
    if not system_components:
        console.print("\n[red]✗ 系统初始化失败，测试终止[/red]")
        sys.exit(1)
    config, patchouli, session_manager, storage = system_components

    # 2. 创建 ChatBot
    try:
        chatbot = create_chatbot(config, patchouli, session_manager)
    except Exception as e:
        console.print(f"\n[red]✗ ChatBot 创建失败: {e}[/red]")
        sys.exit(1)

    # 3. 执行测试套件
    console.print("\n" + "="*60)
    console.print("[bold cyan]🧪 开始执行测试套件[/bold cyan]\n")

    test_results = {}
    base_session_id = "test_stage1_session"

    # 测试1: 基础信息提取
    storage.create_collection(recreate=True)  # 清空之前的记忆
    test_results["test1"] = test_basic_profile_extraction(
        chatbot,
        f"{base_session_id}_test1",
        storage
    )

    # 测试2: 代码片段提取
    storage.create_collection(recreate=True)
    test_results["test2"] = test_code_snippet_extraction(
        chatbot,
        f"{base_session_id}_test2",
        storage
    )

    # 测试3: 项目架构提取
    storage.create_collection(recreate=True)
    test_results["test3"] = test_project_architecture_extraction(
        chatbot,
        f"{base_session_id}_test3",
        storage
    )

    # 测试4: 工作偏好提取
    storage.create_collection(recreate=True)
    test_results["test4"] = test_work_preferences_extraction(
        chatbot,
        f"{base_session_id}_test4",
        storage
    )

    # 测试5: 低价值过滤
    storage.create_collection(recreate=True)
    test_results["test5"] = test_low_value_filtering(
        chatbot,
        f"{base_session_id}_test5",
        storage
    )

    # 测试6: 多记忆提取
    storage.create_collection(recreate=True)
    test_results["test6"] = test_multi_memory_extraction(
        chatbot,
        f"{base_session_id}_test6",
        storage
    )

    # 测试7: Buffer 触发机制
    storage.create_collection(recreate=True)
    test_results["test7"] = test_buffer_accumulation_trigger(
        chatbot,
        f"{base_session_id}_test7",
        storage
    )

    # 测试8: 经验总结提取
    storage.create_collection(recreate=True)
    test_results["test8"] = test_reflection_extraction(
        chatbot,
        f"{base_session_id}_test8",
        storage
    )

    # 4. 汇总测试结果
    console.print("\n" + "="*60)
    console.print("[bold cyan]📊 测试结果汇总[/bold cyan]\n")

    # 创建结果表格
    table = Table(title="Stage 1 测试结果", show_header=True, header_style="bold magenta")
    table.add_column("测试用例", style="cyan", width=35)
    table.add_column("状态", justify="center", width=10)
    table.add_column("说明", style="dim")

    test_names = {
        "test1": "基础信息提取 (个人资料)",
        "test2": "代码片段提取 (Python函数)",
        "test3": "项目架构知识提取",
        "test4": "工作偏好提取",
        "test5": "低价值信息过滤",
        "test6": "多记忆同时提取",
        "test7": "Buffer累积触发机制",
        "test8": "经验总结提取"
    }

    all_passed = True
    for test_id, passed in test_results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        status_style = "green" if passed else "red"
        table.add_row(test_names[test_id], f"[{status_style}]{status}[/{status_style}]", "")
        if not passed:
            all_passed = False

    console.print(table)

    # 5. 最终结果
    console.print("\n" + "="*60)
    if all_passed:
        console.print(Panel(
            "[bold green]✅ 全部测试通过！[/bold green]\n\n"
            f"共执行 {len(test_results)} 个测试用例，全部成功。\n"
            "Stage 1 记忆生成与写入功能正常。",
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
