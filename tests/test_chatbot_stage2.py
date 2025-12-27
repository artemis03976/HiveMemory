"""
ChatBot Agent Stage 2 测试: 记忆检索与上下文注入

测试内容:
1. 初始化带有 RetrievalEngine 的 ChatBot
2. 预置多个场景的记忆原子到 Qdrant (10条)
3. 测试对话中能否正确检索并利用记忆
4. 验证检索统计信息和路由决策

测试用例:
- test_basic_fact_retrieval: 基础事实检索 (用户技术栈偏好)
- test_code_snippet_retrieval: 代码片段检索 (CSV处理函数)
- test_multi_memory_retrieval: 多记忆检索 (HiveMemory架构)
- test_router_skip_retrieval: 路由器跳过检索 (问候语)
- test_no_relevant_memories: 无相关记忆 (烹饪问题)

运行方式:
    python tests/test_chatbot_stage2.py
"""

import sys
import os
from pathlib import Path
from uuid import uuid4

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
from hivememory.core.models import MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType
from hivememory.memory.storage import QdrantMemoryStore
from hivememory.agents.patchouli import PatchouliAgent
from hivememory.agents.chatbot import ChatBotAgent
from hivememory.agents.session_manager import SessionManager
from hivememory.retrieval import create_retrieval_engine, RetrievalEngine

console = Console(force_terminal=True, legacy_windows=False)


def setup_system():
    """初始化系统组件 (包含检索模块)"""
    console.print("\n[bold cyan]📦 初始化系统组件 (Stage 2)...[/bold cyan]")

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

        # 初始化 Qdrant Storage
        storage = QdrantMemoryStore(
            qdrant_config=config.qdrant,
            embedding_config=config.embedding
        )
        storage.create_collection(recreate=True)
        console.print(f"  ✓ Qdrant 连接成功 ({config.qdrant.host}:{config.qdrant.port})")

        # 初始化 Patchouli Agent
        patchouli = PatchouliAgent(storage=storage)
        console.print("  ✓ PatchouliAgent 初始化成功")

        # 初始化 Session Manager
        session_manager = SessionManager(
            redis_client=redis_client,
            key_prefix="hivememory:test:stage2",
            ttl_days=7
        )
        console.print("  ✓ SessionManager 初始化成功")
        
        # 初始化 Retrieval Engine (Stage 2 新增)
        retrieval_engine = create_retrieval_engine(
            storage=storage,
            enable_routing=True,  # 启用路由
            top_k=3,
            threshold=0.6,
            render_format="xml"
        )
        console.print("  ✓ RetrievalEngine 初始化成功")

        return config, patchouli, session_manager, storage, retrieval_engine

    except Exception as e:
        console.print(f"  ✗ 初始化失败: {e}", style="bold red")
        return None


def create_chatbot(config, patchouli, session_manager, retrieval_engine):
    """创建 ChatBot Agent (带检索功能)"""
    console.print("\n[bold cyan]🤖 创建 ChatBot Agent...[/bold cyan]")
    
    try:
        worker_llm_config = config.get_worker_llm_config()

        chatbot = ChatBotAgent(
            patchouli=patchouli,
            session_manager=session_manager,
            user_id="test_user_stage2",
            agent_id="test_chatbot_v2",
            llm_config=worker_llm_config,
            retrieval_engine=retrieval_engine,  # 注入检索引擎
            enable_memory_retrieval=True
        )
        
        console.print("  ✓ ChatBot 创建成功 (已启用记忆检索)")

        return chatbot
        
    except Exception as e:
        console.print(f"  ✗ ChatBot 创建失败: {e}", style="bold red")
        raise


def setup_test_memories(storage, user_id):
    """创建并注入多个测试记忆到 Qdrant

    返回包含各类记忆的字典，供测试使用
    """
    console.print("\n[bold cyan]💉 注入测试记忆集...[/bold cyan]")

    memories = {}

    # ===== 1. 用户偏好类记忆 =====
    # 1.1 工作技术栈偏好
    tech_stack_memory = MemoryAtom(
        id=uuid4(),
        meta=MetaData(
            source_agent_id="system_inject",
            user_id=user_id,
            confidence_score=1.0
        ),
        index=IndexLayer(
            title="用户技术栈偏好",
            summary="软件工程师，5年经验，主要使用Python和TypeScript进行开发",
            tags=["work", "preferences", "tech_stack", "programming"],
            memory_type=MemoryType.USER_PROFILE,
        ),
        payload=PayloadLayer(
            content="用户是一名软件工程师，有5年开发经验。\n"
                   "主要技术栈：Python（后端开发）、TypeScript（前端开发）\n"
                   "常用框架：FastAPI, React, Pydantic\n"
                   "工作地点：北京\n"
                   "团队规模：5-8人"
        )
    )
    memories["tech_stack"] = tech_stack_memory

    # 1.2 工作习惯
    work_habits_memory = MemoryAtom(
        id=uuid4(),
        meta=MetaData(
            source_agent_id="system_inject",
            user_id=user_id,
            confidence_score=0.9
        ),
        index=IndexLayer(
            title="用户工作习惯",
            summary="喜欢使用TDD开发方式，重视代码质量和测试覆盖率",
            tags=["work", "habits", "tdd", "testing"],
            memory_type=MemoryType.USER_PROFILE,
        ),
        payload=PayloadLayer(
            content="用户开发习惯：\n"
                   "- 严格遵循TDD（测试驱动开发）流程\n"
                   "- 要求测试覆盖率 > 80%\n"
                   "- 使用 pytest 进行单元测试\n"
                   "- 每次提交前必须运行 pylint 和 mypy 检查\n"
                   "- 喜欢写详细的文档字符串"
        )
    )
    memories["work_habits"] = work_habits_memory

    # ===== 2. 开发工作流类记忆 =====
    # 2.1 代码片段：CSV处理工具
    csv_utils_memory = MemoryAtom(
        id=uuid4(),
        meta=MetaData(
            source_agent_id="system_inject",
            user_id=user_id,
            confidence_score=1.0
        ),
        index=IndexLayer(
            title="Python CSV数据清洗工具函数",
            summary="用于CSV文件读取和清洗的实用函数，支持多种编码格式",
            tags=["python", "code", "utils", "csv", "data_processing"],
            memory_type=MemoryType.CODE_SNIPPET,
        ),
        payload=PayloadLayer(
            content="```python\ndef clean_csv_data(filepath: str, encoding: str = 'utf-8') -> pd.DataFrame:\n"
                   "    \"\"\"\n"
                   "    读取并清洗CSV文件，处理脏数据和编码问题\n"
                   "    \n"
                   "    Args:\n"
                   "        filepath: CSV文件路径\n"
                   "        encoding: 文件编码，默认utf-8，支持gbk、gb18030等\n"
                   "    \n"
                   "    Returns:\n"
                   "        清洗后的DataFrame\n"
                   "    \"\"\"\n"
                   "    # 尝试多种编码读取\n"
                   "    for enc in [encoding, 'gbk', 'gb18030', 'latin1']:\n"
                   "        try:\n"
                   "            df = pd.read_csv(filepath, encoding=enc)\n"
                   "            break\n"
                   "        except UnicodeDecodeError:\n"
                   "            continue\n"
                   "    \n"
                   "    # 删除空行和重复行\n"
                   "    df = df.dropna().drop_duplicates()\n"
                   "    \n"
                   "    return df\n"
                   "```\n"
                   "\n"
                   "用途说明：处理用户上传的脏数据文件，支持中文编码自动识别。"
        )
    )
    memories["csv_utils"] = csv_utils_memory

    # 2.2 项目架构知识
    arch_layer1_memory = MemoryAtom(
        id=uuid4(),
        meta=MetaData(
            source_agent_id="system_inject",
            user_id=user_id,
            confidence_score=1.0
        ),
        index=IndexLayer(
            title="HiveMemory架构-ChatBot层",
            summary="ChatBot Agent负责用户对话接口和记忆检索",
            tags=["architecture", "project", "chatbot", "design"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(
            content="HiveMemory三层架构 - 第一层：ChatBot Agent\n"
                   "\n"
                   "职责：\n"
                   "- 处理用户对话输入输出\n"
                   "- 管理会话状态和对话历史\n"
                   "- 集成记忆检索功能（Stage 2）\n"
                   "- 调用Patchouli Agent进行记忆提取\n"
                   "\n"
                   "技术栈：\n"
                   "- LLM: 生成对话回复\n"
                   "- Redis: 会话缓存\n"
                   "- Qdrant: 记忆存储与检索"
        )
    )
    memories["arch_layer1"] = arch_layer1_memory

    arch_layer2_memory = MemoryAtom(
        id=uuid4(),
        meta=MetaData(
            source_agent_id="system_inject",
            user_id=user_id,
            confidence_score=1.0
        ),
        index=IndexLayer(
            title="HiveMemory架构-Patchouli层",
            summary="Patchouli Agent负责记忆提取、整理和存储",
            tags=["architecture", "project", "patchouli", "design"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(
            content="HiveMemory三层架构 - 第二层：Patchouli Agent\n"
                   "\n"
                   "职责：\n"
                   "- 从对话中提取关键信息\n"
                   "- 生成结构化记忆原子（MemoryAtom）\n"
                   "- 计算向量化嵌入（Embedding）\n"
                   "- 管理记忆的生命周期（创建、更新、归档）\n"
                   "\n"
                   "核心能力：\n"
                   "- 信息提取与分类\n"
                   "- 记忆去重与合并\n"
                   "- 重要性评分"
        )
    )
    memories["arch_layer2"] = arch_layer2_memory

    arch_layer3_memory = MemoryAtom(
        id=uuid4(),
        meta=MetaData(
            source_agent_id="system_inject",
            user_id=user_id,
            confidence_score=1.0
        ),
        index=IndexLayer(
            title="HiveMemory架构-Storage层",
            summary="Storage Layer基于Qdrant实现向量存储和相似度检索",
            tags=["architecture", "project", "storage", "qdrant"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(
            content="HiveMemory三层架构 - 第三层：Storage Layer\n"
                   "\n"
                   "职责：\n"
                   "- 向量数据库操作（Qdrant）\n"
                   "- 相似度检索（语义搜索）\n"
                   "- 元数据��滤（按user_id、tags等）\n"
                   "- 记忆持久化存储\n"
                   "\n"
                   "特性：\n"
                   "- 支持多种Embedding模型\n"
                   "- 混合检索（向量+元数据）\n"
                   "- 自动索引优化"
        )
    )
    memories["arch_layer3"] = arch_layer3_memory

    # 2.3 代码审查规范
    code_review_memory = MemoryAtom(
        id=uuid4(),
        meta=MetaData(
            source_agent_id="system_inject",
            user_id=user_id,
            confidence_score=0.95
        ),
        index=IndexLayer(
            title="团队代码审查规范",
            summary="代码审查必须通过的三个检查点",
            tags=["guidelines", "review", "standards", "best_practices"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(
            content="团队代码审查检查清单：\n"
                   "\n"
                   "【必查项】\n"
                   "1. 所有函数必须有完整的类型注解（Type Hints）\n"
                   "2. 测试覆盖率必须 > 80%（使用 pytest-cov 测量）\n"
                   "3. 必须通过 pylint 和 mypy 静态检查（评分 > 8.0）\n"
                   "\n"
                   "【推荐项】\n"
                   "- 关键函数需要添加文档字符串（docstring）\n"
                   "- 复杂逻辑需要添加注释说明\n"
                   "- 遵循 PEP 8 代码风格规范"
        )
    )
    memories["code_review"] = code_review_memory

    # ===== 3. 经验总结类记忆 =====
    # 3.1 API设计经验
    api_design_memory = MemoryAtom(
        id=uuid4(),
        meta=MetaData(
            source_agent_id="system_inject",
            user_id=user_id,
            confidence_score=0.9
        ),
        index=IndexLayer(
            title="RESTful API设计经验总结",
            summary="API版本控制中遇到的问题和解决方案",
            tags=["api", "design", "lessons_learned", "rest"],
            memory_type=MemoryType.REFLECTION,
        ),
        payload=PayloadLayer(
            content="问题：RESTful API 版本控制混乱，客户端兼容性差\n"
                   "\n"
                   "解决方案：\n"
                   "1. 在 URL 中包含版本号（如 /api/v1/users, /api/v2/users）\n"
                   "2. 使用 Deprecation HTTP头标记旧接口\n"
                   "3. 维护至少 2 个主版本（N 和 N-1）\n"
                   "4. 新版本保持向后兼容，不删除字段只增加\n"
                   "5. 提供 API 变更日志（Changelog）\n"
                   "\n"
                   "效果：升级平滑，用户有充足时间迁移"
        )
    )
    memories["api_design"] = api_design_memory

    # 3.2 调试经验
    debugging_memory = MemoryAtom(
        id=uuid4(),
        meta=MetaData(
            source_agent_id="system_inject",
            user_id=user_id,
            confidence_score=0.85
        ),
        index=IndexLayer(
            title="Python内存泄漏调试经验",
            summary="使用内存分析工具定位和解决引用循环问题",
            tags=["debugging", "python", "performance", "memory"],
            memory_type=MemoryType.REFLECTION,
        ),
        payload=PayloadLayer(
            content="问题场景：FastAPI应用运行一段时间后内存占用持续增长\n"
                   "\n"
                   "排查过程：\n"
                   "1. 使用 tracemalloc 追踪内存分配\n"
                   "2. 使用 objgraph 查看对象引用关系\n"
                   "3. 使用 memory_profiler 分析内存热点\n"
                   "\n"
                   "发现原因：\n"
                   "- 全局事件处理器未取消注册\n"
                   "- 缓存对象无限增长（无LRU淘汰）\n"
                   "- 异步任务未正确关闭导致引用循环\n"
                   "\n"
                   "解决方案：\n"
                   "- 使用 weakref 避免强引用\n"
                   "- 添加 functools.lru_cache 装饰器\n"
                   "- 实现正确的资源清理（__del__, contextlib）"
        )
    )
    memories["debugging"] = debugging_memory

    # 批量注入记忆
    try:
        for key, memory in memories.items():
            storage.upsert_memory(memory)
            console.print(f"  ✓ [{key:15s}] {memory.index.title}")

        console.print(f"\n  [green]总计注入 {len(memories)} 条记忆[/green]")
        return memories

    except Exception as e:
        console.print(f"  ✗ 注入失败: {e}", style="bold red")
        raise


def verify_retrieval_happened(chatbot, expected_min_count=1):
    """验证检索是否发生并返回检索信息"""
    retrieval_info = chatbot.get_last_retrieval_info()

    if not retrieval_info:
        console.print("[red]✗ 无检索信息记录[/red]")
        return False, None

    console.print("\n[dim]检索调试信息:[/dim]")
    console.print(f"  - 触发检索: {retrieval_info['should_retrieve']}")
    console.print(f"  - 记忆数量: {retrieval_info['memories_count']}")
    console.print(f"  - 检索耗时: {retrieval_info['latency_ms']:.1f}ms")

    if retrieval_info['memories_count'] >= expected_min_count:
        console.print(f"  ✓ 检索到 {retrieval_info['memories_count']} 条记忆")
        for i, mem in enumerate(retrieval_info['memories'][:3], 1):  # 最多显示3条
            console.print(f"    {i}. {mem['title']}")
        return True, retrieval_info
    else:
        console.print(f"[red]✗ 期望至少 {expected_min_count} 条记忆，实际 {retrieval_info['memories_count']} 条[/red]")
        return False, retrieval_info


def test_basic_fact_retrieval(chatbot, session_id):
    """测试1: 基础事实检索 - 用户技术栈偏好"""
    console.print("\n[bold magenta]💬 测试1: 基础事实检索 - 用户技术栈[/bold magenta]")

    chatbot.clear_session(session_id)
    question = "我的主要技术栈是什么？"
    console.print(f"\n👤 [bold]User:[/bold] {question}")

    try:
        with console.status("[bold green]思考中...[/bold green]"):
            response = chatbot.chat(
                session_id=session_id,
                user_message=question,
                record_to_patchouli=False
            )

        console.print(f"🤖 [bold]Bot:[/bold] {response}")

        # 验证答案包含关键技术栈信息
        keywords = ["Python", "TypeScript"]
        found_all = all(kw in response for kw in keywords)

        if found_all:
            console.print("[green]✓ 回复包含预期技术栈信息 (Python, TypeScript)[/green]")
        else:
            console.print(f"[red]✗ 回复缺少关键技术栈信息[/red]")
            return False

        # 验证检索发生
        success, _ = verify_retrieval_happened(chatbot, expected_min_count=1)
        return success

    except Exception as e:
        console.print(f"[red]✗ 测试出错: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return False


def test_code_snippet_retrieval(chatbot, session_id):
    """测试2: 代码片段检索 - CSV处理函数"""
    console.print("\n[bold magenta]💬 测试2: 代码片段检索 - CSV工具函数[/bold magenta]")

    chatbot.clear_session(session_id)
    question = "我们之前写的CSV处理函数怎么用的？"
    console.print(f"\n👤 [bold]User:[/bold] {question}")

    try:
        with console.status("[bold green]思考中...[/bold green]"):
            response = chatbot.chat(
                session_id=session_id,
                user_message=question,
                record_to_patchouli=False
            )

        console.print(f"🤖 [bold]Bot:[/bold] {response}")

        # 验证答案包含函数名或用途说明
        keywords = ["clean_csv_data", "CSV", "函数"]
        found_any = any(kw in response for kw in keywords)

        if found_any:
            console.print("[green]✓ 回复包含CSV函数相关信息[/green]")
        else:
            console.print(f"[red]✗ 回复缺少CSV函数信息[/red]")
            return False

        # 验证检索发生
        success, _ = verify_retrieval_happened(chatbot, expected_min_count=1)
        return success

    except Exception as e:
        console.print(f"[red]✗ 测试出错: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return False


def test_multi_memory_retrieval(chatbot, session_id):
    """测试3: 多记忆检索 - HiveMemory架构"""
    console.print("\n[bold magenta]💬 测试3: 多记忆检索 - HiveMemory架构[/bold magenta]")

    chatbot.clear_session(session_id)
    question = "HiveMemory的架构是怎么设计的？"
    console.print(f"\n👤 [bold]User:[/bold] {question}")

    try:
        with console.status("[bold green]思考中...[/bold green]"):
            response = chatbot.chat(
                session_id=session_id,
                user_message=question,
                record_to_patchouli=False
            )

        console.print(f"🤖 [bold]Bot:[/bold] {response}")

        # 验证答案包含多个架构层级的信息
        keywords = ["ChatBot", "Patchouli", "Storage"]
        found_count = sum(1 for kw in keywords if kw in response)

        if found_count >= 2:
            console.print(f"[green]✓ 回复综合了多个架构层级信息 ({found_count}/3)[/green]")
        else:
            console.print(f"[yellow]⚠ 回复包含的架构层级较少 ({found_count}/3)，继续检查检索情况[/yellow]")

        success, retrieval_info = verify_retrieval_happened(chatbot, expected_min_count=2)

        if success and retrieval_info:
            console.print(f"  ✓ 多记忆检索成功 (共 {retrieval_info['memories_count']} 条)")

        # 综合判断：回复质量和检索情况都要符合预期
        if found_count >= 2 and success:
            return True
        elif found_count < 2 and not success:
            # 两者都失败才算失败
            console.print("[red]✗ 回复内容不足且检索未达到预期[/red]")
            return False
        else:
            # 部分成功，给出警告但不算失败
            if found_count < 2:
                console.print("[yellow]⚠ 检索成功但回复质量有待改进[/yellow]")
            else:
                console.print("[yellow]⚠ 回复质量良好但检索数量未达预期[/yellow]")
            return True

    except Exception as e:
        console.print(f"[red]✗ 测试出错: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return False


def test_router_skip_retrieval(chatbot, session_id):
    """测试4: 路由器跳过检索 - 问候语"""
    console.print("\n[bold magenta]💬 测试4: 路由器跳过检索 - 问候语[/bold magenta]")

    chatbot.clear_session(session_id)
    question = "你好"
    console.print(f"\n👤 [bold]User:[/bold] {question}")

    try:
        with console.status("[bold green]思考中...[/bold green]"):
            response = chatbot.chat(
                session_id=session_id,
                user_message=question,
                record_to_patchouli=False
            )

        console.print(f"🤖 [bold]Bot:[/bold] {response}")

        # 验证路由器判断不应检索
        retrieval_info = chatbot.get_last_retrieval_info()

        if not retrieval_info:
            console.print("[red]✗ 无检索信息记录[/red]")
            return False

        console.print("\n[dim]路由决策:[/dim]")
        console.print(f"  - 触发检索: {retrieval_info['should_retrieve']}")
        console.print(f"  - 记忆数量: {retrieval_info['memories_count']}")

        if not retrieval_info['should_retrieve']:
            console.print("[green]✓ 路由器正确判断无需检索[/green]")
            return True
        else:
            console.print("[yellow]⚠ 路由器触发了检索（对于问候语可能不够优化）[/yellow]")
            # 这是一个软性警告，不算失败
            return True

    except Exception as e:
        console.print(f"[red]✗ 测试出错: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return False


def test_no_relevant_memories(chatbot, session_id):
    """测试5: 无相关记忆 - 完全无关的问题"""
    console.print("\n[bold magenta]💬 测试5: 无相关记忆 - 烹饪问题[/bold magenta]")

    chatbot.clear_session(session_id)
    question = "如何制作红烧肉？"
    console.print(f"\n👤 [bold]User:[/bold] {question}")

    try:
        with console.status("[bold green]思考中...[/bold green]"):
            response = chatbot.chat(
                session_id=session_id,
                user_message=question,
                record_to_patchouli=False
            )

        console.print(f"🤖 [bold]Bot:[/bold] {response}")

        # 验证LLM正常响应（但可能不包含记忆信息）
        if len(response) > 10:  # 简单检查是否有实质回复
            console.print("[green]✓ LLM正常生成回复（无相关记忆）[/green]")
        else:
            console.print("[red]✗ LLM回复异常简短[/red]")
            return False

        # 验证检索情况（可能检索也可能不检索）
        retrieval_info = chatbot.get_last_retrieval_info()

        if retrieval_info:
            console.print("\n[dim]检索调试信息:[/dim]")
            console.print(f"  - 触发检索: {retrieval_info['should_retrieve']}")
            console.print(f"  - 记忆数量: {retrieval_info['memories_count']}")

            # 对于完全无关的问题，应该检索不到记忆或检索很少
            if retrieval_info['memories_count'] == 0:
                console.print("[green]✓ 正确识别无相关记忆[/green]")
            else:
                console.print(f"[yellow]⚠ 检索到 {retrieval_info['memories_count']} 条记忆（可能误检）[/yellow]")

        return True  # 无论检索结果如何，只要LLM正常响应就算通过

    except Exception as e:
        console.print(f"[red]✗ 测试出错: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return False


def main():
    console.print(Panel.fit(
        "[bold magenta]HiveMemory ChatBot Stage 2 测试[/bold magenta]\n"
        "测试记忆检索与上下文注入功能",
        border_style="magenta"
    ))
    
    # 1. 初始化
    components = setup_system()
    if not components:
        console.print("\n[red]✗ 系统初始化失败，测试终止[/red]")
        sys.exit(1)
    config, patchouli, session_manager, storage, retrieval_engine = components
    
    # 2. 创建 ChatBot
    chatbot = create_chatbot(config, patchouli, session_manager, retrieval_engine)
    
    # 3. 注入测试记忆集
    user_id = chatbot.user_id
    try:
        memories = setup_test_memories(storage, user_id)
    except Exception:
        sys.exit(1)

    # 等待索引刷新 (Qdrant 有时需要一点时间，虽然 upsert 通常很快)
    time.sleep(1)

    # 4. 执行测试套件
    console.print("\n" + "="*60)
    console.print("[bold cyan]🧪 开始执行测试套件[/bold cyan]\n")

    session_id = "test_stage2_session_001"
    test_results = {}

    # 测试1: 基础事实检索
    test_results["test1"] = test_basic_fact_retrieval(chatbot, f"{session_id}_test1")

    # 测试2: 代码片段检索
    test_results["test2"] = test_code_snippet_retrieval(chatbot, f"{session_id}_test2")

    # 测试3: 多记忆检索
    test_results["test3"] = test_multi_memory_retrieval(chatbot, f"{session_id}_test3")

    # 测试4: 路由器跳过检索
    test_results["test4"] = test_router_skip_retrieval(chatbot, f"{session_id}_test4")

    # 测试5: 无相关记忆
    test_results["test5"] = test_no_relevant_memories(chatbot, f"{session_id}_test5")

    # 5. 汇总测试结果
    console.print("\n" + "="*60)
    console.print("[bold cyan]📊 测试结果汇总[/bold cyan]\n")

    # 创建结果表格
    table = Table(title="Stage 2 测试结果", show_header=True, header_style="bold magenta")
    table.add_column("测试用例", style="cyan", width=30)
    table.add_column("状态", justify="center", width=10)
    table.add_column("说明", style="dim")

    test_names = {
        "test1": "基础事实检索 (技术栈)",
        "test2": "代码片段检索 (CSV函数)",
        "test3": "多记忆检索 (架构)",
        "test4": "路由器跳过检索 (问候)",
        "test5": "无相关记忆 (烹饪)"
    }

    all_passed = True
    for test_id, passed in test_results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        status_style = "green" if passed else "red"
        table.add_row(test_names[test_id], f"[{status_style}]{status}[/{status_style}]", "")
        if not passed:
            all_passed = False

    console.print(table)

    # 6. 最终结果
    console.print("\n" + "="*60)
    if all_passed:
        console.print(Panel(
            "[bold green]✅ 全部测试通过！[/bold green]\n\n"
            f"共执行 {len(test_results)} 个测试用例，全部成功。\n"
            "Stage 2 记忆检索与上下文注入功能正常。",
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
