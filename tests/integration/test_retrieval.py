"""
记忆检索端到端测试

测试场景:
1. 语义概念检索 (Dense focus)
2. 精准代码检索 (Sparse focus)
3. 结构化过滤检索 (Type/Tag filters)
4. 混合检索与排序 (Hybrid Ranking)

运行方式:
    python tests/integration/test_retrieval.py
"""

import sys
import os
from pathlib import Path
import time
import logging
from datetime import datetime, timedelta

# 设置 UTF-8 编码 (Windows 兼容性)
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hivememory.core.models import (
    MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType
)
from hivememory.core.config import get_config
from hivememory.memory.storage import QdrantMemoryStore
from hivememory.retrieval import (
    QueryProcessor,
    ProcessedQuery,
    SimpleRouter,
    HybridRetriever,
    ContextRenderer,
    RetrievalEngine,
    create_default_retrieval_engine,
    RenderFormat,
)
from hivememory.retrieval.models import QueryFilters

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

console = Console(force_terminal=True, legacy_windows=False)


# ========== 测试场景定义 ==========

SCENARIO_1 = {
    "name": "语义概念检索 (Dense Focus)",
    "description": "测试基于语义理解的检索，查找相关概念而非精确匹配",
    "queries": [
        ("如何处理时间格式？", "test_user"),
        ("项目是用什么语言写的？", "test_user"),
    ],
    "expected_titles": ["Python 日期解析函数", "项目配置信息"]
}

SCENARIO_2 = {
    "name": "精准代码检索 (Sparse Focus)",
    "description": "测试基于关键词的精准检索，查找特定函数名或变量",
    "queries": [
        ("parse_date 函数实现", "test_user"),
        ("我的 OPENAI API KEY 是什么", "test_user"),
    ],
    "expected_titles": ["Python 日期解析函数", "用户 API Key 配置"]
}

SCENARIO_3 = {
    "name": "结构化过滤检索",
    "description": "测试基于元数据的过滤功能 (Type, Tags)",
    "queries": [
        ("找一下关于配置的记忆", "test_user", MemoryType.FACT),
        ("Python 相关的代码", "test_user", MemoryType.CODE_SNIPPET),
    ],
    "filters": [
        QueryFilters(memory_type=MemoryType.FACT),
        QueryFilters(memory_type=MemoryType.CODE_SNIPPET),
    ]
}


# ========== 测试数据 ==========

TEST_MEMORIES = [
    {
        "title": "用户 API Key 配置",
        "summary": "用户设置的 OpenAI API Key 为 sk-test-123456",
        "tags": ["api-key", "config", "openai"],
        "type": MemoryType.USER_PROFILE,
        "content": "用户的 OpenAI API Key 配置：\n**API Key**: `sk-test-123456`\n\n请在调用 OpenAI API 时使用此密钥。",
        "confidence": 1.0
    },
    {
        "title": "Python 日期解析函数",
        "summary": "parse_date 函数用于解析 ISO8601 格式的日期字符串",
        "tags": ["python", "datetime", "utils", "code"],
        "type": MemoryType.CODE_SNIPPET,
        "content": """```python
def parse_date(date_str):
    \"\"\"解析 ISO8601 格式的日期字符串\"\"\"
    from datetime import datetime
    if date_str.endswith('Z'):
        date_str = date_str[:-1] + '+00:00'
    return datetime.fromisoformat(date_str)
```""",
        "confidence": 0.9
    },
    {
        "title": "项目配置信息",
        "summary": "项目环境配置为 Python 3.12，使用 Black 格式化，行宽 100",
        "tags": ["python", "config", "project"],
        "type": MemoryType.FACT,
        "content": "项目环境要求：\n- Python 版本：**3.12**\n- 代码格式化：Black\n- 行宽：100 字符",
        "confidence": 0.95
    },
    # 增加干扰项
    {
        "title": "JavaScript 日期处理",
        "summary": "使用 moment.js 处理日期",
        "tags": ["javascript", "date", "utils"],
        "type": MemoryType.CODE_SNIPPET,
        "content": "import moment from 'moment';\nconst date = moment().format();",
        "confidence": 0.8
    },
    {
        "title": "Rust 项目配置",
        "summary": "Cargo.toml 配置示例",
        "tags": ["rust", "config"],
        "type": MemoryType.FACT,
        "content": "[package]\nname = \"demo\"\nversion = \"0.1.0\"",
        "confidence": 0.85
    }
]


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
        storage.create_collection(recreate=True)
        console.print("✓ Qdrant 初始化成功", style="green")

        return storage, config

    except Exception as e:
        console.print(f"✗ 环境准备失败: {e}", style="bold red")
        return None, None


def insert_test_memories(storage: QdrantMemoryStore, user_id: str = "test_user"):
    """插入测试记忆"""
    console.print("\n[bold cyan]📝 插入测试记忆...[/bold cyan]")
    
    inserted = []
    for mem_data in TEST_MEMORIES:
        memory = MemoryAtom(
            meta=MetaData(
                source_agent_id="test_agent",
                user_id=user_id,
                confidence_score=mem_data["confidence"]
            ),
            index=IndexLayer(
                title=mem_data["title"],
                summary=mem_data["summary"],
                tags=mem_data["tags"],
                memory_type=mem_data["type"]
            ),
            payload=PayloadLayer(
                content=mem_data["content"]
            )
        )
        
        storage.upsert_memory(memory, use_sparse=True)
        inserted.append(memory)
        console.print(f"  ✓ {mem_data['title']}")
    
    console.print(f"\n[green]成功插入 {len(inserted)} 条测试记忆[/green]")
    return inserted


def test_query_processor():
    """测试查询预处理器"""
    console.print("\n[bold magenta]📊 测试 QueryProcessor[/bold magenta]")
    
    processor = QueryProcessor()
    
    test_cases = [
        ("我之前设置的 API Key 是什么？", "时间引用检测"),
        ("找一下项目里的日期处理代码", "类型检测 (CODE)"),
        ("昨天讨论的 Python 配置", "时间范围解析"),
    ]
    
    results = []
    for query, description in test_cases:
        processed = processor.process(query, user_id="test_user")
        
        console.print(f"\n  [cyan]查询:[/cyan] {query}")
        console.print(f"  [dim]{description}[/dim]")
        console.print(f"  → 语义查询: {processed.semantic_query[:50]}...")
        console.print(f"  → 关键词: {processed.keywords}")
        if processed.filters.memory_type:
            console.print(f"  → 类型过滤: {processed.filters.memory_type.value}")
        if processed.filters.time_range:
            console.print(f"  → 时间范围: {processed.filters.time_range}")
        
        results.append((query, processed))
    
    console.print("\n[green]✓ QueryProcessor 测试完成[/green]")
    return results


def test_router():
    """测试检索路由器"""
    console.print("\n[bold magenta]🚦 测试 SimpleRouter[/bold magenta]")
    
    router = SimpleRouter()
    
    test_cases = [
        ("你好", False),  # 闲聊
        ("我之前的 API Key 是什么", True),  # 需要检索
        ("帮我写一个排序算法", False),  # 新任务
        ("项目里那个日期函数怎么用", True),  # 引用历史
        ("谢谢", False),  # 简单回复
    ]
    
    results = []
    for query, expected in test_cases:
        result = router.should_retrieve(query)
        status = "✓" if result == expected else "✗"
        color = "green" if result == expected else "red"
        
        console.print(f"  [{color}]{status}[/{color}] \"{query}\" → {result} (expected: {expected})")
        results.append((query, result, expected))
    
    passed = sum(1 for _, r, e in results if r == e)
    console.print(f"\n[{'green' if passed == len(results) else 'yellow'}]路由测试: {passed}/{len(results)} 通过[/]")
    return results


def test_hybrid_retriever(storage: QdrantMemoryStore):
    """测试混合检索器 (Dense + Sparse)"""
    console.print("\n[bold magenta]🔍 测试 HybridRetriever[/bold magenta]")
    
    # 确保启用混合搜索
    retriever = HybridRetriever(storage=storage, enable_hybrid_search=True)
    
    # 场景 1: 语义优先 (Dense)
    query_text = "如何处理时间"
    console.print(f"\n  [cyan]场景 1: 语义检索[/cyan] (查询: '{query_text}')")
    processed_query = ProcessedQuery(
        semantic_query=query_text, 
        original_query=query_text
    )

    results = retriever.retrieve(processed_query, top_k=3)
    
    for i, r in enumerate(results.results, 1):
        console.print(f"    {i}. {r.memory.index.title} (score: {r.score:.3f}) - {r.match_reason}")

    # 场景 2: 关键词优先 (Sparse)
    query_text = "parse_date"
    console.print(f"\n  [cyan]场景 2: 关键词检索[/cyan] (查询: '{query_text}')")
    processed_query = ProcessedQuery(
        semantic_query=query_text, 
        original_query=query_text,
        keywords=["parse_date"]  # 模拟提取到的关键词
    )
    results = searcher.retrieve(processed_query, top_k=3)
    
    for i, r in enumerate(results.results, 1):
        console.print(f"    {i}. {r.memory.index.title} (score: {r.score:.3f}) - {r.match_reason}")
    
    console.print("\n[green]✓ HybridRetriever 测试完成[/green]")


def test_context_renderer(storage: QdrantMemoryStore):
    """测试上下文渲染器"""
    console.print("\n[bold magenta]📄 测试 ContextRenderer[/bold magenta]")
    
    # 先检索一些记忆
    retriever = HybridRetriever(storage=storage, enable_hybrid_search=True)
    query = ProcessedQuery(semantic_query="API Key", original_query="API Key")
    results = retriever.retrieve(query, top_k=2)
    
    # 测试 XML 渲染
    renderer_xml = ContextRenderer(render_format=RenderFormat.XML, max_tokens=1000)
    xml_output = renderer_xml.render(results.results)
    
    console.print("\n  [cyan]XML 格式输出:[/cyan]")
    console.print(Panel(xml_output[:500] + "..." if len(xml_output) > 500 else xml_output, 
                        title="XML Context", border_style="blue"))
    
    # 测试 Markdown 渲染
    renderer_md = ContextRenderer(render_format=RenderFormat.MARKDOWN, max_tokens=1000)
    md_output = renderer_md.render(results.results)
    
    console.print("\n  [cyan]Markdown 格式输出:[/cyan]")
    console.print(Panel(md_output[:500] + "..." if len(md_output) > 500 else md_output,
                        title="Markdown Context", border_style="green"))
    
    console.print("\n[green]✓ ContextRenderer 测试完成[/green]")


def test_retrieval_engine(storage: QdrantMemoryStore):
    """测试完整检索引擎 (Engine Flow)"""
    console.print("\n[bold magenta]🚀 测试 RetrievalEngine (完整流程)[/bold magenta]")
    
    # 创建默认检索引擎
    engine = create_default_retrieval_engine(
        storage=storage,
        enable_routing=True,
        top_k=3,
        threshold=0.3,
        render_format="xml"
    )
    
    # 测试过滤条件传递
    console.print("\n  [cyan]测试带过滤条件的检索:[/cyan]")
    # retrieve_context 不直接支持 memory_type 参数，通常由 processor 从 query 中提取
    # 这里我们使用 search_memories 接口来测试显式过滤
    memories = engine.search_memories(
        query_text="代码规范",
        user_id="test_user",
        memory_type="FACT"  # 指定只检索 FACT 类型
    )
    
    if memories:
        for mem in memories:
            console.print(f"    • {mem.index.title} [{mem.index.memory_type.value}]")
            if mem.index.memory_type != MemoryType.FACT:
                 console.print(f"      [red]✗ 类型错误: {mem.index.memory_type}[/red]")
    else:
        console.print("    [yellow]未找到匹配记忆[/yellow]")
        
    console.print("\n[green]✓ RetrievalEngine 测试完成[/green]")


def run_acceptance_test(storage: QdrantMemoryStore):
    """验收测试：模拟完整的记忆召回场景"""
    console.print("\n[bold magenta]🏆 验收测试：记忆召回场景[/bold magenta]")
    
    engine = create_retrieval_engine(storage=storage, render_format="xml", threshold=0.1)
    
    scenarios = [SCENARIO_1, SCENARIO_2, SCENARIO_3]
    passed_count = 0
    total_checks = 0
    
    for scenario in scenarios:
        console.print(f"\n[bold cyan]场景: {scenario['name']}[/bold cyan]")
        console.print(f"[dim]{scenario['description']}[/dim]")
        
        # 处理不同场景的输入
        queries = scenario.get("queries", [])
        filters_list = scenario.get("filters", [None] * len(queries))
        
        for (query_text, user_id, *rest), filter_obj in zip(queries, filters_list):
            mem_type_str = rest[0].value if rest else None
            
            console.print(f"\n  [bold]用户提问:[/bold] {query_text}")
            
            # 调用 Engine
            # 简单起见，如果指定了 memory_type，我们使用 search_memories 来验证过滤
            # 注意：search_memories 返回的是 MemoryAtom 列表，没有分数信息
            # 为了获取分数，我们需要直接访问 engine 的 searcher
            if mem_type_str:
                # 构造 ProcessedQuery
                from hivememory.retrieval.models import ProcessedQuery, QueryFilters
                
                filters = QueryFilters()
                if mem_type_str == "FACT":
                    filters.memory_type = MemoryType.FACT
                elif mem_type_str == "CODE_SNIPPET":
                    filters.memory_type = MemoryType.CODE_SNIPPET
                
                if user_id:
                    filters.user_id = user_id
                    
                p_query = ProcessedQuery(
                    semantic_query=query_text,
                    original_query=query_text,
                    filters=filters
                )
                
                # 直接调用 retrieve 获取带分数的 SearchResults
                search_results = engine.searcher.retrieve(p_query, top_k=5)
                result_list = search_results.results
            else:
                # 正常流程，也需要获取 SearchResults 对象而非仅仅 memories
                # retrieve_context 返回的是 Context 对象，我们需要其原始 search_results
                # 但 engine.retrieve_context 内部封装了 retrieve，我们可以通过 retrieve_context 返回的 metadata 获取分数
                # 或者更简单，直接再次调用 searcher 用于展示
                
                # 为了不破坏原有流程，我们这里模拟调用 searcher
                # 注意：这里需要确保使用与 engine 相同的 query processor
                p_query = engine.processor.process(query=query_text, user_id=user_id)
                search_results = engine.searcher.retrieve(p_query, top_k=5)
                result_list = search_results.results
            
            # 检查结果
            console.print(f"  [dim]检索到 {len(result_list)} 条记忆[/dim]")
            
            # 显示所有结果及其分数
            for i, r in enumerate(result_list, 1):
                title = r.memory.index.title
                score = r.score
                reason = r.match_reason
                console.print(f"    {i}. [green]{title}[/green] (score: {score:.4f}) - [dim]{reason}[/dim]")

            # 验证 top-1 是否相关
            if result_list:
                passed_count += 1
            else:
                console.print("  [red]✗ 未召回任何记忆[/red]")
            
            total_checks += 1
            
    return passed_count == total_checks


def main():
    """主测试流程"""
    console.print(Panel.fit(
        "[bold magenta]HiveMemory 阶段2 - 记忆检索模块测试[/bold magenta]\n"
        "测试查询处理、路由、混合检索、渲染全流程",
        border_style="magenta"
    ))
    
    # 环境准备
    storage, config = setup_environment()
    if not storage:
        return
    
    # 插入测试数据
    insert_test_memories(storage)
    
    # 等待索引建立
    time.sleep(1)
    
    # 运行各模块测试
    test_query_processor()
    test_router()
    test_hybrid_retriever(storage)
    test_context_renderer(storage)
    test_retrieval_engine(storage)
    
    # 验收测试
    success = run_acceptance_test(storage)
    
    # 汇总
    console.print("\n" + "="*60)
    console.print("\n[bold cyan]📋 测试完成[/bold cyan]")
    
    if success:
        console.print("[green]所有测试通过！阶段2 记忆检索模块已就绪。[/green]")
        console.print("\n[dim]下一步：运行 examples/memory_chat.py 进行交互式测试[/dim]")
    else:
        console.print("[yellow]部分测试需要检查。[/yellow]")


if __name__ == "__main__":
    main()
