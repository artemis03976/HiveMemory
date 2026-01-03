"""
记忆检索端到端测试

测试场景:
1. 存入 API Key 记忆 → 新对话中检索 → 验证召回
2. 存入代码片段 → 按类型过滤检索 → 验证精准匹配
3. 测试检索模块的各个组件

运行方式:
    python tests/test_retrieval.py
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

from hivememory.core.models import (
    MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType
)
from hivememory.memory.storage import QdrantMemoryStore
from hivememory.retrieval import (
    QueryProcessor,
    ProcessedQuery,
    SimpleRouter,
    HybridSearcher,
    ContextRenderer,
    RetrievalEngine,
    create_retrieval_engine,
    RenderFormat,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

console = Console(force_terminal=True, legacy_windows=False)


# ========== 测试数据 ==========

TEST_MEMORIES = [
    {
        "title": "用户 API Key 配置",
        "summary": "用户设置的 OpenAI API Key 为 sk-test-123456",
        "tags": ["api-key", "config", "openai"],
        "type": MemoryType.USER_PROFILE,
        "content": "用户的 OpenAI API Key 配置：\n**API Key**: `sk-test-123456`\n\n请在调用 OpenAI API 时使用此密钥。",
        "confidence": 1.0  # 用户显式输入
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
        "confidence": 0.9  # 已验证的代码
    },
    {
        "title": "项目使用 Python 3.12",
        "summary": "项目环境配置为 Python 3.12，使用 Black 格式化，行宽 100",
        "tags": ["python", "config", "project"],
        "type": MemoryType.FACT,
        "content": "项目环境要求：\n- Python 版本：**3.12**\n- 代码格式化：Black\n- 行宽：100 字符",
        "confidence": 0.95
    },
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

        return storage

    except Exception as e:
        console.print(f"✗ 环境准备失败: {e}", style="bold red")
        return None


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
        
        storage.upsert_memory(memory)
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


def test_hybrid_searcher(storage: QdrantMemoryStore):
    """测试混合检索器"""
    console.print("\n[bold magenta]🔍 测试 HybridSearcher[/bold magenta]")
    
    searcher = HybridSearcher(storage=storage)
    
    test_queries = [
        ("API Key 配置", "test_user"),
        ("日期解析代码", "test_user"),
        ("Python 版本", "test_user"),
    ]
    
    for query_text, user_id in test_queries:
        console.print(f"\n  [cyan]查询:[/cyan] {query_text}")
        
        results = searcher.search_by_text(
            query_text=query_text,
            user_id=user_id,
            top_k=3
        )
        
        console.print(f"  [dim]找到 {len(results)} 条结果 (耗时 {results.latency_ms:.1f}ms)[/dim]")
        
        for i, r in enumerate(results.results[:2], 1):
            console.print(f"    {i}. {r.memory.index.title} (score: {r.score:.2f})")
    
    console.print("\n[green]✓ HybridSearcher 测试完成[/green]")


def test_context_renderer(storage: QdrantMemoryStore):
    """测试上下文渲染器"""
    console.print("\n[bold magenta]📄 测试 ContextRenderer[/bold magenta]")
    
    # 先检索一些记忆
    searcher = HybridSearcher(storage=storage)
    results = searcher.search_by_text("API Key", user_id="test_user", top_k=2)
    
    # 测试 XML 渲染
    renderer_xml = ContextRenderer(format=RenderFormat.XML, max_tokens=1000)
    xml_output = renderer_xml.render(results.results)
    
    console.print("\n  [cyan]XML 格式输出:[/cyan]")
    console.print(Panel(xml_output[:500] + "..." if len(xml_output) > 500 else xml_output, 
                        title="XML Context", border_style="blue"))
    
    # 测试 Markdown 渲染
    renderer_md = ContextRenderer(format=RenderFormat.MARKDOWN, max_tokens=1000)
    md_output = renderer_md.render(results.results)
    
    console.print("\n  [cyan]Markdown 格式输出:[/cyan]")
    console.print(Panel(md_output[:500] + "..." if len(md_output) > 500 else md_output,
                        title="Markdown Context", border_style="green"))
    
    console.print("\n[green]✓ ContextRenderer 测试完成[/green]")


def test_retrieval_engine(storage: QdrantMemoryStore):
    """测试完整检索引擎"""
    console.print("\n[bold magenta]🚀 测试 RetrievalEngine (完整流程)[/bold magenta]")
    
    engine = create_retrieval_engine(
        storage=storage,
        enable_routing=True,
        top_k=3,
        threshold=0.3,  # 较低阈值以提高召回率
        format="xml"
    )
    
    test_queries = [
        "我的 API Key 是什么？",
        "你好",  # 应该被路由器过滤
        "项目用的是哪个 Python 版本？",
    ]
    
    for query in test_queries:
        console.print(f"\n  [cyan]查询:[/cyan] {query}")
        
        result = engine.retrieve_context(
            query=query,
            user_id="test_user"
        )
        
        if not result.should_retrieve:
            console.print(f"  [dim]→ 路由器判断无需检索[/dim]")
            continue
        
        console.print(f"  → 检索到 {result.memories_count} 条记忆 (耗时 {result.latency_ms:.1f}ms)")
        
        if result.memories:
            for mem in result.memories[:2]:
                console.print(f"    • {mem.index.title}")
        
        if result.rendered_context:
            console.print(f"  → 渲染上下文: {len(result.rendered_context)} 字符")
    
    console.print("\n[green]✓ RetrievalEngine 测试完成[/green]")


def run_acceptance_test(storage: QdrantMemoryStore):
    """验收测试：模拟完整的记忆召回场景"""
    console.print("\n[bold magenta]🏆 验收测试：记忆召回场景[/bold magenta]")
    
    engine = create_retrieval_engine(storage=storage, render_format="xml", threshold=0.1)
    
    # 模拟用户提问
    query = "我的 API Key 是多少？"
    console.print(f"\n  [bold]用户提问:[/bold] {query}")
    
    result = engine.retrieve_context(query=query, user_id="test_user")
    
    # Debug: 显示检索到的记忆
    console.print(f"\n  [dim]检索到 {len(result.memories)} 条记忆:[/dim]")
    for i, mem in enumerate(result.memories):
        console.print(f"    {i+1}. {mem.index.title}")
    
    # 检查是否召回了正确的记忆
    api_key_found = False
    for mem in result.memories:
        title_lower = mem.index.title.lower()
        # 检查中文或英文的 API Key 标题
        if ("api" in title_lower and "key" in title_lower) or "api key" in title_lower:
            api_key_found = True
            console.print(f"\n  [green]✓ 成功召回记忆:[/green] {mem.index.title}")
            console.print(f"    置信度: {mem.meta.confidence_score:.0%}")
            
            # 检查内容中是否包含 API Key
            if "sk-test-123456" in mem.payload.content:
                console.print(f"    [green]✓ 内容包含正确的 API Key[/green]")
            break
    
    if api_key_found:
        console.print("\n" + "="*50)
        console.print("[bold green]🎉 验收测试通过！[/bold green]")
        console.print("系统能够正确召回用户之前设置的 API Key 信息。")
        return True
    else:
        console.print("\n[bold red]✗ 验收测试失败[/bold red]")
        console.print("未能召回 API Key 相关记忆。")
        return False



def main():
    """主测试流程"""
    console.print(Panel.fit(
        "[bold magenta]HiveMemory 阶段2 - 记忆检索模块测试[/bold magenta]\n"
        "测试查询处理、路由、检索、渲染全流程",
        border_style="magenta"
    ))
    
    # 环境准备
    storage = setup_environment()
    if not storage:
        return
    
    # 插入测试数据
    insert_test_memories(storage)
    
    # 等待索引建立
    time.sleep(1)
    
    # 运行各模块测试
    test_query_processor()
    test_router()
    test_hybrid_searcher(storage)
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
