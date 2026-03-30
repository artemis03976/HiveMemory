import asyncio
import os
import sys
from uuid import uuid4
from datetime import datetime, timedelta
from rich.console import Console
from rich.panel import Panel

# 设置控制台输出编码为 utf-8 以解决 Windows GBK 报错问题
if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding='utf-8')

from hivememory.core.models import MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType
from hivememory.engines.retrieval.renderer import (
    FullContextRenderer,
    CascadeContextRenderer,
    CompactContextRenderer
)
from hivememory.patchouli.config import (
    FullRendererConfig,
    CascadeRendererConfig,
    CompactRendererConfig
)

console = Console()

def create_mock_memories():
    now = datetime.utcnow()
    
    # 1. 核心事实 (高置信度，高生命力)
    fact_memory = MemoryAtom(
        id=uuid4(),
        meta=MetaData(
            source_agent_id="test_agent",
            user_id="test_user",
            confidence_score=0.95,
            vitality_score=85.0,
            created_at=now - timedelta(days=2),
            updated_at=now - timedelta(days=1),
            access_count=15
        ),
        index=IndexLayer(
            title="数据库连接配置",
            summary="当前测试环境使用的是本地的 PostgreSQL 数据库，端口为 5432。",
            tags=["database", "config", "postgresql"],
            memory_type=MemoryType.FACT,
            alias="db_config_test"
        ),
        payload=PayloadLayer(
            content="""
DATABASE_URL=postgresql://user:password@localhost:5432/hivememory_test
MAX_CONNECTIONS=20
TIMEOUT=30s
            """.strip()
        )
    )

    # 2. 代码片段 (中等置信度)
    code_memory = MemoryAtom(
        id=uuid4(),
        meta=MetaData(
            source_agent_id="test_agent",
            user_id="test_user",
            confidence_score=0.8,
            vitality_score=60.0,
            created_at=now - timedelta(days=5),
            updated_at=now - timedelta(days=5),
            access_count=3
        ),
        index=IndexLayer(
            title="FastAPI 路由初始化",
            summary="在 main.py 中注册 chat 和 memories 路由的示例代码。",
            tags=["fastapi", "router", "python"],
            memory_type=MemoryType.CODE_SNIPPET,
            alias="fastapi_routes"
        ),
        payload=PayloadLayer(
            content="""
from fastapi import FastAPI
from routers import chat, memories

app = FastAPI(title="HiveMemory API")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(memories.router, prefix="/api/v1")
            """.strip()
        )
    )

    # 3. 待办事项 (陈旧记忆，用于测试降级/截断)
    wip_memory = MemoryAtom(
        id=uuid4(),
        meta=MetaData(
            source_agent_id="test_agent",
            user_id="test_user",
            confidence_score=0.6,
            vitality_score=20.0,
            created_at=now - timedelta(days=100),  # 大于默认的 90 天 stale_days
            updated_at=now - timedelta(days=100),
            access_count=1
        ),
        index=IndexLayer(
            title="前端渲染优化计划",
            summary="计划在下周重构 React 组件，引入懒加载机制以优化长列表的渲染性能。",
            tags=["frontend", "react", "performance", "todo"],
            memory_type=MemoryType.WORK_IN_PROGRESS,
            alias="frontend_todo_1"
        ),
        payload=PayloadLayer(
            content="1. 梳理当前的长列表组件。\n2. 引入 react-window 库。\n3. 测试不同设备下的滚动帧率。\n4. 补充相关的单元测试用例。\n(此处省略几百字的详细测试计划...)"
        )
    )

    return [fact_memory, code_memory, wip_memory]

async def main():
    memories = create_mock_memories()
    
    console.print("\n[bold cyan]=== HiveMemory Context Renderer 渲染效果测试 ===[/bold cyan]\n")
    console.print(f"参与渲染的记忆数量: [yellow]{len(memories)}[/yellow] 条\n")

    # 1. FullContextRenderer 测试
    full_config = FullRendererConfig(
        max_tokens=2000,
        max_content_length=500,
        show_artifacts=False
    )
    full_renderer = FullContextRenderer(full_config)
    full_result = full_renderer.render(memories)
    
    console.print(Panel(
        full_result, 
        title="[bold green]1. FullContextRenderer (完整渲染)[/bold green]", 
        border_style="green",
        expand=False
    ))

    # 2. CascadeContextRenderer 测试
    # 设置 full_payload_count=1，期望第1条完整，后2条降级为摘要
    cascade_config = CascadeRendererConfig(
        max_memory_tokens=2000,
        full_payload_count=1,
        max_content_length=500,
        index_max_summary_length=100
    )
    cascade_renderer = CascadeContextRenderer(cascade_config)
    cascade_result = cascade_renderer.render(memories)
    
    console.print(Panel(
        cascade_result, 
        title="[bold blue]2. CascadeContextRenderer (瀑布式: Top1完整, 其余摘要)[/bold blue]", 
        border_style="blue",
        expand=False
    ))

    # 3. CompactContextRenderer 测试
    compact_config = CompactRendererConfig(
        max_memory_tokens=2000,
        index_max_summary_length=100
    )
    compact_renderer = CompactContextRenderer(compact_config)
    compact_result = compact_renderer.render(memories)
    
    console.print(Panel(
        compact_result, 
        title="[bold magenta]3. CompactContextRenderer (紧凑式: 仅显示摘要和标签)[/bold magenta]", 
        border_style="magenta",
        expand=False
    ))

if __name__ == "__main__":
    asyncio.run(main())
