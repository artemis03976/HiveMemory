"""
环境连接性测试脚本

测试内容:
1. Qdrant 向量数据库连接
2. Redis 连接
3. BGE-M3 Embedding 模型加载
4. LiteLLM (可选,需要API Key)
5. 基础数据模型验证
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

from rich.console import Console
from rich.table import Table
from rich.panel import Panel


console = Console(force_terminal=True, legacy_windows=False)


def test_qdrant_connection():
    """测试 Qdrant 连接"""
    console.print("\n[bold cyan]1. 测试 Qdrant 连接...[/bold cyan]")

    try:
        from qdrant_client import QdrantClient
        from hivememory.core.config import load_app_config

        config = load_app_config()
        client = QdrantClient(
            host=config.qdrant.host,
            port=config.qdrant.port,
            timeout=10
        )

        # 获取集合列表
        collections = client.get_collections()

        console.print(f"✓ Qdrant 连接成功!")
        console.print(f"  主机: {config.qdrant.host}:{config.qdrant.port}")
        console.print(f"  现有集合数: {len(collections.collections)}")

        return True, "连接成功"

    except Exception as e:
        console.print(f"✗ Qdrant 连接失败: {e}", style="bold red")
        console.print(f"  提示: 请确保已运行 'docker-compose up -d'")
        return False, str(e)


def test_redis_connection():
    """测试 Redis 连接"""
    console.print("\n[bold cyan]2. 测试 Redis 连接...[/bold cyan]")

    try:
        from redis import Redis
        from hivememory.core.config import load_app_config

        config = load_app_config()
        client = redis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            password=config.redis.password,
            db=config.redis.db,
            decode_responses=True,
            socket_connect_timeout=5,
        )

        # 测试 ping
        client.ping()

        # 测试写入
        client.set("hivememory:test", "hello", ex=10)
        value = client.get("hivememory:test")

        console.print(f"✓ Redis 连接成功!")
        console.print(f"  主机: {config.redis.host}:{config.redis.port}")
        console.print(f"  测试写入: OK ({value})")

        return True, "连接成功"

    except Exception as e:
        console.print(f"✗ Redis 连接失败: {e}", style="bold red")
        console.print(f"  提示: 请确保已运行 'docker-compose up -d'")
        return False, str(e)


def test_embedding_model():
    """测试 Embedding 模型加载"""
    console.print("\n[bold cyan]3. 测试 Embedding 模型...[/bold cyan]")

    try:
        from sentence_transformers import SentenceTransformer
        from hivememory.core.config import load_app_config

        config = load_app_config()  
        console.print(f"  加载模型: {config.embedding.model_name}")
        console.print(f"  设备: {config.embedding.device}")

        # 加载模型
        model = SentenceTransformer(
            config.embedding.model_name,
            device=config.embedding.device
        )

        # 测试编码
        test_text = "This is a test sentence for embedding."
        embedding = model.encode(test_text, normalize_embeddings=True)

        console.print(f"✓ Embedding 模型加载成功!")
        console.print(f"  向量维度: {len(embedding)}")
        console.print(f"  测试文本: '{test_text[:30]}...'")

        return True, f"加载成功 (维度: {len(embedding)})"

    except Exception as e:
        console.print(f"✗ Embedding 模型加载失败: {e}", style="bold red")
        console.print(f"  提示: 首次运行会自动下载模型,请耐心等待")
        return False, str(e)


def test_pydantic_models():
    """测试数据模型"""
    console.print("\n[bold cyan]4. 测试数据模型...[/bold cyan]")

    try:
        from hivememory.core.models import (
            MemoryAtom,
            MetaData,
            IndexLayer,
            PayloadLayer,
            MemoryType,
        )

        # 创建测试记忆
        memory = MemoryAtom(
            meta=MetaData(
                source_agent_id="test_agent",
                user_id="test_user",
                confidence_score=0.9,
            ),
            index=IndexLayer(
                title="测试记忆",
                summary="这是一个用于测试的记忆原子",
                tags=["test", "demo"],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(
                content="测试内容: HiveMemory 系统正常运行",
            ),
        )

        # 测试序列化
        json_data = memory.model_dump()

        # 测试渲染
        rendered = memory.render_for_context()

        console.print(f"✓ 数据模型验证成功!")
        console.print(f"  记忆ID: {memory.id}")
        console.print(f"  标题: {memory.index.title}")

        return True, "验证成功"

    except Exception as e:
        console.print(f"✗ 数据模型验证失败: {e}", style="bold red")
        return False, str(e)


def test_litellm_optional():
    """测试 LiteLLM (可选)"""
    console.print("\n[bold cyan]5. 测试 LiteLLM 调用 (可选)...[/bold cyan]")

    try:
        import litellm
        from hivememory.core.config import get_librarian_llm_config

        config = get_librarian_llm_config()

        if not config.api_key or config.api_key == "":
            console.print("⊘ 跳过 LiteLLM 测试 (未配置 API Key)", style="yellow")
            return None, "跳过"

        # 简单测试调用
        response = litellm.completion(
            model=config.model,
            messages=[{"role": "user", "content": "Say 'OK' if you can hear me."}],
            api_key=config.api_key,
            api_base=config.api_base,
            max_tokens=10,
            temperature=0,
        )

        reply = response.choices[0].message.content

        console.print(f"✓ LiteLLM 调用成功!")
        console.print(f"  模型: {config.model}")
        console.print(f"  响应: {reply}")

        return True, "调用成功"

    except Exception as e:
        console.print(f"⊘ LiteLLM 测试失败 (可选): {e}", style="yellow")
        return None, str(e)


def main():
    """主测试流程"""
    console.print(Panel.fit(
        "[bold magenta]HiveMemory 环境连接性测试[/bold magenta]\n"
        "检查核心组件是否正常运行",
        border_style="magenta"
    ))

    # 执行测试
    results = []

    results.append(("Qdrant 数据库", *test_qdrant_connection()))
    results.append(("Redis 缓存", *test_redis_connection()))
    results.append(("Embedding 模型", *test_embedding_model()))
    results.append(("数据模型", *test_pydantic_models()))
    results.append(("LiteLLM (可选)", *test_litellm_optional()))

    # 生成结果表格
    table = Table(title="\n测试结果汇总", show_header=True, header_style="bold cyan")
    table.add_column("组件", style="cyan", width=20)
    table.add_column("状态", width=10)
    table.add_column("详情", width=40)

    for component, status, detail in results:
        if status is True:
            status_str = "[green]✓ 成功[/green]"
        elif status is False:
            status_str = "[red]✗ 失败[/red]"
        else:
            status_str = "[yellow]⊘ 跳过[/yellow]"

        table.add_row(component, status_str, detail)

    console.print(table)

    # 统计结果
    success_count = sum(1 for _, status, _ in results if status is True)
    fail_count = sum(1 for _, status, _ in results if status is False)

    if fail_count == 0:
        console.print("\n[bold green]🎉 所有核心组件测试通过! 系统已就绪。[/bold green]")
    else:
        console.print(f"\n[bold red]⚠️  {fail_count} 个组件测试失败, 请检查配置。[/bold red]")

    console.print("\n[dim]提示: 运行 'docker-compose -f docker/docker-compose.yml up -d' 启动服务[/dim]")


if __name__ == "__main__":
    main()
