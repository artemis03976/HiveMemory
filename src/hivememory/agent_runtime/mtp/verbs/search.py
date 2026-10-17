"""SEARCH 指令的语义处理。"""

from __future__ import annotations

from hivememory.agent_runtime.mtp.context import MTPExecutionContext
from hivememory.config.memory_compiler import MemoryCompilerConfig
from hivememory.core.mtp import (
    MTPCommand,
    MTPFilterParser,
    MTPResponse,
    MTPResponseStatus,
    MTPWarningInfo,
)
from hivememory.core.mtp.exceptions import (
    InvalidArgumentError,
)
from hivememory.engines.memory_compiler import (
    MemoryCompileOptions,
    MemoryCompiler,
    MemoryEnvelopeTarget,
)
from hivememory.workspace.contracts import (
    RetrieveRequest,
)


async def handle_search(
    command: MTPCommand,
    context: MTPExecutionContext,
    *,
    filter_parser: MTPFilterParser,
    compiler: MemoryCompiler,
    compiler_config: MemoryCompilerConfig | None,
) -> MTPResponse:
    """
    处理 SEARCH 指令

    提交检索请求，经 workspace 授权与缓存预热后，
    再通过 MemoryCompiler 编译为 Agent 可读文本。

    Type A 数据类响应

    Args:
        command: SEARCH 指令 (target=*, args: query="...", filter="...")

    Returns:
        MTPResponse: 检索到的完整原子经 MemoryCompiler 编译后的上下文
    """
    query = command.args.get("query", "")
    if not query:
        raise InvalidArgumentError(message_key="mtp.search.missing_query")

    # 解析 filter 参数
    # 例如 filter="type:CODE" → QueryFilters(memory_type=CODE_SNIPPET)
    # 宽容解析: 非法 filter 降级为 None，但返回警告
    filter_str = command.args.get("filter", "")
    parsed_filters, filter_warnings = filter_parser.parse(filter_str) if filter_str else (None, [])

    # 让 StorageOfflineError / StorageReadError 继续向上传播到 _route_and_execute 统一处理；
    # 能力层返回完整原子列表，filter 翻译与 MTP 输出编译留在协议边界。
    memories = await context.submit_operation(
        RetrieveRequest(
            semantic_query=query,
            filters=parsed_filters,
        ),
    )

    if not memories:
        return MTPResponse(
            status=MTPResponseStatus.SUCCESS,
            content="",
            warnings=[
                MTPWarningInfo(message_key="mtp.search.no_memories_found"),
                *filter_warnings,
            ],
        )

    content = compiler.compile(
        memories,
        MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
        MemoryCompileOptions(
            retrieval_strategy_config=(
                compiler_config.retrieval_context.strategy if compiler_config is not None else None
            ),
        ),
    ).text
    response_warnings = list(filter_warnings)

    return MTPResponse(
        status=MTPResponseStatus.SUCCESS,
        content=content,
        warnings=response_warnings,
    )
