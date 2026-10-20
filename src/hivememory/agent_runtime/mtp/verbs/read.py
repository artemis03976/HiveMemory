"""READ 指令的语义处理。"""

from __future__ import annotations

from hivememory.agent_runtime.mtp.context import MTPExecutionContext
from hivememory.core.models import MemoryAtom, PendingAtom
from hivememory.core.models.reference import ReferenceResolution
from hivememory.core.mtp import (
    MTPCommand,
    MTPResponse,
    MTPResponseStatus,
    MTPWarningInfo,
)
from hivememory.core.mtp.exceptions import (
    AliasNotFoundError,
    InvalidArgumentError,
)
from hivememory.engines.memory_compiler import (
    MemoryCompileOptions,
    MemoryCompiler,
    MemoryCompileTarget,
)
from hivememory.workspace.contracts import (
    ResolveReferencesRequest,
)


async def handle_read(
    command: MTPCommand,
    context: MTPExecutionContext,
    *,
    compiler: MemoryCompiler,
) -> MTPResponse:
    """
    处理 READ 指令

    获取记忆原子的 Payload 内容。支持列表并行读取。

    Type A 数据类响应

    Args:
        command: READ 指令 (target=alias 或 [alias1, alias2])

    Returns:
        MTPResponse: 记忆内容
    """
    if command.target.is_wildcard:
        raise InvalidArgumentError(message_key="mtp.read.wildcard_not_supported")

    aliases = command.target.aliases
    if not aliases:
        raise InvalidArgumentError(message_key="mtp.read.missing_alias")

    results = await context.submit_operation(ResolveReferencesRequest(aliases=tuple(aliases)))
    output_lines: list[str] = []
    unresolved: list[str] = []
    # 按请求顺序逐项编译，输出与 agent 列出的 alias 顺序一致。
    for alias, result in zip(aliases, results, strict=True):
        source = _readable(result)
        if source is None:
            unresolved.append(alias)
            continue
        # pending 句柄自带 alias；其余结果按请求的 alias 呈现，redirect 才能标出原名。
        options = (
            None if isinstance(source, PendingAtom) else MemoryCompileOptions(requested_alias=alias)
        )
        output_lines.append(compiler.compile(source, MemoryCompileTarget.MTP_READ, options).text)

    # 全部无效：直接返回错误
    if not output_lines:
        raise AliasNotFoundError(
            message_key="mtp.read.alias_not_found",
            params={"aliases": "\n".join(f"  {a}" for a in unresolved)},
        )

    # 局部未解析的 alias 交给 warnings 渲染
    warnings = [
        MTPWarningInfo(message_key="mtp.read.partial_alias_not_found", params={"alias": alias})
        for alias in unresolved
    ]

    # 正式原子的引用记录已由能力层在交付前完成，协议层只负责输出。
    return MTPResponse(
        status=MTPResponseStatus.SUCCESS,
        content="\n".join(output_lines),
        warnings=warnings,
    )


def _readable(result: ReferenceResolution) -> PendingAtom | ReferenceResolution | MemoryAtom | None:
    """READ 交付给编译器的内容；无可交付内容时返回 ``None``，按未解析提示。

    pending 交付意图句柄；带可读原子的 redirect 与 discarded/failed/expired
    终态交付解析结果本身；正式原子交付原子；其余（not_found、不可读的
    redirect、缺失载荷的 pending）都按未解析处理。
    """
    if result.kind == "pending":
        return result.pending
    if result.kind == "redirect":
        return result if result.atom is not None else None
    if result.kind == "atom":
        return result.atom
    if result.kind in {"discarded", "failed", "expired"}:
        return result
    return None
