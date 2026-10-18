"""READ 指令的语义处理。"""

from __future__ import annotations

from typing import Any

from hivememory.agent_runtime.mtp.context import MTPExecutionContext
from hivememory.core.models import MemoryAtom
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

    resolved: list[tuple[str, MemoryAtom]] = []  # (alias, atom)
    resolved_pending: list[tuple[str, Any]] = []  # (alias, PendingAtom)
    resolved_redirects: list[tuple[str, Any]] = []  # (alias, ReferenceResolution)
    resolved_terminal: list[tuple[str, Any]] = []  # (alias, ReferenceResolution)
    unresolved: list[str] = []
    results = await context.submit_operation(ResolveReferencesRequest(aliases=tuple(aliases)))
    for alias, result in zip(aliases, results, strict=True):
        if result.kind == "pending" and result.pending is not None:
            resolved_pending.append((alias, result.pending))
        elif result.kind == "redirect" and result.atom is not None:
            resolved_redirects.append((alias, result))
        elif result.kind == "atom" and result.atom is not None:
            resolved.append((alias, result.atom))
        elif result.kind in {"discarded", "failed", "expired"}:
            resolved_terminal.append((alias, result))
        else:
            unresolved.append(alias)

    # 全部无效：直接返回错误
    if not resolved and not resolved_pending and not resolved_redirects and not resolved_terminal:
        raise AliasNotFoundError(
            message_key="mtp.read.alias_not_found",
            params={"aliases": "\n".join(f"  {a}" for a in unresolved)},
        )

    # 组装输出 — 通过 MemoryCompiler 统一编译，局部未解析 alias 交给 warnings 渲染
    output_lines: list[str] = []
    warnings: list[MTPWarningInfo] = []
    for alias, pending in resolved_pending:
        artifact = compiler.compile(
            pending,
            MemoryCompileTarget.MTP_READ,
        )
        output_lines.append(artifact.text)
    for alias, result in resolved_redirects:
        artifact = compiler.compile(
            result,
            MemoryCompileTarget.MTP_READ,
            MemoryCompileOptions(requested_alias=alias),
        )
        output_lines.append(artifact.text)
    for alias, result in resolved_terminal:
        artifact = compiler.compile(
            result,
            MemoryCompileTarget.MTP_READ,
            MemoryCompileOptions(requested_alias=alias),
        )
        output_lines.append(artifact.text)
    for alias, atom in resolved:
        artifact = compiler.compile(
            atom,
            MemoryCompileTarget.MTP_READ,
            MemoryCompileOptions(requested_alias=alias),
        )
        output_lines.append(artifact.text)
    for alias in unresolved:
        warnings.append(
            MTPWarningInfo(
                message_key="mtp.read.partial_alias_not_found",
                params={"alias": alias},
            )
        )

    # 正式原子的引用记录已由能力层在交付前完成，协议层只负责输出。
    return MTPResponse(
        status=MTPResponseStatus.SUCCESS,
        content="\n".join(output_lines),
        warnings=warnings,
    )
