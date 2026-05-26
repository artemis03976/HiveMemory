"""MemoryCompiler — 记忆到文本编译的统一入口。"""

from __future__ import annotations

from typing import List, Union

from hivememory.core.models import MemoryAtom
from hivememory.engines.memory_compiler.models import (
    CompiledMemoryArtifact,
    MemoryCompileOptions,
    MemoryCompileTarget,
)
from hivememory.engines.memory_compiler.targets import (
    compile_memory_atom,
    compile_pending_atom,
    compile_resolve_result,
)


class MemoryCompiler:
    """
    记忆编译外观层。

    Phase 1: 委托给现有渲染器，收敛所有入口。
    Phase 2+: 内化渲染逻辑，引入 IR 和缓存。
    """

    def compile(
        self,
        source: Union[
            MemoryAtom,
            "PendingAtom",
            "ResolveResult",
            "PendingAtomSettlement",
            List[MemoryAtom],
            List["ResolveResult"],
        ],
        target: MemoryCompileTarget,
        options: MemoryCompileOptions | None = None,
    ) -> CompiledMemoryArtifact | List[CompiledMemoryArtifact]:
        opts = options or MemoryCompileOptions()

        if isinstance(source, list):
            return [self._compile_single(item, target, opts) for item in source]

        return self._compile_single(source, target, opts)

    def _compile_single(
        self,
        source,
        target: MemoryCompileTarget,
        options: MemoryCompileOptions,
    ) -> CompiledMemoryArtifact:
        from hivememory.alice.runtime.models import PendingAtom
        from hivememory.alice.runtime.resolver import ResolveResult
        from hivememory.engines.generation.models import PendingAtomSettlement

        if isinstance(source, MemoryAtom):
            return compile_memory_atom(source, target, options)

        if isinstance(source, PendingAtom):
            return compile_pending_atom(source, target, options)

        if isinstance(source, ResolveResult):
            return compile_resolve_result(source, target, options)

        if isinstance(source, PendingAtomSettlement):
            kind = source.status.lower() if source.status in {"DISCARDED", "FAILED"} else "redirect"
            resolve = ResolveResult(
                kind=kind,
                requested_alias=source.pending_alias,
                canonical_alias=source.canonical_alias,
                settlement=source,
            )
            return compile_resolve_result(resolve, target, options)

        raise TypeError(
            f"Unsupported source type: {type(source).__name__}. "
            f"Expected MemoryAtom, PendingAtom, ResolveResult, or PendingAtomSettlement."
        )
