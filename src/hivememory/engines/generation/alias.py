"""
HiveMemory - 记忆别名生成器 (Alias Generator)

职责:
    按 memory type 前缀 + 后缀构造 MTP 别名候选，并经中期库确认同一
    Workspace 内唯一；候选被占用时追加消歧后缀重试。

唯一性分两层保证（A2 §8 D-4，见 docs/archive/todo/memory-alias-uniqueness.md）:
    - 第一层（本组件）: 生成侧查询中期库后给出空闲候选，让抽取草稿的正常
      路径不触碰冲突错误；
    - 第二层（``MidTermMemoryStore.upsert``）: 写入前兜底校验，覆盖手工
      编辑、Profile 管理创建、revive 等不经生成器的路径。

作者: HiveMemory Team
版本: 0.1.0
"""

from __future__ import annotations

import re
from typing import Protocol
from uuid import UUID

from hivememory.core.errors import MemoryAliasConflictError
from hivememory.core.models import WorkspaceIdentity

# MTP 别名系统: MemoryType -> 别名前缀映射 (Section 2.3.1)
MEMORY_TYPE_ALIAS_PREFIX: dict[str, str] = {
    "CODE_SNIPPET": "code",
    "FACT": "fact",
    "URL_RESOURCE": "url",
    "REFLECTION": "ref",
    "USER_PROFILE": "user",
    "WORK_IN_PROGRESS": "wip",
    "AGENT_PROFILE": "agent",
}

# 后缀清洗后的最大长度；加上前缀与消歧后缀仍低于 IndexLayer.alias 的 60 字符上限。
_MAX_SUFFIX_LENGTH = 40


class AliasHolderLookup(Protocol):
    """alias 占用查询端口；``MidTermMemoryStore`` 满足该协议。"""

    async def list_alias_holders(
        self,
        workspace_identity: WorkspaceIdentity,
        alias: str,
        *,
        limit: int = ...,
    ) -> list[UUID]:
        """返回 Workspace 内占用 ``alias`` 的 memory_id。"""
        ...


class AliasGenerator:
    """
    Workspace 内唯一的 MTP 别名生成器。

    候选序列为 ``<base>``、``<base>_2``、``<base>_3``……，逐个查询中期库，
    返回第一个空闲候选；全部被占用时抛 ``MemoryAliasConflictError``，不静默
    生成无别名记忆。查询与写入之间的竞态由第二层写前校验兜底。

    Examples:
        >>> generator = AliasGenerator(mid_term)
        >>> await generator.generate(
        ...     workspace_identity=workspace,
        ...     memory_type="CODE_SNIPPET",
        ...     alias_suffix="quicksort_impl",
        ...     title="Quick Sort",
        ... )
        'code_quicksort_impl'
    """

    def __init__(self, holders: AliasHolderLookup, *, max_attempts: int = 20) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts 必须至少为 1")
        self._holders = holders
        self._max_attempts = max_attempts

    async def generate(
        self,
        *,
        workspace_identity: WorkspaceIdentity,
        memory_type: str,
        alias_suffix: str,
        title: str,
    ) -> str | None:
        """
        生成 Workspace 内唯一的别名。

        Args:
            workspace_identity: 别名唯一性的作用域
            memory_type: 记忆类型字符串 (e.g. "CODE_SNIPPET")
            alias_suffix: LLM 生成的别名后缀 (可能为空)
            title: 记忆标题 (用于 fallback)

        Returns:
            空闲的完整别名；后缀与标题都无法构造候选时返回 None

        Raises:
            MemoryAliasConflictError: 全部消歧候选均已被占用
        """
        base = self.build_candidate(memory_type, alias_suffix, title)
        if base is None:
            return None
        for attempt in range(1, self._max_attempts + 1):
            candidate = base if attempt == 1 else f"{base}_{attempt}"
            holders = await self._holders.list_alias_holders(
                workspace_identity,
                candidate,
                limit=1,
            )
            if not holders:
                return candidate
        raise MemoryAliasConflictError(
            "alias 候选及其消歧后缀均已被同一 Workspace 内的其他记忆占用",
            details={
                "alias": base,
                "attempts": self._max_attempts,
                "reason": "alias_candidates_exhausted",
            },
        )

    @staticmethod
    def build_candidate(
        memory_type: str,
        alias_suffix: str,
        title: str,
    ) -> str | None:
        """
        构建基础 MTP 别名候选 (Section 2.3.1)，不查询唯一性。

        策略:
            1. 从 MEMORY_TYPE_ALIAS_PREFIX 取前缀
            2. 优先使用 LLM 生成的 alias_suffix
            3. alias_suffix 为空时从 title 派生 fallback suffix
            4. 清洗并验证最终别名格式

        Args:
            memory_type: 记忆类型字符串 (e.g. "CODE_SNIPPET")
            alias_suffix: LLM 生成的别名后缀 (可能为空)
            title: 记忆标题 (用于 fallback)

        Returns:
            完整别名 (e.g. "code_quicksort_impl"), 或 None
        """
        prefix = MEMORY_TYPE_ALIAS_PREFIX.get(memory_type, "mem")

        # 优先使用 LLM 生成的 suffix；清洗后为空（如纯中文或纯符号）时同样
        # 回退 title，避免产出 "fact_" 这类退化别名。
        suffix = _clean_suffix(re.sub(r"[^a-z0-9_]", "", (alias_suffix or "").strip().lower()))
        if not suffix:
            derived = re.sub(r"[^a-z0-9\s_]", "", title.lower().strip())
            suffix = _clean_suffix(re.sub(r"\s+", "_", derived))

        if not suffix:
            return None
        return f"{prefix}_{suffix}"


def _clean_suffix(suffix: str) -> str:
    """合并连续下划线、截断到上限，并去除首尾下划线（含截断产生的尾部下划线）。"""
    suffix = re.sub(r"_+", "_", suffix).strip("_")
    return suffix[:_MAX_SUFFIX_LENGTH].strip("_")


__all__ = [
    "MEMORY_TYPE_ALIAS_PREFIX",
    "AliasGenerator",
    "AliasHolderLookup",
]
