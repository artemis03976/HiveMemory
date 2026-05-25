"""
PendingAtom 渲染器。

负责将 PendingAtom 格式化为 Agent 可读的文本输出，
用于 READ 响应和 WRITE/UPDATE ACK 回填。

作者: HiveMemory Team
版本: 1.0
"""

from __future__ import annotations

from hivememory.alice.runtime.models import PendingAtom, PendingAtomStatus


class PendingAtomRenderer:
    """MVP 阶段的 pending atom 渲染器，未来由 MemoryCompiler 接管。"""

    @staticmethod
    def render_read(pending: PendingAtom) -> str:
        """渲染 READ 响应内容。"""
        if pending.status == PendingAtomStatus.REVISION:
            return PendingAtomRenderer._render_revision_read(pending)
        return PendingAtomRenderer._render_draft_read(pending)

    @staticmethod
    def _render_draft_read(pending: PendingAtom) -> str:
        lines = [f"[{pending.pending_alias}] (runtime pending atom):"]
        lines.append("status: pending")
        lines.append("source: WRITE")
        if pending.title:
            lines.append(f"title: {pending.title}")
        lines.append("")
        lines.append("content:")
        lines.append(pending.content)
        lines.append("")
        lines.append(
            "note: This is a runtime pending atom. "
            "Final memory generation is asynchronous."
        )
        return "\n".join(lines)

    @staticmethod
    def _render_revision_read(pending: PendingAtom) -> str:
        lines = [
            f"[{pending.pending_alias}] "
            f"(pending revision of '{pending.target_alias}'):"
        ]
        lines.append("status: revision")
        if pending.instruction:
            lines.append(f"instruction: {pending.instruction}")
        lines.append("")
        lines.append("new content:")
        lines.append(pending.content)
        lines.append("")
        lines.append(
            "note: This is a pending revision. "
            "The original memory has not been modified yet."
        )
        return "\n".join(lines)

    @staticmethod
    def render_ack(pending: PendingAtom) -> str:
        """渲染 WRITE/UPDATE ACK 回填文案。"""
        if pending.status == PendingAtomStatus.REVISION:
            return (
                f"Memory '{pending.target_alias}' update accepted as "
                f"pending revision '{pending.pending_alias}'.\n"
                f"It is readable during this run via READ. "
                f"Final memory update will complete asynchronously."
            )
        return (
            f"Memory accepted as pending atom '{pending.pending_alias}'.\n"
            f"It is readable during this run via READ. "
            f"Final memory generation will complete asynchronously."
        )


__all__ = ["PendingAtomRenderer"]
