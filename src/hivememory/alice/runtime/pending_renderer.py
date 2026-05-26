"""
PendingAtom 渲染器。

负责将 PendingAtom 格式化为 Agent 可读的文本输出，
用于 READ 响应和 WRITE/UPDATE ACK 回填。

作者: HiveMemory Team
版本: 1.0
"""

from __future__ import annotations

from hivememory.alice.runtime.models import PendingAtom, PendingAtomStatus
from hivememory.core.models import MemoryAtom
from hivememory.engines.generation.models import (
    PendingAtomSettlement,
    UpdateFocus,
    WriteFocus,
)


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
        focus = pending.focus
        if not isinstance(focus, WriteFocus):
            raise TypeError("WRITE pending atom must carry WriteFocus.")

        lines = [f"[{pending.pending_alias}] (runtime pending atom):"]
        lines.append("status: pending")
        lines.append("source: WRITE")
        if focus.title:
            lines.append(f"title: {focus.title}")
        lines.append("")
        lines.append("content:")
        lines.append(focus.content)
        lines.append("")
        lines.append(
            "note: This is a runtime pending atom. "
            "Final memory generation is asynchronous."
        )
        return "\n".join(lines)

    @staticmethod
    def _render_revision_read(pending: PendingAtom) -> str:
        focus = pending.focus
        if not isinstance(focus, UpdateFocus):
            raise TypeError("UPDATE pending atom must carry UpdateFocus.")

        lines = [
            f"[{pending.pending_alias}] "
            f"(pending revision of '{focus.base_alias}'):"
        ]
        lines.append("status: revision")
        if focus.instruction:
            lines.append(f"instruction: {focus.instruction}")
        lines.append("")
        lines.append("new content:")
        lines.append(focus.content or "")
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
            focus = pending.focus
            if not isinstance(focus, UpdateFocus):
                raise TypeError("UPDATE pending atom must carry UpdateFocus.")
            return (
                f"Memory '{focus.base_alias}' update accepted as "
                f"pending revision '{pending.pending_alias}'.\n"
                f"It is readable during this run via READ. "
                f"Final memory update will complete asynchronously."
            )
        return (
            f"Memory accepted as pending atom '{pending.pending_alias}'.\n"
            f"It is readable during this run via READ. "
            f"Final memory generation will complete asynchronously."
        )

    @staticmethod
    def render_redirect_read(
        *,
        requested_alias: str,
        canonical_alias: str,
        atom: MemoryAtom,
        settlement: PendingAtomSettlement | None = None,
    ) -> str:
        """渲染 READ redirect 响应，主体使用 canonical alias。"""
        status = settlement.status.lower() if settlement else "redirected"
        lines = [
            "[Alias Redirected]",
            f"Requested alias: {requested_alias}",
            f"Canonical alias: {canonical_alias}",
            f"Status: {status}",
            "",
            f"[{canonical_alias}]:",
            atom.payload.content,
            "",
            f"Action: Use '{canonical_alias}' for future READ/RUN/UPDATE calls.",
        ]
        return "\n".join(lines)

    @staticmethod
    def render_redirect_run_notice(
        *,
        requested_alias: str,
        canonical_alias: str,
        settlement: PendingAtomSettlement | None = None,
    ) -> str:
        """渲染 RUN redirect 提示头。"""
        status = settlement.status.lower() if settlement else "redirected"
        return "\n".join(
            [
                "[Alias Redirected]",
                f"Requested alias: {requested_alias}",
                f"Canonical alias: {canonical_alias}",
                f"Status: {status}",
                f"Action: Use '{canonical_alias}' for future RUN calls.",
                "",
            ]
        )

    @staticmethod
    def render_settled_without_atom(
        *,
        requested_alias: str,
        settlement: PendingAtomSettlement | None,
    ) -> str:
        """渲染已结算但未物化为 canonical atom 的 pending alias。"""
        status = settlement.status.lower() if settlement else "settled"
        message = settlement.message if settlement and settlement.message else ""
        reason = settlement.reason if settlement and settlement.reason else ""
        lines = [
            f"[{requested_alias}]",
            f"status: {status}",
            "materialized: false",
        ]
        if message:
            lines.append(f"message: {message}")
        if reason:
            lines.append(f"reason: {reason}")
        lines.append("")
        lines.append("Action: Use SEARCH to locate related finalized memory if needed.")
        return "\n".join(lines)


__all__ = ["PendingAtomRenderer"]
