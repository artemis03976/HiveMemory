from __future__ import annotations

from dataclasses import dataclass

from hivememory.gateway.commands.models import (
    CommandDefinition,
    CommandParseResult,
    CommandParseStatus,
)
from hivememory.gateway.commands.parser import (
    CommandParseError,
    build_parse_result,
    tokenize_command,
)


@dataclass(frozen=True)
class _Candidate:
    """一次 registry 匹配中的候选别名。"""

    definition: CommandDefinition
    alias: str
    alias_tokens: tuple[str, ...]


class CommandRegistry:
    """
    全局系统指令注册表。

    Registry 只负责注册、只读 listing 和 L1 解析，不执行任何指令副作用。
    """

    def __init__(self) -> None:
        self._definitions: dict[str, CommandDefinition] = {}
        self._aliases: dict[str, str] = {}

    def register(self, definition: CommandDefinition) -> None:
        """注册单条指令，并在启动期拒绝 command_id / alias 冲突。"""

        command_id = definition.command_id
        if command_id in self._definitions:
            raise ValueError(f"Duplicate command id: {command_id}")

        names = [definition.primary_name, *definition.aliases]
        normalized_names = [_normalize_name(name) for name in names]
        if len(set(normalized_names)) != len(normalized_names):
            raise ValueError(f"Duplicate alias in command definition: {command_id}")

        for name in names:
            normalized = _normalize_name(name)
            if normalized in self._aliases:
                existing = self._aliases[normalized]
                raise ValueError(f"Duplicate command alias: {name} ({existing})")

        self._definitions[command_id] = definition
        for name in names:
            self._aliases[_normalize_name(name)] = command_id

    def get(self, command_id: str) -> CommandDefinition | None:
        return self._definitions.get(command_id)

    def list(self, include_hidden: bool = False) -> list[CommandDefinition]:
        """返回按 priority 和名称排序的只读指令列表。"""

        definitions = [
            definition
            for definition in self._definitions.values()
            if include_hidden or not definition.hidden
        ]
        return sorted(
            definitions,
            key=lambda item: (item.priority, item.primary_name.casefold()),
        )

    def match(self, raw_input: str) -> CommandParseResult | None:
        """
        匹配用户输入。

        非 slash 输入返回 None；slash 输入即使未知也返回 UNKNOWN，防止交给 L2 猜测执行。
        """

        stripped = raw_input.strip()
        if not stripped:
            return None

        try:
            tokens = tokenize_command(stripped)
        except CommandParseError as exc:
            return CommandParseResult(
                raw_input=stripped,
                parse_status=CommandParseStatus.INVALID_ARGS,
                error=str(exc),
            )

        if not tokens or not tokens[0].startswith("/"):
            return None

        candidates = self._matching_candidates(tokens)
        if not candidates:
            return CommandParseResult(
                raw_input=stripped,
                name=tokens[0],
                tokens=tokens,
                parse_status=CommandParseStatus.UNKNOWN,
                error=f"Unknown system command: {tokens[0]}",
            )

        best = candidates[0]
        tied = [
            candidate
            for candidate in candidates
            if (
                len(candidate.alias_tokens) == len(best.alias_tokens)
                and candidate.definition.priority == best.definition.priority
            )
        ]
        if len(tied) > 1:
            return CommandParseResult(
                raw_input=stripped,
                name=" ".join(tokens[: len(best.alias_tokens)]),
                tokens=tokens,
                parse_status=CommandParseStatus.AMBIGUOUS,
                error="Ambiguous system command",
            )

        arg_tokens = tokens[len(best.alias_tokens) :]
        return build_parse_result(
            definition=best.definition,
            raw_input=stripped,
            tokens=tokens,
            matched_alias=best.alias,
            arg_tokens=arg_tokens,
        )

    def _matching_candidates(self, tokens: list[str]) -> list[_Candidate]:
        """查找所有命中候选，长命令优先，其次按 priority 排序。"""

        normalized_tokens = tuple(token.casefold() for token in tokens)
        candidates: list[_Candidate] = []

        for definition in self._definitions.values():
            if not definition.enabled:
                continue
            for alias in [definition.primary_name, *definition.aliases]:
                alias_tokens = tuple(part.casefold() for part in alias.split())
                if normalized_tokens[: len(alias_tokens)] == alias_tokens:
                    candidates.append(
                        _Candidate(
                            definition=definition,
                            alias=alias,
                            alias_tokens=alias_tokens,
                        )
                    )

        return sorted(
            candidates,
            key=lambda item: (
                -len(item.alias_tokens),
                item.definition.priority,
                item.alias.casefold(),
            ),
        )


def _normalize_name(name: str) -> str:
    normalized = " ".join(name.strip().split()).casefold()
    if not normalized.startswith("/"):
        raise ValueError(f"Command name must start with '/': {name}")
    return normalized


__all__ = ["CommandRegistry"]
