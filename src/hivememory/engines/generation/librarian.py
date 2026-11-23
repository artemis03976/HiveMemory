"""
HiveMemory - 馆长 (Librarian)

话题结算的另一种生成方式。一次性生成看不到本话题已经写入的记忆，也看不到库，只能先尽量多地
提取，再靠查重与合并的规则补救；馆长改为先探索再写：模型每一步输出一个 JSON 动作，调用
Patchouli 定义的工具，先看本话题主动写入过的记忆，再检索并阅读库中相关的记忆，然后只写两者
都没有覆盖的内容。

工具只读写 Patchouli 自己的数据：读取以结算的发起者（Workspace 的 system actor）直接访问
中期库，只看得到 PUBLIC 记忆，也不经过检索服务，因此不刷新生命力。写入只暂存：新建的草稿与对
读过的记忆的修订（编辑集）在循环结束后交回引擎，草稿按新建提交，不查重、不触碰、不提名合并，
修订成为以读到的版本为底的 UPDATE。残余的重叠接受。

修订只改本次运行读过的记忆，一条记忆一次结算只修订一次；编辑集在循环中以读到的版本为底应用，
对不上时作为工具错误交回模型，由它在剩余的步数内改正。
"""

import json
import logging
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError

from hivememory.config.patchouli import LibrarianConfig, MemoryGenerationConfig
from hivememory.core.models import ActorIdentity, MemoryAtom, WorkspaceIdentity
from hivememory.engines.generation.edits import (
    EditSetError,
    apply_edit_set,
    describe_validation_error,
)
from hivememory.engines.generation.models import ExtractedMemoryDraft, MemoryEditSet
from hivememory.i18n import get_generation_prompt_text, resolve_language
from hivememory.infrastructure.llm.base import (
    BaseLLMService,
    LLMRequestParams,
    complete_with_provider_retry,
)
from hivememory.utils.json_parser import JSONParseError, parse_llm_json

if TYPE_CHECKING:
    from hivememory.memory_library.stores import MidTermMemoryStore

logger = logging.getLogger(__name__)

# 提示词中列出的工具；日志只按这些名字计数，模型编造的名字记为 unknown。
_TOOLS = (
    "list_topic_writes",
    "search_memories",
    "read_memory",
    "write_memory",
    "revise_memory",
    "finish",
)


class _Action(BaseModel):
    """模型每一步输出的动作。"""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class _SearchArgs(BaseModel):
    query: str = Field(min_length=1)
    mode: Literal["dense", "sparse"] = "dense"


class _ReadArgs(BaseModel):
    alias: str = Field(min_length=1)


class _ReviseArgs(MemoryEditSet):
    alias: str = Field(min_length=1)


@dataclass(frozen=True)
class StagedRevision:
    """馆长暂存的一条修订：本次运行读到的那一版、以它为底的编辑集与应用之后的记忆。"""

    base: MemoryAtom
    edit_set: MemoryEditSet
    revised: MemoryAtom


@dataclass(frozen=True)
class LibrarianResult:
    """一次结算暂存的内容：新建的草稿与对已有记忆的修订，各按暂存的顺序。"""

    drafts: list[ExtractedMemoryDraft]
    revisions: list[StagedRevision]


@dataclass
class _Run:
    """一次结算的工具上下文：读取的归属与发起者、本话题的写入、读过的记忆与已暂存的内容。"""

    belong_to: WorkspaceIdentity
    from_actor: ActorIdentity
    topic_write_memory_ids: list[str]
    staged: list[ExtractedMemoryDraft] = field(default_factory=list)
    # read_memory 读到的版本，按 alias；同一条读了几次时取最后一次。
    read: dict[str, MemoryAtom] = field(default_factory=dict)
    revisions: list[StagedRevision] = field(default_factory=list)


class Librarian:
    """话题结算的馆长：以 JSON 动作调用工具的循环，返回暂存的草稿与修订。

    失败语义:
        动作无法解析、工具不存在、参数不合格或修订不能落地时，把错误作为工具结果交回模型
        并计一步，不中止循环；步数用完时照常返回已暂存的内容。提供商失败在同一步内重试一次、
        不另计步，与提取器的模型调用相同，一次瞬时失败不让整次结算失败。重试耗尽或上下文超窗
        时原样传播，生成任务以对应的原因码失败；取消原样传播。
    """

    def __init__(
        self,
        *,
        mid_term: "MidTermMemoryStore",
        llm_service: BaseLLMService,
        llm_params: LLMRequestParams,
        config: LibrarianConfig,
    ) -> None:
        self._mid_term = mid_term
        self._llm_service = llm_service
        # 每一步只输出一个 JSON 对象；提供商不支持 JSON 输出时退回普通补全，照样按文本解析。
        self._llm_params = llm_params.model_copy(update={"json_output": True})
        self._config = config
        language = resolve_language()
        self._system_prompt = get_generation_prompt_text(
            "librarian", "system_prompt", language
        ).format(max_steps=config.max_steps, search_top_k=config.search_top_k)
        self._user_prompt = get_generation_prompt_text("librarian", "user_prompt", language)
        self._error_unparsable = get_generation_prompt_text(
            "librarian", "error_unparsable", language
        )
        self._error_unknown_tool = get_generation_prompt_text(
            "librarian", "error_unknown_tool", language
        )
        self._error_invalid_args = get_generation_prompt_text(
            "librarian", "error_invalid_args", language
        )
        self._error_memory_not_found = get_generation_prompt_text(
            "librarian", "error_memory_not_found", language
        )
        self._error_revise_unread = get_generation_prompt_text(
            "librarian", "error_revise_unread", language
        )
        self._error_revise_repeated = get_generation_prompt_text(
            "librarian", "error_revise_repeated", language
        )
        self._error_revise_invalid = get_generation_prompt_text(
            "librarian", "error_revise_invalid", language
        )

    async def settle(
        self,
        transcript: str,
        *,
        topic_write_memory_ids: Collection[str],
        belong_to: WorkspaceIdentity,
        from_actor: ActorIdentity,
    ) -> LibrarianResult:
        """对一段话题材料运行馆长循环，返回暂存的草稿与修订。

        Args:
            transcript: 带段标签的话题材料，放在第一条消息中
            topic_write_memory_ids: 本话题主动写入结算到的记忆 ID，由 list_topic_writes 列出
            belong_to: 读取的 Workspace 归属
            from_actor: 读取的发起者；结算时是 Workspace 的 system actor

        Raises:
            ModelProviderError: 提供商失败且重试耗尽，或上下文超窗
                （ModelContextWindowExceededError）
        """
        run = _Run(
            belong_to=belong_to,
            from_actor=from_actor,
            topic_write_memory_ids=sorted(topic_write_memory_ids),
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": self._user_prompt.format(transcript=transcript)},
        ]
        tool_calls: Counter[str] = Counter()
        revise_rejected = 0
        total_tokens = prompt_tokens = cached_prompt_tokens = 0
        steps = 0
        finished = False
        while steps < self._config.max_steps and not finished:
            steps += 1
            # 每一步都重发整段对话；消息只往后追加、前缀不变，所以第二步起的输入大部分
            # 可以命中提供商的上下文缓存，日志分开记下命中的部分。
            completion = await complete_with_provider_retry(
                self._llm_service, messages, self._llm_params
            )
            if completion.usage is not None:
                total_tokens += completion.usage.total_tokens
                prompt_tokens += completion.usage.prompt_tokens
                cached_prompt_tokens += completion.usage.cached_prompt_tokens
            messages.append({"role": "assistant", "content": completion.text})
            action = _parse_action(completion.text)
            if action is None:
                tool_calls["unparsable"] += 1
                reply: dict[str, Any] = {"error": self._error_unparsable}
            else:
                tool_calls[action.tool if action.tool in _TOOLS else "unknown"] += 1
                finished = action.tool == "finish"
                reply = {"tool": action.tool, **await self._run_tool(run, action)}
                if action.tool == "revise_memory" and "error" in reply:
                    revise_rejected += 1
            reply["steps_left"] = self._config.max_steps - steps
            messages.append({"role": "user", "content": json.dumps(reply, ensure_ascii=False)})

        # 步数用完是预算截断：模型没有说写完，已暂存的照常提交。
        logger.log(
            logging.INFO if finished else logging.WARNING,
            "馆长结算结束: end=%s, steps=%d, staged=%d, total_tokens=%d, prompt_tokens=%d, "
            "cached_prompt_tokens=%d, tool_calls=%s, revised=%d, revise_rejected=%d",
            "finish" if finished else "max_steps",
            steps,
            len(run.staged),
            total_tokens,
            prompt_tokens,
            cached_prompt_tokens,
            " ".join(f"{name}={count}" for name, count in sorted(tool_calls.items())),
            len(run.revisions),
            revise_rejected,
        )
        return LibrarianResult(drafts=run.staged, revisions=run.revisions)

    async def _run_tool(self, run: _Run, action: _Action) -> dict[str, Any]:
        """执行一个工具，返回交回模型的结果；参数不合格或读不到记忆时返回错误。"""
        try:
            match action.tool:
                case "list_topic_writes":
                    return {"result": await self._list_topic_writes(run)}
                case "search_memories":
                    args = _SearchArgs.model_validate(action.args)
                    hits = await self._mid_term.search(
                        run.belong_to,
                        query=args.query,
                        top_k=self._config.search_top_k,
                        mode=args.mode,
                        from_actor=run.from_actor,
                    )
                    return {
                        "result": [
                            {
                                **_index_view(hit["memory"]),
                                "memory_type": hit["memory"].index.memory_type.value,
                                "score": round(hit["score"], 3),
                            }
                            for hit in hits
                        ]
                    }
                case "read_memory":
                    alias = _ReadArgs.model_validate(action.args).alias
                    memory = await self._mid_term.get_by_alias(
                        run.belong_to, alias, from_actor=run.from_actor
                    )
                    if memory is None:
                        return {"error": self._error_memory_not_found}
                    run.read[alias] = memory
                    return {"result": _full_view(memory)}
                case "write_memory":
                    # 写入只暂存：是否值得写由馆长在写之前判断，草稿一律视为有价值。
                    run.staged.append(
                        ExtractedMemoryDraft.model_validate({**action.args, "has_value": True})
                    )
                    return {"result": {"staged": len(run.staged)}}
                case "revise_memory":
                    return self._revise_memory(run, _ReviseArgs.model_validate(action.args))
                case "finish":
                    return {"result": {"staged": len(run.staged)}}
                case _:
                    return {"error": self._error_unknown_tool}
        except ValidationError as error:
            return {
                "error": self._error_invalid_args.format(details=describe_validation_error(error))
            }

    def _revise_memory(self, run: _Run, args: _ReviseArgs) -> dict[str, Any]:
        """校验并暂存一条修订；不满足任何一条时返回错误、不暂存。

        只改本次运行读过的记忆：材料中有不可信内容，而馆长能改全库，读过才能给出依据。
        一条记忆一次结算只修订一次，几处修改放进同一个编辑集。编辑集以读到的版本为底应用；
        提交前基础的版本变了时由提交一侧重放。
        """
        base = run.read.get(args.alias)
        if base is None:
            return {"error": self._error_revise_unread}
        if any(revision.base.id == base.id for revision in run.revisions):
            return {"error": self._error_revise_repeated}
        edit_set = MemoryEditSet.model_validate(args.model_dump(exclude={"alias"}))
        try:
            revised = apply_edit_set(base, edit_set)
        except EditSetError as error:
            return {"error": self._error_revise_invalid.format(details=error)}
        run.revisions.append(StagedRevision(base=base, edit_set=edit_set, revised=revised))
        return {"result": {"revised": args.alias}}

    async def _list_topic_writes(self, run: _Run) -> list[dict[str, Any]]:
        """本话题主动写入结算到的记忆；发起者读不到的（PRIVATE、TEAM 或已删除）跳过。"""
        views = []
        for memory_id in run.topic_write_memory_ids:
            memory = await self._mid_term.get(
                run.belong_to, UUID(memory_id), from_actor=run.from_actor
            )
            if memory is not None:
                views.append(_index_view(memory))
        return views


def _parse_action(text: str) -> _Action | None:
    """把模型输出解析为一个动作；不是 JSON 对象或结构不对时返回 None。"""
    try:
        return parse_llm_json(text, as_model=_Action)
    except JSONParseError:
        return None


def _index_view(memory: MemoryAtom) -> dict[str, Any]:
    return {
        "alias": memory.index.alias,
        "title": memory.index.title,
        "summary": memory.index.summary,
    }


def _full_view(memory: MemoryAtom) -> dict[str, Any]:
    return {
        **_index_view(memory),
        "tags": memory.index.tags,
        "memory_type": memory.index.memory_type.value,
        "content": memory.payload.content,
    }


def create_librarian(
    config: MemoryGenerationConfig,
    *,
    mid_term: "MidTermMemoryStore",
    llm_service: BaseLLMService | None,
    llm_params: LLMRequestParams | None,
) -> Librarian | None:
    """话题结算选馆长、提取开启且配置了模型时返回馆长，否则返回 None，结算走一次性生成。

    ``extractor.enabled`` 是两种方式共同的开关：关闭时一次性生成以 No-Op 提取器运行，
    馆长也不装配，结算不产生记忆。没有模型时同理。
    """
    if config.settle_mode != "librarian" or not config.extractor.enabled:
        return None
    if llm_service is None or llm_params is None:
        return None
    return Librarian(
        mid_term=mid_term,
        llm_service=llm_service,
        llm_params=llm_params,
        config=config.librarian,
    )


__all__ = [
    "Librarian",
    "LibrarianResult",
    "StagedRevision",
    "create_librarian",
]
