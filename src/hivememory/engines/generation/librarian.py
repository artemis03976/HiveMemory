"""
HiveMemory - 馆长 (Librarian)

话题结算的另一种生成方式。一次性生成看不到本话题已经写入的记忆，也看不到库，只能先尽量多地
提取，再靠查重与合并的规则补救；馆长改为先探索再写：模型以模型服务的原生工具调用使用
Patchouli 定义的工具，先看本话题生成的记忆，再检索并阅读库中相关的记忆，然后只写两者
都没有覆盖的内容。工具的参数 Schema 由参数模型生成，工作顺序与各参数的含义写在系统提示词里。

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
from hivememory.i18n import Language, get_generation_prompt_text, resolve_language
from hivememory.infrastructure.llm.base import (
    BaseLLMService,
    LLMRequestParams,
    LLMTool,
    LLMToolCall,
    complete_with_provider_retry,
    tool_result_message,
)

if TYPE_CHECKING:
    from hivememory.memory_library.stores import MidTermMemoryStore

logger = logging.getLogger(__name__)


class _SearchArgs(BaseModel):
    query: str = Field(min_length=1)
    mode: Literal["dense", "sparse"] = "dense"


class _ReadArgs(BaseModel):
    alias: str = Field(min_length=1)


class _ReviseArgs(MemoryEditSet):
    alias: str = Field(min_length=1)


def _without_descriptions(node: Any) -> Any:
    """去掉 JSON Schema 中各处的说明文字，只留结构。

    参数的含义写在提示词里、与提示词同一语言；参数模型上的说明文字是中文，原样交给英文
    提示词时，模型会跟着中文说明用中文写记忆。只去掉值是文字的 description，名为
    description 的参数（值是 Schema）保留。
    """
    if isinstance(node, dict):
        return {
            key: _without_descriptions(value)
            for key, value in node.items()
            if not (key == "description" and isinstance(value, str))
        }
    if isinstance(node, list):
        return [_without_descriptions(item) for item in node]
    return node


def _write_parameters() -> dict[str, Any]:
    """write_memory 的参数：草稿去掉 has_value，是否值得写由馆长在写之前判断。"""
    schema: dict[str, Any] = _without_descriptions(ExtractedMemoryDraft.model_json_schema())
    del schema["properties"]["has_value"]
    schema["required"].remove("has_value")
    return schema


_NO_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}}

# 工具名到参数的 JSON Schema，按提示词中工具的顺序；日志只按这些名字计数，模型编造的名字
# 记为 unknown。
_TOOL_PARAMETERS: dict[str, dict[str, Any]] = {
    "list_generated_memories": _NO_PARAMETERS,
    "search_memories": _without_descriptions(_SearchArgs.model_json_schema()),
    "read_memory": _without_descriptions(_ReadArgs.model_json_schema()),
    "revise_memory": _without_descriptions(_ReviseArgs.model_json_schema()),
    "write_memory": _write_parameters(),
    "finish": _NO_PARAMETERS,
}


def _tools(language: Language) -> tuple[LLMTool, ...]:
    """交给模型服务的工具定义；每个工具一句话说明，与提示词同一语言。"""
    return tuple(
        LLMTool(
            name=name,
            description=get_generation_prompt_text("librarian", f"tool_{name}", language),
            parameters=parameters,
        )
        for name, parameters in _TOOL_PARAMETERS.items()
    )


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
    """一次结算的工具上下文：读取的归属与发起者、本话题生成的记忆、读过的记忆与已暂存的内容。"""

    belong_to: WorkspaceIdentity
    from_actor: ActorIdentity
    topic_generated_memory_ids: list[str]
    staged: list[ExtractedMemoryDraft] = field(default_factory=list)
    # read_memory 读到的版本，按 alias；同一条读了几次时取最后一次。
    read: dict[str, MemoryAtom] = field(default_factory=dict)
    revisions: list[StagedRevision] = field(default_factory=list)


class Librarian:
    """话题结算的馆长：以原生工具调用使用工具的循环，返回暂存的草稿与修订。

    循环与步数:
        每次请求模型，把它返回的 assistant 消息原样追加进对话，再按顺序逐个执行其中的工具调用，
        每个调用计一步，每个调用都以一条 tool 消息交回结果与剩余步数。一批调用执行到一半步数
        用完时，余下的调用交回错误、不执行，循环结束；finish 之后同一批中的调用同样交回错误、
        不执行。模型没有调用工具时，交回一条 user 消息要求调用工具，计一步。

    失败语义:
        没有调用工具、参数不是合法的 JSON 对象、工具不存在、参数不合格或修订不能落地时，把错误
        交回模型并计一步，不中止循环；步数用完时照常返回已暂存的内容。提供商失败在同一次请求内
        重试一次、不另计步，与提取器的模型调用相同，一次瞬时失败不让整次结算失败。重试耗尽或
        上下文超窗时原样传播，生成任务以对应的原因码失败；模型不支持工具调用时由提供商拒绝请求，
        同样以提供商失败结束；取消原样传播。
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
        self._config = config
        language = resolve_language()
        # 重新校验而不是 model_copy：工具与 JSON 对象输出互斥，传入的参数要求 JSON 输出时
        # 在装配时失败。
        self._llm_params = LLMRequestParams.model_validate(
            {**llm_params.model_dump(), "tools": _tools(language)}
        )
        self._system_prompt = get_generation_prompt_text(
            "librarian", "system_prompt", language
        ).format(max_steps=config.max_steps, search_top_k=config.search_top_k)
        self._user_prompt = get_generation_prompt_text("librarian", "user_prompt", language)
        self._error_no_tool_call = get_generation_prompt_text(
            "librarian", "error_no_tool_call", language
        )
        self._error_invalid_json = get_generation_prompt_text(
            "librarian", "error_invalid_json", language
        )
        self._error_steps_exhausted = get_generation_prompt_text(
            "librarian", "error_steps_exhausted", language
        )
        self._error_finished = get_generation_prompt_text("librarian", "error_finished", language)
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
        topic_generated_memory_ids: Collection[str],
        belong_to: WorkspaceIdentity,
        from_actor: ActorIdentity,
    ) -> LibrarianResult:
        """对一段话题材料运行馆长循环，返回暂存的草稿与修订。

        Args:
            transcript: 带段标签的话题材料，放在第一条消息中
            topic_generated_memory_ids: 本话题生成的记忆 ID（主动写入结算到的，以及本话题此前
                沉淀新建的），由 list_generated_memories 列出
            belong_to: 读取的 Workspace 归属
            from_actor: 读取的发起者；结算时是 Workspace 的 system actor

        Raises:
            ModelProviderError: 提供商失败且重试耗尽，或上下文超窗
                （ModelContextWindowExceededError）
        """
        run = _Run(
            belong_to=belong_to,
            from_actor=from_actor,
            topic_generated_memory_ids=sorted(topic_generated_memory_ids),
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": self._user_prompt.format(transcript=transcript)},
        ]
        tool_calls: Counter[str] = Counter()
        revise_rejected = 0
        total_tokens = prompt_tokens = cached_prompt_tokens = 0
        requests = steps = 0
        finished = False
        while steps < self._config.max_steps and not finished:
            requests += 1
            # 每次请求都重发整段对话；消息只往后追加、前缀不变，所以第二次起的输入大部分
            # 可以命中提供商的上下文缓存，日志分开记下命中的部分。
            completion = await complete_with_provider_retry(
                self._llm_service, messages, self._llm_params
            )
            if completion.usage is not None:
                total_tokens += completion.usage.total_tokens
                prompt_tokens += completion.usage.prompt_tokens
                cached_prompt_tokens += completion.usage.cached_prompt_tokens
            messages.append(completion.assistant_message)
            if not completion.tool_calls:
                steps += 1
                tool_calls["no_tool_call"] += 1
                reply = {
                    "error": self._error_no_tool_call,
                    "steps_left": self._config.max_steps - steps,
                }
                messages.append({"role": "user", "content": json.dumps(reply, ensure_ascii=False)})
                continue
            # 每个调用 id 都要有一条 tool 消息回应，不执行的调用也交回错误，并计为 not_run。
            for call in completion.tool_calls:
                if finished:
                    tool_calls["not_run"] += 1
                    reply = {"error": self._error_finished}
                elif steps == self._config.max_steps:
                    tool_calls["not_run"] += 1
                    reply = {"error": self._error_steps_exhausted}
                else:
                    steps += 1
                    label, reply = await self._run_call(run, call)
                    tool_calls[label] += 1
                    finished = label == "finish"
                    if label == "revise_memory" and "error" in reply:
                        revise_rejected += 1
                reply["steps_left"] = self._config.max_steps - steps
                messages.append(tool_result_message(call, json.dumps(reply, ensure_ascii=False)))

        # 步数用完是预算截断：模型没有说写完，已暂存的照常提交。步数多于请求次数，说明有请求
        # 一次返回了多个调用；tool_calls 中的 not_run 是交回了错误、没有执行的调用。
        logger.log(
            logging.INFO if finished else logging.WARNING,
            "馆长结算结束: end=%s, steps=%d, requests=%d, staged=%d, total_tokens=%d, "
            "prompt_tokens=%d, cached_prompt_tokens=%d, tool_calls=%s, revised=%d, "
            "revise_rejected=%d",
            "finish" if finished else "max_steps",
            steps,
            requests,
            len(run.staged),
            total_tokens,
            prompt_tokens,
            cached_prompt_tokens,
            " ".join(f"{name}={count}" for name, count in sorted(tool_calls.items())),
            len(run.revisions),
            revise_rejected,
        )
        return LibrarianResult(drafts=run.staged, revisions=run.revisions)

    async def _run_call(self, run: _Run, call: LLMToolCall) -> tuple[str, dict[str, Any]]:
        """执行一个工具调用，返回日志中的计数项与交回模型的结果。

        参数不是合法的 JSON 对象时不执行，计为 invalid_json；工具不存在时计为 unknown。
        """
        arguments = call.parsed_arguments()
        if arguments is None:
            return "invalid_json", {"error": self._error_invalid_json}
        label = call.name if call.name in _TOOL_PARAMETERS else "unknown"
        return label, await self._run_tool(run, call.name, arguments)

    async def _run_tool(self, run: _Run, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """执行一个工具，返回交回模型的结果；参数不合格或读不到记忆时返回错误。"""
        try:
            match name:
                case "list_generated_memories":
                    return {"result": await self._list_generated_memories(run)}
                case "search_memories":
                    args = _SearchArgs.model_validate(arguments)
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
                    alias = _ReadArgs.model_validate(arguments).alias
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
                        ExtractedMemoryDraft.model_validate({**arguments, "has_value": True})
                    )
                    return {"result": {"staged": len(run.staged)}}
                case "revise_memory":
                    return self._revise_memory(run, _ReviseArgs.model_validate(arguments))
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

    async def _list_generated_memories(self, run: _Run) -> list[dict[str, Any]]:
        """本话题生成的记忆；发起者读不到的（PRIVATE、TEAM 或已删除）跳过。"""
        views = []
        for memory_id in run.topic_generated_memory_ids:
            memory = await self._mid_term.get(
                run.belong_to, UUID(memory_id), from_actor=run.from_actor
            )
            if memory is not None:
                views.append(_index_view(memory))
        return views


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
