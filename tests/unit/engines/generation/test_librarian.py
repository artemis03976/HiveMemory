"""Librarian 单元测试：工具调用的循环、步数预算、工具定义、六个工具（含修订的四项校验）与模型服务
失败的重试和传播。"""

import json
import re
from uuid import uuid4

import pytest

from hivememory.config.patchouli import LibrarianConfig
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import ModelContextWindowExceededError, ModelProviderError
from hivememory.core.memory_access import memory_is_readable
from hivememory.core.models import ActorIdentity, IndexLayer, MemoryAtom, MemoryType, PayloadLayer
from hivememory.engines.generation.librarian import Librarian
from hivememory.engines.generation.models import ExtractedMemoryDraft
from hivememory.i18n import set_language
from hivememory.infrastructure.llm.base import LLMCompletion, LLMRequestParams
from hivememory.memory_library.stores import MidTermMemoryStore
from tests.helpers.llm import ScriptedLLMService, tool_call_completion
from tests.helpers.memory import (
    make_memory_identity_scope,
    make_memory_metadata,
)

# CJK 标点、汉字与全角字符；英文提示词中出现任何一个，模型都可能跟着用中文写记忆。
_CJK = re.compile(r"[　-〿一-鿿＀-￯]")

WORKSPACE = make_memory_identity_scope().workspace_identity
# 话题结算的发起者：Workspace owner 加保留的 system agent，只读得到 PUBLIC 记忆。
SYSTEM = ActorIdentity(user_id="u1", agent_id=SYSTEM_AGENT_ID)
PARAMS = LLMRequestParams(model="test-model")


def _memory(alias: str, *, visibility: str = "PUBLIC", tags: tuple[str, ...] = ()) -> MemoryAtom:
    return MemoryAtom(
        meta=make_memory_metadata(source_agent_id="a1", user_id="u1", visibility=visibility),
        index=IndexLayer(
            title=f"{alias} 的标题",
            summary=f"{alias} 的摘要",
            tags=list(tags) or ["t"],
            memory_type=MemoryType.FACT,
            alias=alias,
        ),
        payload=PayloadLayer(content=f"{alias} 的正文"),
    )


class _MemoryPort:
    """中期库后端的内存替身：读取按归属与发起者的读取策略过滤；检索返回打了与 mode
    同名标签的记忆，最多 top_k 条。"""

    def __init__(self, *memories: MemoryAtom) -> None:
        self.memories = {memory.id: memory for memory in memories}

    def _readable(self, belong_to, from_actor) -> list[MemoryAtom]:
        return [
            memory
            for memory in self.memories.values()
            if memory_is_readable(memory, workspace_identity=belong_to, actor_identity=from_actor)
        ]

    async def get(self, belong_to, memory_id, *, from_actor, enforce_actor_visibility=True):
        return next(
            (m for m in self._readable(belong_to, from_actor) if m.id == memory_id),
            None,
        )

    async def get_by_alias(self, belong_to, alias, *, from_actor, enforce_actor_visibility=True):
        return next(
            (m for m in self._readable(belong_to, from_actor) if m.index.alias == alias),
            None,
        )

    async def search(
        self,
        belong_to,
        query,
        top_k,
        filters=None,
        mode="dense",
        score_threshold=0.0,
        *,
        from_actor,
        enforce_actor_visibility=True,
    ):
        hits = [m for m in self._readable(belong_to, from_actor) if mode in m.index.tags]
        return [{"memory": memory, "score": 0.8123} for memory in hits[:top_k]]

    async def patch_payload(self, key, patch):
        memory = self.memories[key.memory_id]
        for dotted_path, value in patch.items():
            *parents, name = dotted_path.split(".")
            target = memory
            for part in parents:
                target = getattr(target, part)
            setattr(target, name, value)
        return memory


def _action(tool: str, **args) -> LLMCompletion:
    """模型一次只调用一个工具的回应。"""
    return tool_call_completion((tool, args))


def _write_args(title: str = "新的结论", **overrides) -> dict:
    return {
        "title": title,
        "summary": f"{title}的一句话摘要",
        "tags": ["结论"],
        "memory_type": "FACT",
        "content": f"{title}的正文",
        "confidence_score": 0.8,
        "alias_suffix": "new_conclusion",
        **overrides,
    }


def _write(title: str = "新的结论", **overrides) -> LLMCompletion:
    return _action("write_memory", **_write_args(title, **overrides))


FINISH = _action("finish")


def _librarian(
    llm: ScriptedLLMService,
    *memories: MemoryAtom,
    max_steps: int = 12,
    search_top_k: int = 5,
) -> Librarian:
    return Librarian(
        mid_term=MidTermMemoryStore(primary=_MemoryPort(*memories)),  # type: ignore[arg-type]
        llm_service=llm,
        llm_params=PARAMS,
        config=LibrarianConfig(max_steps=max_steps, search_top_k=search_top_k),
    )


async def _settle(
    librarian: Librarian, *generated_memories: MemoryAtom, extra_ids: tuple[str, ...] = ()
):
    return await librarian.settle(
        "[Turn 1]\n[Segment 1.1]\n[User]: 以后怎么部署",
        topic_generated_memory_ids=[*(str(memory.id) for memory in generated_memories), *extra_ids],
        belong_to=WORKSPACE,
        from_actor=SYSTEM,
    )


def _reply_messages(llm: ScriptedLLMService) -> list[dict]:
    """交回模型的消息，按顺序：每个工具调用一条 tool 消息，没有调用工具时一条 user 消息。

    替身记下的是馆长的对话列表本身，结算结束后其中也有最后一次请求之后交回的消息。
    """
    return [message for message in llm.calls[-1].messages[2:] if message["role"] != "assistant"]


def _replies(llm: ScriptedLLMService) -> list[dict]:
    return [json.loads(message["content"]) for message in _reply_messages(llm)]


def _call_ids(*completions: LLMCompletion) -> list[str]:
    return [call.id for completion in completions for call in completion.tool_calls]


class TestLibrarianLoop:
    @pytest.mark.asyncio
    async def test_tool_calls_run_and_their_results_return_to_the_model(self):
        """每个工具调用被执行，结果连同剩余步数以对应调用 id 的 tool 消息交回模型；写入的草稿
        在结束后返回。"""
        written = _memory("agent_write")
        listing = _action("list_generated_memories")
        llm = ScriptedLLMService(listing, _write(), FINISH)

        result = await _settle(_librarian(llm, written), written)

        assert [draft.title for draft in result.drafts] == ["新的结论"]
        first = _reply_messages(llm)[0]
        assert (first["role"], first["tool_call_id"]) == ("tool", _call_ids(listing)[0])
        assert json.loads(first["content"]) == {
            "result": [
                {
                    "alias": "agent_write",
                    "title": "agent_write 的标题",
                    "summary": "agent_write 的摘要",
                }
            ],
            "steps_left": 11,
        }

    @pytest.mark.asyncio
    async def test_calls_returned_together_all_run_in_order_one_step_each(self):
        """一次返回的多个调用按顺序全部执行，每个计一步，每个调用 id 都有一条回应。"""
        batch = tool_call_completion(
            ("write_memory", _write_args("第一条")),
            ("write_memory", _write_args("第二条")),
            ("list_generated_memories", {}),
        )
        llm = ScriptedLLMService(batch, FINISH)

        result = await _settle(_librarian(llm))

        assert [draft.title for draft in result.drafts] == ["第一条", "第二条"]
        assert [message["tool_call_id"] for message in _reply_messages(llm)] == _call_ids(
            batch, FINISH
        )
        assert [reply["steps_left"] for reply in _replies(llm)] == [11, 10, 9, 8]

    @pytest.mark.parametrize(
        ("output", "reply_role"),
        [
            (
                LLMCompletion(
                    text="我先看看本话题写过什么",
                    assistant_message={"role": "assistant", "content": "我先看看本话题写过什么"},
                ),
                "user",
            ),
            (tool_call_completion(("search_memories", '{"query": "部署"')), "tool"),
            (_action("delete_memory", alias="agent_write"), "tool"),
        ],
        ids=["no-tool-call", "invalid-json", "unknown-tool"],
    )
    @pytest.mark.asyncio
    async def test_invalid_output_returns_an_error_and_counts_a_step(self, output, reply_role):
        """没有调用工具、参数不是合法的 JSON 对象或工具不存在时把错误交回模型，计一步，循环
        继续；没有工具调用时错误放在 user 消息中，否则放在回应那个调用的 tool 消息中。"""
        llm = ScriptedLLMService(output, _write(), FINISH)

        result = await _settle(_librarian(llm, max_steps=3))

        first = _reply_messages(llm)[0]
        reply = json.loads(first["content"])
        assert (first["role"], "error" in reply, reply["steps_left"]) == (reply_role, True, 2)
        assert [draft.title for draft in result.drafts] == ["新的结论"]

    @pytest.mark.asyncio
    async def test_staged_drafts_are_returned_when_steps_run_out(self):
        """步数用完时不再调用模型，已暂存的草稿照常返回。"""
        llm = ScriptedLLMService(_write("第一条"), _write("第二条"))

        result = await _settle(_librarian(llm, max_steps=2))

        assert [draft.title for draft in result.drafts] == ["第一条", "第二条"]

    @pytest.mark.asyncio
    async def test_calls_left_when_steps_run_out_get_an_error_and_are_not_run(self):
        """步数在一批调用中途用完：余下的调用不执行，各得到一条错误回应，循环结束；已暂存的
        照常返回。"""
        batch = tool_call_completion(
            ("write_memory", _write_args("第一条")),
            ("write_memory", _write_args("第二条")),
            ("write_memory", _write_args("第三条")),
        )
        llm = ScriptedLLMService(batch)

        result = await _settle(_librarian(llm, max_steps=2))

        assert [draft.title for draft in result.drafts] == ["第一条", "第二条"]
        assert [message["tool_call_id"] for message in _reply_messages(llm)] == _call_ids(batch)
        last = _replies(llm)[-1]
        assert ("error" in last, last["steps_left"]) == (True, 0)

    @pytest.mark.asyncio
    async def test_calls_after_finish_in_the_same_batch_are_not_run(self):
        """finish 之后同一批中的调用不执行，得到错误回应，循环结束。"""
        batch = tool_call_completion(
            ("write_memory", _write_args("第一条")),
            ("finish", {}),
            ("write_memory", _write_args("第二条")),
        )
        llm = ScriptedLLMService(batch)

        result = await _settle(_librarian(llm))

        assert [draft.title for draft in result.drafts] == ["第一条"]
        assert [message["tool_call_id"] for message in _reply_messages(llm)] == _call_ids(batch)
        assert "error" in _replies(llm)[-1]

    @pytest.mark.asyncio
    async def test_invalid_write_arguments_return_an_error_without_staging(self):
        llm = ScriptedLLMService(_write(content=None), FINISH)

        result = await _settle(_librarian(llm))

        assert result.drafts == []
        assert "content" in _replies(llm)[0]["error"]

    @pytest.mark.asyncio
    async def test_provider_failure_is_retried_within_the_step(self):
        """提供商失败在同一步内重试一次，不另计步：两步的预算仍够写下草稿并结束。"""
        llm = ScriptedLLMService(ModelProviderError("rate limited"), _write(), FINISH)

        result = await _settle(_librarian(llm, max_steps=2))

        assert [draft.title for draft in result.drafts] == ["新的结论"]
        assert _replies(llm)[0]["steps_left"] == 1

    @pytest.mark.parametrize(
        "errors",
        [
            (ModelProviderError("rate limited"), ModelProviderError("still limited")),
            (ModelContextWindowExceededError("window"),),
        ],
        ids=["provider-retry-exhausted", "context-window"],
    )
    @pytest.mark.asyncio
    async def test_model_service_failure_propagates(self, errors):
        """重试耗尽的提供商失败与上下文超窗不被当作工具错误交回模型，原样传播给生成任务；
        超窗不重试（替身没有多余的预置结果，重试即失败）。"""
        llm = ScriptedLLMService(_write(), *errors)

        with pytest.raises(type(errors[-1])) as raised:
            await _settle(_librarian(llm))

        assert raised.value is errors[-1]


class TestLibrarianToolDefinitions:
    @pytest.mark.asyncio
    async def test_tool_parameters_come_from_the_argument_models(self):
        """请求带六个工具，参数 Schema 由参数模型生成；写入的 Schema 是草稿去掉 has_value，
        是否值得写由馆长在写之前判断。"""
        llm = ScriptedLLMService(FINISH)

        await _settle(_librarian(llm))

        tools = {tool.name: tool.parameters for tool in llm.calls[0].params.tools}
        draft = ExtractedMemoryDraft.model_json_schema()
        assert set(tools) == {
            "list_generated_memories",
            "search_memories",
            "read_memory",
            "revise_memory",
            "write_memory",
            "finish",
        }
        assert tools["search_memories"]["properties"]["mode"]["enum"] == ["dense", "sparse"]
        assert sorted(tools["revise_memory"]["required"]) == ["alias", "changelog"]
        assert (
            sorted(tools["write_memory"]["properties"]),
            tools["write_memory"]["required"],
        ) == (
            sorted(set(draft["properties"]) - {"has_value"}),
            [name for name in draft["required"] if name != "has_value"],
        )

    @pytest.mark.asyncio
    async def test_english_prompt_and_tool_definitions_contain_no_chinese(self):
        """全局语言为英文时，系统提示词与工具定义（说明与参数 Schema）全程英文：参数模型上的
        中文说明不进入 Schema。"""
        set_language("en")
        llm = ScriptedLLMService(FINISH)

        await _settle(_librarian(llm))

        call = llm.calls[0]
        tools = json.dumps([tool.model_dump() for tool in call.params.tools], ensure_ascii=False)
        assert _CJK.findall(call.messages[0]["content"] + tools) == []


class TestLibrarianTools:
    @pytest.mark.asyncio
    async def test_list_generated_memories_skips_memories_the_system_actor_cannot_read(self):
        """PRIVATE 写入与已删除的写入不列出：结算的发起者只读得到 PUBLIC 记忆。"""
        public = _memory("public_write")
        private = _memory("private_write", visibility="PRIVATE")
        llm = ScriptedLLMService(_action("list_generated_memories"), FINISH)

        await _settle(_librarian(llm, public, private), public, private, extra_ids=(str(uuid4()),))

        assert [view["alias"] for view in _replies(llm)[0]["result"]] == ["public_write"]

    @pytest.mark.parametrize("mode", ["dense", "sparse"])
    @pytest.mark.asyncio
    async def test_search_memories_uses_the_requested_mode(self, mode):
        dense = _memory("dense_hit", tags=("dense",))
        sparse = _memory("sparse_hit", tags=("sparse",))
        llm = ScriptedLLMService(_action("search_memories", query="部署", mode=mode), FINISH)

        await _settle(_librarian(llm, dense, sparse))

        assert _replies(llm)[0]["result"] == [
            {
                "alias": f"{mode}_hit",
                "title": f"{mode}_hit 的标题",
                "summary": f"{mode}_hit 的摘要",
                "memory_type": "FACT",
                "score": 0.812,
            }
        ]

    @pytest.mark.asyncio
    async def test_search_memories_returns_at_most_search_top_k(self):
        memories = [_memory(f"hit_{i}", tags=("dense",)) for i in range(3)]
        llm = ScriptedLLMService(_action("search_memories", query="部署"), FINISH)

        await _settle(_librarian(llm, *memories, search_top_k=2))

        assert len(_replies(llm)[0]["result"]) == 2

    @pytest.mark.asyncio
    async def test_search_memories_rejects_an_unknown_mode(self):
        llm = ScriptedLLMService(_action("search_memories", query="部署", mode="hybrid"), FINISH)

        await _settle(_librarian(llm))

        assert "mode" in _replies(llm)[0]["error"]

    @pytest.mark.asyncio
    async def test_read_memory_returns_the_full_memory(self):
        memory = _memory("deploy_rule")
        llm = ScriptedLLMService(_action("read_memory", alias="deploy_rule"), FINISH)

        await _settle(_librarian(llm, memory))

        assert _replies(llm)[0]["result"] == {
            "alias": "deploy_rule",
            "title": "deploy_rule 的标题",
            "summary": "deploy_rule 的摘要",
            "tags": ["t"],
            "memory_type": "FACT",
            "content": "deploy_rule 的正文",
        }

    @pytest.mark.asyncio
    async def test_read_memory_reports_an_unreadable_memory_as_error(self):
        private = _memory("private_rule", visibility="PRIVATE")
        llm = ScriptedLLMService(_action("read_memory", alias="private_rule"), FINISH)

        await _settle(_librarian(llm, private))

        assert set(_replies(llm)[0]) == {"error", "steps_left"}

    @pytest.mark.asyncio
    async def test_searching_and_reading_leave_access_state_unchanged(self):
        """馆长的读取不经过检索服务，不推进访问次数与访问时间，生命力因此不被刷新。"""
        memory = _memory("deploy_rule", tags=("dense",))
        before = memory.meta.lifecycle.model_copy(deep=True)
        llm = ScriptedLLMService(
            _action("search_memories", query="部署"),
            _action("read_memory", alias="deploy_rule"),
            FINISH,
        )

        await _settle(_librarian(llm, memory))

        assert memory.meta.lifecycle == before


def _revise(alias: str, *edits: dict, **fields) -> LLMCompletion:
    return _action("revise_memory", alias=alias, edits=list(edits), changelog="改了", **fields)


def _replace(field: str, old: str, new: str) -> dict:
    return {"op": "replace", "field": field, "old": old, "new": new}


class TestLibrarianRevisions:
    @pytest.mark.asyncio
    async def test_a_revision_of_a_memory_read_in_this_run_is_staged(self):
        """读过之后修订：以读到的那一版为底的编辑集被暂存。"""
        memory = _memory("deploy_rule")
        llm = ScriptedLLMService(
            _action("read_memory", alias="deploy_rule"),
            _revise(
                "deploy_rule",
                _replace("content", "deploy_rule 的正文", "改为手动部署"),
                source_segments=["1.1"],
            ),
            FINISH,
        )

        result = await _settle(_librarian(llm, memory))

        [revision] = result.revisions
        assert (revision.base, revision.edit_set.source_segments) == (memory, ["1.1"])
        assert _replies(llm)[1]["result"] == {"revised": "deploy_rule"}

    @pytest.mark.parametrize(
        ("actions", "error_key"),
        [
            ([_revise("deploy_rule", _replace("content", "deploy_rule 的正文", "x"))], "unread"),
            (
                [
                    _action("read_memory", alias="deploy_rule"),
                    _revise("deploy_rule", _replace("content", "并不存在的原文", "x")),
                ],
                "invalid",
            ),
        ],
        ids=["unread", "invalid-edit-set"],
    )
    @pytest.mark.asyncio
    async def test_a_revision_that_fails_a_check_is_an_error_and_is_not_staged(
        self, actions, error_key
    ):
        """没读过或编辑集不能应用时，错误交回模型并计一步，不暂存。"""
        memory = _memory("deploy_rule")
        llm = ScriptedLLMService(*actions, FINISH)

        result = await _settle(_librarian(llm, memory))

        last = _replies(llm)[len(actions) - 1]
        assert (result.revisions, "error" in last, last["steps_left"]) == (
            [],
            True,
            12 - len(actions),
        )
        assert {
            "unread": "read_memory",
            "invalid": "edit 1 (replace in content)",
        }[
            error_key
        ] in last["error"]

    @pytest.mark.asyncio
    async def test_a_memory_is_revised_at_most_once_per_settlement(self):
        """同一条记忆第二次修订返回错误，只保留第一次；几处修改应放进同一个编辑集。"""
        memory = _memory("deploy_rule")
        llm = ScriptedLLMService(
            _action("read_memory", alias="deploy_rule"),
            _revise("deploy_rule", _replace("content", "deploy_rule 的正文", "第一次")),
            _revise("deploy_rule", _replace("content", "deploy_rule 的正文", "第二次")),
            FINISH,
        )

        result = await _settle(_librarian(llm, memory))

        assert [r.edit_set.edits[0].new for r in result.revisions] == ["第一次"]
        assert "error" in _replies(llm)[2]
