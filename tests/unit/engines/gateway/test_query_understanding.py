"""QueryUnderstandingEngine 单元测试。"""

from __future__ import annotations

import asyncio
import time

import pytest

from hivememory.config.gateway import UserQueryAnalysisConfig
from hivememory.core.errors import ModelProviderError
from hivememory.core.models import ActorIdentity, LogicalBlock, TopicData, TurnRecord
from hivememory.core.protocol.gateway import IntentType, MemoryWriteSignal
from hivememory.engines.gateway.query_understanding import (
    QueryUnderstandingEngine,
    QueryUnderstandingError,
)
from tests.helpers.llm import DEFAULT_LLM_PARAMS, ScriptedLLMService
from tests.helpers.workspace import make_identity_scope


def _build_topic_data() -> TopicData:
    now = time.time()
    return TopicData(
        topic_id="topic-1",
        workspace_identity=make_identity_scope(user_id="u1").workspace_identity,
        topic_title="Docker 部署",
        topic_summary="排查 Docker 部署问题",
        state_summary="已定位到内存溢出",
        blocks=(
            LogicalBlock(
                turn=TurnRecord(
                    identity=ActorIdentity(user_id="u1"),
                    user_query="那个报错怎么修？",
                    assistant_final_text="可以增加内存限制",
                )
            ),
        ),
        last_update=now,
    )


@pytest.mark.asyncio
async def test_analyze_parses_shared_result_and_renders_context() -> None:
    llm = ScriptedLLMService(
        '{"intent_type":"RAG",'
        '"rewritten_query":"Docker OOM 内存溢出报错的排查原因与修复方案",'
        '"search_keywords":["Docker","OOM","内存溢出"],'
        '"memory_write_signal":"WRITE",'
        '"sub_intents":[],'
        '"reason":"技术问答"}'
    )
    engine = QueryUnderstandingEngine(
        config=UserQueryAnalysisConfig(), llm_service=llm, llm_params=DEFAULT_LLM_PARAMS
    )

    result = await engine.analyze("那个报错怎么修？", topic_data=_build_topic_data())

    assert result.intent_type == IntentType.RAG
    assert result.rewritten_query == "Docker OOM 内存溢出报错的排查原因与修复方案"
    assert result.search_keywords == ("Docker", "OOM", "内存溢出")
    assert result.memory_write_signal == MemoryWriteSignal.WRITE
    assert result.sub_intents == ()

    messages = llm.calls[-1].messages
    system_prompt = messages[0]["content"]
    assert "Docker 部署" in system_prompt
    assert "那个报错怎么修？" in system_prompt
    assert messages[1] == {"role": "user", "content": "那个报错怎么修？"}
    # 组件参数原样沿用，只追加 JSON 输出要求。
    assert llm.calls[-1].params == DEFAULT_LLM_PARAMS.model_copy(update={"json_output": True})


@pytest.mark.asyncio
async def test_model_override_replaces_only_the_model() -> None:
    """配置的模型覆盖只替换模型，凭证与采样参数沿用 Gateway 组件配置。"""
    llm = ScriptedLLMService('{"intent_type":"RAG","rewritten_query":"重写结果"}')
    params = DEFAULT_LLM_PARAMS.model_copy(update={"api_key": "gateway-key"})
    engine = QueryUnderstandingEngine(
        config=UserQueryAnalysisConfig(model_override="override-model"),
        llm_service=llm,
        llm_params=params,
    )

    await engine.analyze("查询")

    assert llm.calls[-1].params == params.model_copy(
        update={"model": "override-model", "json_output": True}
    )


@pytest.mark.asyncio
async def test_analyze_without_topic_data_uses_empty_context() -> None:
    llm = ScriptedLLMService(
        '{"intent_type":"WRITE","rewritten_query":"用户偏好少盐饮食",'
        '"search_keywords":["少盐"],"memory_write_signal":"WRITE",'
        '"sub_intents":[],"reason":"偏好陈述"}'
    )
    engine = QueryUnderstandingEngine(
        config=UserQueryAnalysisConfig(), llm_service=llm, llm_params=DEFAULT_LLM_PARAMS
    )

    result = await engine.analyze("以后做菜少放盐", topic_data=None)

    assert result.intent_type == IntentType.WRITE
    messages = llm.calls[-1].messages
    assert "{topic_context}" not in messages[0]["content"]


@pytest.mark.asyncio
async def test_analyze_applies_lenient_defaults_for_optional_fields() -> None:
    llm = ScriptedLLMService(
        '{"intent_type":"NOT_A_REAL_INTENT","rewritten_query":"重写结果",'
        '"search_keywords":"not-a-list","memory_write_signal":"BAD",'
        '"sub_intents":null}'
    )
    engine = QueryUnderstandingEngine(
        config=UserQueryAnalysisConfig(), llm_service=llm, llm_params=DEFAULT_LLM_PARAMS
    )

    result = await engine.analyze("查询", topic_data=None)

    assert result.intent_type == IntentType.RAG
    assert result.memory_write_signal == MemoryWriteSignal.UNKNOWN
    assert result.search_keywords == ()
    assert result.sub_intents == ()


@pytest.mark.asyncio
async def test_analyze_respects_configurable_context_limits() -> None:
    payload = (
        '{"intent_type":"RAG","rewritten_query":"重写结果",'
        '"search_keywords":[],"memory_write_signal":"UNKNOWN",'
        '"sub_intents":[],"reason":""}'
    )
    llm = ScriptedLLMService(payload, payload)
    engine = QueryUnderstandingEngine(
        config=UserQueryAnalysisConfig(context_block_limit=0, context_text_limit=4),
        llm_service=llm,
        llm_params=DEFAULT_LLM_PARAMS,
    )

    await engine.analyze("查询", topic_data=_build_topic_data())

    system_prompt = llm.calls[-1].messages[0]["content"]
    # block_limit=0 时不渲染任何历史轮次
    assert "user: " not in system_prompt
    assert "assistant: " not in system_prompt
    # 标题与摘要仍渲染
    assert "Docker 部署" in system_prompt

    engine = QueryUnderstandingEngine(
        config=UserQueryAnalysisConfig(context_block_limit=1, context_text_limit=4),
        llm_service=llm,
        llm_params=DEFAULT_LLM_PARAMS,
    )

    now = time.time()
    topic_data = TopicData(
        topic_id="topic-1",
        workspace_identity=make_identity_scope(user_id="u1").workspace_identity,
        topic_title="三餐推荐",
        blocks=(
            LogicalBlock(
                turn=TurnRecord(
                    identity=ActorIdentity(user_id="u1"),
                    user_query="上一轮用户输入的很长内容",
                    assistant_final_text="",
                )
            ),
        ),
        last_update=now,
    )

    await engine.analyze("查询", topic_data=topic_data)

    system_prompt = llm.calls[-1].messages[0]["content"]
    # text_limit=4 时历史文本被截断
    assert "user: 上一轮用" in system_prompt
    assert "上一轮用户输入的很长内容" not in system_prompt


@pytest.mark.asyncio
async def test_analyze_raises_without_llm_service() -> None:
    engine = QueryUnderstandingEngine(
        config=UserQueryAnalysisConfig(), llm_service=None, llm_params=DEFAULT_LLM_PARAMS
    )

    with pytest.raises(QueryUnderstandingError):
        await engine.analyze("查询", topic_data=None)


@pytest.mark.asyncio
async def test_analyze_raises_on_missing_rewritten_query() -> None:
    llm = ScriptedLLMService('{"intent_type":"RAG","rewritten_query":"  "}')
    engine = QueryUnderstandingEngine(
        config=UserQueryAnalysisConfig(), llm_service=llm, llm_params=DEFAULT_LLM_PARAMS
    )

    with pytest.raises(QueryUnderstandingError):
        await engine.analyze("查询", topic_data=None)


@pytest.mark.asyncio
async def test_analyze_wraps_llm_failure() -> None:
    failure = ModelProviderError("provider-private")
    llm = ScriptedLLMService(failure)
    engine = QueryUnderstandingEngine(
        config=UserQueryAnalysisConfig(), llm_service=llm, llm_params=DEFAULT_LLM_PARAMS
    )

    with pytest.raises(QueryUnderstandingError, match="提供商调用失败") as caught:
        await engine.analyze("查询", topic_data=None)
    assert caught.value.__cause__ is failure
    assert "provider-private" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [TypeError("broken projection"), asyncio.CancelledError()])
async def test_analyze_propagates_program_errors_and_cancellation(failure):
    """未知故障与取消不得成为可降级的能力失败。"""
    llm = ScriptedLLMService(failure)
    engine = QueryUnderstandingEngine(
        config=UserQueryAnalysisConfig(), llm_service=llm, llm_params=DEFAULT_LLM_PARAMS
    )
    with pytest.raises(type(failure)) as caught:
        await engine.analyze("查询")
    assert caught.value is failure


@pytest.mark.asyncio
async def test_analyze_classifies_invalid_model_json():
    """模型格式无效是预期输出失败，仍可由 Gateway 保守降级。"""
    llm = ScriptedLLMService("not json")
    engine = QueryUnderstandingEngine(
        config=UserQueryAnalysisConfig(), llm_service=llm, llm_params=DEFAULT_LLM_PARAMS
    )
    with pytest.raises(QueryUnderstandingError, match="无效 JSON"):
        await engine.analyze("查询")
