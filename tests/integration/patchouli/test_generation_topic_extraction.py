"""话题结算的提取经真实 Librarian 模型：事实性对话产出记忆，闲聊不产出。

引擎按生产方式渲染生成视图、调用提取提示词并把草稿转为结果；提示词与模型输出的解析、
无产出、截断与重试由提取器的单元测试维护。中期库在本测试的边界之外，替身没有已有记忆，
查重总是判为新记忆。

需要配置中可用的 Librarian 模型与凭证，调用会产生费用。
"""

import pytest

from hivememory.config.app import load_app_config, provider_credentials_from_env
from hivememory.core.models import (
    ActorIdentity,
    resolve_default_workspace_identity,
    system_actor_for_workspace,
)
from hivememory.engines.generation.deduplicator import MemoryDeduplicator
from hivememory.engines.generation.engine import MemoryGenerationEngine
from hivememory.engines.generation.extractor import LLMMemoryExtractor
from hivememory.engines.generation.models import (
    GenerationAction,
    GenerationContext,
    GenerationRequest,
    GenerationTurn,
)
from hivememory.infrastructure.llm import LiteLLMService
from hivememory.system.assembler import component_llm_params
from hivememory.system.model_registry import ModelRegistry
from hivememory.system.provider_registry import ProviderRegistry

pytestmark = pytest.mark.live_llm

_USER_ID = "topic_extraction_user"
_IDENTITY = ActorIdentity(user_id=_USER_ID, agent_id="a1")


class _EmptyMidTerm:
    """测试边界外的中期库：没有已有记忆，别名都空闲。"""

    async def search(self, *args, **kwargs):
        return []

    async def list_alias_holders(self, *args, **kwargs):
        return []


def _engine() -> MemoryGenerationEngine:
    config = load_app_config()
    registry = ModelRegistry(
        provider_registry=ProviderRegistry(env_providers=provider_credentials_from_env())
    )
    llm_params = component_llm_params(config.patchouli.llm, registry)
    assert llm_params is not None, "Librarian 未配置可用模型"
    return MemoryGenerationEngine(
        mid_term=_EmptyMidTerm(),
        extractor=LLMMemoryExtractor(
            config=config.patchouli.generation.extractor,
            llm_service=LiteLLMService(),
            llm_params=llm_params,
        ),
        deduplicator=MemoryDeduplicator(config=config.patchouli.generation.deduplicator),
    )


async def _settle(user_message: str, assistant_message: str):
    belong_to = resolve_default_workspace_identity(_USER_ID)
    context = GenerationContext(
        turns=[
            GenerationTurn(
                user_message=user_message,
                material_entries=[f"[Assistant]: {assistant_message}"],
                identity=_IDENTITY,
            )
        ]
    )
    return await _engine().process(
        GenerationRequest(context=context),
        belong_to=belong_to,
        from_actor=system_actor_for_workspace(belong_to),
    )


@pytest.mark.asyncio
async def test_factual_decision_produces_memory_with_the_fact():
    """一项带理由的配置决定产出新记忆，记忆保留了服务名这一可检索的事实。"""
    outcomes = await _settle(
        "记一下：报表服务 report-svc 的每日汇总任务改到凌晨 2 点执行，"
        "因为 1 点的数据库备份会锁表，汇总经常超时。",
        "好的，report-svc 的每日汇总任务改为 02:00 执行，避开 01:00 的数据库备份。",
    )

    created = [outcome.atom for outcome in outcomes if outcome.action == GenerationAction.CREATE]
    assert created, f"事实性对话应产出新记忆，实际结果：{outcomes}"
    text = " ".join(f"{a.index.title} {a.index.summary} {a.payload.content}" for a in created)
    assert "report-svc" in text


@pytest.mark.asyncio
async def test_small_talk_produces_no_memory():
    """只有寒暄的对话不产出记忆。"""
    outcomes = await _settle("好的，谢谢你的帮助", "不客气，有问题随时问我")

    assert outcomes == []
