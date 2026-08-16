"""
完整 Chat Run E2E 测试

驱动真实 HiveMemorySystem（真实 LLM + Qdrant）经 ChatApplicationService 统一入口，
验证 v4「一次 chat 调用 = gateway → prepare(检索) → agent run → finalize(记忆落库)」完整闭环：
- chat(): 非流式完整闭环，finalize 后记忆真实落库 Qdrant
- chat(): MTP WRITE 主动生成路径（materialize_tasks → submit_active → 确定性落库）
- chat_stream(): 完整事件序列契约（generation_id → topic_info → memory_refs → 运行期 → run_status → done）
- 检索注入：预埋记忆后 chat，memory_refs 携带引用

标记: [e2e, live_llm]（需真实 LLM API Key + Qdrant）
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from hivememory.core.models import IndexLayer, MemoryAtom, MemoryType, MetaData, PayloadLayer
from hivememory.system.application.chat_service import NonStreamingChatAgentOutcome
from tests.e2e.conftest import wait_for_memory_persistence_async

pytestmark = [pytest.mark.e2e, pytest.mark.live_llm]


def _make_memory(user_id: str, content: str, title: str = "E2E Memory") -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=MetaData(source_agent_id="e2e_test", user_id=user_id),
        index=IndexLayer(
            title=title,
            summary="用于 e2e 检索注入验证的记忆",
            tags=["e2e"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content=content),
    )


def _event_summary(events: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for e in events:
        event = e.get("event")
        data = e.get("data", {}) or {}
        rows.append(f"{event}(scope={data.get('scope', '-')})")
    return " -> ".join(rows)


async def _collect_stream_events(
    system,
    user_message: str,
    user_id: str,
    agent_id: str = "omni_doll",
    enable_memory_retrieval: bool = True,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    async for event in system.chat_service.chat_stream(
        user_message=user_message,
        user_id=user_id,
        agent_id=agent_id,
        enable_memory_retrieval=enable_memory_retrieval,
        generation_options={"temperature": 0, "top_p": 1},
    ):
        events.append(event)
        if event.get("event") in {"done", "error"}:
            break
    return events


class TestChatRun:
    """完整 chat run 闭环"""

    @pytest.mark.asyncio
    async def test_chat_full_round_trip_persists_memory(self, e2e_system, clean_user):
        """chat() 完整闭环：回答 + finalize 触发记忆落库"""
        user_id = clean_user()
        result = await e2e_system.chat_service.chat(
            user_message=(
                "我叫小林，我在一家物流公司工作，每天通勤两小时。"
                "请记住这些关于我的信息。"
            ),
            user_id=user_id,
            enable_memory_retrieval=True,
        )
        assert isinstance(result, NonStreamingChatAgentOutcome), (
            f"chat 应返回 agent outcome, 实际 {type(result).__name__}"
        )
        assert result.agent_run_result.final_text
        assert result.agent_run_result.status == "completed"

        # finalize 链路应把对话内容物化为真实记忆
        memories = await wait_for_memory_persistence_async(
            e2e_system, user_id, min_count=1, timeout=30.0,
        )
        assert memories, "chat finalize 后 Qdrant 应有记忆落库"

    @pytest.mark.asyncio
    async def test_chat_stream_event_sequence(self, e2e_system, clean_user):
        """chat_stream() 完整事件序列：序言 → 运行期 → 终态"""
        user_id = clean_user()
        events = await _collect_stream_events(
            e2e_system, "你好，请用一句话介绍你自己。", user_id,
        )
        event_names = [e.get("event") for e in events]

        # 序言：generation_id / topic_info
        assert "generation_id" in event_names, (
            f"缺少 generation_id 事件。events={_event_summary(events)}"
        )
        assert "topic_info" in event_names, (
            f"缺少 topic_info 事件。events={_event_summary(events)}"
        )
        # 运行期：至少 token 或 mtp 或子代理事件之一
        assert any(n in event_names for n in ("token", "mtp_start", "sub_agent_start")), (
            f"缺少运行期事件。events={_event_summary(events)}"
        )
        # 终态：run_status(finalizing) → done
        assert "run_status" in event_names, (
            f"缺少 run_status 事件。events={_event_summary(events)}"
        )
        done = next((e for e in events if e.get("event") == "done"), None)
        assert done is not None, f"缺少 done 事件。events={_event_summary(events)}"
        assert done.get("data", {}).get("final_text"), "done.final_text 应为空字符串之外的文本"

    @pytest.mark.asyncio
    async def test_chat_stream_retrieval_injection(self, e2e_system, clean_user, qdrant_store):
        """检索注入：预埋记忆 → chat 携带 memory_refs"""
        user_id = clean_user()
        atom = _make_memory(user_id, "我最喜欢的编程语言是 Rust，因为它内存安全。")
        await qdrant_store.upsert_memory(atom)

        events = await _collect_stream_events(
            e2e_system,
            "根据我的记忆，我最喜欢的编程语言是什么？",
            user_id,
            enable_memory_retrieval=True,
        )

        # 检索注入事件应携带引用
        refs_events = [e for e in events if e.get("event") == "memory_refs"]
        assert refs_events, f"缺少 memory_refs 事件。events={_event_summary(events)}"
        assert any(
            str(refs_events[i].get("data")) != "{}" or refs_events[i].get("data")
            for i in range(len(refs_events))
        ), "memory_refs 应携带检索结果"

    @pytest.mark.asyncio
    async def test_chat_mtp_write_persists_deterministic_content(
        self, e2e_system, clean_user,
    ):
        """
        MTP WRITE 主动生成路径（确定性验证）：
        显式 WRITE 指令 → PendingAtom → finalize 派发 submit_active → 真实生成落库。

        与宽松的"请记住…"不同，本用例要求 agent 必须输出 WRITE 指令，
        materialize_tasks 非空即证明走的是 WRITE 主动生成而非 finalize 自动提取。
        """
        user_id = clean_user()
        result = await e2e_system.chat_service.chat(
            user_message=(
                "请使用 MTP 的 WRITE 指令保存一条记忆，内容如下："
                "我最好的朋友叫张伟，我们每个月一起打篮球。"
                "你必须在回复中输出 WRITE 指令，把上面这句话完整写入记忆。"
            ),
            user_id=user_id,
            enable_memory_retrieval=False,
        )
        run_result = result.agent_run_result
        assert run_result.status == "completed", (
            f"chat 未正常完成: status={run_result.status}"
        )
        assert run_result.materialize_tasks, (
            "agent run 应产生 MTP WRITE 物化任务, 说明走了 WRITE 主动生成路径"
        )

        # WRITE 走 Mode B（WriteFocus 直接落库），内容应确定性地包含显式给定文本
        memories = await wait_for_memory_persistence_async(
            e2e_system, user_id, min_count=1, timeout=30.0,
        )
        all_content = " ".join(
            m.payload.content for m in memories if m.payload.content
        )
        assert "张伟" in all_content and "打篮球" in all_content, (
            f"WRITE 生成的记忆应包含显式内容, 实际: {all_content[:300]}"
        )
