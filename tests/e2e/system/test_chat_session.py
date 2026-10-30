"""确定性 HTTP 会话链路：真实 Alice/Gateway/Patchouli，外部模型与存储使用替身。"""

import asyncio
import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from hivememory.alice.system import AliceSystem
from hivememory.config.app import HiveMemoryConfig
from hivememory.config.workspace import WorkspaceSessionConfig
from hivememory.server import deps
from hivememory.server.routers.chat import router as chat_router
from hivememory.server.routers.sessions import router as sessions_router
from hivememory.workspace.capability.actor.agent_profiles import ActorAgentService
from hivememory.workspace.capability.actor.memory import ActorMemoryService
from hivememory.workspace.capability.operations import WorkspaceOperationEntry
from hivememory.workspace.capability.root.session import RootSessionService
from hivememory.workspace.sessions import ConversationSessionService
from tests.helpers.llm import CallbackLLMService
from tests.helpers.process import make_task_process_service
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_workspace_read_view,
)

pytestmark = pytest.mark.e2e


def events_from_text(text):
    """解析实际 HTTP SSE 数据，断言不依赖内部进程表。"""
    events, name = [], ""
    for line in text.splitlines():
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            events.append((name, json.loads(line[6:])))
    return events


@pytest.mark.asyncio
async def test_http_session_resume_after_partial_stop(patchouli_chat_stack):
    """三轮中第二轮流出半截后停止；第三轮真实模型输入和读取历史都保留中断轮。"""
    stack = patchouli_chat_stack
    access = make_access_composition(
        [make_actor_access_record(agent_id=agent) for agent in ("omni_doll", "system")],
        adapters=("http",),
    )
    workspace = access.default_workspace
    runtime = make_workspace_read_view(stack.bus)
    runtime.subscribe(stack.bus)
    entry = WorkspaceOperationEntry(
        ActorMemoryService(
            stack.bus,
            operation_authorizer=access.authorizer,
            memory_reader=runtime.memory,
            threads=access.threads,
        ),
        agent=ActorAgentService(
            operation_authorizer=access.authorizer,
            profile_reader=runtime.profiles,
        ),
        threads=access.threads,
    )
    history_inputs = []
    partial_sent = asyncio.Event()
    stopped_native = asyncio.Event()
    observed_process = []

    async def completion(messages, _params):
        history_inputs.append([dict(message) for message in messages])
        round_number = len(history_inputs)
        yield (
            "第一轮答案" if round_number == 1 else "只完成一半" if round_number == 2 else "继续完成"
        )
        if round_number == 2:
            # 第二轮停在半截，等待用户停止的原生取消。
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                stopped_native.set()
                raise

    sessions = ConversationSessionService(WorkspaceSessionConfig())
    alice = AliceSystem(
        config=HiveMemoryConfig().alice,
        operation_entry=entry,
        llm_service=CallbackLLMService(stream=completion),
    )
    service = make_task_process_service(
        stack.bus,
        cpu=alice.cpu_port,
        sessions=sessions,
        access_gateway=access.gateway,
        operation_authorizer=access.authorizer,
        process_table=access.process_table,
        workspace_read_view=runtime,
        operation_entry=entry,
    )
    app = FastAPI()
    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(sessions_router, prefix="/api/v1")
    app.dependency_overrides[deps.get_process_service] = lambda: service
    app.dependency_overrides[deps.get_access_gateway] = lambda: access.gateway
    app.dependency_overrides[deps.get_server_principal_id] = lambda: access.principal.principal_id
    root = RootSessionService(sessions, operation_authorizer=access.authorizer)
    app.dependency_overrides[deps.get_root_session_service] = lambda: root

    async def observed_app(scope, receive, send):
        async def observe(message):
            await send(message)
            if message["type"] == "http.response.body":
                for name, data in events_from_text(message.get("body", b"").decode()):
                    if name == "process_id":
                        observed_process.append(data["process_id"])
                    if name == "token" and data["content"] == "只完成一半":
                        partial_sent.set()

        await app(scope, receive, observe)

    await alice.start()
    try:
        async with AsyncClient(
            transport=ASGITransport(app=observed_app),
            base_url="http://test",
            headers={
                "X-User-ID": workspace.owner_user_id,
                "X-Workspace-ID": workspace.workspace_id,
            },
        ) as client:
            created = await client.post("/api/v1/sessions", json={"title": "中断后续接"})
            assert created.status_code == 201
            session_id = created.json()["session_id"]
            responses = []

            async def chat(message):
                return await client.post(
                    "/api/v1/chat",
                    json={
                        "message": message,
                        "session_id": session_id,
                        "agent_id": "omni_doll",
                        "enable_memory_retrieval": False,
                        "generation_options": {"model": "fake-test-model"},
                    },
                )

            responses.append(await chat("第一轮问题"))
            second = asyncio.create_task(chat("第二轮问题"))
            await asyncio.wait_for(partial_sent.wait(), timeout=5)
            stop = await client.post("/api/v1/chat/stop", json={"process_id": observed_process[-1]})
            assert stop.json()["cancelled"] is True
            responses.append(await asyncio.wait_for(second, timeout=5))
            responses.append(await chat("接着说"))
            loaded = await client.get(f"/api/v1/sessions/{session_id}")
            assert loaded.status_code == 200
            history = loaded.json()
        assert stopped_native.is_set()
        assert [r.status_code for r in responses] == [200, 200, 200]
        terminals = [events_from_text(r.text)[-1][1] for r in responses]
        assert [data["status"] for data in terminals] == ["completed", "cancelled", "completed"]
        assert terminals[1]["final_text"] == "只完成一半"
        assert [turn["payload"]["outcome"] for turn in history["turns"]] == [
            "completed",
            "cancelled",
            "completed",
        ]
        assert history["active_process_id"] is None
        assert [turn["topic_id"] for turn in history["turns"]] == [
            data["topic_id"] for data in terminals
        ]
        for response, data, turn in zip(responses, terminals, history["turns"], strict=True):
            topic = stack.short_term.get(workspace, data["topic_id"])
            block = next(
                block
                for block in topic.blocks
                if block.user_message == turn["payload"]["user_message"]
            )
            assert block.turn.outcome == turn["payload"]["outcome"]
        third_prompt = "\n".join(item["content"] for item in history_inputs[2])
        assert "第一轮答案" in third_prompt
        assert "只完成一半" in third_prompt
        assert "已中断" in third_prompt
        assert len(history_inputs) == 3
    finally:
        await service.shutdown()
        await alice.stop()
        runtime.close()
        access.gateway.close()
        access.gateway.revoke_all_contexts()


@pytest.mark.asyncio
async def test_text_only_cpu_output_is_structured_before_patchouli_submission(
    patchouli_chat_stack,
):
    """CPU 只交回正文时，HTTP 到真实 Patchouli 仍形成可重放的结构化轮次。

    三种有输出结局的封口与会话写入由进程归属层矩阵覆盖，这里只保留
    completed 代表证明公开入口到真实 Patchouli 的链路。
    """
    outcome = "completed"
    from tests.helpers.cpu import ScriptedCPU, make_cpu_result

    access = make_access_composition(
        [make_actor_access_record(agent_id="omni_doll")], adapters=("http",)
    )
    sessions = ConversationSessionService(WorkspaceSessionConfig())
    service = make_task_process_service(
        patchouli_chat_stack.bus,
        cpu=ScriptedCPU(result=make_cpu_result(status=outcome, final_text="仅正文结果")),
        sessions=sessions,
        access_gateway=access.gateway,
        operation_authorizer=access.authorizer,
    )
    app = FastAPI()
    app.include_router(chat_router, prefix="/api/v1")
    app.dependency_overrides[deps.get_process_service] = lambda: service
    app.dependency_overrides[deps.get_server_principal_id] = lambda: access.principal.principal_id
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat",
                headers={"X-User-ID": access.default_workspace.owner_user_id},
                json={
                    "message": "只含正文",
                    "agent_id": "omni_doll",
                    "enable_memory_retrieval": False,
                },
            )
        events = events_from_text(response.text)
        name, done = events[-1]
        assert (name, done["status"]) == ("done", outcome)
        topic = patchouli_chat_stack.short_term.get(access.default_workspace, done["topic_id"])
        (block,) = topic.blocks
        assert block.turn.outcome == outcome
        assert [(e.kind, e.content) for e in block.turn_events] == [
            ("user_message", "只含正文"),
            ("assistant_message", "仅正文结果"),
        ]
        session_id = events[0][1]["session_id"]
        assert sessions.get(access.default_workspace, session_id).turns[
            0
        ].payload.turn_events == list(block.turn_events)
    finally:
        await service.shutdown()
        access.gateway.close()
        access.gateway.revoke_all_contexts()


@pytest.mark.asyncio
async def test_http_long_session_reuses_compacted_history_and_seals_budget_failure(
    patchouli_chat_stack, tmp_path
):
    """多轮 HTTP 压缩后仍可恢复原文；后续大工具结果被守卫拦截并保存失败轮次。"""
    from hivememory.components.events.publisher import RuntimeEventPublisher
    from hivememory.config.alice import CompactionConfig
    from hivememory.core.models import IndexLayer, MemoryAtom, MemoryType, PayloadLayer
    from tests.helpers.memory import make_memory_metadata

    stack = patchouli_chat_stack
    access = make_access_composition(
        [make_actor_access_record(agent_id=agent) for agent in ("omni_doll", "system")],
        adapters=("http",),
    )
    workspace = access.default_workspace
    runtime = make_workspace_read_view(stack.bus)
    runtime.subscribe(stack.bus)
    entry = WorkspaceOperationEntry(
        ActorMemoryService(
            stack.bus,
            operation_authorizer=access.authorizer,
            memory_reader=runtime.memory,
            threads=access.threads,
        ),
        agent=ActorAgentService(
            operation_authorizer=access.authorizer,
            profile_reader=runtime.profiles,
        ),
        threads=access.threads,
    )
    for alias, size in (("fact_context", 4000),):
        await stack.mid_term.upsert(
            MemoryAtom(
                meta=make_memory_metadata(
                    user_id=workspace.owner_user_id, source_agent_id="omni_doll"
                ),
                index=IndexLayer(
                    title="可重取资料", summary="检索材料", alias=alias, memory_type=MemoryType.FACT
                ),
                payload=PayloadLayer(content="EVIDENCE:" + "z" * size),
            )
        )
    answers = [f"第 {index} 轮结论：" + "a" * 4000 for index in range(5)]
    script = iter(
        [item for answer in answers for item in ("⟪ READ | fact_context | ⟫", answer)]
        + ["继续依据历史完成", '⟪ RUN | sys_read_file | path="large.txt" ⟫']
    )
    generation_inputs = []
    summary_inputs = []

    async def summarize(messages, _params):
        """只替换 LLM 服务；实际摘要校验、MTP、预算和会话提交都通过生产路径。"""
        summary_inputs.append(messages)
        return "omni_doll 已检索前序资料；继续验证未完成事项。"

    async def generate(messages, _params):
        generation_inputs.append([dict(item) for item in messages])
        yield next(script)

    llm = CallbackLLMService(complete=summarize, stream=generate)
    config = HiveMemoryConfig().alice
    config.koakuma.workspace_path = str(tmp_path)
    (tmp_path / "large.txt").write_text("z" * 100000)
    config.runtime.default_context_window = 14000
    config.runtime.safety_margin = 0
    config.compaction = CompactionConfig(
        trigger_ratio=0.65,
        target_ratio=0.45,
        recent_raw_ratio=0.15,
        summary_max_tokens=128,
    )
    publisher = RuntimeEventPublisher(stack.events)
    sessions = ConversationSessionService(WorkspaceSessionConfig())
    alice = AliceSystem(
        config=config, operation_entry=entry, event_publisher=publisher, llm_service=llm
    )
    service = make_task_process_service(
        stack.bus,
        cpu=alice.cpu_port,
        sessions=sessions,
        access_gateway=access.gateway,
        operation_authorizer=access.authorizer,
        process_table=access.process_table,
        workspace_read_view=runtime,
        operation_entry=entry,
        event_publisher=publisher,
    )
    app = FastAPI()
    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(sessions_router, prefix="/api/v1")
    app.dependency_overrides[deps.get_process_service] = lambda: service
    app.dependency_overrides[deps.get_access_gateway] = lambda: access.gateway
    app.dependency_overrides[deps.get_server_principal_id] = lambda: access.principal.principal_id
    root = RootSessionService(sessions, operation_authorizer=access.authorizer)
    app.dependency_overrides[deps.get_root_session_service] = lambda: root
    await alice.start()
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers={
                "X-User-ID": workspace.owner_user_id,
                "X-Workspace-ID": workspace.workspace_id,
            },
        ) as client:
            created = await client.post("/api/v1/sessions", json={"title": "长会话"})
            assert created.status_code == 201
            session_id = created.json()["session_id"]

            async def chat(message):
                response = await client.post(
                    "/api/v1/chat",
                    json={
                        "message": message,
                        "session_id": session_id,
                        "agent_id": "omni_doll",
                        "enable_memory_retrieval": False,
                        "generation_options": {"model": "fake-test-model", "max_tokens": 1000},
                    },
                )
                assert response.status_code == 200
                return events_from_text(response.text)

            rounds = [await chat(f"轮次 {index}：" + "u" * 4000) for index in range(5)]
            assert [events[-1][1]["status"] for events in rounds] == ["completed"] * 5
            compacted = (await client.get(f"/api/v1/sessions/{session_id}")).json()
            checkpoint = compacted["compaction_checkpoint"]
            assert checkpoint["summary"] == "omni_doll 已检索前序资料；继续验证未完成事项。"
            assert checkpoint["references"] == ["memory:fact_context"]
            assert [
                turn["payload"]["assistant_final_text"] for turn in compacted["turns"]
            ] == answers
            old_results = [
                event["result"]
                for turn in compacted["turns"]
                for event in turn["payload"]["turn_events"]
                if event["kind"] == "tool_call"
            ]
            assert len(old_results) == 5
            assert old_results == [
                generation_inputs[index][-1]["content"] for index in (1, 3, 5, 7, 9)
            ]
            progress = [
                data for events in rounds for name, data in events if name == "context_compaction"
            ]
            assert [data["status"] for data in progress] == ["started", "completed"]
            assert progress[-1]["layer"] == "summary"
            assert len(summary_inputs) == 1

            resumed = await chat("继续")
            assert resumed[-1][1]["status"] == "completed"
            prompt = "\n".join(item["content"] for item in generation_inputs[-1])
            assert "[较早对话摘要]" in prompt
            assert checkpoint["summary"] in prompt
            assert "可重新 READ" in prompt
            assert "轮次 0：" not in prompt
            assert "轮次 4：" in prompt
            assert "compaction_checkpoint" not in resumed[-1][1]

            failed = await chat("读取大资料")
            assert failed[-1][1]["status"] == "failed"
            assert failed[-1][1]["failure_reason"] == "turn_context_budget_exceeded"
            assert len(generation_inputs) == 12
            loaded = (await client.get(f"/api/v1/sessions/{session_id}")).json()
            assert len(loaded["turns"]) == 7
            assert loaded["turns"][-1]["payload"]["outcome"] == "failed"
            assert loaded["active_process_id"] is None
            (guard,) = [
                e for e in stack.events.events if e.event_type == "agent.context.budget_exceeded"
            ]
            assert guard.process_id == failed[-1][1]["process_id"]
            assert guard.data["input_token_limit"] == 13000
    finally:
        await service.shutdown()
        await alice.stop()
        runtime.close()
        access.gateway.close()
        access.gateway.revoke_all_contexts()
