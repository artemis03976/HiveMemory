"""HTTP chat 按 actor 操作白名单执行。

越权动词在登记意图前被拒绝；无 actor 操作许可时仍完成阶段编排并保留资源可见性。
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from hivememory.alice.system import AliceSystem
from hivememory.config.app import HiveMemoryConfig
from hivememory.core.access import WorkspaceAccessContext, WorkspaceOperation
from hivememory.core.models import (
    ActorIdentity,
    AttachmentSelectionRequest,
    IdentityScope,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MemoryVisibility,
    PayloadLayer,
)
from hivememory.server import deps
from hivememory.server.routers.chat import router
from hivememory.workspace.assets.store import InMemoryWorkspaceAssetStore
from hivememory.workspace.capability.actor.agent_profiles import ActorAgentService
from hivememory.workspace.capability.actor.memory import ActorMemoryService
from hivememory.workspace.capability.operations import WorkspaceOperationEntry
from hivememory.workspace.contracts import (
    CPUInputManifest,
    CPUPort,
    CPURunHandle,
)
from tests.helpers.memory import make_memory_metadata
from tests.helpers.process import make_task_process_service
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
    make_workspace_runtime,
)
from tests.helpers.workspace_assets import make_ready_text_asset

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_http_read_search_agent_rejects_model_write_without_registering_intent(
    monkeypatch, patchouli_chat_stack
):
    """HTTP 白名单贯穿清单、提示词和真实 MTP；模型越权 WRITE 不登记任何意图。"""
    stack = patchouli_chat_stack
    workspace = make_workspace_identity(owner_user_id="u1")
    actor = ActorIdentity(user_id="u1", agent_id="read_search_agent")
    access = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1",
                agent_id=actor.agent_id,
                allowed_operations=frozenset(
                    {WorkspaceOperation.MEMORY_READ, WorkspaceOperation.MEMORY_SEARCH}
                ),
            )
        ],
        adapters=("http",),
        default_workspace=workspace,
    )
    await stack.mid_term.upsert(
        MemoryAtom(
            meta=make_memory_metadata(user_id="u1", source_agent_id=actor.agent_id),
            index=IndexLayer(
                alias=actor.agent_id, title="只读执行图纸", memory_type=MemoryType.AGENT_PROFILE
            ),
            payload=PayloadLayer(
                content="根据已有资料回答",
                # 旧客户端残留全量许可也不能扩大访问登记的能力。
                agent_config={
                    "model_name": "fake-test-model",
                    "allowed_mtp_verbs": None,
                    "allowed_sys_tools": None,
                },
            ),
        )
    )
    runtime = make_workspace_runtime(stack.bus)
    entry = WorkspaceOperationEntry(
        ActorMemoryService(
            stack.bus,
            operation_authorizer=access.authorizer,
            memory_reader=runtime.aliases,
            dispatch_topics=access.thread_spawner,
        ),
        agent=ActorAgentService(
            operation_authorizer=access.authorizer,
            thread_spawner=access.thread_spawner,
            profile_reader=runtime.profiles,
        ),
        operation_authorizer=access.authorizer,
    )
    llm_messages = []
    responses = iter(['⟪ WRITE | * | content="越权写入" ⟫', "权限拒绝后正常回答"])

    async def completion(*, messages, **_kwargs):
        """只替换 LLM 协议；MTP 解析、策略、操作入口和 finalize 都走真实链路。"""
        llm_messages.append([dict(item) for item in messages])
        text = next(responses)

        async def chunks():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=text),
                        finish_reason="stop",
                    )
                ]
            )

        return chunks()

    monkeypatch.setattr("hivememory.agent_runtime.model.client.litellm.acompletion", completion)
    alice = AliceSystem(config=HiveMemoryConfig().alice, operation_entry=entry)
    service = make_task_process_service(
        stack.bus,
        cpu=alice.cpu_port,
        access_gateway=access.gateway,
        operation_authorizer=access.authorizer,
        process_table=access.process_table,
        workspace_runtime=runtime,
        operation_entry=entry,
    )
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[deps.get_process_service] = lambda: service
    app.dependency_overrides[deps.get_server_principal_id] = lambda: access.principal.principal_id
    await alice.start()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await asyncio.wait_for(
                client.post(
                    "/api/v1/chat",
                    headers={"X-User-ID": "u1", "X-Workspace-ID": workspace.workspace_id},
                    json={
                        "message": "根据已有资料回答",
                        "agent_id": actor.agent_id,
                        "generation_options": {"model": "fake-test-model"},
                    },
                ),
                timeout=10,
            )
        assert response.status_code == 200
        events = []
        event_name = ""
        for line in response.text.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                events.append((event_name, json.loads(line.removeprefix("data: "))))
        terminal, done = events[-1]
        assert terminal == "done"
        assert done["status"] == "completed"
        assert done["final_text"] == "权限拒绝后正常回答"
        initial, resumed = llm_messages
        prompt = initial[0]["content"]
        assert "READ" in prompt and "SEARCH" in prompt and "RUN" in prompt
        assert "WRITE" not in prompt and "UPDATE" not in prompt and "CALL" not in prompt
        assert "sys_write_file" in prompt
        # 结构化权限错误在真实交互记录中验收，不能只依赖自然语言回复。
        topic = stack.short_term.get(workspace, done["topic_id"])
        (block,) = topic.blocks
        (tool_result,) = [event for event in block.turn.turn_events if event.kind == "tool_result"]
        assert tool_result.status == "error"
        assert '<error code="mtp.permission.denied"' in tool_result.content
        assert "WRITE" in resumed[-1]["content"]
        assert runtime.intents.size == 0
    finally:
        await alice.stop()
        runtime.close()
        access.gateway.close()
        access.gateway.revoke_all_contexts()


class _RecordingCPU:
    """记录公开 CPU 端口的输入，全部执行仍交给真实 Alice CPU。"""

    def __init__(self, delegate: CPUPort) -> None:
        self._delegate = delegate
        self.manifests: list[CPUInputManifest] = []

    def execute(
        self,
        manifest: CPUInputManifest,
        *,
        access: WorkspaceAccessContext,
        generation_options: dict[str, Any] | None,
        stream: bool,
    ) -> CPURunHandle:
        self.manifests.append(manifest)
        return self._delegate.execute(
            manifest,
            access=access,
            generation_options=generation_options,
            stream=stream,
        )


@pytest.mark.asyncio
async def test_http_chat_without_actor_operations_completes_with_actor_visible_inputs(
    monkeypatch, patchouli_chat_stack
):
    """空白名单不阻断阶段，Profile、附件和预检索仍按进程 actor 校验归属与可见性。"""
    stack = patchouli_chat_stack
    workspace = make_workspace_identity(owner_user_id="u1")
    actor = ActorIdentity(user_id="u1", agent_id="agent_no_operations")
    access = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1",
                agent_id=actor.agent_id,
                allowed_operations=frozenset(),
            )
        ],
        adapters=("http",),
        default_workspace=workspace,
    )
    profile = MemoryAtom(
        meta=make_memory_metadata(
            user_id="u1", source_agent_id=actor.agent_id, visibility=MemoryVisibility.PRIVATE
        ),
        index=IndexLayer(
            alias=actor.agent_id,
            title="无操作许可的执行 Profile",
            memory_type=MemoryType.AGENT_PROFILE,
        ),
        payload=PayloadLayer(
            content="仅生成本轮回复的专属人设",
            agent_config={"model_name": "fake-test-model", "allowed_mtp_verbs": []},
        ),
    )
    await stack.mid_term.upsert(profile)
    memories = []
    for alias, source_agent, visibility, content in (
        ("fact_public", "other_agent", MemoryVisibility.PUBLIC, "本轮公共资料"),
        ("fact_own", actor.agent_id, MemoryVisibility.PRIVATE, "本轮专属资料"),
        ("fact_foreign", "other_agent", MemoryVisibility.PRIVATE, "他人的私有资料"),
    ):
        atom = MemoryAtom(
            meta=make_memory_metadata(
                user_id="u1", source_agent_id=source_agent, visibility=visibility
            ),
            index=IndexLayer(
                alias=alias, title=alias, summary=content, memory_type=MemoryType.FACT
            ),
            payload=PayloadLayer(content=content),
        )
        await stack.mid_term.upsert(atom)
        memories.append(atom)
    message = "查找本轮可见资料"
    # 外部向量服务返回全部候选，真实资源所有者负责剔除其他 actor 的 PRIVATE 记忆。
    stack.vector_store.search_matches[message] = tuple(atom.index.alias for atom in memories)
    assets = InMemoryWorkspaceAssetStore()
    asset_ref = make_ready_text_asset(
        assets,
        IdentityScope(actor_identity=actor, workspace_identity=workspace),
        operation_id="stage-chat-attachment",
        content="本轮选中的附件正文",
    )
    runtime = make_workspace_runtime(stack.bus)
    entry = WorkspaceOperationEntry(
        ActorMemoryService(
            stack.bus,
            operation_authorizer=access.authorizer,
            memory_reader=runtime.aliases,
            dispatch_topics=access.thread_spawner,
        ),
        agent=ActorAgentService(
            operation_authorizer=access.authorizer,
            thread_spawner=access.thread_spawner,
            profile_reader=runtime.profiles,
        ),
        operation_authorizer=access.authorizer,
    )
    llm_messages = []

    async def completion(*, messages, **_kwargs):
        """替换外部 LLM 流式协议；输入编译、Alice 循环与交互封口仍走真实组件。"""
        llm_messages.append([dict(item) for item in messages])

        async def chunks():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="阶段授权验收完成"), finish_reason="stop"
                    )
                ]
            )

        return chunks()

    monkeypatch.setattr("hivememory.agent_runtime.model.client.litellm.acompletion", completion)
    alice = AliceSystem(
        config=HiveMemoryConfig().alice,
        operation_entry=entry,
    )
    cpu = _RecordingCPU(alice.cpu_port)
    service = make_task_process_service(
        stack.bus,
        cpu=cpu,
        access_gateway=access.gateway,
        operation_authorizer=access.authorizer,
        asset_reader=assets,
        process_table=access.process_table,
        workspace_runtime=runtime,
        operation_entry=entry,
    )
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[deps.get_process_service] = lambda: service
    app.dependency_overrides[deps.get_server_principal_id] = lambda: access.principal.principal_id
    await alice.start()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await asyncio.wait_for(
                client.post(
                    "/api/v1/chat",
                    headers={"X-User-ID": "u1", "X-Workspace-ID": workspace.workspace_id},
                    json={
                        "message": message,
                        "agent_id": actor.agent_id,
                        "generation_options": {"model": "fake-test-model"},
                        "attachments": [
                            AttachmentSelectionRequest(asset_ref=asset_ref).model_dump(mode="json")
                        ],
                    },
                ),
                timeout=10,
            )
        assert response.status_code == 200
        events = []
        event_name = ""
        for line in response.text.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                events.append((event_name, json.loads(line.removeprefix("data: "))))
        terminal, done = events[-1]
        assert terminal == "done"
        assert done["status"] == "completed"
        assert done["final_text"] == "阶段授权验收完成"

        # 端口输入同时保护检索结果与编译正文，不能只检查模型回复是否成功。
        (manifest,) = cpu.manifests
        assert {atom.index.alias for atom in manifest.memories} == {"fact_public", "fact_own"}
        assert "本轮公共资料" in manifest.turn_context
        assert "本轮专属资料" in manifest.turn_context
        assert "他人的私有资料" not in manifest.turn_context
        assert manifest.agent_profile.agent_id == actor.agent_id
        assert manifest.agent_profile.persona == profile.payload.content
        assert "本轮选中的附件正文" in manifest.turn_context
        (messages,) = llm_messages
        assert profile.payload.content in messages[0]["content"]
        assert "他人的私有资料" not in "\n".join(item["content"] for item in messages)

        # finalize 后真实话题记录保留本轮 actor，阶段授权没有把发起者改成 system。
        topic = stack.short_term.get(workspace, done["topic_id"])
        (block,) = topic.blocks
        assert block.turn.identity == actor
        assert block.turn.user_query == message
        assert block.turn.assistant_final_text == "阶段授权验收完成"
        assert assets.close_and_clear().leases_cleared == 0
    finally:
        await alice.stop()
        runtime.close()
        assets.close_and_clear()
        access.gateway.close()
        access.gateway.revoke_all_contexts()
