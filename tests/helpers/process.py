"""任务进程测试的装配辅助。

与生产组合根同构：编排依赖（总线、CPU 端口、CPU 分配器、操作授权者、
线程管理、操作入口、root 交互与记忆 API、Gateway 超时）只交给执行器
``TaskProcessRunner``；线程管理经真实 root 话题 API 确定话题；CPU 持有真实
操作入口，以每次执行收到的线程句柄绑定提交函数。注册入口
``TaskProcessService`` 只持有执行器与生命周期依赖（认证网关、系统自身的
principal、取消的 root 授权、事件发布器）。

:class:`PatchouliTurnStub` 是进程一轮所用的 Patchouli 公开路由的替身：预检索、
存储健康、话题准备、检索命中与交互提交，按真实契约回答并逐次记录参数。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.config.attachments import AttachmentCompilerConfig
from hivememory.config.workspace import WorkspaceSessionConfig
from hivememory.core.access import CallerPrincipal
from hivememory.core.models import MemoryAtom
from hivememory.core.ports.workspace_assets import WorkspaceAssetReaderPort
from hivememory.core.protocol.models import RetrievalRequest
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.interaction_submission import (
    InteractionProgress,
    InteractionStatus,
    InteractionSubmitResult,
)
from hivememory.system.access import LOCAL_ADAPTER, SYSTEM_PRINCIPAL_ID
from hivememory.workspace.assets.store import InMemoryWorkspaceAssetStore
from hivememory.workspace.authentication import ActorAuthenticationGateway
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
from hivememory.workspace.capability.actor.agent_profiles import ActorAgentService
from hivememory.workspace.capability.actor.memory import ActorMemoryService
from hivememory.workspace.capability.backing import BusCanonicalReadBackend
from hivememory.workspace.capability.operations import WorkspaceOperationEntry
from hivememory.workspace.capability.root.interaction import RootInteractionService
from hivememory.workspace.capability.root.memory import RootMemoryService
from hivememory.workspace.capability.root.topic import RootTopicService
from hivememory.workspace.contracts import CPUPort, OperationEntry
from hivememory.workspace.process.allocation import CPUAllocator
from hivememory.workspace.process.runner import TaskProcessRunner
from hivememory.workspace.process.service import ProcessHandle, TaskProcessService
from hivememory.workspace.process.table import ProcessTable
from hivememory.workspace.process.threads import ProcessThreads
from hivememory.workspace.read_view import WorkspaceReadView
from hivememory.workspace.sessions import ConversationSessionService
from tests.helpers.cpu import ScriptedCPU
from tests.helpers.intent_dispatch import install_intent_admission
from tests.helpers.workspace import INTENT_RECORD_FILE, INTENT_RETENTION


def make_task_process_service(
    global_bus: Any,
    *,
    cpu: CPUPort,
    access_gateway: ActorAuthenticationGateway,
    operation_authorizer: WorkspaceOperationAuthorizer,
    event_publisher: RuntimeEventPublisher | None = None,
    asset_reader: WorkspaceAssetReaderPort | None = None,
    attachment_compiler_config: AttachmentCompilerConfig | None = None,
    workspace_read_view: WorkspaceReadView | None = None,
    operation_entry: OperationEntry | None = None,
    process_table: ProcessTable | None = None,
    sessions: ConversationSessionService | None = None,
    gateway_session_turn_limit: int = 3,
    shutdown_wait_seconds: float = 10.0,
) -> TaskProcessService:
    """按生产顺序装配能力与操作入口，再构建 CPU 分配器、执行器与注册入口。

    注册入口与生产一样以系统自身的 principal 经 ``local`` adapter 认证系统的
    root 工作 actor：认证网关须登记该 principal（``make_access_composition``
    缺省登记），Workspace 访问登记须有该用户 ``system`` 的记录。
    """
    if (
        isinstance(global_bus, GlobalSystemBus)
        and PatchouliRoutes.MEMORY_INTENT_SUBMIT not in global_bus.list_routes()
    ):
        install_intent_admission(global_bus)
    runtime = workspace_read_view or WorkspaceReadView(
        backing=BusCanonicalReadBackend(global_bus),
        atom_capacity=64,
        profile_capacity=32,
        intent_record_file=INTENT_RECORD_FILE,
        intent_retention=INTENT_RETENTION,
    )
    table = process_table if process_table is not None else ProcessTable()
    threads = ProcessThreads(
        table,
        admission=access_gateway,
        authorizer=operation_authorizer,
        topics=RootTopicService(global_bus, operation_authorizer=operation_authorizer),
    )
    memory = ActorMemoryService(
        global_bus,
        operation_authorizer=operation_authorizer,
        memory_reader=runtime.memory,
        threads=threads,
    )
    agent = ActorAgentService(
        operation_authorizer=operation_authorizer,
        profile_reader=runtime.profiles,
    )
    entry = operation_entry or WorkspaceOperationEntry(memory, agent=agent, threads=threads)
    if isinstance(cpu, ScriptedCPU):
        cpu.bind_operation_entry(entry)
    allocator = CPUAllocator(
        operation_authorizer=operation_authorizer,
        profile_reader=runtime.profiles,
        asset_reader=asset_reader or InMemoryWorkspaceAssetStore(),
        attachment_compiler_config=attachment_compiler_config,
    )
    sessions = sessions or ConversationSessionService(WorkspaceSessionConfig())
    runner = TaskProcessRunner(
        global_bus,
        cpu=cpu,
        sessions=sessions,
        allocator=allocator,
        operation_authorizer=operation_authorizer,
        threads=threads,
        operations=entry,
        interactions=RootInteractionService(
            global_bus,
            operation_authorizer=operation_authorizer,
            gateway_request_timeout_ms=8000,
        ),
        memories=RootMemoryService(global_bus, operation_authorizer=operation_authorizer),
        gateway_session_turn_limit=gateway_session_turn_limit,
    )
    return TaskProcessService(
        runner,
        process_table=table,
        sessions=sessions,
        access_gateway=access_gateway,
        operation_authorizer=operation_authorizer,
        system_principal=CallerPrincipal(SYSTEM_PRINCIPAL_ID),
        system_adapter=LOCAL_ADAPTER,
        event_publisher=event_publisher,
        shutdown_wait_seconds=shutdown_wait_seconds,
    )


_NEW_TOPIC = "NEW_TOPIC"


@dataclass
class PatchouliTurnStub:
    """任务进程一轮所用的 Patchouli 公开路由替身，按真实契约回答并逐次记录。

    - 预检索（``memory.retrieve``）返回 ``memories``，原子须属于被测 Workspace
      并对执行者可读，否则读取视图按真实规则丢弃；
    - 存储健康返回 ``storage_available``；
    - 话题准备：已有话题原样返回，``NEW_TOPIC`` 建立为 ``new_topic_id``；
    - 交互提交：进度查询恒为未接纳；提交按 ``submit_state`` 给出结局，成功时
      实际所在的话题是请求的话题（``NEW_TOPIC`` 解析为 ``new_topic_id``）；
    - 检索命中只记录。

    需要不同回答的测试在 :meth:`install` 之后以 ``bus.register`` 覆盖单条路由
    之前先 ``unregister``。
    """

    memories: list[MemoryAtom] = field(default_factory=list)
    storage_available: bool = True
    new_topic_id: str = "topic-new"
    submit_state: InteractionProgress = "succeeded"
    retrievals: list[RetrievalRequest] = field(default_factory=list)
    prepared_topics: list[dict[str, Any]] = field(default_factory=list)
    submissions: list[dict[str, Any]] = field(default_factory=list)
    hits: list[dict[str, Any]] = field(default_factory=list)

    def install(self, bus: GlobalSystemBus) -> PatchouliTurnStub:
        """在总线上登记六条路由，返回自身便于链式使用。"""
        bus.register(PatchouliRoutes.MEMORY_RETRIEVE, self._retrieve)
        bus.register(PatchouliRoutes.STORAGE_AVAILABLE, self._storage_available)
        bus.register(PatchouliRoutes.PREPARE_TOPIC, self._prepare_topic)
        bus.register(PatchouliRoutes.RECORD_RETRIEVAL_HITS, self._record_hits)
        bus.register(PatchouliRoutes.INTERACTION_STATUS, self._interaction_status)
        bus.register(PatchouliRoutes.INTERACTION_SUBMIT, self._submit)
        return self

    def _resolve_topic(self, topic_id: str) -> str:
        return self.new_topic_id if topic_id == _NEW_TOPIC else topic_id

    async def _retrieve(self, request: RetrievalRequest) -> list[MemoryAtom]:
        self.retrievals.append(request)
        return [memory.model_copy(deep=True) for memory in self.memories]

    async def _storage_available(self) -> bool:
        return self.storage_available

    async def _prepare_topic(self, **kwargs: Any) -> str:
        self.prepared_topics.append(kwargs)
        return self._resolve_topic(kwargs["target_topic_id"])

    async def _record_hits(self, memory_ids: Any, **kwargs: Any) -> None:
        self.hits.append({"memory_ids": tuple(memory_ids), **kwargs})

    async def _interaction_status(self, **_kwargs: Any) -> InteractionStatus | None:
        return None

    async def _submit(self, **kwargs: Any) -> InteractionSubmitResult:
        self.submissions.append(kwargs)
        requested = kwargs["requested_topic_id"]
        succeeded = self.submit_state == "succeeded"
        return InteractionSubmitResult(
            interaction_id=kwargs["interaction_id"],
            state=self.submit_state,
            topic_id=self._resolve_topic(requested) if succeeded else None,
            error_class=None if succeeded else "RuntimeError",
            new_topic=requested == _NEW_TOPIC,
        )


async def collect_events(
    service: TaskProcessService, handle: ProcessHandle
) -> list[dict[str, Any]]:
    """经流式交付把进程运行到结束，返回全部流式事件；最后一项读出时进程已经关闭。"""
    return [event async for event in service.run_process(handle)]


def done_data(events: list[dict[str, Any]]) -> dict[str, Any]:
    """取流式交付终态 ``done`` 的数据；最后一项不是 ``done`` 时以实际事件失败。"""
    assert events[-1]["event"] == "done", events[-1]
    return events[-1]["data"]


def error_code(events: list[dict[str, Any]]) -> str | None:
    """取流式交付终态 ``error`` 的错误码：Workspace 领域错误带码，其余错误为 ``None``。"""
    assert events[-1]["event"] == "error", events[-1]
    return events[-1]["data"].get("code")


def process_state(table: ProcessTable, process_id: str) -> tuple[str, str, str | None] | None:
    """读取已登记进程的 (阶段, 结局, 停止原因)；未登记或已注销时为 ``None``。"""
    process = table.get(process_id)
    if process is None:
        return None
    record = process.record
    return record.phase.value, record.outcome.value, record.stop_reason


async def wait_for_process_exit(
    table: ProcessTable, process_id: str, *, timeout: float = 2.0
) -> None:
    """等待已启动进程的进程 task 结束，不请求停止；进程已注销时立即返回。"""
    process = table.get(process_id)
    if process is None:
        return
    if process.task is None:
        raise AssertionError(f"进程尚未启动: {process_id}")
    await asyncio.wait_for(asyncio.wait((process.task,)), timeout)


__all__ = [
    "PatchouliTurnStub",
    "collect_events",
    "done_data",
    "error_code",
    "make_task_process_service",
    "process_state",
    "wait_for_process_exit",
]
