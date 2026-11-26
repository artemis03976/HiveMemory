"""PatchouliLocalRoutes 的类型化路由契约。"""

# route 声明在类体中生成常量，不是实例方法。
# ruff: noqa: N802, N805

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.contracts.route import route

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from typing import Any
    from uuid import UUID

    from hivememory.core.models import (
        ActorIdentity,
        AgentProfile,
        MemoryAccessPolicy,
        MemoryAtom,
        MemoryType,
        TopicData,
        TopicSnapshot,
        WorkspaceIdentity,
    )
    from hivememory.core.models.pending import PendingAtomMaterializeTask
    from hivememory.engines.perception.models import TopicSettleMaterial
    from hivememory.engines.retrieval.models import RetrievalQuery
    from hivememory.patchouli.contracts.memory_management import MemoryListPage
    from hivememory.patchouli.contracts.memory_tasks import (
        MemoryGenerationAdmission,
        MemoryGenerationTask,
    )
    from hivememory.patchouli.contracts.topic_management import (
        TopicDeleteResult,
        TopicSettleResult,
    )
    from hivememory.patchouli.control.memory_generation.models import (
        MemoryGenerationResult,
        MemoryGenerationTaskSpec,
    )


class PatchouliLocalRoutes:
    """PatchouliBus 内部路由名。

    局部路由描述可组合的 Patchouli 原语；对外的用例由公开路由组合它们。
    """

    @route("memory.create")
    async def MEMORY_CREATE(
        belong_to: WorkspaceIdentity,
        *,
        from_actor: ActorIdentity,
        memory_type: MemoryType,
        title: str,
        summary: str = "",
        content: str = "",
        tags: list[str] | None = None,
        alias: str | None = None,
        access_policy: MemoryAccessPolicy | None = None,
    ) -> MemoryAtom:
        """以外部提交的材料创建记忆原子并持久化，原子的身份与时间在此确定，策略缺省为 PUBLIC。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("memory.list")
    async def MEMORY_LIST(
        *,
        belong_to: WorkspaceIdentity,
        from_actor: ActorIdentity,
        query: str | None = None,
        filters: dict[str, Any] | None = None,
        exclude_types: list[str] | None = None,
        limit: int = 20,
        cursor: str | None = None,
        enforce_actor_visibility: bool = True,
    ) -> MemoryListPage:
        """根据查询和过滤条件列出一页中期记忆原子。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("memory.get")
    async def MEMORY_GET(
        memory_id: UUID | str,
        *,
        belong_to: WorkspaceIdentity,
        from_actor: ActorIdentity,
        enforce_actor_visibility: bool = True,
    ) -> MemoryAtom | None:
        """根据记忆 ID 读取中期记忆原子。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("memory.update")
    async def MEMORY_UPDATE(
        memory_id: UUID,
        *,
        belong_to: WorkspaceIdentity,
        title: str | None = None,
        summary: str | None = None,
        content: str | None = None,
        alias: str | None = None,
        tags: list[str] | None = None,
        access_policy: MemoryAccessPolicy | None = None,
    ) -> MemoryAtom | None:
        """对外部手动的记忆编辑进行持久化处理；``access_policy`` 为 ``None`` 时不修改策略。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("memory.delete")
    async def MEMORY_DELETE(belong_to: WorkspaceIdentity, memory_id: UUID) -> bool:
        """删除中期记忆（供公共路由绑定）。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("memory.retrieve")
    async def MEMORY_RETRIEVE(query: RetrievalQuery) -> list[MemoryAtom]:
        """异步总线入口：只执行检索与活跃度刷新。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("memory.retrieve_by_aliases")
    async def MEMORY_RETRIEVE_BY_ALIASES(
        aliases: list[str], belong_to: WorkspaceIdentity, *, from_actor: ActorIdentity
    ) -> list[MemoryAtom]:
        """精确别名检索的异步总线入口。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("memory.record_hit")
    async def MEMORY_RECORD_HIT(
        memory_id: UUID | str, *, belong_to: WorkspaceIdentity, source: str = "system"
    ) -> Any:
        """记录一次命中事件。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("memory.record_citation")
    async def MEMORY_RECORD_CITATION(
        memory_id: UUID | str, *, belong_to: WorkspaceIdentity, source: str = "system"
    ) -> Any:
        """记录一次引用事件。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("memory.record_feedback")
    async def MEMORY_RECORD_FEEDBACK(
        memory_id: UUID | str, *, belong_to: WorkspaceIdentity, positive: bool, source: str = "user"
    ) -> Any:
        """记录用户反馈事件。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("memory.revive")
    async def MEMORY_REVIVE(memory_id: UUID | str, *, belong_to: WorkspaceIdentity) -> None:
        """从长期存储复活记忆到中期存储。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("lifecycle.refresh_memory_vitality")
    async def REFRESH_MEMORY_VITALITY(
        memories: Iterable[MemoryAtom], persist: bool = False
    ) -> list[tuple[UUID, float]]:
        """批量刷新记忆生命力。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("lifecycle.run_gardening_once")
    async def LIFECYCLE_RUN_GARDENING_ONCE() -> dict[str, Any]:
        """执行一次生命周期垃圾回收，供全局调度器调用。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("generation.submit_settlement")
    async def GENERATION_SUBMIT_SETTLEMENT(
        material: TopicSettleMaterial,
    ) -> MemoryGenerationTask | None:
        """将话题的沉淀材料转为 SETTLE 任务，统一以归属对应的 system 为发起者。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("generation.submit_active")
    async def GENERATION_SUBMIT_ACTIVE(
        task: PendingAtomMaterializeTask, topic_id: str, *, belong_to: WorkspaceIdentity
    ) -> MemoryGenerationAdmission:
        """将一个 WRITE/UPDATE 意图转为主动生成任务，返回它的接纳结果。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("generation.execute_spec")
    async def GENERATION_EXECUTE_SPEC(
        spec: MemoryGenerationTaskSpec,
        *,
        on_commit_started: Callable[[], None],
    ) -> list[MemoryGenerationResult]:
        """执行统一生成任务规范，只返回结果不发布事件。

        开始写入记忆之前同步调用 ``on_commit_started``；此后的写入不被取消打断。
        """
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("memory_task.submit_generation")
    async def MEMORY_TASK_SUBMIT_GENERATION(
        spec: MemoryGenerationTaskSpec,
    ) -> MemoryGenerationTask:
        """提交一个不可变工作定义，并返回其接纳快照。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("memory_task.admit_generation")
    async def MEMORY_TASK_ADMIT_GENERATION(
        spec: MemoryGenerationTaskSpec,
    ) -> MemoryGenerationAdmission:
        """接纳一项生成任务，返回区分接纳、拒绝与结果未知的接纳结果。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("memory_task.list")
    async def MEMORY_TASK_LIST() -> list[MemoryGenerationTask]:
        """列出控制器当前仍保留的任务快照。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("memory_task.get")
    async def MEMORY_TASK_GET(task_id: str) -> MemoryGenerationTask | None:
        """按任务标识查询当前只读领域快照。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("memory_task.cancel")
    async def MEMORY_TASK_CANCEL(task_id: str, *, reason: str = "user_requested") -> bool:
        """请求取消指定任务，并发布被接纳的取消请求快照。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("memory_task.wait")
    async def MEMORY_TASK_WAIT(
        task_id: str, timeout: float | None = None
    ) -> MemoryGenerationTask | None:
        """等待指定任务并返回最新快照；不存在时返回 ``None``。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("memory_task.wait_many")
    async def MEMORY_TASK_WAIT_MANY(
        task_ids: list[str], timeout: float | None = None
    ) -> list[MemoryGenerationTask | None]:
        """在共享超时窗口内按请求顺序返回任务最新快照。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("memory_task.wait_all")
    async def MEMORY_TASK_WAIT_ALL(
        timeout: float | None = None,
    ) -> list[MemoryGenerationTask]:
        """等待调用时所有尚未进入领域终态的记忆生成任务。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("memory.get_agent_profile")
    async def GET_AGENT_PROFILE(agent_alias: str | None) -> AgentProfile:
        """Profile 解析的唯一实现：内置 Profile、按 agent_id 读取文件与三种失败原因。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("topic.prepare")
    async def TOPIC_PREPARE(
        target_topic_id: str,
        new_topic_title: str | None,
        new_topic_summary: str | None,
        belong_to: WorkspaceIdentity,
    ) -> str:
        """确保目标短期话题存在（建新话题时池满，先让最久没用的话题退役），返回真实 topic_id。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("topic.get")
    async def TOPIC_GET(topic_id: str, *, belong_to: WorkspaceIdentity) -> TopicData | None:
        """读取短期话题上下文（纯读，无访问追踪副作用）。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("topic.get_generated_memories")
    async def TOPIC_GET_GENERATED_MEMORIES(
        topic_id: str, *, belong_to: WorkspaceIdentity
    ) -> frozenset[str] | None:
        """读取本话题生成的记忆；退役话题的日志删除之前同样交出，话题不存在时为空值。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("topic.record_generated_memories")
    async def TOPIC_RECORD_GENERATED_MEMORIES(
        topic_id: str,
        *,
        belong_to: WorkspaceIdentity,
        generated: frozenset[str],
        absorbed_into: Mapping[str, str],
    ) -> None:
        """把一次生成执行后本话题新生成的记忆与合并改指记到话题上；退役话题的日志删除之前照样记。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("topic.record_settled")
    async def TOPIC_RECORD_SETTLED(
        topic_id: str, *, belong_to: WorkspaceIdentity, settled_until: int
    ) -> None:
        """一次沉淀到达终态，在话题的日志中记下它覆盖到第几轮；话题已被删除时不记。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("topic.list_active")
    async def TOPIC_LIST_ACTIVE(
        *, belong_to: WorkspaceIdentity, include_empty: bool = False, sort_by_recency: bool = True
    ) -> tuple[TopicSnapshot, ...]:
        """列出指定用户的话题快照（短期检索入口）。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("topic.settle_and_retire")
    async def TOPIC_SETTLE_AND_RETIRE(
        belong_to: WorkspaceIdentity, topic_id: str
    ) -> TopicSettleResult:
        """手动沉淀并退役指定话题。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("topic.delete")
    async def TOPIC_DELETE(belong_to: WorkspaceIdentity, topic_id: str) -> TopicDeleteResult:
        """删除话题的日志与内存中的状态，不沉淀；正被占用时报 ``removed=False``。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("runtime.models.warmup")
    async def RUNTIME_MODELS_WARMUP() -> None:
        """后台预热所有推理模型 (Embedding + Reranker)"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("runtime.models.prefetch")
    async def RUNTIME_MODELS_PREFETCH() -> None:
        """在后台开始加载尚未加载的推理模型并立即返回"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("runtime.models.ready")
    async def RUNTIME_MODELS_READY() -> bool:
        """检查推理模型是否就绪：每个启用的模型至少成功加载过一次，且当前不处于加载失败状态"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("runtime.models.failed")
    async def RUNTIME_MODELS_FAILED() -> list[str]:
        """列出最近一次加载失败的推理模型名；重新开始加载后不再列出"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("runtime.storage_health")
    async def RUNTIME_STORAGE_HEALTH() -> bool:
        """存储层健康检查"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    ALL = (
        MEMORY_CREATE,
        MEMORY_LIST,
        MEMORY_GET,
        MEMORY_UPDATE,
        MEMORY_DELETE,
        MEMORY_RETRIEVE,
        MEMORY_RETRIEVE_BY_ALIASES,
        MEMORY_RECORD_HIT,
        MEMORY_RECORD_CITATION,
        MEMORY_RECORD_FEEDBACK,
        MEMORY_REVIVE,
        REFRESH_MEMORY_VITALITY,
        LIFECYCLE_RUN_GARDENING_ONCE,
        GENERATION_SUBMIT_SETTLEMENT,
        GENERATION_SUBMIT_ACTIVE,
        GENERATION_EXECUTE_SPEC,
        MEMORY_TASK_SUBMIT_GENERATION,
        MEMORY_TASK_ADMIT_GENERATION,
        MEMORY_TASK_LIST,
        MEMORY_TASK_GET,
        MEMORY_TASK_CANCEL,
        MEMORY_TASK_WAIT,
        MEMORY_TASK_WAIT_MANY,
        MEMORY_TASK_WAIT_ALL,
        GET_AGENT_PROFILE,
        TOPIC_PREPARE,
        TOPIC_GET,
        TOPIC_GET_GENERATED_MEMORIES,
        TOPIC_RECORD_GENERATED_MEMORIES,
        TOPIC_RECORD_SETTLED,
        TOPIC_LIST_ACTIVE,
        TOPIC_SETTLE_AND_RETIRE,
        TOPIC_DELETE,
        RUNTIME_MODELS_WARMUP,
        RUNTIME_MODELS_PREFETCH,
        RUNTIME_MODELS_READY,
        RUNTIME_MODELS_FAILED,
        RUNTIME_STORAGE_HEALTH,
    )
