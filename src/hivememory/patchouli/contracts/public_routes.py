"""PatchouliRoutes 的类型化路由契约。"""

# route 声明在类体中生成常量，不是实例方法。
# ruff: noqa: N802, N805

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.contracts.route import route

if TYPE_CHECKING:
    from typing import Any
    from uuid import UUID

    from hivememory.core.models import (
        IdentityScope,
        MemoryAtom,
        ResolvedAgentProfile,
        TopicData,
        TopicSnapshot,
    )
    from hivememory.core.protocol.gateway import GatewayDecision
    from hivememory.core.protocol.models import InteractionPayload, RetrievalRequest
    from hivememory.patchouli.contracts.interaction_submission import (
        InteractionOrigin,
        InteractionStatus,
        InteractionSubmitResult,
    )
    from hivememory.patchouli.contracts.memory_intent import (
        MemoryIntent,
        MemoryIntentSubmissionResult,
    )
    from hivememory.patchouli.contracts.memory_management import (
        AgentProfileCreation,
        MemoryCreation,
        MemoryEdit,
    )
    from hivememory.patchouli.contracts.memory_tasks import MemoryGenerationTask
    from hivememory.patchouli.contracts.prepare import PreparedAgentRun
    from hivememory.patchouli.contracts.topic_management import (
        TopicEvictionResult,
        TopicSettleResult,
    )


class PatchouliRoutes:
    """由 Patchouli 提供的全局 RPC 名称与签名。"""

    @route("patchouli.public.memory.create")
    async def MEMORY_CREATE(identity_scope: IdentityScope, creation: MemoryCreation) -> MemoryAtom:
        """以调用方提交的材料创建属于目标 Workspace 的记忆，返回持久化后的原子。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.memory.list")
    async def MEMORY_LIST(
        *,
        identity_scope: IdentityScope,
        query: str | None = None,
        filters: dict[str, str] | None = None,
        limit: int = 20,
        exclude_types: list[str] | None = None,
        refresh_vitality: bool = True,
    ) -> list[MemoryAtom]:
        """按管理读取视角列出目标 Workspace 的记忆。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.memory.get")
    async def MEMORY_GET(
        memory_id: UUID | str, *, identity_scope: IdentityScope, refresh_vitality: bool = True
    ) -> MemoryAtom | None:
        """读取单条记忆，并按请求刷新生命力。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.memory.update")
    async def MEMORY_UPDATE(
        memory_id: UUID | str, *, identity_scope: IdentityScope, edit: MemoryEdit
    ) -> MemoryAtom | None:
        """以调用方提交的部分编辑修改目标归属下的记忆；记忆不存在时返回 None。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.memory.delete")
    async def MEMORY_DELETE(memory_id: UUID | str, *, identity_scope: IdentityScope) -> bool:
        """删除目标 Workspace 的指定记忆。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.memory.record_feedback")
    async def MEMORY_RECORD_FEEDBACK(
        memory_id: UUID | str, *, identity_scope: IdentityScope, positive: bool, source: str
    ) -> Any:
        """记录目标记忆的正向或负向反馈。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.memory_task.list")
    async def MEMORY_TASK_LIST(*, identity_scope: IdentityScope) -> list[MemoryGenerationTask]:
        """列出目标 Workspace 的记忆生成任务快照。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.memory_task.get")
    async def MEMORY_TASK_GET(
        task_id: str, *, identity_scope: IdentityScope
    ) -> MemoryGenerationTask:
        """读取本 Workspace 的任务；不存在与越域统一抛出 ``ResourceNotFoundError``。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.memory_task.cancel")
    async def MEMORY_TASK_CANCEL(task_id: str, *, identity_scope: IdentityScope) -> bool:
        """请求取消目标 Workspace 的记忆生成任务。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.agent_profile.create")
    async def AGENT_PROFILE_CREATE(
        identity_scope: IdentityScope, creation: AgentProfileCreation
    ) -> MemoryAtom:
        """以调用方提交的材料创建目标 Workspace 的 Agent Profile，返回持久化后的原子。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.agent_profile.list")
    async def AGENT_PROFILE_LIST(
        *, identity_scope: IdentityScope, limit: int = 100
    ) -> list[MemoryAtom]:
        """列出当前发起者可见的 Agent Profile 原子。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.topic.list_active")
    async def TOPIC_LIST_ACTIVE(
        *, identity_scope: IdentityScope, include_empty: bool = False
    ) -> tuple[TopicSnapshot, ...]:
        """列出目标 Workspace 的活跃话题快照。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.topic.get_data")
    async def TOPIC_GET_DATA(*, identity_scope: IdentityScope, topic_id: str) -> TopicData | None:
        """无副作用读取调用方可见的完整话题数据。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.memory.retrieve")
    async def MEMORY_RETRIEVE(request: RetrievalRequest) -> list[MemoryAtom]:
        """语义检索 backing（能力层授权 ``memory.search``），返回完整原子列表。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.memory.retrieve_by_aliases")
    async def MEMORY_RETRIEVE_BY_ALIASES(
        aliases: list[str], identity_scope: IdentityScope
    ) -> list[MemoryAtom]:
        """alias 批量读取 backing，按 READ/RUN/UPDATE 本次授权 scope 返回，只含实际可读的完整原子。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.get_agent_profile")
    async def GET_AGENT_PROFILE(
        agent_alias: str | None, *, identity_scope: IdentityScope
    ) -> ResolvedAgentProfile:
        """Profile 定义解析 backing：唯一解析规则 + 源原子 policy 依据与关联。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.memory.read")
    async def MEMORY_READ(
        memory_id: UUID | str, *, identity_scope: IdentityScope, refresh_vitality: bool = True
    ) -> MemoryAtom | None:
        """Actor-visible 的 canonical UUID 点读 backing，读取视图按本次授权 scope 交付。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.interaction.submit")
    async def INTERACTION_SUBMIT(
        *,
        identity_scope: IdentityScope,
        payload: InteractionPayload,
        requested_topic_id: str = "NEW_TOPIC",
        interaction_id: str | None = None,
        origin: InteractionOrigin = "workspace_port",
        new_topic_title: str | None = None,
        new_topic_summary: str | None = None,
    ) -> InteractionSubmitResult:
        """提交一份已发生的交互载荷；队列按 interaction_id 幂等接纳。

        目标为 ``NEW_TOPIC`` 时可携带新话题的标题与摘要；队列容量已满或已
        停止以 ``InteractionAdmissionUnavailableError`` 拒绝，可用同一标识重试。
        """
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.interaction.status")
    async def INTERACTION_STATUS(
        *, identity_scope: IdentityScope, interaction_id: str
    ) -> InteractionStatus | None:
        """查询交互在提交队列中的进度；未被接纳时返回 ``None``。

        同名交互属于其他 Workspace 时报冲突；成功时给出真实落点。
        """
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.interaction.wait")
    async def INTERACTION_WAIT(
        *, identity_scope: IdentityScope, interaction_id: str, timeout_seconds: float
    ) -> InteractionStatus | None:
        """等待交互应用到终态或超时，返回此时的进度；未被接纳时返回 ``None``。

        超时时进度仍为 ``pending``；归属校验与状态查询相同。
        """
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.memory_intent.submit")
    async def MEMORY_INTENT_SUBMIT(
        *, intent: MemoryIntent, identity_scope: IdentityScope
    ) -> MemoryIntentSubmissionResult:
        """提交记忆意图；结果由 Patchouli 生成链决定并经任务观察 API 查询。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.prepare_agent_run")
    async def PREPARE_AGENT_RUN(
        *,
        identity_scope: IdentityScope,
        interaction_id: str,
        gateway_decision: GatewayDecision,
        enable_memory_retrieval: bool = True,
    ) -> PreparedAgentRun:
        """准备本轮的 Topic 与未编译检索结果（prepare 只做 Topic 与检索）。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.finalize_agent_run")
    async def FINALIZE_AGENT_RUN(
        prepared_run: PreparedAgentRun,
        payload: InteractionPayload,
        *,
        identity_scope: IdentityScope,
    ) -> None:
        """原样提交封口好的交互记录，并把 post-apply 工作交给 Patchouli 持有。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.cleanup_prepared_agent_run")
    async def CLEANUP_PREPARED_AGENT_RUN(
        prepared_run: PreparedAgentRun, *, identity_scope: IdentityScope
    ) -> bool:
        """清理已 prepare 但未 finalize 的预创建空话题。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.manual_settle_topic")
    async def MANUAL_SETTLE_TOPIC(
        *, identity_scope: IdentityScope, topic_id: str
    ) -> TopicSettleResult:
        """通过本地总线结算 Topic（生命周期变更授权在能力层），返回稳定业务结果。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.evict_topic")
    async def EVICT_TOPIC(*, identity_scope: IdentityScope, topic_id: str) -> TopicEvictionResult:
        """通过本地总线驱逐 Topic（生命周期变更授权在能力层），不触发记忆结算。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.record_memory_citation")
    async def RECORD_MEMORY_CITATION(
        memory_id: str | UUID, *, identity_scope: IdentityScope, source: str = "mtp"
    ) -> Any:
        """记录一次记忆引用事件。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.models.warmup")
    async def WARMUP_MODELS() -> None:
        """预热 Patchouli 使用的模型。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("patchouli.public.models.ready")
    async def MODELS_READY() -> bool:
        """读取模型是否已完成预热。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    ALL = (
        MEMORY_CREATE,
        MEMORY_LIST,
        MEMORY_GET,
        MEMORY_UPDATE,
        MEMORY_DELETE,
        MEMORY_RECORD_FEEDBACK,
        MEMORY_TASK_LIST,
        MEMORY_TASK_GET,
        MEMORY_TASK_CANCEL,
        AGENT_PROFILE_CREATE,
        AGENT_PROFILE_LIST,
        TOPIC_LIST_ACTIVE,
        TOPIC_GET_DATA,
        MEMORY_RETRIEVE,
        MEMORY_RETRIEVE_BY_ALIASES,
        GET_AGENT_PROFILE,
        MEMORY_READ,
        INTERACTION_SUBMIT,
        INTERACTION_STATUS,
        INTERACTION_WAIT,
        MEMORY_INTENT_SUBMIT,
        PREPARE_AGENT_RUN,
        FINALIZE_AGENT_RUN,
        CLEANUP_PREPARED_AGENT_RUN,
        MANUAL_SETTLE_TOPIC,
        EVICT_TOPIC,
        RECORD_MEMORY_CITATION,
        WARMUP_MODELS,
        MODELS_READY,
    )


__all__ = ["PatchouliRoutes"]
