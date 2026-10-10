"""CPU 分配 — 任务进程在进入 Actor 之前准备的全部输入。

CPU 分配由进程完成而不是交给 Actor：Profile 经 workspace 能力层解析，
附件租借经注入的 reader port 取得并登记进进程工作集，附件与记忆文本由
进程调用共享编译引擎生成，最后组装与 CPU 无关的输入清单。Actor 只消费
清单，不接触租借或编译配置。

附件租借由本分配器取得，也由本分配器释放（:meth:`CPUAllocator.release`）：
工作集只登记租借，不持有 reader。

授权边界（A1 访问边界返工第 4.4 节）：本层是任务进程的阶段授权点——
Profile 解析绑定 ``profile.read``、附件租借绑定 ``asset.acquire``，检查
都以任务目标 workspace 为目标、在对应副作用前执行；CPU 输入清单的
``IdentityScope`` 由操作授权者的 CPU 执行身份过渡方法组装（I-9），不取自
调用方。
"""

from __future__ import annotations

import logging
from typing import Protocol

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.attachments import AttachmentCompilerConfig
from hivememory.config.memory_compiler import MemoryCompilerConfig
from hivememory.core.access import WorkspaceAccessContext, WorkspaceOperation
from hivememory.core.errors import AssetOperationConflictError, WorkspaceDomainError
from hivememory.core.models import (
    AgentProfile,
    AttachmentSelectionRequest,
    WorkspaceIdentity,
)
from hivememory.core.models.workspace_asset import RepresentationLease
from hivememory.core.ports.workspace_assets import WorkspaceAssetReaderPort
from hivememory.engines.attachment_compiler import AttachmentCompiler
from hivememory.engines.memory_compiler import (
    MemoryCompileOptions,
    MemoryCompiler,
    MemoryEnvelopeTarget,
)
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
from hivememory.workspace.contracts import CPUInputManifest
from hivememory.workspace.process.working_set import ProcessWorkingSet

logger = logging.getLogger(__name__)


class _AgentProfileReader(Protocol):
    """CPU 分配所需的 Profile 读取端口，由组合根注入能力层实现。"""

    async def get_agent_profile(
        self,
        agent_alias: str | None,
        *,
        target_workspace: WorkspaceIdentity,
        access: WorkspaceAccessContext,
    ) -> AgentProfile:
        """解析本进程的执行 Profile，并在交付前执行逐次授权。"""
        ...


class CPUAllocator:
    """解析 Profile、取得并编译附件、编译记忆并组装 CPU 输入清单。

    ``asset_reader`` 是进程级唯一 WorkspaceAssetStore 的只读 reader 端口；
    两个编译配置段驱动进程侧的记忆/附件编译（与拆分前 Patchouli prepare
    使用相同的引擎与配置段）。``operation_authorizer`` 是组合根注入的操作授权者：
    Profile 解析与附件租借的阶段操作授权在本层、对应副作用前执行。
    """

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        *,
        operation_authorizer: WorkspaceOperationAuthorizer,
        agent_service: _AgentProfileReader,
        asset_reader: WorkspaceAssetReaderPort | None = None,
        memory_compiler_config: MemoryCompilerConfig | None = None,
        attachment_compiler_config: AttachmentCompilerConfig | None = None,
    ) -> None:
        self._bus = global_bus
        self._authorizer = operation_authorizer
        self._agent_service = agent_service
        self._asset_reader = asset_reader
        self._memory_compiler_config = memory_compiler_config or MemoryCompilerConfig()
        self._memory_compiler = MemoryCompiler()
        self._attachment_compiler = AttachmentCompiler(
            attachment_compiler_config or AttachmentCompilerConfig(),
        )

    async def resolve_agent_profile(
        self,
        *,
        access: WorkspaceAccessContext,
        target_workspace: WorkspaceIdentity,
    ) -> AgentProfile:
        """经能力层解析本进程的执行 Profile（``profile.read``）。

        能力层逐次授权，读取视图负责缓存及 canonical 变更失效；运行上下文
        只需要能力描述，源原子的授权依据不进入 run。

        中间态（2026-09-29）：Profile 属于 CPU 分配，但暂时在 Patchouli prepare
        之前解析。当前 prepare 会按 Gateway 的路由决定预先新建 Topic，话题池已满
        时还会先按 LRU 结算一个已有话题；若在 prepare 之后才发现 Profile 缺失，
        失败的请求已经留下这些不可逆的副作用。Topic 的新建与驱逐改到 interaction
        提交之后以后，Profile 解析可以回到 prepare 之后的 CPU 分配步骤。
        """
        # 当前执行 Agent 由授权点确定，不能从入口附带的 Profile alias 推断。
        # 能力层仍独立执行 profile.read 授权并负责交付可见性检查。
        scope = self._authorizer.authorize_operation(
            access, WorkspaceOperation.PROFILE_READ, target_workspace
        )
        return await self._agent_service.get_agent_profile(
            scope.actor_identity.agent_id,
            target_workspace=target_workspace,
            access=access,
        )

    def allocate(
        self,
        working_set: ProcessWorkingSet,
        *,
        process_id: str,
        user_message: str,
        agent_profile: AgentProfile,
        selections: list[AttachmentSelectionRequest],
        access: WorkspaceAccessContext,
        target_workspace: WorkspaceIdentity,
    ) -> CPUInputManifest:
        """取得附件租借并编译附件与记忆，组装输入清单。

        在 prepare 之后执行，读取工作集中的 prepare 结果；Profile 已由
        :meth:`resolve_agent_profile` 提前解析。附件租借绑定 ``asset.acquire``，
        授权检查在租借副作用前执行（``_acquire_selected_attachment``）。
        CPU 执行身份由操作授权者的过渡方法组装（I-9）：清单携带的
        ``IdentityScope`` 不取自调用方。stop 请求不打断分配，由调用方在
        进入 Actor 之前统一检查。任何失败沿异常路径上抛，已取得的租借由
        本分配器在进程关闭时释放。
        """
        prepared = working_set.prepared
        if prepared is None:
            raise RuntimeError("CPU 分配必须在 prepare 结果写入工作集之后执行")

        # 1. 附件：按用户选择顺序 acquire READY representation 并核对版本摘要。
        #    取得的 lease 由 _acquire_selected_attachment 直接登记进工作集。
        for selection in selections:
            self._acquire_selected_attachment(
                working_set,
                access,
                target_workspace,
                selection,
            )

        # 2. 编译：附件与记忆文本由进程生成，CPU 只消费成品。检索为空时
        #    memory_context 为空字符串（与拆分前 prepare 的行为一致）。
        attachment_compile_result = self._attachment_compiler.compile(
            leases=tuple(working_set.attachment_leases),
        )
        working_set.used_attachments = attachment_compile_result.used_attachments
        memories = list(prepared.retrieval_result.memories)
        memory_context = (
            self._memory_compiler.compile(
                memories,
                MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
                MemoryCompileOptions(
                    retrieval_strategy_config=(
                        self._memory_compiler_config.retrieval_context.strategy
                    ),
                ),
            ).text
            if memories
            else ""
        )

        # 3. 清单：组装与 CPU 无关的输入清单交给 Actor。执行身份是过渡期
        #    的授权点产物（I-9）：只做目标与 owner 检查、不检查 operation，
        #    Alice 的能力层调用迁移完成后随本方法一并调整。
        manifest = CPUInputManifest(
            process_id=process_id,
            identity_scope=self._authorizer.cpu_execution_identity(access, target_workspace),
            user_message=user_message,
            agent_profile=agent_profile,
            memories=memories,
            memory_context=memory_context,
            attachment_context=attachment_compile_result.attachment_context,
            storage_available=prepared.storage_available,
            topic_id=prepared.topic_id,
            topic_context=prepared.topic_context,
        )
        return manifest

    def _acquire_selected_attachment(
        self,
        working_set: ProcessWorkingSet,
        access: WorkspaceAccessContext,
        target_workspace: WorkspaceIdentity,
        selection: AttachmentSelectionRequest,
    ) -> RepresentationLease:
        """acquire 单个选中附件并核对客户端提供的版本摘要。

        附件租借绑定 ``asset.acquire``：授权检查先于租借副作用执行，租借
        使用操作授权者返回的可信 scope。reader 的同一 Store 临界区已完成
        Workspace/ref、asset READY 与 representation READY 校验并建立
        lease，无需先做 resolve_asset。取得的 lease 先登记进工作集；版本
        摘要不一致时立即移出工作集并释放该租借，拒绝整轮，不留游离租借。
        """
        if self._asset_reader is None:
            raise WorkspaceDomainError(
                "当前系统未装配附件读取能力，不能处理附件选择",
                details={"reason": "asset_reader_unavailable"},
            )
        scope = self._authorizer.authorize_operation(
            access, WorkspaceOperation.ASSET_ACQUIRE, target_workspace
        )
        lease = self._asset_reader.acquire_ready_representation(
            scope,
            selection.asset_ref,
        )
        working_set.register_lease(lease)
        representation = lease.representation
        mismatch = (
            lease.asset_ref != selection.asset_ref
            or (
                selection.representation_id is not None
                and representation.representation_id != selection.representation_id
            )
            or (selection.revision is not None and representation.revision != selection.revision)
            or (
                selection.content_hash is not None
                and representation.content_hash != selection.content_hash
            )
        )
        if mismatch:
            working_set.remove_lease(lease)
            self._release_lease(lease)
            raise AssetOperationConflictError(
                "所选附件版本与当前可用表示不一致，请重新选择附件",
                details={
                    "reason": "selection_version_mismatch",
                    "asset_id": representation.asset_id,
                },
            )
        return lease

    # ========== 租借释放 ==========

    def release(self, working_set: ProcessWorkingSet) -> None:
        """释放工作集登记的全部附件租借（幂等：租借只取出一次）。

        进程关闭时由编排骨架同步调用，先于关闭流程中的任何 ``await``。
        逐项释放；Store 关闭等 ``WorkspaceDomainError`` 只记录警告，不改变
        进程已经确定的终态。
        """
        for lease in working_set.take_leases():
            self._release_lease(lease)

    def _release_lease(self, lease: RepresentationLease) -> None:
        if self._asset_reader is None:
            return
        try:
            self._asset_reader.release_representation_lease(lease.lease_id)
        except WorkspaceDomainError as exc:
            # Store 已关闭等清理路径：记录摘要，不改变进程终态。
            logger.warning(
                "释放附件 lease 失败: lease_id=%s, code=%s",
                lease.lease_id,
                exc.code,
            )


__all__ = ["CPUAllocator"]
