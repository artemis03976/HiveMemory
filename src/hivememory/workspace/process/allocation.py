"""CPU 分配 — 任务进程在进入 Actor 之前准备的全部输入。

CPU 分配由进程完成而不是交给 Actor：Profile 经 Patchouli 公开路由解析，
附件租借经注入的 reader port 取得并登记进进程工作集，附件与记忆文本由
进程调用共享编译引擎生成，最后组装与 CPU 无关的输入清单。Actor 只消费
清单，不接触租借或编译配置。
"""

from __future__ import annotations

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.attachments import AttachmentCompilerConfig
from hivememory.config.memory_compiler import MemoryCompilerConfig
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import AssetOperationConflictError, WorkspaceDomainError
from hivememory.core.models import (
    AgentProfile,
    AttachmentSelectionRequest,
    IdentityScope,
    ResolvedAgentProfile,
)
from hivememory.core.models.workspace_asset import RepresentationLease
from hivememory.core.ports.workspace_assets import WorkspaceAssetReaderPort
from hivememory.engines.attachment_compiler import AttachmentCompiler
from hivememory.engines.memory_compiler import (
    MemoryCompileOptions,
    MemoryCompiler,
    MemoryEnvelopeTarget,
)
from hivememory.workspace.contracts import CPUInputManifest
from hivememory.workspace.process.working_set import ProcessWorkingSet


class CPUAllocator:
    """解析 Profile、取得并编译附件、编译记忆并组装 CPU 输入清单。

    ``asset_reader`` 是进程级唯一 WorkspaceAssetStore 的只读 reader 端口；
    两个编译配置段驱动进程侧的记忆/附件编译（与拆分前 Patchouli prepare
    使用相同的引擎与配置段）。
    """

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        *,
        asset_reader: WorkspaceAssetReaderPort | None = None,
        memory_compiler_config: MemoryCompilerConfig | None = None,
        attachment_compiler_config: AttachmentCompilerConfig | None = None,
    ) -> None:
        self._bus = global_bus
        self._asset_reader = asset_reader
        self._memory_compiler_config = memory_compiler_config or MemoryCompilerConfig()
        self._memory_compiler = MemoryCompiler()
        self._attachment_compiler = AttachmentCompiler(
            attachment_compiler_config or AttachmentCompilerConfig(),
        )

    def new_working_set(self) -> ProcessWorkingSet:
        """为一次进程创建工作集；租借经同一 reader 释放。"""
        return ProcessWorkingSet(asset_reader=self._asset_reader)

    async def resolve_agent_profile(self, identity_scope: IdentityScope) -> AgentProfile:
        """经 Patchouli 公开路由解析本进程的执行 Profile。

        与拆分前 prepare 使用的本地路由是同一条解析规则；运行上下文只需要能力
        描述，源原子 policy 依据不进入 run（A2 §2.3）。暂不经能力层：能力层需要
        访问上下文（生产入口要到 A1 返工才取得），它依赖的 Profile 缓存也还没有
        失效机制（见任务进程 Idea 1.2）。

        中间态（2026-09-29）：Profile 属于 CPU 分配，但暂时在 Patchouli prepare
        之前解析。当前 prepare 会按 Gateway 的路由决定预先新建 Topic，话题池已满
        时还会先按 LRU 结算一个已有话题；若在 prepare 之后才发现 Profile 缺失，
        失败的请求已经留下这些不可逆的副作用。Topic 的新建与驱逐改到 interaction
        提交之后以后，Profile 解析可以回到 prepare 之后的 CPU 分配步骤。
        """
        resolved_profile: ResolvedAgentProfile = await self._bus.request(
            GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE,
            identity_scope.actor_identity.agent_id,
            identity_scope=identity_scope,
        )
        return resolved_profile.profile

    def allocate(
        self,
        working_set: ProcessWorkingSet,
        *,
        process_id: str,
        identity_scope: IdentityScope,
        user_message: str,
        agent_profile: AgentProfile,
        selections: list[AttachmentSelectionRequest],
    ) -> CPUInputManifest:
        """取得附件租借并编译附件与记忆，组装输入清单。

        在 prepare 之后执行，读取工作集中的 prepare 结果；Profile 已由
        :meth:`resolve_agent_profile` 提前解析。stop 请求不打断分配，由调用方
        在进入 Actor 之前统一检查。任何失败沿异常路径上抛，已取得的租借由
        工作集在进程关闭时释放。
        """
        prepared = working_set.prepared
        if prepared is None:
            raise RuntimeError("CPU 分配必须在 prepare 结果写入工作集之后执行")

        # 1. 附件：按用户选择顺序 acquire READY representation 并核对版本摘要。
        #    取得的 lease 由 _acquire_selected_attachment 直接登记进工作集。
        for selection in selections:
            self._acquire_selected_attachment(working_set, identity_scope, selection)

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

        # 3. 清单：组装与 CPU 无关的输入清单交给 Actor。
        manifest = CPUInputManifest(
            process_id=process_id,
            identity_scope=identity_scope,
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
        identity_scope: IdentityScope,
        selection: AttachmentSelectionRequest,
    ) -> RepresentationLease:
        """acquire 单个选中附件并核对客户端提供的版本摘要。

        reader 的同一 Store 临界区已完成 Workspace/ref、asset READY 与
        representation READY 校验并建立 lease，无需先做 resolve_asset。
        取得的 lease 先登记进工作集；版本摘要不一致时经工作集释放该租借
        并拒绝整轮，不留游离租借。
        """
        if self._asset_reader is None:
            raise WorkspaceDomainError(
                "当前系统未装配附件读取能力，不能处理附件选择",
                details={"reason": "asset_reader_unavailable"},
            )
        lease = self._asset_reader.acquire_ready_representation(
            identity_scope,
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
            working_set.discard_lease(lease)
            raise AssetOperationConflictError(
                "所选附件版本与当前可用表示不一致，请重新选择附件",
                details={
                    "reason": "selection_version_mismatch",
                    "asset_id": representation.asset_id,
                },
            )
        return lease


__all__ = ["CPUAllocator"]
