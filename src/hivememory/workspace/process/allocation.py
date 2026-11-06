"""CPU 分配 — 任务进程在进入 Actor 之前准备的全部输入。

CPU 分配由进程完成而不是交给 Actor：Profile 经注入的读取端口解析，
附件租借经注入的 reader port 取得并登记进进程工作集，附件与记忆文本由
进程调用共享编译引擎生成，再定稿人格、环境通知与本轮材料并组装输入清单。
Actor 只消费编译结果，不接触租借或编译配置。

附件租借由本分配器取得，也由本分配器释放（:meth:`CPUAllocator.release`）：
工作集只登记租借，不持有 reader。

授权边界：本层是任务进程的系统阶段授权点——Profile 与附件是用户的选择，
对用户可见而不是对 Agent 可见，两者都由系统以 root 工作 actor 授权，不借用
actor 操作。Profile 按管理语义读取，只检查 Workspace 归属；资产库只核对
Workspace。检查在对应副作用前执行；CPU 输入清单的观测标签沿用注册时的
绑定值，不从工作 actor 读取或组装执行身份。
"""

from __future__ import annotations

import logging
from typing import Protocol

from hivememory.config.attachments import AttachmentCompilerConfig
from hivememory.config.memory_compiler import MemoryCompilerConfig
from hivememory.core.access import WorkingActor
from hivememory.core.errors import AssetOperationConflictError, WorkspaceDomainError
from hivememory.core.models import (
    AgentProfile,
    AttachmentSelectionRequest,
    ExecutionLabels,
    IdentityScope,
    WorkspaceIdentity,
)
from hivememory.core.models.workspace_asset import RepresentationLease
from hivememory.core.ports.workspace_assets import WorkspaceAssetReaderPort
from hivememory.core.protocol.models import ConversationTurn, SessionCompactionCheckpoint
from hivememory.engines.attachment_compiler import AttachmentCompiler
from hivememory.engines.memory_compiler import (
    MemoryCompileOptions,
    MemoryCompiler,
    MemoryEnvelopeTarget,
)
from hivememory.prompts.compilation import compile_prompt_input
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
from hivememory.workspace.contracts import CPUInputManifest
from hivememory.workspace.process.working_set import ProcessWorkingSet

logger = logging.getLogger(__name__)


class _AgentProfileReader(Protocol):
    """CPU 分配所需的 Profile 读取端口，由组合根注入读取视图实现。"""

    async def get(
        self,
        agent_alias: str | None,
        *,
        scope: IdentityScope,
        enforce_actor_visibility: bool = True,
    ) -> AgentProfile:
        """按授权的 scope 解析 Profile；``enforce_actor_visibility=False`` 是管理读取。"""
        ...


class CPUAllocator:
    """解析 Profile、租借附件并编译提示词，组装 CPU 输入清单。

    ``asset_reader`` 是进程级唯一 WorkspaceAssetStore 的只读 reader 端口；
    两个编译配置段驱动进程侧的记忆/附件编译。
    ``operation_authorizer`` 是组合根注入的授权者：
    Profile 解析与附件租借的 root 授权在本层、对应副作用前执行。
    """

    def __init__(
        self,
        *,
        operation_authorizer: WorkspaceOperationAuthorizer,
        profile_reader: _AgentProfileReader,
        asset_reader: WorkspaceAssetReaderPort,
        memory_compiler_config: MemoryCompilerConfig | None = None,
        attachment_compiler_config: AttachmentCompilerConfig | None = None,
    ) -> None:
        self._authorizer = operation_authorizer
        self._profile_reader = profile_reader
        self._asset_reader = asset_reader
        self._memory_compiler_config = memory_compiler_config or MemoryCompilerConfig()
        self._memory_compiler = MemoryCompiler()
        self._attachment_compiler = AttachmentCompiler(
            attachment_compiler_config or AttachmentCompilerConfig(),
        )

    async def resolve_agent_profile(
        self,
        *,
        agent_id: str,
        system_actor: WorkingActor,
        target_workspace: WorkspaceIdentity,
    ) -> AgentProfile:
        """以系统的 root 工作 actor 授权后，经读取视图按管理语义解析执行者的 Profile。

        ``agent_id`` 是注册时认证得到的执行者，不能从入口附带的 Profile alias
        推断。Profile 是用户的选择：读取只检查 Workspace 归属，不按 Agent 的
        可见性裁剪，指向其他 Agent 的 PRIVATE 图纸也能被它自己的进程加载；
        读取视图负责缓存及 canonical 变更失效。运行上下文只需要能力描述，源
        原子的授权依据不进入 run。

        Profile 属于 CPU 分配，但在 Patchouli prepare 之前解析。prepare 保留
        Topic 预建，话题池已满时还会先按 LRU 结算已有话题；先解析 Profile
        可避免缺少图纸的请求留下这些副作用，也保证 Actor 开始前已有派发话题。
        """
        scope = self._authorizer.authorize_root(system_actor, target_workspace)
        return await self._profile_reader.get(
            agent_id,
            scope=scope,
            enforce_actor_visibility=False,
        )

    def allocate(
        self,
        working_set: ProcessWorkingSet,
        *,
        process_id: str,
        labels: ExecutionLabels,
        user_message: str,
        agent_profile: AgentProfile,
        selections: list[AttachmentSelectionRequest],
        session_history: tuple[ConversationTurn, ...] = (),
        compaction_checkpoint: SessionCompactionCheckpoint | None = None,
        system_actor: WorkingActor,
        target_workspace: WorkspaceIdentity,
    ) -> CPUInputManifest:
        """取得附件租借并编译附件与记忆，组装输入清单。

        在 prepare 之后执行，读取工作集中的 prepare 结果；Profile 已由
        :meth:`resolve_agent_profile` 提前解析，只用于编译人格段，图纸与操作
        快照随主线程交给 CPU，不进入清单。附件租借以系统的 root 工作 actor
        授权，授权检查在租借副作用前执行（``_acquire_selected_attachment``）。
        ``labels`` 由注册入口认证成功后绑定，只供 CPU 的提示词与观测使用，
        不参与授权。stop 请求不打断分配，由调用方在
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
                system_actor,
                target_workspace,
                selection,
            )

        # 2. 编译：附件与记忆文本由进程生成，检索为空时不产生记忆段。
        attachment_compile_result = self._attachment_compiler.compile(
            leases=tuple(working_set.attachment_leases),
        )
        working_set.used_attachments = attachment_compile_result.used_attachments
        memories = list(prepared.memories)
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

        # 3. 提示词：在 CPU 边界前定稿环境通知、人格与本轮材料，语言只解析一次。
        prompt = compile_prompt_input(
            agent_profile=agent_profile,
            storage_available=prepared.storage_available,
            memory_context=memory_context,
            attachment_context=attachment_compile_result.attachment_context,
        )

        # 4. 清单：沿用注册时的只读标签，CPU 不接收身份或工作 actor。
        #    标签与资源授权分离；附件租借仍在各次 acquire 前独立授权。
        manifest = CPUInputManifest(
            process_id=process_id,
            labels=labels,
            user_message=user_message,
            language=prompt.language,
            instructions=prompt.instructions,
            turn_context=prompt.turn_context,
            topic_id=prepared.topic_id,
            session_history=session_history,
            compaction_checkpoint=compaction_checkpoint,
        )
        working_set.dispatch_topic_id = manifest.topic_id
        return manifest

    def _acquire_selected_attachment(
        self,
        working_set: ProcessWorkingSet,
        system_actor: WorkingActor,
        target_workspace: WorkspaceIdentity,
        selection: AttachmentSelectionRequest,
    ) -> RepresentationLease:
        """acquire 单个选中附件并核对客户端提供的版本摘要。

        附件是用户的选择，由系统以 root 授权租借：授权检查先于租借副作用执行，
        租借使用授权者返回的可信 scope，资产库只核对其中的 Workspace。
        reader 的同一 Store 临界区已完成
        Workspace/ref、asset READY 与 representation READY 校验并建立
        lease，无需先做 resolve_asset。取得的 lease 先登记进工作集；版本
        摘要不一致时立即移出工作集并释放该租借，拒绝整轮，不留游离租借。
        """
        scope = self._authorizer.authorize_root(system_actor, target_workspace)
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
