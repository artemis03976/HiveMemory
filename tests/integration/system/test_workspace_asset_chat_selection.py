"""附件选择跨阶段集成验收（prepare 拆分后由进程 CPU 分配承担）。

从 W1-C 的真实出口出发：真实上传应用服务（真实 text parser）把文件推进
到 EXTRACTED_TEXT READY，随后 chat 任务进程在 CPU 分配边界按用户选择顺序
resolve/acquire 并冻结坐标。捕获选择绕过 READY 门槛、版本摘要漂移或
removed 竞态下继续使用 representation 的缺陷。

访问边界（A1 访问边界返工第 4.4/4.5 节）：上传 access 与进程 context 都
由同一真实网关组合签发（``register_process`` 完成两阶段认证并绑定
process_id）；进程内各阶段授权按 operation 白名单执行。
"""

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.attachments import AttachmentParserConfig
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import AssetRemovedError
from hivememory.core.models import ActorIdentity, AttachmentSelectionRequest
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.workspace.assets.store import InMemoryWorkspaceAssetStore
from tests.helpers.attachment_parsing import ChunkedSource, make_upload_access, make_upload_service
from tests.helpers.chat_handoff import make_gateway_decision
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.process import make_task_process_service
from tests.helpers.workspace import make_identity_scope
from tests.unit.workspace.process.test_gateway_chat_flow import (
    _profile_route,
    _scoped_prepared_route,
)

ACTOR = ActorIdentity(user_id="user-1", agent_id="omni_doll")


def _decision_outcome():
    return GatewayDecisionOutcome(decision=make_gateway_decision())


async def _gateway_route(**_kwargs):
    """GATEWAY_PROCESS 替身：恒返回常规 RAG 决定。"""
    return _decision_outcome()


async def _upload(
    upload_service,
    composition,
    upload_access,
    *,
    file_name: str,
    content: bytes,
    operation_id: str,
    media_type: str = "text/markdown",
):
    """经组合签发的 access 上传（授权组装可信 scope）。"""
    return await upload_service.upload_asset(
        target_workspace=composition.default_workspace,
        file_name=file_name,
        declared_media_type=media_type,
        source=ChunkedSource(content),
        client_operation_id=operation_id,
        access=upload_access,
    )


def _process_service(bus: GlobalSystemBus, composition, store, cpu: ScriptedCPU):
    """注册入口与上传共享同一访问组合：签发与授权读取同一份访问登记。"""
    return make_task_process_service(
        bus,
        asset_reader=store,
        cpu=cpu,
        access_gateway=composition.gateway,
        operation_authorizer=composition.authorizer,
    )


@pytest.mark.asyncio
async def test_uploaded_ready_asset_can_be_selected_by_task_process() -> None:
    """捕获选择坐标与上传产物漂移，或 PROCESSING 资产被提前选择。"""
    store = InMemoryWorkspaceAssetStore()
    composition = make_upload_access(user_id="user-1")
    upload_access = await composition.authenticate(agent_id="omni_doll")
    upload_service = make_upload_service(
        store=store,
        parser_config=AttachmentParserConfig(),
        access_composition=composition,
    )
    upload_service_2 = make_upload_service(
        store=store,
        parser_config=AttachmentParserConfig(),
        access_composition=composition,
    )

    first = await _upload(
        upload_service,
        composition,
        upload_access,
        file_name="first.md",
        content="# 第一份\n".encode(),
        operation_id="op-first",
    )
    second = await _upload(
        upload_service_2,
        composition,
        upload_access,
        file_name="second.md",
        content="# 第二份\n".encode(),
        operation_id="op-second",
    )
    assert (first.handle.asset.state.value, second.handle.asset.state.value) == (
        "ready",
        "ready",
    )

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _scoped_prepared_route())
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    finalize_kwargs: dict = {}

    async def finalize(**kwargs):
        finalize_kwargs.update(kwargs)
        return []

    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)

    service = _process_service(
        bus, composition, store, ScriptedCPU(result=make_cpu_result(final_text="完成"))
    )
    process = await service.register_process(
        adapter="local",
        principal=composition.principal,
        actor=ACTOR,
        workspace=composition.default_workspace,
        process_id="process-selection",
        message="总结这两份附件",
        attachments=[
            # 用户顺序：第二份在前。
            AttachmentSelectionRequest(
                asset_ref=second.handle.asset_ref,
                revision=1,
                content_hash=second.handle.asset.representations[1].content_hash,
            ),
            AttachmentSelectionRequest(asset_ref=first.handle.asset_ref),
        ],
    )
    result = await service.run_process(process, stream=False)

    # 用户选择只作为 compiler input：实际使用顺序由编译产物冻结，经封口 payload 交给 finalize。
    assert result.kind == "agent"
    assert list(finalize_kwargs["payload"].used_attachments) == [
        second.handle.asset_ref,
        first.handle.asset_ref,
    ]
    assert store.close_and_clear().leases_cleared == 0


@pytest.mark.asyncio
async def test_removed_asset_rejects_selection_after_upload() -> None:
    """捕获 removed 后的选择绕过 not-found/removed 语义进入本轮。"""
    store = InMemoryWorkspaceAssetStore()
    scope = make_identity_scope(user_id="user-1", agent_id="omni_doll")
    composition = make_upload_access(user_id="user-1")
    upload_access = await composition.authenticate(agent_id="omni_doll")
    upload_service = make_upload_service(
        store=store,
        parser_config=AttachmentParserConfig(),
        access_composition=composition,
    )
    receipt = await _upload(
        upload_service,
        composition,
        upload_access,
        file_name="gone.txt",
        content="正文".encode(),
        operation_id="op-gone",
        media_type="text/plain",
    )
    store.remove_asset(scope, receipt.handle.asset_ref)

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _scoped_prepared_route())
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, _async_true)

    service = _process_service(
        bus, composition, store, ScriptedCPU(result=make_cpu_result(final_text="完成"))
    )
    process = await service.register_process(
        adapter="local",
        principal=composition.principal,
        actor=ACTOR,
        workspace=composition.default_workspace,
        process_id="process-removed",
        message="使用已删除附件",
        attachments=[
            AttachmentSelectionRequest(asset_ref=receipt.handle.asset_ref),
        ],
    )
    with pytest.raises(AssetRemovedError):
        await service.run_process(process, stream=False)
    # remove 清除全部 representation：同一 ref 不可能复活，也不会残留 lease。
    # 同 Workspace 内已知 ref 的既有 Store 语义是 AssetRemovedError。
    assert store.close_and_clear().leases_cleared == 0


async def _async_true(*_args, **_kwargs):
    return True


@pytest.mark.asyncio
async def test_chat_bus_route_reaches_real_prepare_with_attachments() -> None:
    """回归：真实 Patchouli prepare（精简签名）+ 进程 CPU 分配的完整链路。

    TaskProcessService 经 GlobalSystemBus 调用真实 ``prepare_agent_run``，
    随后在进程侧取得附件租借并编译，最终以清单调用 Alice。
    """
    store = InMemoryWorkspaceAssetStore()
    composition = make_upload_access(user_id="user-1")
    upload_access = await composition.authenticate(agent_id="omni_doll")
    upload_service = make_upload_service(
        store=store,
        parser_config=AttachmentParserConfig(),
        access_composition=composition,
    )
    receipt = await _upload(
        upload_service,
        composition,
        upload_access,
        file_name="chat.md",
        content="# 选中正文\n".encode(),
        operation_id="op-chat",
    )
    assert receipt.handle.asset.state.value == "ready"

    from hivememory.patchouli.control.interaction_submission import (
        InteractionSubmissionQueue,
    )
    from hivememory.patchouli.service import PatchouliService
    from tests.unit.patchouli.test_phase3f_gateway_decision import _prepare_bus

    async def apply_interaction(_payload, **_kwargs):
        return "topic-1"

    prepare_bus, _retrieve, _submit = _prepare_bus()
    patchouli_service = PatchouliService(
        prepare_bus,
        interaction_queue=InteractionSubmissionQueue(apply_interaction),
    )
    bus = GlobalSystemBus()
    bus.register(
        GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
        patchouli_service.prepare_agent_run,
    )
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())
    bus.register(GlobalRoutes.GATEWAY_PROCESS, _gateway_route)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, _async_empty_tasks)

    service = _process_service(
        bus, composition, store, ScriptedCPU(result=make_cpu_result(final_text="完成"))
    )
    process = await service.register_process(
        adapter="local",
        principal=composition.principal,
        actor=ACTOR,
        workspace=composition.default_workspace,
        process_id="process-via-bus",
        message="总结这份附件",
        attachments=[
            AttachmentSelectionRequest(
                asset_ref=receipt.handle.asset_ref,
                revision=1,
                content_hash=receipt.handle.asset.representations[1].content_hash,
            ),
        ],
    )
    result = await service.run_process(process, stream=False)

    # 真实 prepare 与进程 CPU 分配完整走通。
    assert result.kind == "agent"
    assert result.execution_result.final_text == "完成"
    assert store.close_and_clear().leases_cleared == 0


async def _async_empty_tasks(**_kwargs):
    return []
