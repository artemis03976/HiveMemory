"""真实 root 服务、授权者与总线协作，逐方法验收 root 边界与 system 交付。"""

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.attachments import AttachmentParserConfig
from hivememory.core.access import WorkspaceOperation
from hivememory.core.errors import OperationDeniedError
from hivememory.core.models import (
    IdentityScope,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    WorkspaceAssetState,
)
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.memory_tasks import MemoryGenerationSource, MemoryGenerationTask
from hivememory.patchouli.contracts.topic_management import TopicEvictionResult, TopicSettleResult
from hivememory.workspace.assets.store import InMemoryWorkspaceAssetStore
from hivememory.workspace.capability.root import (
    RootAgentProfileService,
    RootMemoryService,
    RootMemoryTaskService,
    RootTopicService,
)
from tests.helpers.attachment_parsing import ChunkedSource, make_upload_service
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_identity_scope,
)

_MEMORY_ID = UUID("da42f85f-63c6-44c1-b7c8-5b7084133508")


@dataclass(frozen=True)
class _Case:
    """一个管理方法对应的公共路由与合法输入，不携带授权结果。"""

    service: str
    method: str
    route: str | None
    arguments: tuple = ()
    keywords: tuple[tuple[str, object], ...] = ()


_ROUTED_CASES = [
    _Case(
        "memory",
        "create_memory",
        PatchouliRoutes.MEMORY_CREATE,
        keywords=(
            ("title", "记忆"),
            ("summary", "管理创建"),
            ("content", "正文"),
            ("memory_type", "FACT"),
            ("tags", []),
        ),
    ),
    _Case("memory", "list_memories", PatchouliRoutes.MEMORY_LIST),
    _Case("memory", "get_memory", PatchouliRoutes.MEMORY_GET, (_MEMORY_ID,)),
    _Case("memory", "update_memory", PatchouliRoutes.MEMORY_UPDATE, (_MEMORY_ID,)),
    _Case("memory", "delete_memory", PatchouliRoutes.MEMORY_DELETE, (_MEMORY_ID,)),
    _Case(
        "memory",
        "record_feedback",
        PatchouliRoutes.MEMORY_RECORD_FEEDBACK,
        (_MEMORY_ID,),
        (("positive", True), ("source", "root.test")),
    ),
    _Case(
        "profiles",
        "create_agent_profile",
        PatchouliRoutes.AGENT_PROFILE_CREATE,
        keywords=(("title", "Worker"), ("alias", "worker"), ("tags", [])),
    ),
    _Case("profiles", "list_agent_profiles", PatchouliRoutes.AGENT_PROFILE_LIST),
    _Case("topic", "list_active_topics", PatchouliRoutes.TOPIC_LIST_ACTIVE),
    _Case(
        "topic",
        "settle_topic",
        PatchouliRoutes.MANUAL_SETTLE_TOPIC,
        keywords=(("topic_id", "topic-1"),),
    ),
    _Case(
        "topic",
        "evict_topic",
        PatchouliRoutes.EVICT_TOPIC,
        keywords=(("topic_id", "topic-1"),),
    ),
    _Case("tasks", "list_memory_tasks", PatchouliRoutes.MEMORY_TASK_LIST),
    _Case("tasks", "get_memory_task", PatchouliRoutes.MEMORY_TASK_GET, ("task-1",)),
    _Case("tasks", "cancel_memory_task", PatchouliRoutes.MEMORY_TASK_CANCEL, ("task-1",)),
]
# 上传不经 Patchouli 路由，副作用落在进程内资产 Store。
_UPLOAD_CASE = _Case(
    "assets",
    "upload_asset",
    None,
    keywords=(
        ("file_name", "sample.txt"),
        ("declared_media_type", "text/plain"),
        ("client_operation_id", "upload-1"),
    ),
)
_CASES = [*_ROUTED_CASES, _UPLOAD_CASE]


class _OwnerBoundary:
    """资源 owner 的外部替身；记录收到的请求并提供管理服务所需的领域结果。"""

    def __init__(self, bus: GlobalSystemBus) -> None:
        self.accepted: list[tuple[str, IdentityScope]] = []
        for case in _ROUTED_CASES:
            bus.register(case.route, self._handler(case.route))

    def _handler(self, route):
        async def handle(*args, **kwargs):
            scope = kwargs.get("identity_scope")
            if scope is None:
                scope = args[0]
            self.accepted.append((route, scope))
            if route in {
                PatchouliRoutes.MEMORY_CREATE,
                PatchouliRoutes.AGENT_PROFILE_CREATE,
            }:
                return args[1]
            if route in {PatchouliRoutes.MEMORY_GET, PatchouliRoutes.MEMORY_UPDATE}:
                return MemoryAtom(
                    id=_MEMORY_ID,
                    meta=make_memory_metadata(user_id="u1", source_agent_id="writer"),
                    index=IndexLayer(title="记忆", summary="管理读取", memory_type=MemoryType.FACT),
                    payload=PayloadLayer(content="正文"),
                )
            if route in {
                PatchouliRoutes.MEMORY_LIST,
                PatchouliRoutes.AGENT_PROFILE_LIST,
                PatchouliRoutes.MEMORY_TASK_LIST,
            }:
                return []
            if route == PatchouliRoutes.TOPIC_LIST_ACTIVE:
                return ()
            if route == PatchouliRoutes.MANUAL_SETTLE_TOPIC:
                return TopicSettleResult(topic_id=kwargs["topic_id"])
            if route == PatchouliRoutes.EVICT_TOPIC:
                return TopicEvictionResult(topic_id=kwargs["topic_id"], removed=True)
            if route == PatchouliRoutes.MEMORY_TASK_GET:
                return MemoryGenerationTask(
                    task_id=args[0],
                    topic_id="topic-1",
                    label="管理任务",
                    source=MemoryGenerationSource.WRITE,
                    from_actor=scope.actor_identity,
                    belong_to=scope.workspace_identity,
                    created_at=datetime(2026, 10, 10, tzinfo=UTC),
                )
            return True

        return handle


def _services(composition):
    """root 服务不装配 actor 读取视图，外部 owner 只在公共总线边界替换。"""
    bus = GlobalSystemBus()
    owner = _OwnerBoundary(bus)
    store = InMemoryWorkspaceAssetStore()
    services = {
        "memory": RootMemoryService(bus, operation_authorizer=composition.authorizer),
        "profiles": RootAgentProfileService(bus, operation_authorizer=composition.authorizer),
        "topic": RootTopicService(bus, operation_authorizer=composition.authorizer),
        "tasks": RootMemoryTaskService(bus, operation_authorizer=composition.authorizer),
        "assets": make_upload_service(
            store=store, parser_config=AttachmentParserConfig(), access_composition=composition
        ),
    }
    return services, owner, store


class _UploadSource(ChunkedSource):
    """记录读取副作用的字节源，验证 root 拒绝不会消费上传内容。"""

    def __init__(self) -> None:
        super().__init__(b"root upload")
        self.read_count = 0

    async def read(self, size: int = -1) -> bytes:
        self.read_count += 1
        return await super().read(size)


async def _invoke(case, services, composition, access, upload_source=None):
    """以请求输入调用真实管理入口，上传字节源由每次场景独立创建。"""
    kwargs = dict(case.keywords)
    if case.service == "assets":
        kwargs["source"] = upload_source or _UploadSource()
    return await getattr(services[case.service], case.method)(
        *case.arguments, **kwargs, target_workspace=composition.default_workspace, access=access
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.method)
async def test_each_root_method_rejects_non_system_before_owner_or_asset_effects(case):
    """全部 actor 操作许可也不能授予任何管理方法，拒绝先于 owner 与资产注册。"""
    composition = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1", allowed_operations=frozenset(WorkspaceOperation)
            )
        ]
    )
    services, owner, store = _services(composition)
    access = await composition.authenticate()
    upload_source = _UploadSource()

    with pytest.raises(OperationDeniedError) as excinfo:
        await _invoke(case, services, composition, access, upload_source=upload_source)

    assert excinfo.value.details["reason"] == "root_required"
    assert owner.accepted == []
    assert upload_source.read_count == 0
    assert store.list_workspace_assets(make_identity_scope(user_id="u1")) == []


async def _system_root_services():
    """只登记 system 的空白名单准入，root 能力不依赖任何 actor 操作。"""
    composition = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1", agent_id="system", allowed_operations=frozenset()
            )
        ]
    )
    services, owner, store = _services(composition)
    access = await composition.authenticate(agent_id="system")
    return composition, services, owner, store, access


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _ROUTED_CASES, ids=lambda case: case.method)
async def test_each_routed_root_method_reaches_owner_as_system_without_actor_operations(case):
    """system 只经准入与目标校验即可调用管理路由，owner 收到 system 发起的可信 scope。"""
    composition, services, owner, _store, access = await _system_root_services()

    await _invoke(case, services, composition, access)

    assert len(owner.accepted) == 1
    route, scope = owner.accepted[0]
    assert route == case.route
    assert scope.workspace_identity == composition.default_workspace
    assert scope.actor_identity.agent_id == "system"
    assert scope.actor_identity.user_id == "u1"


@pytest.mark.asyncio
async def test_root_upload_registers_asset_as_system_without_actor_operations():
    """上传经 root 授权后注册并解析资产，不经过 Patchouli 路由。"""
    composition, services, owner, store, access = await _system_root_services()

    receipt = await _invoke(_UPLOAD_CASE, services, composition, access)

    assert receipt.handle.asset.state == WorkspaceAssetState.READY
    assert receipt.handle.asset.representations[1].content_object["text"] == "root upload"
    assert len(store.list_workspace_assets(make_identity_scope(user_id="u1"))) == 1
    assert owner.accepted == []
