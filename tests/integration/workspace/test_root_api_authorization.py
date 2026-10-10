"""真实 root 服务、授权者与总线协作，逐方法验收 root 边界与 system 交付。"""

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.attachments import AttachmentParserConfig
from hivememory.core.access import WorkspaceOperation
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import OperationDeniedError
from hivememory.core.models import (
    IdentityScope,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    WorkspaceAssetState,
)
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


_CASES = [
    _Case(
        "memory",
        "create_memory",
        GlobalRoutes.PATCHOULI_MEMORY_CREATE,
        keywords=(
            ("title", "记忆"),
            ("summary", "管理创建"),
            ("content", "正文"),
            ("memory_type", "FACT"),
            ("tags", []),
        ),
    ),
    _Case("memory", "list_memories", GlobalRoutes.PATCHOULI_MEMORY_LIST),
    _Case("memory", "get_memory", GlobalRoutes.PATCHOULI_MEMORY_GET, (_MEMORY_ID,)),
    _Case("memory", "update_memory", GlobalRoutes.PATCHOULI_MEMORY_UPDATE, (_MEMORY_ID,)),
    _Case("memory", "delete_memory", GlobalRoutes.PATCHOULI_MEMORY_DELETE, (_MEMORY_ID,)),
    _Case(
        "memory",
        "record_feedback",
        GlobalRoutes.PATCHOULI_MEMORY_RECORD_FEEDBACK,
        (_MEMORY_ID,),
        (("positive", True), ("source", "root.test")),
    ),
    _Case(
        "profiles",
        "create_agent_profile",
        GlobalRoutes.PATCHOULI_AGENT_PROFILE_CREATE,
        keywords=(("title", "Worker"), ("alias", "worker"), ("tags", [])),
    ),
    _Case("profiles", "list_agent_profiles", GlobalRoutes.PATCHOULI_AGENT_PROFILE_LIST),
    _Case("topic", "list_active_topics", GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE),
    _Case(
        "topic",
        "settle_topic",
        GlobalRoutes.PATCHOULI_MANUAL_SETTLE_TOPIC,
        keywords=(("topic_id", "topic-1"),),
    ),
    _Case(
        "topic",
        "evict_topic",
        GlobalRoutes.PATCHOULI_EVICT_TOPIC,
        keywords=(("topic_id", "topic-1"),),
    ),
    _Case("tasks", "list_memory_tasks", GlobalRoutes.PATCHOULI_MEMORY_TASK_LIST),
    _Case("tasks", "get_memory_task", GlobalRoutes.PATCHOULI_MEMORY_TASK_GET, ("task-1",)),
    _Case("tasks", "cancel_memory_task", GlobalRoutes.PATCHOULI_MEMORY_TASK_CANCEL, ("task-1",)),
    _Case(
        "assets",
        "upload_asset",
        None,
        keywords=(
            ("file_name", "sample.txt"),
            ("declared_media_type", "text/plain"),
            ("client_operation_id", "upload-1"),
        ),
    ),
]


class _OwnerBoundary:
    """资源 owner 的外部替身；记录收到的请求并提供管理服务所需的领域结果。"""

    def __init__(self, bus: GlobalSystemBus) -> None:
        self.accepted: list[tuple[str, IdentityScope]] = []
        for case in _CASES:
            if case.route is not None:
                bus.register(case.route, self._handler(case.route))

    def _handler(self, route):
        async def handle(*args, **kwargs):
            scope = kwargs.get("identity_scope")
            if scope is None:
                scope = args[0]
            self.accepted.append((route, scope))
            if route in {
                GlobalRoutes.PATCHOULI_MEMORY_CREATE,
                GlobalRoutes.PATCHOULI_AGENT_PROFILE_CREATE,
            }:
                return args[1]
            if route in {GlobalRoutes.PATCHOULI_MEMORY_GET, GlobalRoutes.PATCHOULI_MEMORY_UPDATE}:
                return MemoryAtom(
                    id=_MEMORY_ID,
                    meta=make_memory_metadata(user_id="u1", source_agent_id="writer"),
                    index=IndexLayer(title="记忆", summary="管理读取", memory_type=MemoryType.FACT),
                    payload=PayloadLayer(content="正文"),
                )
            if route in {
                GlobalRoutes.PATCHOULI_MEMORY_LIST,
                GlobalRoutes.PATCHOULI_AGENT_PROFILE_LIST,
                GlobalRoutes.PATCHOULI_MEMORY_TASK_LIST,
            }:
                return []
            if route == GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE:
                return ()
            if route == GlobalRoutes.PATCHOULI_MANUAL_SETTLE_TOPIC:
                return TopicSettleResult(topic_id=kwargs["topic_id"])
            if route == GlobalRoutes.PATCHOULI_EVICT_TOPIC:
                return TopicEvictionResult(topic_id=kwargs["topic_id"], removed=True)
            if route == GlobalRoutes.PATCHOULI_MEMORY_TASK_GET:
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
@pytest.mark.parametrize(
    "actor_operations",
    [frozenset(), frozenset(WorkspaceOperation)],
    ids=["no_operations", "all_actor_operations"],
)
async def test_each_root_method_rejects_non_system_before_owner_or_asset_effects(
    case, actor_operations
):
    """actor 操作许可不能授予任何管理方法，拒绝先于 owner 与资产注册。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", allowed_operations=actor_operations)]
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


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.method)
async def test_each_root_method_allows_system_without_actor_operations(case):
    """system 的 root 能力只依赖准入与目标，空 actor 操作登记也能完成管理调用。"""
    composition = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1", agent_id="system", allowed_operations=frozenset()
            )
        ]
    )
    services, owner, store = _services(composition)
    access = await composition.authenticate(agent_id="system")

    result = await _invoke(case, services, composition, access)

    if case.service == "assets":
        assert result.handle.asset.state == WorkspaceAssetState.READY
        assert result.handle.asset.representations[1].content_object["text"] == "root upload"
        assert len(store.list_workspace_assets(make_identity_scope(user_id="u1"))) == 1
        assert owner.accepted == []
    else:
        assert len(owner.accepted) == 1
        route, scope = owner.accepted[0]
        assert route == case.route
        assert scope.workspace_identity == composition.default_workspace
        assert scope.actor_identity.agent_id == "system"
        assert scope.actor_identity.user_id == "u1"
