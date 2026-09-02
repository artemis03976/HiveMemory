"""SystemAssembler、HiveMemorySystem 与真实 AssetStore 的生命周期集成测试。"""

from __future__ import annotations

from typing import Any

import pytest

from hivememory.core.errors import (
    AssetNotFoundError,
    AssetRemovedError,
)
from hivememory.core.models import (
    ActorIdentity,
    AssetRepresentationKind,
    IdentityScope,
    WorkspaceAssetMetadata,
    WorkspaceIdentity,
)
from hivememory.system.assembler import (
    SystemAssembler,
    _RegistriesBundle,
    _RuntimeBundle,
    _ServicesBundle,
    _SubsystemBundle,
)
from hivememory.system.config import HiveMemoryConfig
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.events import RecordingRuntimeEventSink
from hivememory.system.runtime.publisher import RuntimeEventPublisher
from hivememory.system.runtime.workspace import (
    InMemoryWorkspaceAssetStore,
    WorkspaceAssetCommandPort,
    WorkspaceAssetReaderPort,
)
from hivememory.system.system import HiveMemorySystem


class _Scheduler:
    """记录生命周期顺序的外部 scheduler fake。"""

    def __init__(self, calls: list[str]) -> None:
        self._calls = calls
        self.is_running = False

    def start(self) -> None:
        self.is_running = True

    async def stop(self) -> None:
        self._calls.append("scheduler")
        self.is_running = False


class _Ingress:
    """提供 System 生命周期所需的最小 ingress 边界。"""

    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    async def start(self) -> None:
        return None

    async def shutdown_drain(self) -> dict[str, bool]:
        self._calls.append("ingress")
        return {"success": True}


class _LeaseHoldingSubsystem:
    """停止时真实持有并释放 READY representation lease 的消费者。

    用 ``release_representation_lease`` 的返回值证明消费者是显式释放活跃 lease，
    而不是在 ``close_and_clear`` 已经强制清空后无意义地释放（那会返回 False）。
    """

    def __init__(
        self,
        calls: list[str],
        store: InMemoryWorkspaceAssetStore,
        scope: IdentityScope,
        asset_ref: Any,
    ) -> None:
        self._calls = calls
        self._store = store
        self._scope = scope
        self._asset_ref = asset_ref
        self._lease: Any = None
        self.released_active_lease: bool | None = None
        self.asset_id_seen_on_stop: str | None = None

    async def start(self) -> None:
        # 消费者在运行期间取得 READY lease，并在 shutdown drain 期间持有。
        self._lease = self._store.acquire_ready_representation(
            self._scope,
            self._asset_ref,
        )

    async def stop(self) -> None:
        self._calls.append("lease_consumer")
        # 持有期间 Store 必须仍可读取。
        self.asset_id_seen_on_stop = self._lease.representation.asset_id
        # 消费者完成使用后显式 release；返回 True 证明 lease 未被 close 提前清除。
        self.released_active_lease = self._store.release_representation_lease(
            self._lease.lease_id
        )

    async def health(self) -> dict[str, str]:
        return {"status": "ok"}


class _AssetReadingSubsystem:
    """停止时真实读取资产，用可观察结果证明 Store 尚未关闭。"""

    def __init__(
        self,
        name: str,
        calls: list[str],
        store: InMemoryWorkspaceAssetStore,
        scope: IdentityScope,
        asset_ref: Any,
        *,
        fail_on_stop: bool = False,
    ) -> None:
        self.name = name
        self._calls = calls
        self._store = store
        self._scope = scope
        self._asset_ref = asset_ref
        self._fail_on_stop = fail_on_stop
        self.asset_id_seen_on_stop: str | None = None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        self._calls.append(self.name)
        self.asset_id_seen_on_stop = self._store.resolve_asset(
            self._scope,
            self._asset_ref,
        ).asset_id
        if self._fail_on_stop:
            raise RuntimeError(f"{self.name} stop failed")

    async def health(self) -> dict[str, str]:
        return {"status": "ok"}


def _scope() -> IdentityScope:
    """构造集成场景的 main Workspace scope。"""
    return IdentityScope(
        actor_identity=ActorIdentity(user_id="user-1", agent_id="agent-1"),
        workspace_identity=WorkspaceIdentity(
            owner_user_id="user-1",
            workspace_key="main_workspace",
            workspace_id="main_workspace",
        ),
    )


def _ready_asset(store: InMemoryWorkspaceAssetStore, scope: IdentityScope):
    """建立可供所有 shutdown 消费者读取的 READY 资产。"""
    handle = store.create_asset(
        scope,
        WorkspaceAssetMetadata(
            kind="binary",
            display_name="payload.bin",
            media_type="application/octet-stream",
            size_bytes=7,
            required_representation_kind=AssetRepresentationKind.RAW,
        ),
        "upload-1",
    )
    store.register_raw_representation(
        scope,
        handle.asset_ref,
        content_object=b"payload",
        content_hash="payload-hash",
        producer="upload",
        producer_version="1",
    )
    return handle


def _build_system(
    store: InMemoryWorkspaceAssetStore,
    scope: IdentityScope,
    asset_ref: Any,
    *,
    failing_subsystem: str | None = None,
) -> tuple[
    HiveMemorySystem,
    list[str],
    dict[str, _AssetReadingSubsystem],
    RecordingRuntimeEventSink,
]:
    """组合真实 System+Store，并以 fake 隔离本测试边界外的子系统。"""
    calls: list[str] = []
    sink = RecordingRuntimeEventSink()
    subsystems = {
        name: _AssetReadingSubsystem(
            name,
            calls,
            store,
            scope,
            asset_ref,
            fail_on_stop=name == failing_subsystem,
        )
        for name in ("gateway", "patchouli", "alice")
    }
    runtime = _RuntimeBundle(
        global_bus=GlobalSystemBus(),
        scheduler=_Scheduler(calls),  # type: ignore[arg-type]
        workspace_asset_store=store,
        event_bus=None,
        event_sink=sink,
        event_publisher=RuntimeEventPublisher(sink),
    )
    services = _ServicesBundle(
        chat=object(),  # type: ignore[arg-type]
        ingress=_Ingress(calls),  # type: ignore[arg-type]
        memory=object(),  # type: ignore[arg-type]
        memory_task=object(),  # type: ignore[arg-type]
        agent=object(),  # type: ignore[arg-type]
        topic=object(),  # type: ignore[arg-type]
        readiness=object(),  # type: ignore[arg-type]
    )
    system = HiveMemorySystem(
        config=HiveMemoryConfig(runtime_events={"enabled": False}),
        runtime=runtime,
        registries=_RegistriesBundle(
            provider_registry=object(),  # type: ignore[arg-type]
            model_registry=object(),  # type: ignore[arg-type]
        ),
        subsystems=_SubsystemBundle(
            gateway=subsystems["gateway"],  # type: ignore[arg-type]
            patchouli=subsystems["patchouli"],  # type: ignore[arg-type]
            alice=subsystems["alice"],  # type: ignore[arg-type]
        ),
        services=services,
    )
    return system, calls, subsystems, sink


def test_assembler_constructs_one_store_implementing_both_narrow_ports() -> None:
    """捕获 runtime 遗漏 Store、重复容器或只提供宽泛 service locator。"""
    runtime = SystemAssembler(
        HiveMemoryConfig(runtime_events={"enabled": False})
    )._build_runtime()

    assert isinstance(runtime.workspace_asset_store, InMemoryWorkspaceAssetStore)
    assert isinstance(runtime.workspace_asset_store, WorkspaceAssetReaderPort)
    assert isinstance(runtime.workspace_asset_store, WorkspaceAssetCommandPort)


def test_workspace_assets_and_refs_are_isolated_across_workspaces() -> None:
    """捕获同名资产或 opaque ref 被错误合并、跨域读取或删除的缺陷。"""
    store = InMemoryWorkspaceAssetStore()
    main = _scope()
    isolated = IdentityScope(
        actor_identity=ActorIdentity(user_id="user-1", agent_id="agent-1"),
        workspace_identity=WorkspaceIdentity(
            owner_user_id="user-1",
            workspace_key="isolation_workspace",
            workspace_id="isolation_workspace",
        ),
    )
    other_owner = IdentityScope(
        actor_identity=ActorIdentity(user_id="user-2", agent_id="agent-1"),
        workspace_identity=WorkspaceIdentity(
            owner_user_id="user-2",
            workspace_key="main_workspace",
            workspace_id="main_workspace",
        ),
    )
    main_handle = _ready_asset(store, main)
    isolated_handle = _ready_asset(store, isolated)

    assert main_handle.asset.asset_id != isolated_handle.asset.asset_id
    assert main_handle.asset_ref != isolated_handle.asset_ref
    assert [handle.asset.asset_id for handle in store.list_workspace_assets(main)] == [
        main_handle.asset.asset_id
    ]
    assert [
        handle.asset.asset_id for handle in store.list_workspace_assets(isolated)
    ] == [isolated_handle.asset.asset_id]

    with pytest.raises(AssetNotFoundError):
        store.resolve_asset(isolated, main_handle.asset_ref)
    with pytest.raises(AssetNotFoundError):
        store.acquire_ready_representation(isolated, main_handle.asset_ref)
    assert store.list_workspace_assets(other_owner) == []
    with pytest.raises(AssetNotFoundError):
        store.resolve_asset(other_owner, main_handle.asset_ref)

    removed = store.remove_asset(main, main_handle.asset_ref)
    assert removed.state.value == "removed"
    assert store.list_workspace_assets(main) == []
    assert [
        handle.asset.asset_id for handle in store.list_workspace_assets(isolated)
    ] == [isolated_handle.asset.asset_id]
    with pytest.raises(AssetRemovedError):
        store.acquire_ready_representation(main, main_handle.asset_ref)
    assert store.resolve_asset(isolated, isolated_handle.asset_ref).asset_id == (
        isolated_handle.asset.asset_id
    )

    # Store 是进程内 working set；新建实例不应恢复旧 ref 或旧资产。
    fresh_store = InMemoryWorkspaceAssetStore()
    with pytest.raises(AssetNotFoundError):
        fresh_store.resolve_asset(main, main_handle.asset_ref)


@pytest.mark.asyncio
async def test_system_closes_store_only_after_all_asset_consumers_stop() -> None:
    """捕获 Topic settlement 或其他消费者停止前提前清空 AssetStore。"""
    store = InMemoryWorkspaceAssetStore()
    scope = _scope()
    handle = _ready_asset(store, scope)
    system, calls, subsystems, sink = _build_system(store, scope, handle.asset_ref)

    await system.start()
    sink.events.clear()
    await system.stop()

    assert calls == ["scheduler", "ingress", "alice", "patchouli", "gateway"]
    assert {
        name: subsystem.asset_id_seen_on_stop for name, subsystem in subsystems.items()
    } == {
        "gateway": handle.asset.asset_id,
        "patchouli": handle.asset.asset_id,
        "alice": handle.asset.asset_id,
    }
    assert store.is_closed is True
    assert [event.event_type for event in sink.events] == [
        RuntimeEventType.SYSTEM_SHUTTING_DOWN,
        RuntimeEventType.SYSTEM_STOPPED,
    ]
    assert sink.events[-1].data["completed_steps"][-1] == (
        "workspace_asset_store.close_and_clear"
    )


@pytest.mark.asyncio
async def test_upstream_stop_failure_does_not_clear_store_or_report_stopped() -> None:
    """捕获上游 shutdown 失败后仍清空 Store 并谎报有序停止成功。"""
    store = InMemoryWorkspaceAssetStore()
    scope = _scope()
    handle = _ready_asset(store, scope)
    system, calls, _, sink = _build_system(
        store,
        scope,
        handle.asset_ref,
        failing_subsystem="patchouli",
    )
    await system.start()
    sink.events.clear()

    with pytest.raises(RuntimeError, match="patchouli stop failed"):
        await system.stop()

    assert calls == ["scheduler", "ingress", "alice", "patchouli"]
    assert store.is_closed is False
    assert store.resolve_asset(scope, handle.asset_ref).asset_id == handle.asset.asset_id
    assert [event.event_type for event in sink.events] == [
        RuntimeEventType.SYSTEM_SHUTTING_DOWN,
        RuntimeEventType.SYSTEM_STOP_FAILED,
    ]


@pytest.mark.asyncio
async def test_system_waits_for_lease_release_before_close_and_clear() -> None:
    """真实 lease 贯穿 shutdown：消费者先显式 release，close_and_clear 不掩盖泄漏。"""
    store = InMemoryWorkspaceAssetStore()
    scope = _scope()
    handle = _ready_asset(store, scope)
    calls: list[str] = []
    sink = RecordingRuntimeEventSink()
    lease_consumer = _LeaseHoldingSubsystem(calls, store, scope, handle.asset_ref)
    reader_consumers = {
        name: _AssetReadingSubsystem(name, calls, store, scope, handle.asset_ref)
        for name in ("gateway", "alice")
    }

    runtime = _RuntimeBundle(
        global_bus=GlobalSystemBus(),
        scheduler=_Scheduler(calls),  # type: ignore[arg-type]
        workspace_asset_store=store,
        event_bus=None,
        event_sink=sink,
        event_publisher=RuntimeEventPublisher(sink),
    )
    services = _ServicesBundle(
        chat=object(),  # type: ignore[arg-type]
        ingress=_Ingress(calls),  # type: ignore[arg-type]
        memory=object(),  # type: ignore[arg-type]
        memory_task=object(),  # type: ignore[arg-type]
        agent=object(),  # type: ignore[arg-type]
        topic=object(),  # type: ignore[arg-type]
        readiness=object(),  # type: ignore[arg-type]
    )
    system = HiveMemorySystem(
        config=HiveMemoryConfig(runtime_events={"enabled": False}),
        runtime=runtime,
        registries=_RegistriesBundle(
            provider_registry=object(),  # type: ignore[arg-type]
            model_registry=object(),  # type: ignore[arg-type]
        ),
        subsystems=_SubsystemBundle(
            gateway=reader_consumers["gateway"],  # type: ignore[arg-type]
            patchouli=lease_consumer,  # type: ignore[arg-type]
            alice=reader_consumers["alice"],  # type: ignore[arg-type]
        ),
        services=services,
    )

    await system.start()
    await system.stop()

    # patchouli 在 Topic settlement/generation drain 期间持有 lease 时 Store 仍可读，
    # 并成功释放了活跃 lease（而非被 close_and_clear 提前清除）。
    assert lease_consumer.asset_id_seen_on_stop == handle.asset.asset_id
    assert lease_consumer.released_active_lease is True
    assert store.is_closed is True
    assert calls == [
        "scheduler",
        "ingress",
        "alice",
        "lease_consumer",
        "gateway",
    ]
