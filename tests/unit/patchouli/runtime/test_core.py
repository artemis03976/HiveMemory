from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

import pytest

from hivememory.components.events.bus import RecordingRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.config.patchouli import PatchouliConfig
from hivememory.config.shared import SharedConfig
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.models import WorkspaceIdentity
from hivememory.engines.perception.models import TriggerReason
from hivememory.infrastructure.model_service import ModelLoadState
from hivememory.memory_library.adapters.short_term import JsonlShortTermStorage
from hivememory.memory_library.models import (
    StorageHealthComponent,
    StorageHealthReport,
)
from hivememory.memory_library.stores import ShortTermMemoryStore
from hivememory.memory_library.topic_log import (
    TopicCreatedRecord,
    TopicInteractionRecord,
    TopicRetiredRecord,
    encode_topic_record,
)
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.memory_generation.saved_work import SavedWorkReport
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.runtime.core import PatchouliRuntime
from hivememory.patchouli.services.perception import PerceptionFamiliar
from hivememory.patchouli.services.topic_working_set import TopicWorkingSet
from tests.helpers.sessions import make_logical_block
from tests.helpers.workspace import make_identity_scope


def _create_runtime():
    with (
        patch.object(PatchouliRuntime, "_init_infrastructure"),
        patch.object(
            PatchouliRuntime,
            "_build_engines",
            return_value={
                "perception": Mock(),
                "generation": Mock(),
                "lifecycle": Mock(),
                "retrieval": Mock(),
                "artifact": Mock(),
            },
        ),
        patch.object(PatchouliRuntime, "_register_services"),
    ):
        patchouli_config = Mock()
        patchouli_config.shutdown.pending_work_file = "saved/pending_generation.json"
        runtime_events = RecordingRuntimeEventSink()
        runtime = PatchouliRuntime(
            patchouli_config=patchouli_config,
            shared_config=Mock(),
            runtime_events=RuntimeEventPublisher(runtime_events),
        )
        runtime._test_runtime_events = runtime_events
        runtime._services = {
            "perception": Mock(),
            "retrieval": Mock(),
            "generation": Mock(),
            "generation_coordinator": Mock(),
            "lifecycle": Mock(),
        }
        runtime._task_controller = Mock()
        runtime._task_controller.stop_dispatch = AsyncMock()
        runtime._task_controller.save_unfinished = AsyncMock(return_value=SavedWorkReport())
        runtime.storage = Mock()
        runtime.memory_library = Mock()
        runtime.memory_library.mid_term.upsert = Mock()
        runtime.memory_library.mid_term.delete = Mock()
        return runtime


class TestRuntimeShutdownDrain:
    @pytest.mark.asyncio
    async def test_shutdown_drain_stops_dispatch_before_saving_unfinished(self):
        runtime = _create_runtime()
        steps = Mock()
        stop_dispatch = AsyncMock()
        save = AsyncMock(
            return_value=SavedWorkReport(saved_by_source={"WRITE": 1}, file_written=True)
        )
        steps.attach_mock(stop_dispatch, "stop_dispatch")
        steps.attach_mock(save, "save")
        runtime._task_controller.stop_dispatch = stop_dispatch
        runtime._task_controller.save_unfinished = save

        result = await runtime.shutdown_drain()

        # 停止取新工作先于保存：保存时排队中的工作不会再被取走执行。
        assert steps.mock_calls == [
            call.stop_dispatch(),
            call.save(Path("saved/pending_generation.json")),
        ]
        assert result["reentrant"] is False
        assert result["success"] is True
        events = runtime._test_runtime_events.events
        assert [event.event_type for event in events] == [
            RuntimeEventType.SUBSYSTEM_OPERATION_STARTED,
            RuntimeEventType.SUBSYSTEM_OPERATION_COMPLETED,
        ]
        assert events[0].source == "patchouli.shutdown_drain"
        assert events[0].data["operation_kind"] == "shutdown"
        completed = events[-1]
        assert completed.status == "completed"
        assert completed.data["generation"]["saved"] == 1
        assert completed.data["generation"]["file_written"] is True

    @pytest.mark.asyncio
    async def test_shutdown_drain_is_reentrant(self):
        runtime = _create_runtime()

        first = await runtime.shutdown_drain()
        second = await runtime.shutdown_drain()

        assert first["reentrant"] is False
        assert second["reentrant"] is True
        runtime._task_controller.save_unfinished.assert_awaited_once()
        events = runtime._test_runtime_events.events
        assert [event.event_type for event in events] == [
            RuntimeEventType.SUBSYSTEM_OPERATION_STARTED,
            RuntimeEventType.SUBSYSTEM_OPERATION_COMPLETED,
            RuntimeEventType.SUBSYSTEM_OPERATION_STARTED,
            RuntimeEventType.SUBSYSTEM_OPERATION_COMPLETED,
        ]
        assert events[-2].data["reentrant"] is True
        assert events[-1].data["reentrant"] is True
        assert events[-1].data["generation"]["saved"] == 0

    @pytest.mark.asyncio
    async def test_shutdown_drain_failure_emits_failed_event(self):
        runtime = _create_runtime()
        runtime._task_controller.save_unfinished = AsyncMock(side_effect=RuntimeError("save boom"))

        with pytest.raises(RuntimeError, match="save boom"):
            await runtime.shutdown_drain()

        events = runtime._test_runtime_events.events
        assert [event.event_type for event in events] == [
            RuntimeEventType.SUBSYSTEM_OPERATION_STARTED,
            RuntimeEventType.SUBSYSTEM_OPERATION_FAILED,
        ]
        failed = events[-1]
        assert failed.status == "failed"
        assert failed.severity == "error"
        assert failed.reason == "operation_failed"
        assert failed.data["success"] is False
        assert failed.data["generation"] is None
        assert failed.data["error_class"] == "RuntimeError"


def _runtime_with_topic_logs(tmp_path, *, max_resident=5):
    """话题的读回落在真实的短期存储、文件端口与驻留工作集上，日志写在临时目录。"""
    runtime = _create_runtime()
    store = ShortTermMemoryStore(JsonlShortTermStorage(tmp_path / "topics"))
    working_set = TopicWorkingSet(max_resident=max_resident)
    runtime.memory_library.short_term = store
    runtime._services["perception"] = PerceptionFamiliar(
        engine=Mock(build_block=Mock()),
        fold_summary_writer=Mock(),
        progress_writer=None,
        store=store,
        working_set=working_set,
        bus=runtime.local_bus,
        config=SimpleNamespace(idle_timeout_seconds=60),
    )
    return runtime, store, working_set


def _workspace(user_id="u1") -> WorkspaceIdentity:
    return make_identity_scope(user_id=user_id).workspace_identity


def _write_log(tmp_path, topic_id: str, *, at: float, retired: TriggerReason | None = None):
    """写一份上次运行留下的话题日志：在 ``at`` 建立并接收一轮交互，可选地随后退役。"""
    port = JsonlShortTermStorage(tmp_path / "topics")
    records = [
        TopicCreatedRecord(at=at, workspace_identity=_workspace(), topic_title=topic_id),
        TopicInteractionRecord(at=at, block=make_logical_block(f"question {topic_id}", "answer")),
    ]
    if retired is not None:
        records.append(TopicRetiredRecord(at=at, reason=retired))
    for record in records:
        port.append(topic_id, encode_topic_record(record))


class TestRuntimeTopicPool:
    @pytest.mark.asyncio
    async def test_restore_registers_residency_from_oldest_to_newest_update(self, tmp_path):
        """读回按最近写入时间从旧到新登记驻留：最久没有写入的话题是第一个 LRU 候选。"""
        for topic_id, at in (("middle", 2.0), ("newest", 3.0), ("oldest", 1.0)):
            _write_log(tmp_path, topic_id, at=at)
        runtime, _, working_set = _runtime_with_topic_logs(tmp_path)

        await runtime.restore_topic_pool()

        order: list[str] = []
        for _ in range(3):
            order.append(working_set.select_lru_candidate(_workspace(), exclude=set(order)))
        assert order == ["oldest", "middle", "newest"]

    @pytest.mark.asyncio
    async def test_topics_beyond_capacity_are_all_read_back_and_retired_by_lru(self, tmp_path):
        """驻留容量在两次运行之间调小时全部读回；下一次需要新话题时由 LRU 逐个退役，退役前先
        沉淀未沉淀的交互。"""
        for topic_id, at in (("oldest", 1.0), ("middle", 2.0), ("newest", 3.0)):
            _write_log(tmp_path, topic_id, at=at)
        runtime, store, _ = _runtime_with_topic_logs(tmp_path, max_resident=2)
        submitted = []

        async def submit_settlement(task):
            submitted.append((task.topic_id, task.retires_topic))
            return None

        runtime.local_bus.register(
            PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT, submit_settlement
        )

        report = await runtime.restore_topic_pool()
        topic_id = await runtime.perception_familiar.prepare_topic(
            "NEW_TOPIC", "新话题", "", _workspace()
        )

        assert report.restored == 3
        assert submitted == [("oldest", True), ("middle", True)]
        assert sorted(topic.topic_id for topic in store.list_by_workspace(_workspace())) == sorted(
            ["newest", topic_id]
        )

    @pytest.mark.asyncio
    async def test_restore_reports_retired_and_rejected_logs_and_resettles_after_queue(
        self, tmp_path
    ):
        """读回报告回到话题池、待重新沉淀与读不回的日志数；退役话题不回到话题池，它没有完成的
        最后一次沉淀在之后带着原来的退役原因重新发起。"""
        _write_log(tmp_path, "live", at=1.0)
        _write_log(tmp_path, "retired", at=2.0, retired=TriggerReason.MANUAL_SETTLE)
        (tmp_path / "topics" / "broken.jsonl").write_bytes(b"{not json\n")
        runtime, store, working_set = _runtime_with_topic_logs(tmp_path)
        submitted = []

        async def submit_settlement(task):
            submitted.append((task.topic_id, task.reason, task.retires_topic))
            return Mock()

        runtime.local_bus.register(
            PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT, submit_settlement
        )

        report = await runtime.restore_topic_pool()
        resident = working_set.select_lru_candidate(_workspace())
        await runtime.resettle_retired_topics()

        assert (report.restored, report.retired, report.rejected) == (1, 1, 1)
        assert ([topic.topic_id for topic in store.list_by_workspace(_workspace())], resident) == (
            ["live"],
            "live",
        )
        assert submitted == [("retired", TriggerReason.MANUAL_SETTLE, True)]
        completed = runtime._test_runtime_events.events[-1]
        assert (completed.source, completed.status, completed.severity) == (
            "patchouli.restore_topic_pool",
            "completed_with_rejections",
            "error",
        )


class TestRuntimeLocalRoutes:
    def test_local_routes_exclude_public_workflow_mirrors(self):
        routes = set(PatchouliLocalRoutes.ALL)
        assert not any(route.startswith("patchouli.public.") for route in routes)

    def test_mount_local_routes_registers_all_declared_routes(self):
        runtime = _create_runtime()
        service = Mock()

        runtime.mount_local_routes(service)

        assert set(PatchouliLocalRoutes.ALL).issubset(set(runtime.list_local_routes()))

    def test_unmount_local_routes_removes_declared_routes(self):
        runtime = _create_runtime()
        runtime.mount_local_routes(Mock())

        runtime.unmount_local_routes()

        assert set(runtime.list_local_routes()).isdisjoint(set(PatchouliLocalRoutes.ALL))

    @pytest.mark.asyncio
    async def test_missing_local_route_raises_explicit_key_error(self):
        bus = PatchouliBus()

        with pytest.raises(KeyError, match="missing.route"):
            await bus.request("missing.route")


class TestRuntimeStorageHealth:
    @pytest.mark.asyncio
    async def test_check_storage_health_uses_memory_library_report(self):
        runtime = _create_runtime()
        runtime.memory_library.check_storage_health = AsyncMock(
            return_value=StorageHealthReport(
                components=(
                    StorageHealthComponent("short_term", True),
                    StorageHealthComponent("mid_term", True),
                    StorageHealthComponent("long_term", True),
                )
            )
        )

        assert await runtime.check_storage_health() is True
        runtime.memory_library.check_storage_health.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_check_storage_health_returns_false_on_required_failure(self):
        runtime = _create_runtime()
        runtime.memory_library.check_storage_health = AsyncMock(
            return_value=StorageHealthReport(
                components=(
                    StorageHealthComponent("short_term", True),
                    StorageHealthComponent(
                        "mid_term",
                        False,
                        detail="qdrant down",
                    ),
                )
            )
        )

        assert await runtime.check_storage_health() is False
        runtime.memory_library.check_storage_health.assert_awaited_once()


class TestRuntimeRerankerAssembly:
    """重排序服务按 retrieval.reranker 装配，与召回策略无关。"""

    @staticmethod
    def _build_runtime(retrieval: dict) -> tuple[PatchouliRuntime, Mock]:
        reranker_factory = Mock()
        with (
            patch("hivememory.infrastructure.storage.QdrantMemoryStore"),
            patch(
                "hivememory.infrastructure.rerank.get_fast_embed_reranker_service",
                reranker_factory,
            ),
            patch.object(PatchouliRuntime, "_build_memory_library"),
            patch.object(PatchouliRuntime, "_build_engines", return_value={}),
            patch.object(PatchouliRuntime, "_register_services"),
        ):
            runtime = PatchouliRuntime(
                patchouli_config=PatchouliConfig.model_validate({"retrieval": retrieval}),
                shared_config=SharedConfig(),
            )
        return runtime, reranker_factory

    def test_enabled_reranker_is_built_for_any_retriever_type(self):
        """装配只读取 retrieval.reranker，召回类型不参与判断，取 dense 代表。"""
        _, reranker_factory = self._build_runtime(
            {
                "retriever": {"type": "dense"},
                "reranker": {"enabled": True, "model_name": "test/reranker"},
            }
        )

        reranker_factory.assert_called_once()
        assert reranker_factory.call_args.kwargs["config"].model_name == "test/reranker"

    def test_disabled_reranker_is_not_built(self):
        runtime, reranker_factory = self._build_runtime(
            {"retriever": {"type": "dense"}, "reranker": {"enabled": False}}
        )

        assert runtime.reranker_service is None
        reranker_factory.assert_not_called()


class TestRuntimeModelReadiness:
    """模型就绪要求每个启用的模型都就绪；关闭的重排模型不参与判断。"""

    @staticmethod
    def _model_service(*, ready: bool) -> Mock:
        service = Mock()
        service.is_ready.return_value = ready
        return service

    @pytest.mark.parametrize(
        ("embedding_ready", "reranker_ready"),
        [(False, True), (True, False)],
        ids=["embedding-not-ready", "reranker-not-ready"],
    )
    def test_any_enabled_model_not_ready_makes_models_not_ready(
        self, embedding_ready, reranker_ready
    ):
        runtime = _create_runtime()
        runtime.storage.embedding_service = self._model_service(ready=embedding_ready)
        runtime.reranker_service = self._model_service(ready=reranker_ready)

        assert runtime.is_models_ready() is False

    def test_failed_models_lists_only_models_whose_last_load_failed(self):
        runtime = _create_runtime()
        runtime.storage.embedding_service = Mock(model_name="embed", state=ModelLoadState.FAILED)
        runtime.reranker_service = Mock(model_name="rerank", state=ModelLoadState.LOADING)

        assert runtime.failed_models() == ["embed"]

    def test_disabled_reranker_does_not_block_readiness(self):
        runtime = _create_runtime()
        runtime.storage.embedding_service = self._model_service(ready=True)
        runtime.reranker_service = None

        assert runtime.is_models_ready() is True

    @pytest.mark.asyncio
    async def test_unload_idle_models_reports_only_models_unloaded_this_cycle(self):
        """只返回本次卸载的模型名；都没有卸载时为空，调度器据此不发事件。"""
        runtime = _create_runtime()
        runtime.storage.embedding_service = Mock(model_name="embedding-model")
        runtime.storage.embedding_service.unload_if_idle = AsyncMock(return_value=True)
        runtime.reranker_service = Mock(model_name="reranker-model")
        runtime.reranker_service.unload_if_idle = AsyncMock(return_value=False)

        assert await runtime.unload_idle_models() == ["embedding-model"]

    @pytest.mark.asyncio
    async def test_warmup_failure_of_one_model_does_not_skip_the_others(self):
        """预热失败只记录日志，其余模型照常加载。"""
        runtime = _create_runtime()
        runtime.storage.embedding_service = Mock()
        runtime.storage.embedding_service.load = AsyncMock(side_effect=OSError("model missing"))
        runtime.reranker_service = Mock()
        runtime.reranker_service.load = AsyncMock()

        await runtime.warmup_models()

        runtime.reranker_service.load.assert_awaited_once()
