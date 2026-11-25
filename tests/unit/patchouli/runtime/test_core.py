import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

import pytest

from hivememory.components.events.bus import RecordingRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.config.patchouli import PatchouliConfig
from hivememory.config.shared import SharedConfig
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.models import TopicData, WorkspaceIdentity, record_topic_writes
from hivememory.infrastructure.model_service import ModelLoadState
from hivememory.memory_library.models import (
    StorageHealthComponent,
    StorageHealthReport,
)
from hivememory.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.memory_generation.saved_work import SavedWorkReport
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.runtime.core import PatchouliRuntime
from hivememory.patchouli.runtime.models import TopicShutdownFlushReport
from hivememory.patchouli.runtime.topic_pool import save_topic_pool
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
        patchouli_config.shutdown.topic_pool_file = "saved/topic_pool.json"
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
        runtime.memory_library.short_term = ShortTermMemoryStore()
        runtime.memory_library.mid_term.upsert = Mock()
        runtime.memory_library.mid_term.delete = Mock()
        return runtime


class TestRuntimeShutdownDrain:
    @pytest.mark.asyncio
    async def test_shutdown_drain_stops_dispatch_before_settling_and_saves_unfinished(self):
        runtime = _create_runtime()
        steps = Mock()
        stop_dispatch = AsyncMock()
        flush = AsyncMock(
            return_value=TopicShutdownFlushReport(
                settled_topic_ids=("t1", "t2"),
                retired_topic_ids=("t3",),
                resident_block_count=2,
            )
        )
        save = AsyncMock(
            return_value=SavedWorkReport(saved_by_source={"SETTLE": 1}, file_written=True)
        )
        steps.attach_mock(stop_dispatch, "stop_dispatch")
        steps.attach_mock(flush, "flush")
        steps.attach_mock(save, "save")
        runtime._task_controller.stop_dispatch = stop_dispatch
        runtime.perception_familiar.flush_all_for_shutdown = flush
        runtime._task_controller.save_unfinished = save

        result = await runtime.shutdown_drain()

        # 停止取新工作先于结算：结算产生的任务只入队，与排队中的工作一起保存。
        assert steps.mock_calls == [
            call.stop_dispatch(),
            call.flush(),
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
        assert completed.data["perception"] == {
            "settled_topic_count": 2,
            "retired_topic_count": 1,
            "failed_topic_count": 0,
            "resident_block_count": 2,
        }
        assert completed.data["generation"]["saved"] == 1
        assert completed.data["generation"]["file_written"] is True

    @pytest.mark.asyncio
    async def test_shutdown_drain_is_reentrant(self):
        runtime = _create_runtime()
        runtime.perception_familiar.flush_all_for_shutdown = AsyncMock(
            return_value=TopicShutdownFlushReport(
                settled_topic_ids=(),
                resident_block_count=0,
            )
        )

        first = await runtime.shutdown_drain()
        second = await runtime.shutdown_drain()

        assert first["reentrant"] is False
        assert second["reentrant"] is True
        runtime.perception_familiar.flush_all_for_shutdown.assert_awaited_once()
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
        assert events[-1].data["perception"]["settled_topic_count"] == 0
        assert events[-1].data["generation"]["saved"] == 0

    @pytest.mark.asyncio
    async def test_shutdown_drain_failure_emits_failed_event(self):
        runtime = _create_runtime()
        runtime.perception_familiar.flush_all_for_shutdown = AsyncMock(
            side_effect=RuntimeError("flush boom")
        )

        with pytest.raises(RuntimeError, match="flush boom"):
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
        assert failed.data["perception"] is None
        assert failed.data["generation"] is None
        assert failed.data["error_class"] == "RuntimeError"


def _runtime_with_topic_pool(tmp_path, *, max_resident=5):
    """话题池的保存与读回落在真实的短期存储与驻留工作集上，快照写在临时目录。"""
    runtime = _create_runtime()
    runtime._patchouli_config.shutdown.topic_pool_file = str(tmp_path / "topic_pool.json")
    store = ShortTermMemoryStore()
    working_set = TopicWorkingSet(max_resident=max_resident)
    runtime.memory_library.short_term = store
    runtime._services["perception"] = PerceptionFamiliar(
        engine=None,
        store=store,
        working_set=working_set,
        bus=runtime.local_bus,
        config=SimpleNamespace(idle_timeout_seconds=60),
    )
    return runtime, store, working_set


def _workspace(user_id="u1") -> WorkspaceIdentity:
    return make_identity_scope(user_id=user_id).workspace_identity


def _settled_topic(topic_id: str, *, last_update: float, workspace=None) -> TopicData:
    """接收过一轮交互、已经全部沉淀的话题：关闭时不再沉淀，只随快照保存。"""
    block = make_logical_block(f"question {topic_id}", f"answer {topic_id}")
    return TopicData(
        topic_id=topic_id,
        workspace_identity=workspace or _workspace(),
        topic_title=f"title {topic_id}",
        interactions=(block,),
        last_update=last_update,
        settled_interaction_count=1,
    )


class TestRuntimeTopicPool:
    @pytest.mark.asyncio
    async def test_drain_saves_topic_pool_after_committing_work_records_its_writes(self, tmp_path):
        """关闭时话题池在保存生成工作之后写出：等待中写完的任务记回话题的写入在快照里；下次
        启动读回话题，读回即删除快照。"""
        runtime, store, _ = _runtime_with_topic_pool(tmp_path)
        topic = _settled_topic("t1", last_update=10.0)
        store.put(topic)

        async def save_unfinished(path):
            # 保存生成工作时等已开始写入的任务写完，它的写入在此记回话题。
            await runtime.perception_familiar.record_writes(
                "t1",
                belong_to=topic.workspace_identity,
                written=frozenset({"m"}),
                absorbed_into={},
            )
            return SavedWorkReport()

        runtime._task_controller.save_unfinished = AsyncMock(side_effect=save_unfinished)

        result = await runtime.shutdown_drain()
        restarted, restored_store, _ = _runtime_with_topic_pool(tmp_path)
        report = await restarted.restore_topic_pool()

        assert (result["topic_pool"].saved, result["topic_pool"].file_written) == (1, True)
        assert (report.read, report.restored, report.rejected) == (1, 1, 0)
        assert restored_store.list_all() == [record_topic_writes(topic, frozenset({"m"}), {})]
        assert not (tmp_path / "topic_pool.json").exists()

    @pytest.mark.asyncio
    async def test_drain_completes_when_a_topic_cannot_be_encoded(self, tmp_path):
        """话题里有不能编码为 UTF-8 的文本（半个代理对）时话题池写不出，关闭照常完成，
        完成事件报告没有保存。"""
        runtime, store, _ = _runtime_with_topic_pool(tmp_path)
        broken = _settled_topic("t1", last_update=1.0)
        store.put(TopicData.model_validate({**dict(broken), "topic_title": "deploy \ud83d"}))

        result = await runtime.shutdown_drain()

        assert (result["topic_pool"].saved, result["topic_pool"].file_written) == (1, False)
        completed = runtime._test_runtime_events.events[-1]
        assert (completed.status, completed.severity) == ("completed_without_saved_work", "error")
        assert not (tmp_path / "topic_pool.json").exists()

    @pytest.mark.asyncio
    async def test_restore_registers_residency_from_oldest_to_newest_update(self, tmp_path):
        """读回按 last_update 从旧到新登记驻留：最久没有写入的话题是第一个 LRU 候选。"""
        save_topic_pool(
            tmp_path / "topic_pool.json",
            [
                _settled_topic("middle", last_update=2.0),
                _settled_topic("newest", last_update=3.0),
                _settled_topic("oldest", last_update=1.0),
            ],
        )
        runtime, _, working_set = _runtime_with_topic_pool(tmp_path)

        await runtime.restore_topic_pool()

        assert [topic_id for _, topic_id in working_set.list_shutdown_candidates()] == [
            "oldest",
            "middle",
            "newest",
        ]

    @pytest.mark.asyncio
    async def test_restore_sets_aside_the_oldest_topics_beyond_capacity(self, tmp_path):
        """同一 Workspace 读回的话题超过驻留容量时只读回最近的若干个，最久的写进 rejected 文件；
        其他 Workspace 的话题不受影响。"""
        save_topic_pool(
            tmp_path / "topic_pool.json",
            [
                _settled_topic("oldest", last_update=1.0),
                _settled_topic("middle", last_update=2.0),
                _settled_topic("newest", last_update=3.0),
                _settled_topic("other", last_update=0.5, workspace=_workspace("u2")),
            ],
        )
        runtime, store, _ = _runtime_with_topic_pool(tmp_path, max_resident=2)

        report = await runtime.restore_topic_pool()

        assert sorted(topic.topic_id for topic in store.list_all()) == ["middle", "newest", "other"]
        rejected = json.loads(Path(report.rejected_file).read_text(encoding="utf-8"))["topics"]
        assert [item["topic_id"] for item in rejected] == ["oldest"]
        completed = runtime._test_runtime_events.events[-1]
        assert (completed.source, completed.status, completed.severity) == (
            "patchouli.restore_topic_pool",
            "completed_with_rejections",
            "error",
        )

    @pytest.mark.asyncio
    async def test_restore_sets_aside_invalid_topics_and_restores_the_rest(self, tmp_path):
        """单个话题校验失败（沉淀进度超过交互数）时写进 rejected 文件，其余照常读回。"""
        path = tmp_path / "topic_pool.json"
        valid = _settled_topic("valid", last_update=1.0)
        invalid = _settled_topic("invalid", last_update=2.0).model_dump(mode="json")
        invalid["settled_interaction_count"] = 5
        document = {"format_version": 1, "topics": [valid.model_dump(mode="json"), invalid]}
        path.write_text(json.dumps(document), encoding="utf-8")
        runtime, store, _ = _runtime_with_topic_pool(tmp_path)

        report = await runtime.restore_topic_pool()

        assert store.list_all() == [valid]
        rejected = json.loads(Path(report.rejected_file).read_text(encoding="utf-8"))["topics"]
        assert rejected == [invalid]
        assert not path.exists()

    @pytest.mark.asyncio
    async def test_restore_sets_aside_unreadable_snapshot_and_starts_empty(self, tmp_path):
        """整份快照读不回时改名为 rejected 文件，话题池从空开始，启动照常继续。"""
        path = tmp_path / "topic_pool.json"
        path.write_text("{not json", encoding="utf-8")
        runtime, store, _ = _runtime_with_topic_pool(tmp_path)

        report = await runtime.restore_topic_pool()

        assert store.list_all() == []
        assert not path.exists()
        assert Path(report.rejected_file).read_text(encoding="utf-8") == "{not json"


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
