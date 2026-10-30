from unittest.mock import AsyncMock, Mock, patch

import pytest

from hivememory.components.events.bus import RecordingRuntimeEventSink
from hivememory.config.patchouli import PatchouliConfig
from hivememory.config.shared import SharedConfig
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskStatus,
)
from hivememory.patchouli.memory_library.models import (
    StorageHealthComponent,
    StorageHealthReport,
)
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.runtime.core import PatchouliRuntime
from hivememory.patchouli.runtime.models import TopicShutdownFlushReport
from tests.helpers.workspace import make_identity_scope


def _memory_task(
    task_id: str,
    status: MemoryGenerationTaskStatus,
) -> MemoryGenerationTask:
    return MemoryGenerationTask(
        task_id=task_id,
        topic_id="topic-1",
        label=task_id,
        source=MemoryGenerationSource.WRITE,
        status=status,
        belong_to=make_identity_scope().workspace_identity,
        from_actor=make_identity_scope().actor_identity,
    )


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
        patchouli_config.shutdown.generation_wait_timeout_seconds = 30.0
        runtime_events = RecordingRuntimeEventSink()
        runtime = PatchouliRuntime(
            patchouli_config=patchouli_config,
            shared_config=Mock(),
            runtime_events=runtime_events,
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
        runtime._task_controller.wait_all = AsyncMock(return_value=[])
        runtime._task_controller.cancel_many = AsyncMock(return_value=0)
        runtime.storage = Mock()
        runtime.memory_library = Mock()
        runtime.memory_library.mid_term.upsert = Mock()
        runtime.memory_library.mid_term.delete = Mock()
        return runtime


class TestRuntimeShutdownDrain:
    @pytest.mark.asyncio
    async def test_shutdown_drain_flushes_perception_once(self):
        runtime = _create_runtime()
        runtime.perception_familiar.flush_all_for_shutdown = AsyncMock(
            return_value=TopicShutdownFlushReport(
                settled_topic_ids=("t1", "t2"),
                generation_skipped_topic_ids=("t2",),
                resident_block_count=2,
            )
        )

        result = await runtime.shutdown_drain()

        runtime.perception_familiar.flush_all_for_shutdown.assert_awaited_once()
        runtime._task_controller.wait_all.assert_awaited_once_with(timeout=30.0)
        runtime._task_controller.cancel_many.assert_not_awaited()
        assert result["reentrant"] is False
        assert result["perception"].settled_topic_ids == ("t1", "t2")
        assert result["perception"].generation_skipped_topic_ids == ("t2",)
        assert result["generation"]["timed_out"] == 0
        assert result["generation_cancelled_after_timeout"] == 0
        events = runtime._test_runtime_events.events
        assert [event.event_type for event in events] == [
            RuntimeEventType.SUBSYSTEM_OPERATION_STARTED,
            RuntimeEventType.SUBSYSTEM_OPERATION_COMPLETED,
        ]
        assert events[0].status == "started"
        assert events[0].source == "patchouli.shutdown_drain"
        assert events[0].subsystem == "patchouli"
        assert events[0].component == "patchouli_runtime"
        assert events[0].data["operation_key"] == "patchouli.shutdown_drain"
        assert events[0].data["operation_name"] == "shutdown_drain"
        assert events[0].data["operation_kind"] == "shutdown"
        completed = events[-1]
        assert completed.status == "completed"
        assert completed.data["operation_key"] == "patchouli.shutdown_drain"
        assert completed.data["success"] is True
        assert completed.data["perception"] == {
            "settled_topic_count": 2,
            "generation_skipped_topic_count": 1,
            "resident_block_count": 2,
        }
        assert completed.data["generation"]["timed_out"] == 0
        assert isinstance(completed.data["duration_ms"], float)

    @pytest.mark.asyncio
    async def test_shutdown_drain_reports_generation_timeout(self):
        runtime = _create_runtime()
        runtime.perception_familiar.flush_all_for_shutdown = AsyncMock(
            return_value=TopicShutdownFlushReport(
                settled_topic_ids=("t1",),
                resident_block_count=1,
            )
        )
        runtime._task_controller.wait_all = AsyncMock(
            return_value=[
                _memory_task(
                    "memory-task-timeout",
                    MemoryGenerationTaskStatus.RUNNING,
                )
            ]
        )
        runtime._task_controller.cancel_many = AsyncMock(return_value=1)

        result = await runtime.shutdown_drain()

        runtime._task_controller.cancel_many.assert_awaited_once_with(
            ["memory-task-timeout"],
            reason="shutdown_timeout",
        )
        assert result["success"] is False
        assert result["generation"]["timed_out"] == 1
        assert result["generation_cancelled_after_timeout"] == 1
        completed = runtime._test_runtime_events.events[-1]
        assert completed.event_type == RuntimeEventType.SUBSYSTEM_OPERATION_COMPLETED
        assert completed.status == "completed_with_timeout"
        assert completed.severity == "warning"
        assert completed.data["success"] is False
        assert completed.data["generation"]["running"] == 1
        assert completed.data["generation"]["timed_out"] == 1
        assert completed.data["generation_cancelled_after_timeout"] == 1

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
        runtime._task_controller.wait_all.assert_awaited_once()
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
        assert events[-1].data["generation"]["requested"] == 0

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
        assert failed.reason == "flush boom"
        assert failed.data["success"] is False
        assert failed.data["perception"] is None
        assert failed.data["generation"] is None
        assert failed.data["error"] == "flush boom"


class TestRuntimeLocalRoutes:
    def test_local_routes_exclude_public_workflow_mirrors(self):
        routes = set(PatchouliLocalRoutes.ALL)
        assert not any(route.startswith("patchouli.public.") for route in routes)
        assert not any("prepare_agent_run" in route for route in routes)
        assert not any("finalize_agent_run" in route for route in routes)
        assert not any("cleanup_prepared_agent_run" in route for route in routes)

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

    def test_disabled_reranker_does_not_block_readiness(self):
        runtime = _create_runtime()
        runtime.storage.embedding_service = self._model_service(ready=True)
        runtime.reranker_service = None

        assert runtime.is_models_ready() is True

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
