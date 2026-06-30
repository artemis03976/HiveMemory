from unittest.mock import AsyncMock, Mock, patch

import pytest

from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.memory_library.models import (
    StorageHealthComponent,
    StorageHealthReport,
)
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.runtime.core import PatchouliRuntime
from hivememory.patchouli.runtime.memory_tasks import (
    MemoryGenerationTaskStatus,
    MemoryGenerationTaskWaitResult,
    MemoryGenerationTaskWaitSummary,
)
from hivememory.patchouli.services.perception import ShutdownFlushResult
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.events import RecordingRuntimeEventSink


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
        runtime._task_controller.wait_all = AsyncMock(
            return_value=MemoryGenerationTaskWaitSummary.from_results([])
        )
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
            return_value=ShutdownFlushResult(
                success=True,
                trigger_reason="shutdown",
                flushed_topics=["t1"],
                skipped_topics=[],
                archived_blocks=1,
            )
        )

        result = await runtime.shutdown_drain()

        runtime.perception_familiar.flush_all_for_shutdown.assert_awaited_once()
        runtime._task_controller.wait_all.assert_awaited_once_with(timeout=30.0)
        runtime._task_controller.cancel_many.assert_not_awaited()
        assert result["reentrant"] is False
        assert result["perception"].trigger_reason == "shutdown"
        assert result["generation"].timed_out == 0
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
            "success": True,
            "trigger_reason": "shutdown",
            "flushed_topic_count": 1,
            "skipped_topic_count": 0,
            "archived_blocks": 1,
        }
        assert completed.data["generation"]["timed_out"] == 0
        assert isinstance(completed.data["duration_ms"], float)

    @pytest.mark.asyncio
    async def test_shutdown_drain_reports_generation_timeout(self):
        runtime = _create_runtime()
        runtime.perception_familiar.flush_all_for_shutdown = AsyncMock(
            return_value=ShutdownFlushResult(
                success=True,
                trigger_reason="shutdown",
                flushed_topics=["t1"],
                skipped_topics=[],
                archived_blocks=1,
            )
        )
        runtime._task_controller.wait_all = AsyncMock(
            return_value=MemoryGenerationTaskWaitSummary(
                requested=1,
                found=1,
                missing=0,
                completed=0,
                failed=0,
                cancelled=0,
                pending=0,
                running=1,
                timed_out=1,
                results=(
                    MemoryGenerationTaskWaitResult(
                        task_id="memory-task-timeout",
                        found=True,
                        timed_out=True,
                        status=MemoryGenerationTaskStatus.RUNNING,
                    ),
                ),
            )
        )
        runtime._task_controller.cancel_many = AsyncMock(return_value=1)

        result = await runtime.shutdown_drain()

        runtime._task_controller.cancel_many.assert_awaited_once_with(
            ["memory-task-timeout"],
            reason="shutdown_timeout",
        )
        assert result["success"] is False
        assert result["generation"].timed_out == 1
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
            return_value=ShutdownFlushResult(
                success=True,
                trigger_reason="shutdown",
                flushed_topics=[],
                skipped_topics=[],
                archived_blocks=0,
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
        assert events[-1].data["perception"]["flushed_topic_count"] == 0
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

    @pytest.mark.asyncio
    async def test_ensure_storage_ready_delegates_to_store(self):
        runtime = _create_runtime()
        runtime.storage.ensure_ready = AsyncMock()

        await runtime.ensure_storage_ready()

        runtime.storage.ensure_ready.assert_awaited_once()
