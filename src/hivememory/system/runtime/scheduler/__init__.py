from hivememory.system.runtime.scheduler.async_scheduler import AsyncMaintenanceScheduler
from hivememory.system.runtime.scheduler.global_scheduler import GlobalMaintenanceScheduler
from hivememory.system.runtime.scheduler.models import MaintenanceTaskSpec, TaskRuntimeState

# Legacy compat: patchouli types used by PassiveIngressService until migration
from hivememory.patchouli.kernel.runtime.maintenance_scheduler import (
    MaintenanceTaskSpec as LegacyMaintenanceTaskSpec,
    SystemAsyncScheduler,
)

__all__ = [
    "AsyncMaintenanceScheduler",
    "GlobalMaintenanceScheduler",
    "MaintenanceTaskSpec",
    "TaskRuntimeState",
    "LegacyMaintenanceTaskSpec",
    "SystemAsyncScheduler",
]
