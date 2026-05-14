from hivememory.system.runtime.scheduler.async_scheduler import AsyncMaintenanceScheduler


class GlobalMaintenanceScheduler(AsyncMaintenanceScheduler):
    """Global maintenance scheduler — held by RuntimeHost, serves all subsystems and application services."""
