from hivememory.system.runtime.scheduler.async_scheduler import AsyncMaintenanceScheduler


class GlobalMaintenanceScheduler(AsyncMaintenanceScheduler):
    """Global maintenance scheduler — held by HiveMemorySystem, serves subsystems and application services."""
