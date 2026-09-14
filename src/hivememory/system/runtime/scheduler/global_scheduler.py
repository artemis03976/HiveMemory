from hivememory.system.runtime.scheduler.async_scheduler import AsyncMaintenanceScheduler


class GlobalMaintenanceScheduler(AsyncMaintenanceScheduler):
    """全局维护调度器 — 由 HiveMemorySystem 持有，服务各子系统与应用服务。"""
