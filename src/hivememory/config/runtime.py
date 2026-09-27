"""运行时配置段：全局维护调度器（含各维护任务周期）与运行时事件总线。"""

from pydantic import BaseModel, ConfigDict, Field


class MaintenanceTasksConfig(BaseModel):
    observer_idle_flush_interval_seconds: float = Field(default=5.0)
    observer_idle_flush_timeout_seconds: float = Field(default=30.0)
    enable_observer_idle_flush: bool = Field(default=True)
    perception_idle_flush_interval_seconds: float = Field(default=30.0)
    enable_perception_idle_flush: bool = Field(default=True)
    lifecycle_gc_interval_hours: int = Field(default=24)
    enable_lifecycle_gc: bool = Field(default=True)

    model_config = ConfigDict(extra="ignore")


class SchedulerConfig(BaseModel):
    enabled: bool = Field(default=True)
    tick_seconds: float = Field(default=1.0)
    shutdown_wait_seconds: float = Field(default=5.0)
    tasks: MaintenanceTasksConfig = Field(default_factory=MaintenanceTasksConfig)

    model_config = ConfigDict(extra="ignore")


class RuntimeEventsConfig(BaseModel):
    enabled: bool = Field(default=True)
    buffer_size: int = Field(default=1000)
    subscriber_queue_size: int = Field(default=100)

    model_config = ConfigDict(extra="ignore")


__all__ = [
    "MaintenanceTasksConfig",
    "RuntimeEventsConfig",
    "SchedulerConfig",
]
