"""运行时配置段：全局维护调度器（含各维护任务周期）与运行时事件总线。"""

from pydantic import BaseModel, ConfigDict, Field


class MaintenanceTasksConfig(BaseModel):
    perception_idle_settle_interval_seconds: float = Field(default=30.0)
    enable_perception_idle_settle: bool = Field(default=True)
    # 检查话题是否停顿到该写一条进展记录；停顿阈值由感知配置决定。
    perception_progress_record_interval_seconds: float = Field(default=30.0)
    enable_perception_progress_record: bool = Field(default=True)
    lifecycle_gc_interval_hours: int = Field(default=24)
    enable_lifecycle_gc: bool = Field(default=True)
    # 检查 idle_unload 推理模型是否空闲超时；空闲时限由各模型配置决定。
    model_idle_unload_interval_seconds: float = Field(default=60.0)
    enable_model_idle_unload: bool = Field(default=True)

    model_config = ConfigDict(extra="forbid")


class SchedulerConfig(BaseModel):
    tick_seconds: float = Field(default=1.0)
    shutdown_wait_seconds: float = Field(default=5.0)
    tasks: MaintenanceTasksConfig = Field(default_factory=MaintenanceTasksConfig)

    model_config = ConfigDict(extra="forbid")


class RuntimeEventsConfig(BaseModel):
    enabled: bool = Field(default=True)
    buffer_size: int = Field(default=1000)
    subscriber_queue_size: int = Field(default=100)

    model_config = ConfigDict(extra="forbid")


__all__ = [
    "MaintenanceTasksConfig",
    "RuntimeEventsConfig",
    "SchedulerConfig",
]
