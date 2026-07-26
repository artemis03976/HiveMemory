"""Passive conversation ingress 配置。

按 v0.6.0 设计，Passive ingress 配置属于 System application 层，
不并入 Gateway 配置；Gateway 只识别固定的 `GatewayIngressMode`。

注意：idle flush 的扫描间隔与超时阈值仍由 `scheduler.tasks` 单一持有
（`observer_idle_flush_interval_seconds` / `observer_idle_flush_timeout_seconds`），
此处不重复定义，避免出现两个真相源。
"""

from pydantic import BaseModel, ConfigDict, Field


class PassiveIngressConfig(BaseModel):
    """被动接入的有界进程内资源约束。

    v0.6.0 只承诺进程内幂等与进程内 outbox，不承诺跨进程 exactly-once；
    后续可替换为持久化 ingress store 而保持同一提交语义。
    """

    dedup_ttl_seconds: float = Field(
        default=300.0,
        description="external_event_id 幂等窗口（秒）",
    )
    max_dedup_entries: int = Field(
        default=4096,
        description="幂等 registry 的最大条目数，超出按最旧淘汰",
    )
    max_buffered_events_per_turn: int = Field(
        default=256,
        description="单个 turn 允许累计的最大事件数，超出丢弃并记录",
    )
    max_outbox_items_per_conversation: int = Field(
        default=32,
        description="单个外部会话允许挂起的最大 sealed turn 数，超出按最旧淘汰",
    )

    model_config = ConfigDict(extra="ignore")


__all__ = ["PassiveIngressConfig"]
