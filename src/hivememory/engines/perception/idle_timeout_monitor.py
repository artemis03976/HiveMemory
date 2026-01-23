"""
HiveMemory - 空闲超时监控器 (Idle Timeout Monitor)

职责:
    异步监控所有 Buffer，对超时的 Buffer 触发 Flush。

特性:
    - 使用 APScheduler 后台定时扫描
    - 周期性检查所有 Buffer 的空闲状态
    - 自动触发超时 Buffer 的 Flush
    - 可配置扫描间隔和超时时间

参考: PROJECT.md 4.1 节

作者: HiveMemory Team
版本: 1.0.0
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Any, Optional, List

from hivememory.core.models import FlushReason

if TYPE_CHECKING:
    from hivememory.engines.perception.semantic_flow_perception_layer import SemanticFlowPerceptionLayer

logger = logging.getLogger(__name__)


class IdleTimeoutMonitor:
    """
    异步空闲超时监控器

    使用 APScheduler 后台定时扫描所有 Buffer，
    对超时的 Buffer 触发 Flush。

    工作原理:
        1. 启动后，定期（默认 30s）扫描所有 Buffer
        2. 检查每个 Buffer 的 last_update 时间
        3. 如果超过 idle_timeout_seconds，触发 Flush
        4. Flush 后重置 Buffer 状态

    Examples:
        >>> perception = SemanticFlowPerceptionLayer()
        >>> monitor = IdleTimeoutMonitor(
        ...     perception_layer=perception,
        ...     idle_timeout_seconds=900,  # 15 minutes
        ...     scan_interval_seconds=30,  # scan every 30s
        ... )
        >>> monitor.start()
        >>> # ... 后台自动监控 ...
        >>> monitor.stop()
    """

    def __init__(
        self,
        perception_layer: "SemanticFlowPerceptionLayer",
        idle_timeout_seconds: int = 900,  # 15 minutes default
        scan_interval_seconds: int = 30,  # scan every 30s
        enable_schedule: bool = True,
    ):
        """
        初始化空闲超时监控器

        Args:
            perception_layer: 感知层实例（用于访问 Buffer 池和触发 Flush）
            idle_timeout_seconds: 空闲超时时间（秒），默认 900（15 分钟）
            scan_interval_seconds: 扫描间隔（秒），默认 30
            enable_schedule: 是否启用定时调度，默认 True
        """
        self._perception_layer = perception_layer
        self.idle_timeout_seconds = idle_timeout_seconds
        self.scan_interval_seconds = scan_interval_seconds
        self.enable_schedule = enable_schedule

        self._scheduler = None
        self._is_running = False

        # 统计信息
        self._stats: Dict[str, Any] = {
            "last_scan": None,
            "total_scans": 0,
            "total_flushes": 0,
            "buffers_flushed": [],
        }

        logger.info(
            f"IdleTimeoutMonitor 初始化: "
            f"timeout={idle_timeout_seconds}s, "
            f"interval={scan_interval_seconds}s"
        )

    def start(self) -> None:
        """
        启动监控器

        使用 APScheduler 创建后台任务，定期扫描所有 Buffer。
        """
        if self._is_running:
            logger.warning("IdleTimeoutMonitor 已在运行中")
            return

        if not self.enable_schedule:
            logger.info("IdleTimeoutMonitor 调度已禁用")
            return

        try:
            from apscheduler.schedulers.background import BackgroundScheduler

            self._scheduler = BackgroundScheduler()

            # 添加定时任务
            self._scheduler.add_job(
                self._scan_and_flush_idle_buffers,
                "interval",
                seconds=self.scan_interval_seconds,
                id="idle_timeout_scan",
                replace_existing=True,
            )

            self._scheduler.start()
            self._is_running = True

            logger.info(
                f"IdleTimeoutMonitor 已启动: "
                f"扫描间隔={self.scan_interval_seconds}s"
            )

        except ImportError:
            logger.warning(
                "apscheduler 未安装，IdleTimeoutMonitor 已禁用。"
                "安装方式: pip install apscheduler"
            )
            self.enable_schedule = False

    def stop(self) -> None:
        """
        停止监控器

        关闭 APScheduler 调度器。
        """
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
            self._is_running = False
            logger.info("IdleTimeoutMonitor 已停止")

    def _scan_and_flush_idle_buffers(self) -> List[str]:
        """
        扫描所有 Buffer 并刷新超时的 Buffer

        Returns:
            List[str]: 被刷新的 Buffer key 列表
        """
        flushed_keys = []

        try:
            # 获取所有活跃 Buffer
            buffer_keys = self._perception_layer.list_active_buffers()
            current_time = datetime.now().timestamp()

            logger.debug(f"开始扫描 {len(buffer_keys)} 个 Buffer")

            for key in buffer_keys:
                try:
                    # 解析 key
                    parts = key.split(":")
                    if len(parts) != 3:
                        continue

                    user_id, agent_id, session_id = parts

                    # 获取 Buffer
                    buffer = self._perception_layer.get_buffer(
                        user_id, agent_id, session_id
                    )

                    if buffer is None:
                        continue

                    # 检查是否有内容需要 Flush
                    if not buffer.blocks and buffer.current_block is None:
                        continue

                    # 检查是否超时
                    idle_duration = current_time - buffer.last_update
                    if idle_duration > self.idle_timeout_seconds:
                        logger.info(
                            f"Buffer 超时: {key}, "
                            f"空闲时长={idle_duration:.1f}s"
                        )

                        # 触发 Flush
                        self._flush_buffer(user_id, agent_id, session_id)
                        flushed_keys.append(key)

                except Exception as e:
                    logger.error(f"处理 Buffer {key} 时出错: {e}")

            # 更新统计
            self._update_stats(flushed_keys)

            if flushed_keys:
                logger.info(f"本次扫描刷新了 {len(flushed_keys)} 个 Buffer")

        except Exception as e:
            logger.error(f"扫描 Buffer 时出错: {e}")

        return flushed_keys

    def _flush_buffer(
        self,
        user_id: str,
        agent_id: str,
        session_id: str
    ) -> None:
        """
        刷新指定的 Buffer

        调用感知层的 _flush 方法，原因为 IDLE_TIMEOUT。

        Args:
            user_id: 用户ID
            agent_id: Agent ID
            session_id: 会话ID
        """
        try:
            # 获取 Buffer
            buffer = self._perception_layer.get_buffer(
                user_id, agent_id, session_id
            )

            if buffer is None:
                return

            # 使用感知层的锁来保证线程安全
            with self._perception_layer._lock:
                # 将当前 Block 加入（如果存在且完整）
                if buffer.current_block and buffer.current_block.is_complete:
                    buffer.add_block(buffer.current_block)
                    buffer.current_block = None

                if not buffer.blocks:
                    return

                # 调用感知层的 _flush 方法
                self._perception_layer._flush(buffer, FlushReason.IDLE_TIMEOUT)

                # 重置话题核心
                self._perception_layer.adsorber.reset_topic_kernel(buffer)

        except Exception as e:
            logger.error(
                f"刷新 Buffer 失败: {user_id}:{agent_id}:{session_id}, "
                f"错误: {e}"
            )

    def _update_stats(self, flushed_keys: List[str]) -> None:
        """
        更新统计信息

        Args:
            flushed_keys: 被刷新的 Buffer key 列表
        """
        self._stats["last_scan"] = datetime.now().isoformat()
        self._stats["total_scans"] += 1
        self._stats["total_flushes"] += len(flushed_keys)
        self._stats["buffers_flushed"] = flushed_keys

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            Dict[str, Any]: 统计信息字典
        """
        return {
            **self._stats,
            "is_running": self._is_running,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "scan_interval_seconds": self.scan_interval_seconds,
        }

    def scan_now(self) -> List[str]:
        """
        立即执行一次扫描（手动触发）

        用于测试或立即检查空闲 Buffer。

        Returns:
            List[str]: 被刷新的 Buffer key 列表
        """
        logger.info("手动触发空闲 Buffer 扫描")
        return self._scan_and_flush_idle_buffers()

    @property
    def is_running(self) -> bool:
        """
        监控器是否正在运行

        Returns:
            bool: 是否运行中
        """
        return self._is_running


__all__ = [
    "IdleTimeoutMonitor",
]
