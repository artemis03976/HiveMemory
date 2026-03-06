"""
HiveMemory 感知层抽象接口

定义感知层各组件的抽象接口，遵循依赖倒置原则。

参考: PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 1.0.0
"""

import logging
import datetime
from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict, Callable, TYPE_CHECKING
from hivememory.core.models import Identity, StreamMessage
from hivememory.engines.perception.models import (
    FlushReason,
    InteractionPayload,
)

if TYPE_CHECKING:
    from hivememory.engines.perception.models import SemanticBuffer

logger = logging.getLogger(__name__)


class BaseArbiter(ABC):
    """
    灰度仲裁器接口

    职责：
        - 处理语义相似度处于灰度区间（0.40-0.75）的模糊情况
        - 使用更精细的模型判断两个意图是否属于同一任务流
        - 返回是否应该继续当前话题

    判定流程：
        1. 接收上一轮上下文和当前查询
        2. 使用 Reranker/SLM 等模型进行二分类
        3. 返回 YES（继续）或 NO（切分）

    Examples:
        >>> arbiter = RerankerArbiter(reranker_service)
        >>> result = arbiter.should_continue_topic(
        ...     previous_context="写贪吃蛇游戏代码",
        ...     current_query="部署到服务器",
        ...     similarity_score=0.55
        ... )
        >>> # result = True (同一任务流的不同阶段)
    """

    @abstractmethod
    def should_continue_topic(
        self,
        previous_context: str,
        current_query: str,
        similarity_score: float,
    ) -> bool:
        """
        判断是否应该继续当前话题

        Args:
            previous_context: 上一轮对话的上下文摘要
            current_query: 当前的用户查询（rewritten_query）
            similarity_score: 语义相似度分数（可选，用于记录或调整决策）

        Returns:
            bool: True 表示应该继续（吸附），False 表示应该切分
        """
        pass

    def is_available(self) -> bool:
        """
        检查仲裁器是否可用

        Returns:
            bool: 是否可用
        """
        return True


class BasePerceptionLayer(ABC):
    """
    感知层抽象基类

    定义所有类型的 PerceptionLayer 的统一接口，并提供空闲超时监控的默认实现。

    实现策略：
        - SemanticFlowPerceptionLayer: 语义流策略（LogicalBlock + 语义吸附 + MMU）

    空闲超时监控：
        所有子类都继承统一的空闲超时监控功能，通过 start_idle_monitor() 启动。

    Examples:
        >>> from hivememory.core.models import Identity
        >>> perception = SemanticFlowPerceptionLayer()
        >>> perception.start_idle_monitor()  # 启动空闲监控
        >>> identity = Identity(user_id="user1", agent_id="agent1", session_id="sess1")
        >>> perception.ingest_payload(payload)
        >>> messages = perception.flush_buffer(identity)
    """

    def __init__(self, *args, **kwargs):
        """
        基类构造函数，初始化空闲超时监控相关属性。

        注意：使用 *args, **kwargs 以兼容子类的不同构造函数签名。
        """
        super().__init__(*args, **kwargs)
        # 空闲超时监控配置
        self._idle_timeout_seconds: int = 900  # 15 分钟默认
        self._scan_interval_seconds: int = 30  # 扫描间隔 30 秒
        self._idle_monitor_scheduler = None
        self._idle_monitor_running: bool = False

    def set_flush_callback(self, callback: Callable[[List[StreamMessage], FlushReason], None]) -> None:
        """
        设置缓冲区刷新回调函数

        Args:
            callback: 刷新时调用的函数，接收 StreamMessage 列表和 FlushReason 参数
        """
        self.on_flush_callback = callback

    # ========== 空闲超时监控（默认实现）==========

    def start_idle_monitor(
        self,
        idle_timeout_seconds: int = 900,
        scan_interval_seconds: int = 30,
    ) -> None:
        """
        启动空闲超时监控器

        使用 APScheduler 后台定时扫描所有 Buffer，
        对超时的 Buffer 自动触发 Flush。

        Args:
            idle_timeout_seconds: 空闲超时时间（秒），默认 900（15 分钟）
            scan_interval_seconds: 扫描间隔（秒），默认 30

        Examples:
            >>> perception = SemanticFlowPerceptionLayer()
            >>> perception.start_idle_monitor()
            >>> # 后台自动监控空闲 Buffer
        """
        if self._idle_monitor_running:
            logger.warning("空闲超时监控器已在运行中")
            return

        self._idle_timeout_seconds = idle_timeout_seconds
        self._scan_interval_seconds = scan_interval_seconds

        try:
            from apscheduler.schedulers.background import BackgroundScheduler

            self._idle_monitor_scheduler = BackgroundScheduler()

            # 添加定时任务
            self._idle_monitor_scheduler.add_job(
                self._scan_and_flush_idle_buffers,
                "interval",
                seconds=self._scan_interval_seconds,
                id="idle_timeout_scan",
                replace_existing=True,
            )

            self._idle_monitor_scheduler.start()
            self._idle_monitor_running = True

            logger.info(
                f"空闲超时监控器已启动: "
                f"timeout={idle_timeout_seconds}s, "
                f"interval={scan_interval_seconds}s"
            )

        except ImportError:
            logger.warning(
                "apscheduler 未安装，空闲超时监控器已禁用。"
                "安装方式: pip install apscheduler"
            )

    def stop_idle_monitor(self) -> None:
        """
        停止空闲超时监控器

        Examples:
            >>> perception.stop_idle_monitor()
        """
        if self._idle_monitor_scheduler:
            self._idle_monitor_scheduler.shutdown(wait=False)
            self._idle_monitor_scheduler = None
            self._idle_monitor_running = False
            logger.info("空闲超时监控器已停止")

    def scan_idle_buffers_now(self) -> List[str]:
        """
        立即执行一次空闲 Buffer 扫描（手动触发）

        用于测试或立即检查空闲 Buffer。

        Returns:
            List[str]: 被刷新的 Buffer key 列表

        Examples:
            >>> flushed_keys = perception.scan_idle_buffers_now()
            >>> print(f"刷新了 {len(flushed_keys)} 个 Buffer")
        """
        logger.info("手动触发空闲 Buffer 扫描")
        return self._scan_and_flush_idle_buffers()

    def _scan_and_flush_idle_buffers(self) -> List[str]:
        """
        扫描所有 Buffer 并刷新超时的 Buffer（内部方法）

        子类可以重写此方法以定制扫描逻辑。

        Returns:
            List[str]: 被刷新的 Buffer key 列表
        """
        flushed_keys = []
        current_time = datetime.datetime.now().timestamp()

        try:
            # 获取所有活跃 Buffer
            buffer_keys = self.list_active_buffers()

            logger.debug(f"开始扫描 {len(buffer_keys)} 个 Buffer")

            for key in buffer_keys:
                try:
                    # 解析 key
                    parts = key.split(":")
                    if len(parts) != 3:
                        continue

                    user_id, agent_id, session_id = parts

                    # 获取 Buffer
                    buffer = self.get_buffer(
                        Identity(
                            user_id=user_id,
                            agent_id=agent_id,
                            session_id=session_id
                        )
                    )

                    if buffer is None:
                        continue

                    # 检查是否有内容需要 Flush
                    # SemanticBuffer: 检查 blocks
                    has_content = False
                    if hasattr(buffer, "blocks"):
                        has_content = len(buffer.blocks) > 0

                    if not has_content:
                        continue

                    # 检查是否超时
                    if hasattr(buffer, "is_idle"):
                        is_timeout = buffer.is_idle(self._idle_timeout_seconds)
                    else:
                        # 回退方案：直接检查 last_update
                        idle_duration = current_time - buffer.last_update
                        is_timeout = idle_duration > self._idle_timeout_seconds

                    if is_timeout:
                        logger.info(
                            f"Buffer 超时: {key}, "
                            f"空闲时长={current_time - buffer.last_update:.1f}s"
                        )

                        # 触发 Flush
                        self.flush_buffer(
                            Identity(
                                user_id=user_id,
                                agent_id=agent_id,
                                session_id=session_id
                            ),
                            FlushReason.IDLE_TIMEOUT
                        )
                        flushed_keys.append(key)

                except Exception as e:
                    logger.error(f"处理 Buffer {key} 时出错: {e}")

            if flushed_keys:
                logger.info(f"本次扫描刷新了 {len(flushed_keys)} 个 Buffer")

        except Exception as e:
            logger.error(f"扫描 Buffer 时出错: {e}")

        return flushed_keys

    @property
    def idle_monitor_running(self) -> bool:
        """
        监控器是否正在运行

        Returns:
            bool: 是否运行中
        """
        return self._idle_monitor_running

    # ========== Kernel 模式载荷摄入 (v3.0) ==========

    @abstractmethod
    def ingest_payload(self, payload: InteractionPayload) -> None:
        """
        摄入 Kernel 递归循环的完整交互载荷

        感知层唯一合法入口。子类必须实现此方法。

        Args:
            payload: Kernel → Perception 的原子传输包
        """
        pass

    # ========== MMU 路由与话题管理 (Phase 4.5) ==========

    def route_and_ingest(
        self,
        topic_id: str,
        payload: InteractionPayload,
    ) -> None:
        """
        路由到指定话题并摄入载荷 (MMU 模式)

        默认实现：忽略 topic_id，直接调用 ingest_payload。
        SemanticFlowPerceptionLayer 重写此方法实现真正的路由。

        Args:
            topic_id: 目标话题 ID 或 "NEW_TOPIC"
            payload: 原子传输包
        """
        self.ingest_payload(payload)

    def get_active_topics_menu(self) -> List[Dict[str, str]]:
        """
        获取活跃话题菜单，供 TheEye 路由决策使用

        默认实现：返回空列表（无话题路由能力）。
        SemanticFlowPerceptionLayer 重写此方法。

        Returns:
            List[Dict]: [{"topic_id": ..., "title": ..., "buffer_key": ...}, ...]
        """
        return []

    # ========== 抽象接口 ==========

    @abstractmethod
    def flush_buffer(
        self,
        topic_id: str,
        reason: FlushReason = FlushReason.MANUAL,
    ) -> List[StreamMessage]:
        """
        手动刷新缓冲区，返回消息列表

        注意：统一的返回类型，便于 orchestrator 处理

        Args:
            topic_id: 话题 ID
            reason: 刷新原因

        Returns:
            List[StreamMessage]: 缓冲区的消息列表
        """
        pass

    @abstractmethod
    def get_buffer(
        self,
        topic_id: str,
    ) -> Optional[Any]:
        """
        获取缓冲区对象

        返回类型: SemanticBuffer

        Args:
            topic_id: 话题 ID

        Returns:
            缓冲区对象，不存在返回 None
        """
        pass

    @abstractmethod
    def clear_buffer(
        self,
        topic_id: str,
    ) -> bool:
        """清理指定的缓冲区"""
        pass

    @abstractmethod
    def list_active_buffers(self) -> List[str]:
        """列出所有活跃的缓冲区 key"""
        pass

    @abstractmethod
    def get_buffer_info(
        self,
        topic_id: str,
    ) -> Dict[str, Any]:
        """
        获取缓冲区信息

        Args:
            topic_id: 话题 ID

        Returns:
            Dict: 缓冲区信息字典
        """
        pass


__all__ = [
    "BaseArbiter",
    "BasePerceptionLayer",
]
