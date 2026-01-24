"""
HiveMemory 感知层抽象接口

定义感知层各组件的抽象接口，遵循依赖倒置原则。

参考: PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 1.0.0
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any, Dict, TYPE_CHECKING

from hivememory.core.models import FlushReason, Identity, StreamMessage, StreamMessageType
from hivememory.engines.perception.models import (
    LogicalBlock,
    SemanticBuffer,
)

if TYPE_CHECKING:
    from hivememory.engines.perception.models import SimpleBuffer

logger = logging.getLogger(__name__)


class TriggerStrategy(ABC):
    """
    触发策略接口

    职责:
        判断是否应该触发记忆处理。

    实现策略:
        - MessageCountTrigger: 消息数阈值
        - IdleTimeoutTrigger: 超时触发
        - SemanticBoundaryTrigger: 语义边界检测
    """

    @abstractmethod
    def should_trigger(
        self,
        messages: List[StreamMessage],
        context: Dict[str, Any]
    ) -> tuple[bool, Optional[FlushReason]]:
        """
        判断是否应该触发

        Args:
            messages: 当前缓冲区的消息列表
            context: 上下文信息 (last_trigger_time, buffer_size 等)

        Returns:
            tuple[bool, FlushReason]: (是否触发, 触发原因)

        Examples:
            >>> trigger = MessageCountTrigger(threshold=5)
            >>> should, reason = trigger.should_trigger(messages, {})
            >>> if should:
            ...     print(f"触发原因: {reason.value}")
        """
        pass


class StreamParser(ABC):
    """
    流式解析器接口

    职责：
        - 将原始消息流转换为 StreamMessage
        - 构建 LogicalBlock
        - 判断是否需要创建新的 LogicalBlock

    Examples:
        >>> parser = UnifiedStreamParser()
        >>> message = parser.parse_message({"role": "user", "content": "hello"})
        >>> print(message.message_type)  # StreamMessageType.USER
    """

    @abstractmethod
    def parse_message(self, raw_message: Any) -> StreamMessage:
        """
        解析原始消息为统一格式

        支持的格式：
            - LangChain: AIMessage, HumanMessage, ToolMessage
            - OpenAI: {"role": "...", "content": "...", "tool_calls": ...}
            - 简单文本: 默认为 user 消息

        Args:
            raw_message: 原始消息对象

        Returns:
            StreamMessage: 统一格式的流式消息
        """
        pass

    @abstractmethod
    def should_create_new_block(self, message: StreamMessage) -> bool:
        """
        判断是否需要创建新的 LogicalBlock

        规则：
            - User Query 永远开启新 Block
            - 其他消息延续当前 Block

        Args:
            message: 流式消息

        Returns:
            bool: 是否需要创建新 Block
        """
        pass


class SemanticAdsorber(ABC):
    """
    语义吸附器接口

    职责：
        - 计算语义相似度
        - 判断是吸附还是漂移
        - 更新话题核心向量

    判定流程：
        1. 短文本强吸附（< 50 tokens）
        2. 语义相似度判定（阈值 0.6）

    注意：
        - Token 溢出检测由 RelayController 负责
        - 空闲超时检测由 BasePerceptionLayer 的 start_idle_monitor() 处理（异步）

    Examples:
        >>> adsorber = SemanticBoundaryAdsorber()
        >>> should_adsorb, reason = adsorber.should_adsorb(new_block, buffer)
        >>> if not should_adsorb:
        ...     print(f"触发 Flush: {reason}")
    """

    @abstractmethod
    def compute_similarity(
        self,
        anchor_text: str,
        topic_kernel: Optional[List[float]]
    ) -> float:
        """
        计算语义相似度

        Args:
            anchor_text: 锚点文本（新 Block 的 User Query）
            topic_kernel: 话题核心向量

        Returns:
            float: 相似度 (0-1)，如果 topic_kernel 为 None 返回 0
        """
        pass

    @abstractmethod
    def should_adsorb(
        self,
        new_block: LogicalBlock,
        buffer: SemanticBuffer
    ) -> Tuple[bool, Optional[FlushReason]]:
        """
        判断是否吸附

        Args:
            new_block: 新的 LogicalBlock
            buffer: 当前语义缓冲区

        Returns:
            Tuple[bool, Optional[FlushReason]]:
                - 是否吸附
                - 漂移原因（如果不吸附）
        """
        pass

    @abstractmethod
    def update_topic_kernel(
        self,
        buffer: SemanticBuffer,
        new_block: LogicalBlock
    ) -> None:
        """
        更新话题核心向量

        策略：指数移动平均 (EMA)
            new_kernel = alpha * new_vector + (1 - alpha) * old_kernel

        Args:
            buffer: 当前语义缓冲区
            new_block: 新的 LogicalBlock
        """
        pass


class RelayController(ABC):
    """
    接力控制器接口

    职责：
        - 检测 Token 溢出
        - 生成中间态摘要
        - 维护跨 Block 的上下文连贯性

    Examples:
        >>> controller = TokenOverflowRelayController()
        >>> if controller.should_trigger_relay(buffer, new_block):
        ...     summary = controller.generate_summary(buffer.blocks)
    """

    @abstractmethod
    def should_trigger_relay(
        self,
        buffer: SemanticBuffer,
        new_block: LogicalBlock
    ) -> bool:
        """
        检测是否需要接力（Token 溢出）

        Args:
            buffer: 当前语义缓冲区
            new_block: 新的 LogicalBlock

        Returns:
            bool: 是否需要触发接力
        """
        pass

    @abstractmethod
    def generate_summary(self, blocks: List[LogicalBlock]) -> str:
        """
        生成中间态摘要

        Args:
            blocks: LogicalBlock 列表

        Returns:
            str: 生成的摘要文本
        """
        pass


class BasePerceptionLayer(ABC):
    """
    感知层抽象基类

    定义所有类型的 PerceptionLayer 的统一接口，并提供空闲超时监控的默认实现。

    两种策略实现：
        - SimplePerceptionLayer: 简单缓冲策略（StreamMessage + 三重触发）
        - SemanticFlowPerceptionLayer: 语义流策略（LogicalBlock + 语义吸附）

    空闲超时监控：
        所有子类都继承统一的空闲超时监控功能，通过 start_idle_monitor() 启动。

    Examples:
        >>> from hivememory.core.models import Identity
        >>> perception = SimplePerceptionLayer()
        >>> perception.start_idle_monitor()  # 启动空闲监控
        >>> identity = Identity(user_id="user1", agent_id="agent1", session_id="sess1")
        >>> perception.add_message("user", "hello", identity)
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
        current_time = datetime.now().timestamp()

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
                    # SimpleBuffer: 检查 messages
                    # SemanticBuffer: 检查 blocks 或 current_block
                    has_content = False
                    if hasattr(buffer, "messages"):
                        has_content = len(buffer.messages) > 0
                    elif hasattr(buffer, "blocks"):
                        has_content = (
                            len(buffer.blocks) > 0 or buffer.current_block is not None
                        )

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

    # ========== 抽象接口 ==========

    @abstractmethod
    def add_message(
        self,
        role: str,
        content: str,
        identity: Identity,
        rewritten_query: Optional[str] = None,
        gateway_intent: Optional[str] = None,
        worth_saving: Optional[bool] = None,
    ) -> None:
        """
        添加消息到感知层

        Args:
            role: 角色 (user/assistant/system)
            content: 消息内容
            identity: 身份标识对象
            rewritten_query: Gateway 重写后的查询（可选）
            gateway_intent: Gateway 意图分类结果（可选）
            worth_saving: Gateway 价值判断（可选）
        """
        pass

    @abstractmethod
    def flush_buffer(
        self,
        identity: Identity,
        reason: FlushReason = FlushReason.MANUAL,
    ) -> List[StreamMessage]:
        """
        手动刷新缓冲区，返回消息列表

        注意：统一的返回类型，便于 orchestrator 处理

        Args:
            identity: 身份标识对象
            reason: 刷新原因

        Returns:
            List[StreamMessage]: 缓冲区的消息列表
        """
        pass

    @abstractmethod
    def get_buffer(
        self,
        identity: Identity,
    ) -> Optional[Any]:
        """
        获取缓冲区对象

        注意：返回类型取决于具体实现
            - SimplePerceptionLayer: 返回 SimpleBuffer
            - SemanticFlowPerceptionLayer: 返回 SemanticBuffer

        Args:
            identity: 身份标识对象

        Returns:
            缓冲区对象，不存在返回 None
        """
        pass

    @abstractmethod
    def clear_buffer(
        self,
        identity: Identity,
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
        identity: Identity,
    ) -> Dict[str, Any]:
        """
        获取缓冲区信息

        Args:
            identity: 身份标识对象

        Returns:
            Dict: 缓冲区信息字典
        """
        pass


class GreyAreaArbiter(ABC):
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
            bool: 是否可用（模型是否已加载）
        """
        return True


__all__ = [
    "TriggerStrategy",
    "StreamParser",
    "SemanticAdsorber",
    "RelayController",
    "BasePerceptionLayer",
    "GreyAreaArbiter",
]
