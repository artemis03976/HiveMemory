"""
帕秋莉体系 (The Patchouli System / The Facility)

系统的外层容器，持有 TheEye (Ingress Gateway) 和 PatchouliKernel (内核调度器)。
这是用户（开发者）唯一需要 import 的东西。

架构 (v3.0):
    PatchouliSystem 是"大图书馆"的完整设施 (The Facility)，包含：
    - TheEye (真理之眼): Ingress Gateway，独立于 Kernel 之外
    - PatchouliKernel (帕秋莉内核): 中心调度器，管理微服务

    数据流: User → PatchouliSystem → TheEye.gaze() → Kernel.handle() → Services

    ┌─────────────────────────────────────────┐
    │  PatchouliSystem (The Facility)         │
    │                                         │
    │  TheEye (Gateway) ──→ PatchouliKernel   │
    │                        ├── Retrieval    │
    │                        ├── Librarian    │
    │                        └── (Koakuma)    │
    └─────────────────────────────────────────┘

作者: HiveMemory Team
版本: 3.0
"""

import logging
from typing import List, Optional, Dict, Any

from hivememory.core.models import Identity, StreamMessage
from hivememory.patchouli.protocol.models import Observation

from hivememory.patchouli.config import HiveMemoryConfig, load_app_config
from hivememory.patchouli.eye import TheEye
from hivememory.patchouli.kernel import PatchouliKernel
from hivememory.patchouli.kernel.retrieval_familiar import RetrievalFamiliar
from hivememory.patchouli.kernel.librarian_core import LibrarianCore

logger = logging.getLogger(__name__)


class PatchouliSystem:
    """
    帕秋莉体系 (The Facility) - HiveMemory 的完整封装 v3.0

    外层容器，持有 TheEye (Ingress Gateway) 和 PatchouliKernel (内核调度器)。
    TheEye 独立于 Kernel 之外，做第一道拦截与信息重整，
    处理完后将标准化请求传入 Kernel 进行调度。

    架构:
        - TheEye (真理之眼): Ingress Gateway，流量入口、意图判断、查询重写
        - PatchouliKernel (帕秋莉内核): 中心调度器
            - RetrievalFamiliar (检索使魔): 上下文检索 (Hot)
            - LibrarianCore (馆长本体): 后台记忆维护 (Cold)

    使用示例:
        >>> from hivememory.patchouli.system import PatchouliSystem
        >>>
        >>> system = PatchouliSystem()
        >>>
        >>> result = system.process_interaction(
        ...     role="user",
        ...     content="帮我写贪吃蛇游戏",
        ...     context=[],
        ...     user_id="user123"
        ... )
    """

    def __init__(
        self,
        config: Optional[HiveMemoryConfig] = None,
    ):
        """
        初始化帕秋莉系统

        Args:
            config: 完整的 HiveMemory 配置（可选）

        Examples:
            >>> system = PatchouliSystem()
            >>>
            >>> from hivememory.patchouli.config import load_app_config
            >>> config = load_app_config("path/to/config.yaml")
            >>> system = PatchouliSystem(config=config)
        """
        self.config = config or load_app_config()

        # 1. 初始化 Kernel（内核管理 Retrieval + Librarian 微服务）
        self.kernel = PatchouliKernel(config=self.config)

        # 2. 初始化 Gateway
        self._init_gateway()

        # 3. 构建 TheEye
        self.eye = TheEye(engine=self._gateway_engine)

        logger.info("PatchouliSystem 帕秋莉系统初始化完成")

    def _init_gateway(self) -> None:
        """
        初始化 Gateway 相关基础设施

        Gateway LLM 和 Gateway Engine 属于 TheEye 的依赖，
        独立于 Kernel 管理。
        """
        from hivememory.infrastructure.llm import get_gateway_llm_service
        self._gateway_llm_service = get_gateway_llm_service(
            config=self.config.llm.gateway
        )

        from hivememory.engines.gateway import (
            GatewayEngine,
            BaseInterceptor, create_interceptor,
            BaseSemanticAnalyzer, create_semantic_analyzer,
        )

        config = self.config.gateway

        interceptor: BaseInterceptor = create_interceptor(config.interceptor)

        semantic_analyzer: BaseSemanticAnalyzer = create_semantic_analyzer(
            config.analyzer,
            self._gateway_llm_service
        )

        self._gateway_engine = GatewayEngine(
            interceptor=interceptor,
            semantic_analyzer=semantic_analyzer,
        )

    # ========== 向后兼容属性 ==========

    @property
    def retrieval_familiar(self) -> RetrievalFamiliar:
        """访问检索使魔（代理到 Kernel）"""
        return self.kernel.retrieval_familiar

    @property
    def librarian_core(self) -> LibrarianCore:
        """访问馆长本体（代理到 Kernel）"""
        return self.kernel.librarian_core

    @property
    def storage(self):
        """访问存储层（代理到 Kernel）"""
        return self.kernel.storage

    # ========== 公开 API ==========

    def process_interaction(
        self,
        role: str,
        content: str,
        user_id: str,
        agent_id: str = "default",
        session_id: Optional[str] = None,
        context: Optional[List[StreamMessage]] = None,
    ) -> Dict[str, Any]:
        """
        统一交互入口 (Unified Interaction Entry)

        自动根据角色分流处理：
        - User: Eye 拦截重写 → Kernel 调度 Retrieval + Librarian
        - Assistant/System: 直接投递 Kernel 冷路径

        Args:
            role: 消息角色 (user/assistant/system)
            content: 消息内容
            user_id: 用户 ID
            agent_id: Agent ID
            session_id: 会话 ID
            context: 对话历史上下文 (仅 User 消息需要，用于指代消解)

        Returns:
            Dict: 处理结果
                - intent: 意图 (Chat/RAG/Record)
                - memory: 检索到的记忆 (仅 User RAG)
                - rewritten: 重写后的查询 (仅 User)
        """
        identity = Identity(
            user_id=user_id, agent_id=agent_id, session_id=session_id
        )

        if role == "user":
            return self._process_hot(
                query=content,
                context=context or [],
                identity=identity,
            )
        else:
            self._process_cold(
                role=role,
                content=content,
                identity=identity,
            )
            return {
                "intent": "record_only",
                "memory": None,
                "rewritten": None,
                "worth_saving": True,
            }

    def _process_hot(
        self,
        query: str,
        context: List[StreamMessage],
        identity: Identity,
    ) -> Dict[str, Any]:
        """
        [Hot Path] Eye 拦截 → Kernel 调度

        Step 1: TheEye.gaze() — 意图识别、查询重写 → EyeGazeResult
        Step 2: Kernel.handle_hot() — 数据格式转换 + 调度 Retrieval + Librarian
        """
        # Eye: 拦截与信息重整
        gaze_result = self.eye.gaze(
            query=query,
            context=context,
            identity=identity,
        )

        # Kernel: 数据格式转换 + 调度微服务
        result = self.kernel.handle_hot(gaze_result=gaze_result)

        return result.model_dump()

    def _process_cold(
        self,
        role: str,
        content: str,
        identity: Identity,
    ) -> None:
        """
        [Cold Path] 直接投递 Kernel
        """
        observation = Observation(
            role=role,
            raw_message=content,
            identity=identity,
        )

        self.kernel.handle_cold(observation)

    def retrieve(
        self,
        query: str,
        user_id: str,
        **kwargs
    ) -> str:
        """
        直接检索记忆（快捷入口，委托给 Kernel）

        Args:
            query: 查询文本
            user_id: 用户 ID
            **kwargs: 其他检索参数

        Returns:
            str: 渲染后的记忆上下文
        """
        return self.kernel.retrieve(query, user_id, **kwargs)

    def flush_buffer(
        self,
        identity: Identity,
    ) -> None:
        """手动触发感知层 Flush（委托给 Kernel）"""
        self.kernel.flush_buffer(identity)

    def get_buffer_info(
        self,
        identity: Identity,
    ) -> Dict[str, Any]:
        """获取 Buffer 信息（委托给 Kernel）"""
        return self.kernel.get_buffer_info(identity)

    def add_flush_observer(self, observer) -> None:
        """添加 Flush 事件观察者（委托给 Kernel）"""
        self.kernel.add_flush_observer(observer)


__all__ = [
    "PatchouliSystem",
]
