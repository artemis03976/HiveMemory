"""依赖注入 — HiveMemorySystem 单例管理与身份解析唯一入口"""

import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from hivememory.core.constants import DEFAULT_USER_ID, SYSTEM_AGENT_ID
from hivememory.core.models import ActorIdentity, IdentityScope
from hivememory.core.models.workspace import MAIN_WORKSPACE_ID, resolve_default_workspace_identity
from hivememory.infrastructure.log_handler import WebSocketLogHandler
from hivememory.infrastructure.websocket_manager import WebSocketConnectionManager
from hivememory.system.application.agent_service import AgentApplicationService
from hivememory.system.application.chat_service import ChatApplicationService
from hivememory.system.application.memory_service import MemoryApplicationService
from hivememory.system.application.memory_task_service import MemoryTaskApplicationService
from hivememory.system.application.passive_ingress_service import PassiveIngressService
from hivememory.system.application.topic_service import TopicApplicationService
from hivememory.system.config import HiveMemoryConfig
from hivememory.system import HiveMemorySystem
from hivememory.system.model_registry import ModelRegistry
from hivememory.system.provider_registry import ProviderRegistry

logger = logging.getLogger(__name__)

_system: Optional[HiveMemorySystem] = None
_ws_manager: Optional[WebSocketConnectionManager] = None


def init_system(config: Optional[HiveMemoryConfig] = None) -> HiveMemorySystem:
    """lifespan startup 时调用，组装并返回 HiveMemorySystem"""
    global _system
    _system = HiveMemorySystem.build(config=config)
    logger.info("HiveMemorySystem 组装完成")
    return _system


async def shutdown_system() -> None:
    """lifespan shutdown 时调用"""
    global _system
    if _system:
        await _system.stop()
        logger.info("HiveMemorySystem 已关闭")
    _system = None


def get_system() -> HiveMemorySystem:
    """FastAPI Depends 注入 — 获取 HiveMemorySystem 单例"""
    if _system is None:
        raise RuntimeError("HiveMemorySystem 未初始化，服务未正确启动")
    return _system


def get_memory_service() -> MemoryApplicationService:
    """FastAPI Depends 注入 — 获取记忆 API 应用服务。"""
    return get_system().memory_service


def get_memory_task_service() -> MemoryTaskApplicationService:
    """FastAPI Depends 注入：获取记忆生成任务 API 应用服务。"""
    return get_system().memory_task_service


def get_chat_service() -> ChatApplicationService:
    """FastAPI Depends 注入 — 获取主动对话应用服务。"""
    return get_system().chat_service


def get_ingress_service() -> PassiveIngressService:
    """FastAPI Depends 注入 — 获取被动接入应用服务。"""
    return get_system().ingress_service


def get_agent_service() -> AgentApplicationService:
    """FastAPI Depends 注入 — 获取 Agent API 应用服务。"""
    return get_system().agent_service


def get_topic_service() -> TopicApplicationService:
    """FastAPI Depends 注入 — 获取话题 API 应用服务。"""
    return get_system().topic_service


@dataclass(frozen=True)
class RequestIdentitySelection:
    """用户导向身份选择 — 从统一请求头提取的顶层身份上下文。

    基础选择为 ``user_id + workspace_id``；Agent 选择（``agent_id``）与
    ``session_id`` 只由具体请求的 body/query 提供，不放在公共请求头里。
    ``workspace_id`` 为 ``None`` 表示请求未显式选择 Workspace，由解析器
    在唯一回退点解析到公共默认 Workspace。
    """

    user_id: str | None
    workspace_id: str | None


def get_identity_selection(
    x_user_id: str | None = Header(default=None),
    x_workspace_id: str | None = Header(default=None),
) -> RequestIdentitySelection:
    """FastAPI Depends 注入 — 从请求头提取用户导向身份选择。

    请求头缺省时字段为 ``None``（区别于"显式提供了 default"），交由
    :func:`resolve_request_identity_scope` 在唯一回退点处理。
    """
    return RequestIdentitySelection(user_id=x_user_id, workspace_id=x_workspace_id)


def _merge_identity_field(
    *,
    header_value: str | None,
    explicit_value: str | None,
    field_name: str,
) -> str | None:
    """合并 header 与 body/query 携带的同一身份字段。

    两个来源同时出现且不一致时显式拒绝，不静默选择其中一个；
    空白值视为非法输入，同样显式失败。
    """
    header = header_value.strip() if header_value else None
    explicit = explicit_value.strip() if explicit_value else None
    if header is not None and explicit is not None and header != explicit:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"身份选择冲突：header 与请求参数提供了不一致的 {field_name}，"
                "请在同一请求中只表达一种选择"
            ),
        )
    value = explicit if explicit is not None else header
    if value is not None and not value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"身份字段 {field_name} 不能为空",
        )
    return value


def resolve_request_identity_scope(
    selection: RequestIdentitySelection,
    *,
    require_agent: bool = False,
    agent_id: str | None = None,
    session_id: str | None = None,
    explicit_user_id: str | None = None,
    explicit_workspace_id: str | None = None,
) -> IdentityScope:
    """server 边界唯一身份解析入口 — 一次性冻结 IdentityScope。

    解析规则（v0.6.2 身份收敛）：

    1. ``user_id``：body/query 显式值与 header 选择同时出现且不一致时
       返回 409；全部缺省时在此唯一回退到 :data:`DEFAULT_USER_ID`。
    2. ``workspace_id``：同样做冲突检测；公共产品入口只允许声明的
       ``main_workspace``，其余值返回 404；全部缺省时解析默认 Workspace。
    3. Agent action（``require_agent=True``，如 Chat）必须显式给出具体
       ``agent_id``，缺失返回 400，不得回退到保留 ``system`` actor；
       非 Agent action 一律注入 :data:`SYSTEM_AGENT_ID`，表示"没有具体
       Agent 作为操作来源主体"。
    4. same-owner 约束由 ``IdentityScope`` 模型校验兜底。

    应用服务不得再次解析身份；本函数是默认身份回退的唯一合法位置。
    """
    user_id = _merge_identity_field(
        header_value=selection.user_id,
        explicit_value=explicit_user_id,
        field_name="user_id",
    ) or DEFAULT_USER_ID
    workspace_id = _merge_identity_field(
        header_value=selection.workspace_id,
        explicit_value=explicit_workspace_id,
        field_name="workspace_id",
    )
    if workspace_id is not None and workspace_id != MAIN_WORKSPACE_ID:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Workspace '{workspace_id}' 不存在：公共入口当前只开放 "
                f"'{MAIN_WORKSPACE_ID}'"
            ),
        )

    if require_agent:
        concrete_agent_id = (agent_id or "").strip() if agent_id else None
        if not concrete_agent_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="该操作由具体 Agent 执行，必须显式提供 agent_id",
            )
        resolved_agent_id = concrete_agent_id
    else:
        # 非 Agent action：actor 只表达"没有具体 Agent 作为操作来源主体"，
        # 不代表任何可选 Agent，也不参与 MemoryAccessPolicy 授权。
        resolved_agent_id = SYSTEM_AGENT_ID

    return IdentityScope(
        actor_identity=ActorIdentity(
            user_id=user_id,
            agent_id=resolved_agent_id,
            session_id=session_id,
        ),
        workspace_identity=resolve_default_workspace_identity(user_id),
    )


def get_identity_scope(
    selection: RequestIdentitySelection = Depends(get_identity_selection),
) -> IdentityScope:
    """FastAPI Depends 注入 — 非 Agent action 的统一身份解析。

    供只依赖统一请求头（``x-user-id`` / ``x-workspace-id``）的资源路由
    使用；body/query 携带身份选择的路由应改用
    :func:`resolve_request_identity_scope` 做冲突检测后自行解析。
    """
    return resolve_request_identity_scope(selection)


def init_websocket_log_broadcasting(
    config: HiveMemoryConfig
) -> Optional[WebSocketConnectionManager]:
    """
    初始化 WebSocket 日志广播系统

    Steps:
    1. 检查配置是否启用
    2. 创建 WebSocketConnectionManager
    3. 创建 WebSocketLogHandler 并配置命名空间过滤
    4. 将 handler 附加到 root logger
    5. 配置日志级别和速率限制

    Args:
        config: HiveMemory 配置对象

    Returns:
        WebSocketConnectionManager 实例，如果未启用则返回 None
    """
    global _ws_manager

    if not config.logging.websocket_enabled:
        logger.info("WebSocket log broadcasting disabled")
        return None

    # 创建连接管理器
    _ws_manager = WebSocketConnectionManager(
        buffer_size=config.logging.websocket_buffer_size
    )

    # 创建日志处理器
    handler = WebSocketLogHandler(
        ws_manager=_ws_manager,
        namespaces=config.logging.websocket_namespaces,
        level=getattr(logging, config.logging.websocket_level),
        max_rate=config.logging.websocket_max_rate,
    )

    # 附加到 root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    # 注册追踪上下文过滤器
    from hivememory.infrastructure.trace_context import TraceInjectFilter
    handler.addFilter(TraceInjectFilter())

    logger.info(
        f"WebSocket log broadcasting initialized: "
        f"namespaces={config.logging.websocket_namespaces}, "
        f"level={config.logging.websocket_level}"
    )

    return _ws_manager


async def shutdown_websocket_log_broadcasting(
    manager: Optional[WebSocketConnectionManager]
) -> None:
    """
    关闭 WebSocket 日志广播系统

    清理所有客户端连接并从 root logger 移除 handler

    Args:
        manager: WebSocketConnectionManager 实例
    """
    if manager:
        # 断开所有客户端
        await manager.disconnect_all()

        # 从 root logger 移除 handler
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, WebSocketLogHandler):
                root_logger.removeHandler(handler)
                logger.info("WebSocket log handler removed")


def get_ws_manager() -> WebSocketConnectionManager:
    """FastAPI Depends 注入 — 获取 WebSocket 连接管理器"""
    if _ws_manager is None:
        raise RuntimeError("WebSocket manager not initialized")
    return _ws_manager


def get_model_registry() -> ModelRegistry:
    """FastAPI Depends 注入 — 获取 ModelRegistry 单例（来自 HiveMemorySystem）。"""
    return get_system().model_registry


def get_provider_registry() -> ProviderRegistry:
    """FastAPI Depends 注入 — 获取 ProviderRegistry 单例（来自 HiveMemorySystem）。"""
    return get_system().provider_registry
