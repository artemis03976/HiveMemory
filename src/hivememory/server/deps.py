"""依赖注入 — HiveMemorySystem 单例管理与身份解析唯一入口"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import uuid4

from fastapi import Depends, Header

from hivememory.config.app import HiveMemoryConfig
from hivememory.core.access import CallerPrincipal, RunBinding, WorkspaceAccessContext
from hivememory.core.constants import DEFAULT_USER_ID, SYSTEM_AGENT_ID
from hivememory.core.models import ActorIdentity, WorkspaceIdentity
from hivememory.core.models.workspace import MAIN_WORKSPACE_ID, resolve_default_workspace_identity
from hivememory.infrastructure.log_handler import WebSocketLogHandler
from hivememory.infrastructure.websocket_manager import WebSocketConnectionManager
from hivememory.server.error_handlers import APIError
from hivememory.system import HiveMemorySystem
from hivememory.system.access import HTTP_ADAPTER
from hivememory.system.application.import_bus_service import ImportBusService
from hivememory.system.model_registry import ModelRegistry
from hivememory.system.provider_registry import ProviderRegistry
from hivememory.workspace.authentication import ActorAuthenticationGateway
from hivememory.workspace.capability.root.agent_profiles import RootAgentProfileService
from hivememory.workspace.capability.root.assets import RootWorkspaceAssetService
from hivememory.workspace.capability.root.memory import RootMemoryService
from hivememory.workspace.capability.root.memory_tasks import RootMemoryTaskService
from hivememory.workspace.capability.root.session import RootSessionService
from hivememory.workspace.capability.root.topic import RootTopicService
from hivememory.workspace.process.service import TaskProcessService

logger = logging.getLogger(__name__)

_system: HiveMemorySystem | None = None
_ws_manager: WebSocketConnectionManager | None = None


def init_system(config: HiveMemoryConfig | None = None) -> HiveMemorySystem:
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


def get_root_memory_service() -> RootMemoryService:
    """FastAPI Depends 注入 — 从门面获取 root 记忆管理 API。"""
    return get_system().root_memory_service


def get_root_memory_task_service() -> RootMemoryTaskService:
    """FastAPI Depends 注入 — 从门面获取 root 记忆任务管理 API。"""
    return get_system().root_memory_task_service


def get_process_service() -> TaskProcessService:
    """FastAPI Depends 注入 — 获取任务进程编排服务。"""
    return get_system().process_service


def get_import_bus_service() -> ImportBusService:
    """FastAPI Depends 注入 — 获取 Import Bus 应用服务。"""
    return get_system().import_bus_service


def get_import_request_limit() -> int:
    """FastAPI Depends 注入 — 单次导入请求体的字节上限。"""
    return get_system().config.import_bus.max_request_bytes


def get_root_agent_service() -> RootAgentProfileService:
    """FastAPI Depends 注入 — 从门面获取 root Agent Profile 管理 API。"""
    return get_system().root_agent_service


def get_root_session_service() -> RootSessionService:
    """从门面取得会话管理 root API。"""
    return get_system().root_session_service


def get_root_topic_service() -> RootTopicService:
    """FastAPI Depends 注入 — 从门面获取 root 话题管理 API。"""
    return get_system().root_topic_service


def get_root_workspace_asset_service() -> RootWorkspaceAssetService:
    """FastAPI Depends 注入 — 从门面获取 root WorkspaceAsset 上传 API。"""
    return get_system().root_workspace_asset_service


@dataclass(frozen=True)
class RequestIdentitySelection:
    """用户导向身份选择 — 从统一请求头提取的顶层身份上下文。

    基础选择为 ``user_id + workspace_id``；Agent 选择（``agent_id``）
    由具体请求的 body 提供。外部会话字段不参与身份选择。
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
    :func:`resolve_request_identity_claims` 在唯一回退点处理。
    """
    return RequestIdentitySelection(user_id=x_user_id, workspace_id=x_workspace_id)


def _normalize_identity_header(value: str | None) -> str | None:
    """规范化请求头携带的身份字段：缺省或空串为 ``None``，只含空白显式失败。"""
    if not value:
        return None
    normalized = value.strip()
    if not normalized:
        raise APIError("request.identity_empty")
    return normalized


def _resolve_identity_coordinates(
    selection: RequestIdentitySelection,
    *,
    require_agent: bool,
    agent_id: str | None,
) -> tuple[str, str]:
    """解析 user_id 与 agent_id 两项声明坐标（含 workspace 取值校验）。

    解析规则：

    1. ``user_id``：只取统一请求头的选择；缺省时在此唯一回退到
       :data:`DEFAULT_USER_ID`。
    2. ``workspace_id``：公共产品入口只允许声明的 ``main_workspace``，其余值
       返回 404；缺省时解析默认 Workspace。
    3. Agent action（``require_agent=True``，如 Chat 与导入）必须显式给出
       具体 ``agent_id``，缺失返回 400，不得回退到保留 ``system`` actor；
       非 Agent action 一律使用保留 :data:`SYSTEM_AGENT_ID`，表示"没有
       具体 Agent 作为操作来源主体"。

    应用服务不得再次解析身份；本函数是默认身份回退的唯一合法位置。
    workspace 只校验取值合法性，不参与返回——调用方统一解析公共默认
    Workspace。
    """
    user_id = _normalize_identity_header(selection.user_id) or DEFAULT_USER_ID
    workspace_id = _normalize_identity_header(selection.workspace_id)
    if workspace_id is not None and workspace_id != MAIN_WORKSPACE_ID:
        raise APIError("request.workspace_not_found")
    if require_agent:
        concrete_agent_id = (agent_id or "").strip() if agent_id else None
        if not concrete_agent_id:
            raise APIError("request.agent_required")
        resolved_agent_id = concrete_agent_id
    else:
        # 非 Agent action：actor 只表达"没有具体 Agent 作为操作来源主体"，
        # 不代表任何可选 Agent，也不参与 MemoryAccessPolicy 授权。
        resolved_agent_id = SYSTEM_AGENT_ID

    return user_id, resolved_agent_id


@dataclass(frozen=True)
class RequestIdentityClaims:
    """认证前的请求身份声明。

    server 在认证前只持有 actor 声明与请求进入的 workspace；
    ``IdentityScope`` 在两阶段认证通过之后才由授权点组装，不由入口预先
    冻结。外部会话信息不属于 ActorIdentity，也不进入认证声明。
    """

    actor: ActorIdentity
    workspace: WorkspaceIdentity


def resolve_request_identity_claims(
    selection: RequestIdentitySelection,
    *,
    require_agent: bool = False,
    agent_id: str | None = None,
) -> RequestIdentityClaims:
    """server 边界唯一声明解析入口 — 只产出 actor 声明与请求进入的 workspace。

    身份只取自统一请求头的选择，默认回退与 Workspace 取值规则见
    :func:`_resolve_identity_coordinates`。经认证网关的请求一律使用本函数；声明只作为认证输入，认证通过后经过
    验证的身份只存在于密封的 context 中，server 不把声明当作已确认
    的身份继续使用。Agent action（``require_agent=True``）显式给出保留的
    :data:`SYSTEM_AGENT_ID` 时返回 400：它表示"没有具体 Agent"，不能作为
    任务进程的执行 Agent（注册入口仍保留同一检查）。
    """
    user_id, resolved_agent_id = _resolve_identity_coordinates(
        selection,
        require_agent=require_agent,
        agent_id=agent_id,
    )
    if require_agent and resolved_agent_id == SYSTEM_AGENT_ID:
        raise APIError("request.agent_reserved")
    return RequestIdentityClaims(
        actor=ActorIdentity(
            user_id=user_id,
            agent_id=resolved_agent_id,
        ),
        workspace=resolve_default_workspace_identity(user_id),
    )


# ---------------------------------------------------------------------------
# 统一认证网关接入
#
# server 是一个登记过的调用来源：以自身 principal 与 ``http`` adapter 对
# 每个与 workspace 相关的请求经网关取得 context。外部入口（MCP）不使用
# server 的 principal，而是把请求声明的 principal 与接入凭据交给同一个网关
# 核对。请求头中的用户身份不做证明——这是本地单用户部署的信任假设。
# 认证前 server 只持有声明（:class:`RequestIdentityClaims`），声明只作为
# 认证输入：认证通过后，经过验证的身份只存在于密封的 context 中，server
# 不把声明当作已确认的身份继续使用，也不把声明交给路由处理函数。
# ---------------------------------------------------------------------------


def get_access_gateway() -> ActorAuthenticationGateway:
    """FastAPI Depends 注入 — 统一认证网关。"""
    return get_system().access_gateway


def get_server_principal_id() -> str:
    """FastAPI Depends 注入 — server 自身经网关认证使用的 principal 标识。"""
    return get_system().config.system.server_principal_id


def resolve_principal_claim(x_principal_id: str | None) -> str:
    """外部入口声明的调用来源：缺省或空串返回 400，只含空白同样显式失败。

    声明只是认证输入，经统一认证网关核对接入凭据后才成立；入口不据此
    判定身份。
    """
    principal_id = _normalize_identity_header(x_principal_id)
    if principal_id is None:
        raise APIError("request.principal_required")
    return principal_id


def bearer_credential(authorization: str | None) -> str | None:
    """取出 ``Authorization: Bearer`` 中的接入凭据；缺省或不是 Bearer 时为 ``None``。

    入口不判定凭据：``None`` 同样交给统一认证网关，由它以
    ``invalid_credential`` 拒绝。
    """
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer":
        return None
    return token.strip() or None


def _request_run_id() -> str:
    """为请求级 context 生成一次性的运行标识（server 为本次请求生成）。"""
    return f"request_{uuid4().hex}"


async def authenticate_request_access(
    claims: RequestIdentityClaims,
    *,
    gateway: ActorAuthenticationGateway,
    principal_id: str,
    adapter: str,
    credential: str | None,
) -> WorkspaceAccessContext:
    """以给定 principal、adapter 与凭据经统一认证网关取得请求级访问 context。

    ``claims`` 是 :func:`resolve_request_identity_claims` 解析的请求身份
    声明，只在这里作为认证输入使用；运行绑定为本次请求的标识。网关认证
    失败（未登记 principal、adapter 不匹配、凭据无效、未获准入）以
    ``AdmissionDeniedError`` 拒绝，由访问错误映射转为 HTTP 状态码。
    """
    return await gateway.authenticate(
        adapter=adapter,
        principal=CallerPrincipal(principal_id),
        credential=credential,
        actor=claims.actor,
        workspace=claims.workspace,
        binding=RunBinding.for_request(_request_run_id()),
    )


@dataclass(frozen=True)
class RequestAccess:
    """一次请求的请求级访问凭据与目标 workspace。

    声明只作为认证输入：路由处理函数只取得 ``access`` 与
    ``target_workspace``（这次操作的参数，当前取请求进入的 workspace），
    不接触声明、认证后的身份或任何预先组装的 ``IdentityScope``。
    """

    access: WorkspaceAccessContext
    target_workspace: WorkspaceIdentity


@asynccontextmanager
async def request_access_for_claims(
    claims: RequestIdentityClaims,
    *,
    gateway: ActorAuthenticationGateway,
    principal_id: str,
    adapter: str,
    credential: str | None,
) -> AsyncIterator[RequestAccess]:
    """以给定声明取得请求级访问凭据；退出时使 context 失效。

    取得 context、以 :class:`RequestAccess` 交出凭据与目标 workspace，声明
    不越过认证边界。principal、adapter 与凭据由调用方给出：管理路由经
    :func:`get_request_access` 传入 server 自身的 principal、``http`` 与
    ``None``，以 (user, ``system``) 声明调用；导入入口的 Agent 来自请求体，
    以 (user, 具体 Agent) 声明直接调用；MCP 入口传入请求声明的 principal、
    ``mcp`` 与请求中的凭据，以 (user, 具体 Agent) 声明调用。
    """
    access = await authenticate_request_access(
        claims,
        gateway=gateway,
        principal_id=principal_id,
        adapter=adapter,
        credential=credential,
    )
    try:
        yield RequestAccess(access=access, target_workspace=claims.workspace)
    finally:
        gateway.revoke_context(access)


async def get_request_access(
    selection: RequestIdentitySelection = Depends(get_identity_selection),
    gateway: ActorAuthenticationGateway = Depends(get_access_gateway),
    principal_id: str = Depends(get_server_principal_id),
) -> AsyncIterator[RequestAccess]:
    """FastAPI yield 依赖 — 管理员请求的请求级访问凭据与目标 workspace。

    以 (user, ``system``) 声明经网关认证取得 context，一次请求一个；
    请求结束（含失败）时失效。全部管理路由与取消入口共用这一依赖。
    """
    claims = resolve_request_identity_claims(selection)
    async with request_access_for_claims(
        claims,
        gateway=gateway,
        principal_id=principal_id,
        adapter=HTTP_ADAPTER,
        credential=None,
    ) as request_access:
        yield request_access


def init_websocket_log_broadcasting(
    config: HiveMemoryConfig,
) -> WebSocketConnectionManager | None:
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
    _ws_manager = WebSocketConnectionManager(buffer_size=config.logging.websocket_buffer_size)

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
    from hivememory.components.trace_context import TraceInjectFilter

    handler.addFilter(TraceInjectFilter())

    logger.info(
        f"WebSocket log broadcasting initialized: "
        f"namespaces={config.logging.websocket_namespaces}, "
        f"level={config.logging.websocket_level}"
    )

    return _ws_manager


async def shutdown_websocket_log_broadcasting(
    manager: WebSocketConnectionManager | None,
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
