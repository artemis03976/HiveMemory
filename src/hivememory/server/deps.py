"""依赖注入 — HiveMemorySystem 单例管理与身份解析唯一入口"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import uuid4

from fastapi import Depends, Header, HTTPException, status

from hivememory.config.app import HiveMemoryConfig
from hivememory.core.access import CallerPrincipal, RunBinding, WorkspaceAccessContext
from hivememory.core.constants import DEFAULT_USER_ID, SYSTEM_AGENT_ID
from hivememory.core.models import ActorIdentity, IdentityScope, WorkspaceIdentity
from hivememory.core.models.workspace import MAIN_WORKSPACE_ID, resolve_default_workspace_identity
from hivememory.infrastructure.log_handler import WebSocketLogHandler
from hivememory.infrastructure.websocket_manager import WebSocketConnectionManager
from hivememory.system import HiveMemorySystem
from hivememory.system.application.passive_ingress_service import PassiveIngressService
from hivememory.system.model_registry import ModelRegistry
from hivememory.system.provider_registry import ProviderRegistry
from hivememory.workspace.authentication import ActorAuthenticationGateway
from hivememory.workspace.capability.agent_profiles import AgentApplicationService
from hivememory.workspace.capability.assets import WorkspaceAssetApplicationService
from hivememory.workspace.capability.memory import MemoryApplicationService
from hivememory.workspace.capability.memory_tasks import MemoryTaskApplicationService
from hivememory.workspace.capability.topic import TopicApplicationService
from hivememory.workspace.process.service import TaskProcessService

logger = logging.getLogger(__name__)

_system: HiveMemorySystem | None = None
_ws_manager: WebSocketConnectionManager | None = None

#: server 作为 system actor 的 adapter 接入统一认证网关（A1 访问边界返工 4.1）。
HTTP_ADAPTER = "http"


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


def get_memory_service() -> MemoryApplicationService:
    """FastAPI Depends 注入 — 获取记忆 API 应用服务。"""
    return get_system().memory_service


def get_memory_task_service() -> MemoryTaskApplicationService:
    """FastAPI Depends 注入：获取记忆生成任务 API 应用服务。"""
    return get_system().memory_task_service


def get_process_service() -> TaskProcessService:
    """FastAPI Depends 注入 — 获取任务进程编排服务。"""
    return get_system().process_service


def get_ingress_service() -> PassiveIngressService:
    """FastAPI Depends 注入 — 获取被动接入应用服务。"""
    return get_system().ingress_service


def get_agent_service() -> AgentApplicationService:
    """FastAPI Depends 注入 — 获取 Agent API 应用服务。"""
    return get_system().agent_service


def get_topic_service() -> TopicApplicationService:
    """FastAPI Depends 注入 — 获取话题 API 应用服务。"""
    return get_system().topic_service


def get_workspace_asset_service() -> WorkspaceAssetApplicationService:
    """FastAPI Depends 注入 — 获取附件上传应用服务。"""
    return get_system().workspace_asset_service


@dataclass(frozen=True)
class RequestIdentitySelection:
    """用户导向身份选择 — 从统一请求头提取的顶层身份上下文。

    基础选择为 ``user_id + workspace_id``；Agent 选择（``agent_id``）
    由具体请求的 body/query 提供。外部会话字段不参与身份选择。
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


def _resolve_identity_coordinates(
    selection: RequestIdentitySelection,
    *,
    require_agent: bool,
    agent_id: str | None,
    explicit_user_id: str | None,
    explicit_workspace_id: str | None,
) -> tuple[str, str]:
    """解析 user_id 与 agent_id 两项声明坐标（含 workspace 取值校验与冲突检测）。

    解析规则（v0.6.2 身份收敛）：

    1. ``user_id``：body/query 显式值与 header 选择同时出现且不一致时
       返回 409；全部缺省时在此唯一回退到 :data:`DEFAULT_USER_ID`。
    2. ``workspace_id``：同样做冲突检测；公共产品入口只允许声明的
       ``main_workspace``，其余值返回 404；全部缺省时解析默认 Workspace。
    3. Agent action（``require_agent=True``，如 Chat）必须显式给出具体
       ``agent_id``，缺失返回 400，不得回退到保留 ``system`` actor；
       非 Agent action 一律使用保留 :data:`SYSTEM_AGENT_ID`，表示"没有
       具体 Agent 作为操作来源主体"。

    应用服务不得再次解析身份；本函数是默认身份回退的唯一合法位置。
    workspace 只校验取值合法性，不参与返回——调用方统一解析公共默认
    Workspace。
    """
    user_id = (
        _merge_identity_field(
            header_value=selection.user_id,
            explicit_value=explicit_user_id,
            field_name="user_id",
        )
        or DEFAULT_USER_ID
    )
    workspace_id = _merge_identity_field(
        header_value=selection.workspace_id,
        explicit_value=explicit_workspace_id,
        field_name="workspace_id",
    )
    if workspace_id is not None and workspace_id != MAIN_WORKSPACE_ID:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Workspace '{workspace_id}' 不存在：公共入口当前只开放 " f"'{MAIN_WORKSPACE_ID}'"
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

    return user_id, resolved_agent_id


@dataclass(frozen=True)
class RequestIdentityClaims:
    """认证前的请求身份声明（A1 访问边界返工第 4.1 节）。

    server 在认证前只持有 actor 声明与请求进入的 workspace（不变量 1）；
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
    explicit_user_id: str | None = None,
    explicit_workspace_id: str | None = None,
) -> RequestIdentityClaims:
    """server 边界唯一声明解析入口 — 只产出 actor 声明与请求进入的 workspace。

    身份选择与冲突检测规则见 :func:`_resolve_identity_coordinates`。经
    认证网关的请求一律使用本函数；声明只作为认证输入，认证通过后经过
    验证的身份只存在于密封的 context 中，server 不把声明当作已确认
    的身份继续使用。Agent action（``require_agent=True``）显式给出保留的
    :data:`SYSTEM_AGENT_ID` 时返回 400：它表示"没有具体 Agent"，不能作为
    任务进程的执行 Agent（注册入口仍保留同一检查）。
    """
    user_id, resolved_agent_id = _resolve_identity_coordinates(
        selection,
        require_agent=require_agent,
        agent_id=agent_id,
        explicit_user_id=explicit_user_id,
        explicit_workspace_id=explicit_workspace_id,
    )
    if require_agent and resolved_agent_id == SYSTEM_AGENT_ID:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该操作由具体 Agent 执行，不能使用保留的 system 作为 agent_id",
        )
    return RequestIdentityClaims(
        actor=ActorIdentity(
            user_id=user_id,
            agent_id=resolved_agent_id,
        ),
        workspace=resolve_default_workspace_identity(user_id),
    )


def resolve_request_identity_scope(
    selection: RequestIdentitySelection,
    *,
    require_agent: bool = False,
    agent_id: str | None = None,
    explicit_user_id: str | None = None,
    explicit_workspace_id: str | None = None,
) -> IdentityScope:
    """认证前一次性组装完整 ``IdentityScope``（不变量 1 的已知例外）。

    **只供 ``/ingest``（Import Bus）使用**：被动摄入不在现有系统内、不经
    认证网关，维持认证前组装 scope 的既有行为；经网关的请求不得使用本
    函数，应改用 :func:`resolve_request_identity_claims`（A1 访问边界返工
    第 3 节非目标）。
    """
    user_id, resolved_agent_id = _resolve_identity_coordinates(
        selection,
        require_agent=require_agent,
        agent_id=agent_id,
        explicit_user_id=explicit_user_id,
        explicit_workspace_id=explicit_workspace_id,
    )
    return IdentityScope(
        actor_identity=ActorIdentity(
            user_id=user_id,
            agent_id=resolved_agent_id,
        ),
        workspace_identity=resolve_default_workspace_identity(user_id),
    )


# ---------------------------------------------------------------------------
# 统一认证网关接入（A1 访问边界返工第 4.1/4.3 节）
#
# server 是一个登记过的调用来源：以自身 principal 与 ``http`` adapter 对
# 每个与 workspace 相关的请求经网关取得 context。请求头中的用户身份不做
# 证明——这是本地单用户部署的信任假设。认证前 server 只持有声明
# （:class:`RequestIdentityClaims`），声明只作为认证输入：认证通过后，
# 经过验证的身份只存在于密封的 context 中，server 不把声明当作已确认
# 的身份继续使用，也不把声明交给路由处理函数。
# ---------------------------------------------------------------------------


def get_access_gateway() -> ActorAuthenticationGateway:
    """FastAPI Depends 注入 — 统一认证网关（未装配时显式失败）。"""
    gateway = get_system().access_gateway
    if gateway is None:
        raise RuntimeError("统一认证网关未装配，无法完成请求认证")
    return gateway


def get_server_principal_id() -> str:
    """FastAPI Depends 注入 — server 自身经网关认证使用的 principal 标识。"""
    return get_system().config.system.server_principal_id


def _request_run_id() -> str:
    """为请求级 context 生成一次性的运行标识（server 为本次请求生成）。"""
    return f"request_{uuid4().hex}"


async def authenticate_request_access(
    claims: RequestIdentityClaims,
    *,
    gateway: ActorAuthenticationGateway,
    principal_id: str,
) -> WorkspaceAccessContext:
    """以 server 自身 principal 经统一认证网关取得请求级访问 context。

    ``claims`` 是 :func:`resolve_request_identity_claims` 解析的请求身份
    声明，只在这里作为认证输入使用；运行绑定为本次请求的标识
    （P-9b/P-9c）。网关认证失败（未登记 principal、adapter 不匹配、未获
    准入）以 ``AdmissionDeniedError`` 拒绝，由访问错误映射转为 HTTP 状态码。
    """
    return await gateway.authenticate(
        adapter=HTTP_ADAPTER,
        principal=CallerPrincipal(principal_id),
        actor=claims.actor,
        workspace=claims.workspace,
        binding=RunBinding.for_request(_request_run_id()),
    )


@dataclass(frozen=True)
class RequestAccess:
    """一次管理员请求的请求级访问凭据与目标 workspace。

    声明只作为认证输入：路由处理函数只取得 ``access`` 与
    ``target_workspace``（这次操作的参数，当前取请求进入的 workspace），
    不接触声明、认证后的身份或任何预先组装的 ``IdentityScope``（A1 访问
    边界返工第 4.1/4.3 节）。
    """

    access: WorkspaceAccessContext
    target_workspace: WorkspaceIdentity


@asynccontextmanager
async def request_access_for_claims(
    claims: RequestIdentityClaims,
    *,
    gateway: ActorAuthenticationGateway,
    principal_id: str,
) -> AsyncIterator[RequestAccess]:
    """以给定声明取得请求级访问凭据；退出时使 context 失效。

    供身份解析带 body/query 冲突检测的路由（topics、chat/stop）与
    :func:`get_request_access` 共用：取得 context、以
    :class:`RequestAccess` 交出凭据与目标 workspace，声明不越过认证边界。
    """
    access = await authenticate_request_access(claims, gateway=gateway, principal_id=principal_id)
    try:
        yield RequestAccess(access=access, target_workspace=claims.workspace)
    finally:
        gateway.invalidate_context(access)


async def get_request_access(
    selection: RequestIdentitySelection = Depends(get_identity_selection),
    gateway: ActorAuthenticationGateway = Depends(get_access_gateway),
    principal_id: str = Depends(get_server_principal_id),
) -> AsyncIterator[RequestAccess]:
    """FastAPI yield 依赖 — 管理员请求的请求级访问凭据与目标 workspace。

    以 (user, ``system``) 声明经网关认证取得 context，一次请求一个；
    请求结束（含失败）时失效（A1 访问边界返工第 4.8 节）。供 memories、
    agents、memory-tasks、workspace assets 等管理路由使用；身份解析带
    body/query 冲突检测的路由改用 :func:`request_access_for_claims`。
    """
    claims = resolve_request_identity_claims(selection)
    async with request_access_for_claims(
        claims, gateway=gateway, principal_id=principal_id
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
