"""外部 harness 插件的管理路由：状态、安装与卸载。

安装与卸载改动 harness 的配置、接入登记与当前用户的环境变量，只由管理凭据保护；
访问记录由插件服务经 root API 以请求的工作 actor 补上。命令失败不是 HTTP 错误：
响应照常返回状态，``failure`` 说明哪一步没有完成。
"""

from fastapi import APIRouter, Body, Depends

from hivememory.core.access import WorkingActor
from hivememory.server.deps import get_plugin_service, get_request_actor
from hivememory.server.error_handlers import error_responses
from hivememory.server.models.plugins import (
    PluginActionResponse,
    PluginFailureResponse,
    PluginInstallRequest,
    PluginListResponse,
    PluginStatusResponse,
)
from hivememory.system.plugins import HarnessId, PluginActionResult, PluginInstallationService

router = APIRouter(
    prefix="/plugins",
    responses=error_responses(400, 403, 409, 422, 500, 503),
    tags=["plugins"],
)


@router.get("", response_model=PluginListResponse)
async def list_plugins(
    service: PluginInstallationService = Depends(get_plugin_service),
) -> PluginListResponse:
    """插件目录、版本与四个支持的 harness 的插件状态。"""
    return PluginListResponse(
        integrations_dir=str(service.integrations_dir),
        version=service.version,
        harnesses=[PluginStatusResponse.from_status(status) for status in service.statuses()],
    )


@router.post("/{harness}/install", response_model=PluginActionResponse)
async def install_plugin(
    harness: HarnessId,
    body: PluginInstallRequest | None = Body(default=None),
    actor: WorkingActor = Depends(get_request_actor),
    service: PluginInstallationService = Depends(get_plugin_service),
) -> PluginActionResponse:
    """安装插件并登记接入；已经安装时重新安装，原有登记与凭据沿用。"""
    request = body or PluginInstallRequest()
    result = await service.install(
        harness, actor=actor, run_plugin_commands=request.run_plugin_commands
    )
    return _action_response(result)


@router.post("/{harness}/uninstall", response_model=PluginActionResponse)
async def uninstall_plugin(
    harness: HarnessId,
    service: PluginInstallationService = Depends(get_plugin_service),
) -> PluginActionResponse:
    """卸载插件并删除接入；Agent 的访问记录保留。"""
    return _action_response(await service.uninstall(harness))


def _action_response(result: PluginActionResult) -> PluginActionResponse:
    return PluginActionResponse(
        status=PluginStatusResponse.from_status(result.status),
        failure=(
            PluginFailureResponse.from_failure(result.failure)
            if result.failure is not None
            else None
        ),
    )
