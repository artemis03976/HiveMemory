"""总线类型回归样例：只供 mypy 门禁检查，不运行、不由 pytest 收集。

运行 ``mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py``。
每个 ignore 都对应故意写错的调用；若错误被 Any 或宽泛重载放行，门禁必须失败。
"""

from typing import assert_type

from hivememory.components.bus import AsyncSystemBus, RouteBinding
from hivememory.core.contracts.route import route
from hivememory.core.models import IdentityScope, MemoryAtom, WorkspaceIdentity
from hivememory.core.protocol.gateway import GatewayIngressMode, GatewayProcessResult
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliLocalRoutes, PatchouliRoutes


@route("typecheck.prepare")
async def prepare_route(*, selected_attachments: list[str]) -> int:
    """复现附件关键字重命名后调用方与提供方漂移的接缝。"""
    raise NotImplementedError


async def prepare_handler(*, selected_attachments: list[str]) -> int:
    return len(selected_attachments)


async def renamed_handler(*, attachments: list[str]) -> int:
    return len(attachments)


async def wrong_result_handler(*, selected_attachments: list[str]) -> str:
    return str(len(selected_attachments))


async def check_routes(
    bus: AsyncSystemBus,
    scope: IdentityScope,
    workspace: WorkspaceIdentity,
) -> None:
    """正例验证精确推断；反例保护注册、调用、绑定与返回值类型。"""
    bus.register(prepare_route, prepare_handler)
    RouteBinding.of(prepare_route, prepare_handler)
    assert_type(await bus.request(prepare_route, selected_attachments=[]), int)

    # handler 参数名和返回类型都必须符合 route，不能把返回类型扩宽后接受。
    bus.register(prepare_route, renamed_handler)  # type: ignore[arg-type]
    bus.register(prepare_route, wrong_result_handler)  # type: ignore[misc]
    RouteBinding.of(prepare_route, renamed_handler)  # type: ignore[arg-type]
    RouteBinding.of(prepare_route, wrong_result_handler)  # type: ignore[misc]

    # 关键字漂移、遗漏参数、把仅关键字参数按位置传入都必须报错。
    await bus.request(prepare_route, attachments=[])  # type: ignore[call-arg]
    await bus.request(prepare_route)  # type: ignore[call-arg]
    await bus.request(prepare_route, [])  # type: ignore[call-arg]

    # 身份整体不能冒充资源归属，按位置传参同样需要静态约束。
    assert_type(
        await bus.request(PatchouliLocalRoutes.TOPIC_DISCARD_IF_EMPTY, workspace, "topic"), bool
    )
    await bus.request(PatchouliLocalRoutes.TOPIC_DISCARD_IF_EMPTY, scope, "topic")  # type: ignore[arg-type]

    # 普通字符串只保留运行时兼容性，不能进入生产代码的静态 API。
    bus.register("typecheck.prepare", prepare_handler)  # type: ignore[arg-type]
    await bus.request("typecheck.prepare", selected_attachments=[])  # type: ignore[arg-type]

    assert_type(
        await bus.request(PatchouliRoutes.MEMORY_GET, "memory", identity_scope=scope),
        MemoryAtom | None,
    )
    assert_type(
        await bus.request(
            GatewayPublicRoutes.PROCESS,
            "message",
            identity_scope=scope,
            ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
        ),
        GatewayProcessResult,
    )
