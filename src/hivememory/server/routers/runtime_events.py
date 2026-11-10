"""RuntimeEvent SSE 传输适配器。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse, ServerSentEvent

from hivememory.core.contracts.runtime_events import RuntimeEvent
from hivememory.server.deps import get_system
from hivememory.server.error_handlers import error_responses
from hivememory.server.models.runtime_event import (
    RuntimeEventDisabledResponse,
    RuntimeEventResponse,
    RuntimeEventStatusResponse,
)

router = APIRouter(
    responses=error_responses(422, 500), prefix="/runtime-events", tags=["runtime-events"]
)


def _format_sse(event: RuntimeEvent) -> ServerSentEvent:
    data = RuntimeEventResponse.from_domain(event).model_dump_json()
    return ServerSentEvent(id=event.event_id, event="runtime_event", data=data)


@router.get("/status", response_model=RuntimeEventStatusResponse)
async def runtime_events_status(system=Depends(get_system)) -> RuntimeEventStatusResponse:
    bus = system.runtime_events
    enabled = bus is not None and system.config.runtime_events.enabled
    return RuntimeEventStatusResponse(
        enabled=enabled,
        buffer_size=system.config.runtime_events.buffer_size,
        subscriber_queue_size=system.config.runtime_events.subscriber_queue_size,
        latest_sequence=bus.latest_sequence if bus is not None else None,
    )


@router.get("/stream", responses={503: {"model": RuntimeEventDisabledResponse}})
async def stream_runtime_events(
    last_event_id_query: str | None = Query(default=None, alias="last_event_id"),
    last_event_id_header: str | None = Header(default=None, alias="Last-Event-ID"),
    replay_last: int = Query(default=100, ge=1),
    system=Depends(get_system),
):
    bus = system.runtime_events
    if bus is None or not system.config.runtime_events.enabled:
        content = RuntimeEventDisabledResponse(
            detail="RuntimeEvent stream is disabled"
        ).model_dump()
        return JSONResponse(
            status_code=503,
            content=content,
        )

    query_id = last_event_id_query if isinstance(last_event_id_query, str) else None
    header_id = last_event_id_header if isinstance(last_event_id_header, str) else None
    # 浏览器自动重连沿用首次连接的 URL，查询参数里是那时的回放点；请求头带的是
    # 此后最新收到的事件，两者都在时以请求头为准，否则会把收到过的事件再推一遍。
    last_event_id = header_id or query_id
    # 没有回放点的新连接回放缓冲中最近 replay_last 条，超过缓冲大小时就是整个缓冲；
    # 带回放点的续传回放其后的全部，不受它影响。
    replay_count = replay_last if isinstance(replay_last, int) else 100

    async def _event_generator():
        subscription = bus.subscribe(replay_last=replay_count, last_event_id=last_event_id)
        try:
            async for event in subscription.events():
                yield _format_sse(event)
        finally:
            subscription.close()

    # 订阅没有尽头，只能由响应结束：客户端断开或服务器开始关闭时 sse_starlette
    # 取消它。服务器关闭时要先等进行中的请求结束，才执行应用的关闭流程；不随关闭
    # 结束的流会让关闭停在等待连接上，话题也就无法结算。
    return EventSourceResponse(_event_generator())


__all__ = ["router"]
