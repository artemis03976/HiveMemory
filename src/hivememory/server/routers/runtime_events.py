"""RuntimeEvent SSE transport adapter."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from hivememory.server.deps import get_system
from hivememory.server.models.runtime_event import (
    RuntimeEventDisabledResponse,
    RuntimeEventResponse,
    RuntimeEventStatusResponse,
)
from hivememory.system.contracts.runtime_events import RuntimeEvent

router = APIRouter(prefix="/runtime-events", tags=["runtime-events"])


def _format_sse(event: RuntimeEvent) -> str:
    data = RuntimeEventResponse.from_domain(event).model_dump_json()
    return f"id: {event.event_id}\nevent: runtime_event\ndata: {data}\n\n"


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


@router.get("/stream")
async def stream_runtime_events(
    request: Request,
    last_event_id_query: str | None = Query(default=None, alias="last_event_id"),
    last_event_id_header: str | None = Header(default=None, alias="Last-Event-ID"),
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
    last_event_id = query_id or header_id

    async def _event_generator():
        subscription = bus.subscribe(last_event_id=last_event_id)
        try:
            async for event in subscription.events():
                if await request.is_disconnected():
                    break
                yield _format_sse(event)
                await asyncio.sleep(0)
        finally:
            subscription.close()

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


__all__ = ["router"]
