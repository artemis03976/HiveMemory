from fastapi import WebSocketDisconnect
import pytest

from hivememory.server.routers.logs import websocket_logs_endpoint


class FakeWebSocket:
    def __init__(self, incoming):
        self.incoming = list(incoming)
        self.sent_text = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def receive_text(self):
        if not self.incoming:
            raise WebSocketDisconnect()
        value = self.incoming.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    async def send_text(self, text):
        self.sent_text.append(text)


class FakeManager:
    def __init__(self):
        self.connected = []
        self.disconnected = []
        self.sent_buffered = []

    def generate_client_id(self):
        return "client-1"

    async def connect(self, websocket, client_id):
        self.connected.append((websocket, client_id))
        await websocket.accept()

    async def send_buffered_logs(self, client_id):
        self.sent_buffered.append(client_id)

    async def disconnect(self, client_id):
        self.disconnected.append(client_id)


@pytest.mark.asyncio
async def test_websocket_logs_endpoint_accepts_catches_up_and_pongs():
    websocket = FakeWebSocket(["ping"])
    manager = FakeManager()

    await websocket_logs_endpoint(websocket, manager)

    assert websocket.accepted is True
    assert manager.connected == [(websocket, "client-1")]
    assert manager.sent_buffered == ["client-1"]
    assert websocket.sent_text == ["pong"]
    assert manager.disconnected == ["client-1"]


@pytest.mark.asyncio
async def test_websocket_logs_endpoint_disconnects_on_unexpected_error():
    websocket = FakeWebSocket([RuntimeError("boom")])
    manager = FakeManager()

    await websocket_logs_endpoint(websocket, manager)

    assert manager.disconnected == ["client-1"]
