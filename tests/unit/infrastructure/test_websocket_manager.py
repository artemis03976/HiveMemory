import pytest

from hivememory.infrastructure.websocket_manager import WebSocketConnectionManager


class FakeWebSocket:
    def __init__(self, *, fail_send: bool = False) -> None:
        self.accepted = False
        self.closed = False
        self.sent_json = []
        self.fail_send = fail_send

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, message) -> None:
        if self.fail_send:
            raise RuntimeError("send failed")
        self.sent_json.append(message)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_connect_registers_connection_and_accepts_websocket():
    manager = WebSocketConnectionManager()
    websocket = FakeWebSocket()

    await manager.connect(websocket, "client-1")

    assert websocket.accepted is True
    assert manager.get_connection_count() == 1


@pytest.mark.asyncio
async def test_broadcast_buffers_message_when_no_clients():
    manager = WebSocketConnectionManager(buffer_size=2)

    await manager.broadcast({"message": "first"})
    await manager.broadcast({"message": "second"})

    websocket = FakeWebSocket()
    await manager.connect(websocket, "client-1")
    await manager.send_buffered_logs("client-1")

    assert websocket.sent_json == [{"message": "first"}, {"message": "second"}]


@pytest.mark.asyncio
async def test_broadcast_sends_to_clients_and_removes_failed_connection():
    manager = WebSocketConnectionManager()
    healthy = FakeWebSocket()
    failing = FakeWebSocket(fail_send=True)
    await manager.connect(healthy, "healthy")
    await manager.connect(failing, "failing")

    await manager.broadcast({"message": "hello"})

    assert healthy.sent_json == [{"message": "hello"}]
    assert manager.get_connection_count() == 1


@pytest.mark.asyncio
async def test_disconnect_all_closes_and_clears_connections():
    manager = WebSocketConnectionManager()
    websocket = FakeWebSocket()
    await manager.connect(websocket, "client-1")

    await manager.disconnect_all()

    assert websocket.closed is True
    assert manager.get_connection_count() == 0


def test_generate_client_id_returns_unique_string_ids():
    manager = WebSocketConnectionManager()

    first = manager.generate_client_id()
    second = manager.generate_client_id()

    assert isinstance(first, str)
    assert first != second
