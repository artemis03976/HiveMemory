"""上传传输边界的资源清理；上传业务和 Store 协作由集成测试覆盖。"""

import io
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from hivememory.server import deps
from hivememory.server.error_handlers import install_error_handlers
from hivememory.server.routers.workspace_assets import router


def test_rejected_extra_file_closes_every_parsed_upload(monkeypatch):
    """未知字段也可能包含临时文件，拒绝请求时不能只关闭名为 file 的部分。"""
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(router, prefix="/api/v1")
    service = AsyncMock()
    app.dependency_overrides[deps.get_root_workspace_asset_service] = lambda: service
    app.dependency_overrides[deps.get_request_access] = lambda: None
    closed = []
    original_close = UploadFile.close

    async def track_close(upload):
        await original_close(upload)
        closed.append((upload.filename, upload.file.closed))

    monkeypatch.setattr(UploadFile, "close", track_close)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/workspace/assets",
            headers={"Idempotency-Key": "op-1"},
            files=[
                ("file", ("first.txt", io.BytesIO(b"first"), "text/plain")),
                ("extra", ("second.txt", io.BytesIO(b"second"), "text/plain")),
            ],
        )
    assert response.status_code == 400
    assert response.json()["error"] == "request.upload_fields_invalid"
    assert closed == [("first.txt", True), ("second.txt", True)]
    service.upload_asset.assert_not_awaited()


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({}, {"error": "request.file_required", "detail": "缺少 file 文件字段"}),
        (
            {"Content-Type": "multipart/form-data"},
            {"error": "request.multipart_invalid", "detail": "multipart 表单解析失败"},
        ),
    ],
)
def test_invalid_form_keeps_named_transport_error(headers, expected):
    """缺文件与缺 boundary 属于不同解析分支，但都只返回受控传输提示。"""
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[deps.get_root_workspace_asset_service] = lambda: None
    app.dependency_overrides[deps.get_request_access] = lambda: None
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/workspace/assets",
            headers={"Idempotency-Key": "op-1", **headers},
            content=b"secret",
        )
    assert response.status_code == 400
    assert response.json() == expected
