"""附件上传公开入口的集成测试：HTTP 路由 + 上传应用服务 + 真实 Store + 真实 parser。

覆盖计划 A 门/B/C 门验收：HTTP 上传结果与同一 Workspace 的 Store 快照及
资产列表一致；上传响应在请求内到达 READY/FAILED 终态；幂等重放不重复
解析；Workspace 隔离与各错误路径都有稳定的 HTTP 状态。
"""

import io
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.core.models import WorkspaceAssetRef
from hivememory.server import deps
from hivememory.server.routers.workspace_assets import router
from hivememory.system.application.workspace_asset_service import (
    WorkspaceAssetApplicationService,
)
from hivememory.system.config import AttachmentParserConfig
from hivememory.system.runtime.workspace.store import InMemoryWorkspaceAssetStore
from tests.helpers.workspace import make_identity_scope

USER_HEADERS = {"x-user-id": "user-1"}


def _upload_files(
    client: TestClient,
    *,
    file_name: str = "notes.md",
    content: bytes = b"# hello\n",
    content_type: str = "text/markdown",
    operation_id: str = "op-1",
    headers: dict[str, str] | None = None,
    **kwargs,
):
    merged_headers = {**USER_HEADERS, "Idempotency-Key": operation_id, **(headers or {})}
    return client.post(
        "/api/v1/workspace/assets",
        files={"file": (file_name, io.BytesIO(content), content_type)},
        headers=merged_headers,
        **kwargs,
    )


def _make_docx(body_xml: str) -> bytes:
    """构造最小有效 DOCX 包（真实 parser 转换链路使用）。"""
    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            (
                '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/word/document.xml" ContentType='
                '"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'
            ),
        )
        archive.writestr(
            "_rels/.rels",
            (
                '<?xml version="1.0"?><Relationships '
                'xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type='
                '"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                'Target="word/document.xml"/></Relationships>'
            ),
        )
        archive.writestr(
            "word/document.xml",
            (
                f'<?xml version="1.0"?><w:document xmlns:w="{namespace}">'
                f"<w:body>{body_xml}</w:body></w:document>"
            ),
        )
    return buffer.getvalue()


@pytest.fixture
def upload_stack():
    """构造真实 router + 应用服务 + Store 的测试应用。"""
    store = InMemoryWorkspaceAssetStore()
    service = WorkspaceAssetApplicationService(store=store, parser_config=AttachmentParserConfig())
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[deps.get_workspace_asset_service] = lambda: service
    return TestClient(app), store


def test_upload_returns_201_with_terminal_summary_matching_store_snapshot(upload_stack) -> None:
    """捕获 HTTP 摘要与 Store 快照漂移、或响应泄露内部字段。"""
    client, store = upload_stack

    response = _upload_files(client)

    assert response.status_code == 201
    body = response.json()
    assert set(body) == {
        "asset_ref",
        "kind",
        "display_name",
        "media_type",
        "size_bytes",
        "state",
        "safe_error",
        "required_representation",
        "raw_representation",
    }
    assert (
        body["kind"],
        body["display_name"],
        body["media_type"],
        body["size_bytes"],
        body["state"],
    ) == ("document", "notes.md", "text/markdown", 8, "ready")
    raw = body["raw_representation"]
    assert (raw["kind"], raw["revision"], raw["state"]) == ("raw", 1, "ready")
    assert raw["content_hash"] == "9e8b62f81ea5c66fa06ee53da032751386b37702153070c0e14dd1d316282fa7"
    assert (raw["producer"], raw["producer_version"]) == ("upload", "1")
    # W1-C：请求内解析完成，required representation 与 asset 同响应到达终态。
    required = body["required_representation"]
    assert (required["kind"], required["revision"], required["state"]) == (
        "extracted_text",
        1,
        "ready",
    )
    assert body["safe_error"] is None

    # A 门：HTTP 结果与同一 Workspace 的 Store 快照及资产列表一致。
    handles = store.list_workspace_assets(make_identity_scope(user_id="user-1"))
    assert [handle.asset.asset_id for handle in handles] == [body["asset_ref"]["asset_id"]]
    asset = handles[0].asset
    assert asset.representations[0].content_hash == raw["content_hash"]
    assert asset.representations[0].revision == raw["revision"]
    # required text READY 后 reader 才开放（B/C 门）。
    resolved = store.resolve_asset(
        make_identity_scope(user_id="user-1"),
        WorkspaceAssetRef.model_validate(body["asset_ref"]),
    )
    assert resolved.state.value == "ready"


def test_public_chain_reaches_terminal_for_all_approved_formats(upload_stack) -> None:
    """捕获三类批准格式无法在公开入口到达 READY，或损坏 DOCX 未反馈解析失败。"""
    client, store = upload_stack

    txt = _upload_files(
        client,
        file_name="plain.txt",
        content="中文正文\n".encode(),
        content_type="text/plain",
        operation_id="op-txt",
    )
    markdown = _upload_files(
        client,
        file_name="doc.md",
        content=b"# title\n\nbody\n",
        content_type="text/markdown",
        operation_id="op-md",
    )
    docx = _upload_files(
        client,
        file_name="real.docx",
        content=_make_docx(
            "<w:p><w:r><w:t>正文段落</w:t></w:r></w:p>",
        ),
        content_type=("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        operation_id="op-docx",
    )
    corrupt = _upload_files(
        client,
        file_name="broken.docx",
        content=b"PK\x03\x04 not really a zip",
        content_type=("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        operation_id="op-corrupt",
    )

    assert txt.status_code == 201
    assert txt.json()["state"] == "ready"
    assert markdown.status_code == 201
    assert markdown.json()["state"] == "ready"
    assert docx.status_code == 201
    assert docx.json()["state"] == "ready"
    assert docx.json()["required_representation"]["state"] == "ready"

    # 损坏 DOCX：RAW 保留，解析失败以 failed 终态与安全摘要反馈，不改写为上传失败。
    assert corrupt.status_code == 201
    corrupt_body = corrupt.json()
    assert corrupt_body["state"] == "failed"
    assert corrupt_body["safe_error"]["code"] == "workspace.asset.failed"
    assert corrupt_body["raw_representation"]["state"] == "ready"
    assert corrupt_body["required_representation"]["state"] == "failed"

    states = [
        handle.asset.state.value
        for handle in store.list_workspace_assets(
            make_identity_scope(user_id="user-1"),
        )
    ]
    assert sorted(states) == ["failed", "ready", "ready", "ready"]


def test_idempotent_replay_returns_200_with_same_logical_asset(upload_stack) -> None:
    """捕获重放丢失原 ref、重复创建资产或错报 201。"""
    client, store = upload_stack

    first = _upload_files(client)
    replay = _upload_files(client)

    assert first.status_code == 201
    assert replay.status_code == 200
    assert replay.json()["asset_ref"] == first.json()["asset_ref"]
    assert len(store.list_workspace_assets(make_identity_scope(user_id="user-1"))) == 1


def test_same_key_with_different_content_returns_409(upload_stack) -> None:
    """捕获同一 operation 携带另一份文件被静默接受。"""
    client, store = upload_stack

    _upload_files(client, content=b"# hello\n")
    conflict = _upload_files(client, content=b"# changed\n")

    assert conflict.status_code == 409
    assert len(store.list_workspace_assets(make_identity_scope(user_id="user-1"))) == 1


def test_replay_after_remove_returns_410_and_does_not_recreate(upload_stack) -> None:
    """捕获已移除资产的重放被复活。"""
    client, store = upload_stack

    first = _upload_files(client)
    scope = make_identity_scope(user_id="user-1")
    store.remove_asset(scope, WorkspaceAssetRef.model_validate(first.json()["asset_ref"]))
    replay = _upload_files(client)

    assert replay.status_code == 410
    assert store.list_workspace_assets(scope) == []


def test_store_closed_returns_503(upload_stack) -> None:
    """捕获 Store 关闭后的上传被误报为 409/500。"""
    client, store = upload_stack

    store.close_and_clear()
    closed = _upload_files(client)

    assert closed.status_code == 503


def test_unopened_workspace_returns_404(upload_stack) -> None:
    """捕获公共入口对未开放 Workspace 的身份选择未按既有语义拒绝。"""
    client, _ = upload_stack

    response = _upload_files(
        client,
        headers={"x-workspace-id": "another_workspace"},
    )

    assert response.status_code == 404


def test_other_owner_workspace_stays_isolated(upload_stack) -> None:
    """捕获不同用户（不同 Workspace）的同名上传互相串味。"""
    client, store = upload_stack

    first = _upload_files(client)
    other = _upload_files(client, headers={"x-user-id": "user-2"})

    assert other.status_code == 201
    assert other.json()["asset_ref"] != first.json()["asset_ref"]
    assert len(store.list_workspace_assets(make_identity_scope(user_id="user-1"))) == 1
    assert len(store.list_workspace_assets(make_identity_scope(user_id="user-2"))) == 1


def test_empty_file_returns_400(upload_stack) -> None:
    """捕获空文件被接受为 0 字节资产。"""
    client, _ = upload_stack

    response = _upload_files(client, content=b"")

    assert response.status_code == 400


def test_unsupported_format_returns_415_with_stable_hint(upload_stack) -> None:
    """捕获不批准格式得到 500 或空文案。"""
    client, _ = upload_stack

    legacy = _upload_files(
        client,
        file_name="old.doc",
        content=b"legacy",
        content_type="application/msword",
        operation_id="op-doc",
    )
    pdf = _upload_files(
        client,
        file_name="paper.pdf",
        content=b"%PDF-1.4",
        content_type="application/pdf",
        operation_id="op-pdf",
    )

    assert legacy.status_code == 415
    assert "另存为 .docx" in legacy.json()["detail"]
    assert pdf.status_code == 415
    assert pdf.json()["detail"]


def test_oversized_file_returns_413(upload_stack) -> None:
    """捕获超限文件未被稳定拒绝。"""
    store = InMemoryWorkspaceAssetStore()
    service = WorkspaceAssetApplicationService(
        store=store,
        parser_config=AttachmentParserConfig(max_raw_bytes=8),
    )
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[deps.get_workspace_asset_service] = lambda: service
    client = TestClient(app)

    response = _upload_files(client, content=b"123456789")

    assert response.status_code == 413


def test_duplicate_file_parts_and_extra_fields_are_rejected(upload_stack) -> None:
    """捕获多 file part、重复 file part 或额外业务字段被静默接受。"""
    client, _ = upload_stack

    duplicate = client.post(
        "/api/v1/workspace/assets",
        files=[
            ("file", ("a.md", io.BytesIO(b"1"), "text/markdown")),
            ("file", ("b.md", io.BytesIO(b"2"), "text/markdown")),
        ],
        headers={**USER_HEADERS, "Idempotency-Key": "op-dup"},
    )
    extra = client.post(
        "/api/v1/workspace/assets",
        files={"file": ("a.md", io.BytesIO(b"1"), "text/markdown")},
        data={"agent_id": "smuggled"},
        headers={**USER_HEADERS, "Idempotency-Key": "op-extra"},
    )
    missing_key = client.post(
        "/api/v1/workspace/assets",
        files={"file": ("a.md", io.BytesIO(b"1"), "text/markdown")},
        headers=USER_HEADERS,
    )

    assert duplicate.status_code == 400
    assert extra.status_code == 400
    assert missing_key.status_code == 400


def test_missing_file_part_returns_400(upload_stack) -> None:
    """捕获缺少文件字段的请求被误放行到应用服务。"""
    client, _ = upload_stack

    response = client.post(
        "/api/v1/workspace/assets",
        data={"not_a_file": "text"},
        headers={**USER_HEADERS, "Idempotency-Key": "op-missing"},
    )

    assert response.status_code == 400
