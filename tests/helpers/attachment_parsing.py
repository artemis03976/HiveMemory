"""附件解析交接测试的可控替身与有界等待辅助构造器。

供 W1-C 解析交接的单元/集成验收使用：``ScriptedAttachmentParser`` 是
兼容 ``AttachmentParser`` 协议的可控替身，可记录输入、观察 Store 快照、
用事件门控解析线程；``wait_until_condition`` 提供有界轮询等待。
"""

from __future__ import annotations

import asyncio
import threading

from hivememory.system.application.workspace_asset_service import WorkspaceAssetApplicationService
from hivememory.system.config import AttachmentParserConfig
from hivememory.system.runtime.workspace.ports import WorkspaceAssetCommandPort
from hivememory.system.services.attachments import AttachmentContentBuilder
from hivememory.system.services.attachments.parse_service import AttachmentParseService
from tests.helpers.workspace import make_access_composition, make_actor_access_record


def make_upload_service(
    store: WorkspaceAssetCommandPort,
    parser_config: AttachmentParserConfig,
    parser_factory=None,
) -> WorkspaceAssetApplicationService:
    """用同一 Store 和配置装配真实上传用例，仅允许替换解析算法。

    A1：注入共享行为检查（全操作本地注册表）以满足构造契约；上传测试
    走无 access 的迁移期兼容路径，不触发行为授权。
    """
    return WorkspaceAssetApplicationService(
        store=store,
        parser_config=parser_config,
        parse_service=AttachmentParseService(store, parser_config, parser_factory),
        access_guard=make_access_composition([make_actor_access_record()]).guard,
    )


class ChunkedSource:
    """按块返回固定内容的受控上传源，兼容 ``SupportsAsyncRead`` 协议。

    每个位置参数是一次 ``read()`` 返回的块；块耗尽后返回空 bytes 表示 EOF。
    """

    def __init__(self, *chunks: bytes) -> None:
        self._chunks = list(chunks)

    async def read(self, size: int = -1) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class ScriptedAttachmentParser:
    """可控解析协议替身。

    ``parse`` 运行在解析服务的线程内：可记录收到的 RAW bytes、
    在解析开始时观察 Store 快照、用 ``gate``（threading.Event）阻塞
    模拟耗时解析，并按注入的 ``error``/``producer_override`` 制造
    受控失败或来源漂移。
    """

    producer = "scripted"
    producer_version = "1"

    def __init__(
        self,
        store=None,
        scope=None,
        *,
        error: Exception | None = None,
        producer_override: str | None = None,
        result_text: str = "scripted body",
        gate: threading.Event | None = None,
    ) -> None:
        self._store = store
        self._scope = scope
        self._error = error
        self._producer_override = producer_override
        self._result_text = result_text
        self._gate = gate
        self.calls: list[bytes] = []
        self.snapshots_at_parse = []
        self.started = threading.Event()
        self.finished = threading.Event()

    def parse(
        self,
        raw: bytes,
        *,
        config,
        source_raw_revision: int,
        source_raw_hash: str,
        clock=None,
    ):
        self.calls.append(raw)
        if self._store is not None and self._scope is not None:
            self.snapshots_at_parse.append(self._store.list_workspace_assets(self._scope))
        self.started.set()
        if self._gate is not None:
            self._gate.wait(timeout=10)
        self.finished.set()
        if self._error is not None:
            raise self._error

        builder = AttachmentContentBuilder(
            content_format="plain_text",
            source_raw_revision=source_raw_revision,
            source_raw_hash=source_raw_hash,
            config=config,
        )
        builder.append_text(self._result_text)
        builder.add_locator(kind="paragraph", number=1, start=0, end=len(self._result_text))
        return builder.build(
            producer=self._producer_override or self.producer,
            producer_version=self.producer_version,
        )


def scripted_factory(parser: ScriptedAttachmentParser):
    """返回把所有媒体类型分派到同一替身的注入工厂。"""
    return lambda media_type: parser


async def wait_until_condition(predicate, *, timeout: float = 5.0) -> None:
    """有界轮询等待异步落定；不使用固定 sleep 后断言终态。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("等待超时：条件未在预算内成立")
        await asyncio.sleep(0.01)
