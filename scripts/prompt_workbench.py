"""本地的提示词工作台：在导出的结算材料上修改馆长与一次性生成的提示词，运行并把两个版本并排对照。

工作台只读，不进入产品：

- 材料来自 ``memory_replay.py export`` 导出的数据集目录（``settlements.jsonl`` 与
  ``memories.jsonl``），按生产的方式渲染成话题材料；
- 馆长的检索与读取只读连接 Qdrant，缺省是开发实例的 6343/6344，``write_memory`` 照常只暂存，
  运行结束后显示暂存的记忆，不提交，也不写任何记忆；一次性生成只调用模型，显示草稿；
- 模型按主配置的 ``patchouli.llm`` 解析，与 ``memory_replay`` 相同，凭证同样来自环境变量和
  ``.env``；语言按主配置的 ``i18n.language``；馆长的步数与检索条数按主配置的
  ``patchouli.generation.librarian``。

用法：

1. 从开发实例的 Artifact 目录导出结算材料（只读）::

       python scripts/memory_replay.py export .hivememory/artifacts/<owner>/<workspace> .hivememory/prompt_workbench/materials

2. 确认开发实例的 Qdrant 在运行（``powershell -File scripts/hivememory-dev.ps1 status``），
   然后启动工作台::

       python scripts/prompt_workbench.py .hivememory/prompt_workbench/materials

   ``--port`` 改页面端口（缺省 8790，只监听本机）；``--qdrant-port`` 与
   ``--qdrant-grpc-port`` 改 Qdrant 端口；``--owner`` 与 ``--workspace`` 给出馆长读取的
   Workspace（缺省 ``default`` / ``main_workspace``）。

3. 打开打印出的地址：选目标（馆长或一次性生成）与材料，左右两栏各载入或编辑一个提示词版本，
   运行一边或两边，每边可以重复几次。馆长的每一步显示模型输出的动作、交回模型的工具结果与
   ``steps_left``，结束后显示暂存的记忆与 token；一次性生成显示模型输出与草稿。材料面板里有
   话题材料原文，以及生产中这份材料当时由结算产出的记忆。馆长的 ``list_topic_writes`` 列出的
   本话题写入要在页面上按 alias 给出，因为导出的材料不记录它们。

编辑稿可以保存为命名版本，存在 ``.hivememory/prompt_workbench/versions/<语言>/<目标>/``。
工作台不写回 ``prompts.py``：“与代码对比”给出编辑稿与代码中版本的 diff，供手动改回。
运行结果只保存在工作台进程的内存中，重启后清空。

替换提示词的方式：馆长与提取器在构造时经各自模块中的 ``get_generation_prompt_text`` 读取
提示词，工作台只在构造期间把它换成返回编辑稿的函数，运行的仍是生产的循环、工具与解析。
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import contextlib
import dataclasses
import difflib
import functools
import json
import logging
import re
import string
import threading
import time
import traceback
import uuid
from collections.abc import AsyncGenerator, Callable, Coroutine, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, override
from urllib.parse import parse_qs, urlparse

from memory_replay import librarian_model, read_jsonl, upgrade_turn

from hivememory.config.app import load_app_config
from hivememory.config.patchouli import ExtractorConfig, LibrarianConfig, QdrantConfig
from hivememory.config.shared import EmbeddingConfig
from hivememory.core.errors import StorageOfflineError
from hivememory.core.errors.model import ModelProviderError
from hivememory.core.models import WorkspaceIdentity, system_actor_for_workspace
from hivememory.engines.generation import extractor as extractor_module
from hivememory.engines.generation import librarian as librarian_module
from hivememory.engines.generation.extractor import LLMMemoryExtractor
from hivememory.engines.generation.librarian import Librarian
from hivememory.engines.generation.models import GenerationContext
from hivememory.i18n import Language, get_generation_prompt_text, set_language
from hivememory.infrastructure.llm.base import (
    BaseLLMService,
    LLMCompletion,
    LLMRequestParams,
    LLMStreamChunk,
)
from hivememory.infrastructure.storage import QdrantMemoryStore
from hivememory.memory_library import MidTermMemoryStore, QdrantStorageAdapter
from hivememory.prompts.transcript import GenerationTranscriptBuilder
from hivememory.utils.json_parser import JSONParseError, parse_llm_json

logger = logging.getLogger("prompt_workbench")

WORKBENCH_ROOT = Path(__file__).resolve().parents[1] / ".hivememory" / "prompt_workbench"

# 可编辑的提示词：目标（prompts.py 中的生成模式）→ 键 → 渲染时提供的占位符。
TARGETS: dict[str, dict[str, frozenset[str]]] = {
    "librarian": {
        "system_prompt": frozenset({"max_steps", "search_top_k"}),
        "user_prompt": frozenset({"transcript"}),
    },
    "passive": {
        "system_prompt": frozenset({"format_instructions"}),
        "user_prompt": frozenset({"transcript"}),
    },
}

# 版本名就是文件名：不含路径分隔符与点，首尾不是空格。
VERSION_NAME = re.compile(r"[\w-]+(?: [\w-]+)*")
VERSION_NAME_MAX = 60

MAX_REPEATS = 5

# HTTP 线程等待事件循环的上限；只覆盖启动运行与读取状态，不覆盖运行本身。
CALL_TIMEOUT_SECONDS = 60


class WorkbenchError(Exception):
    """页面的请求不合格；以 400 与这条说明交回页面。"""


# ---------------------------------------------------------------- prompts


@contextlib.contextmanager
def edited_prompts(target: str, prompts: dict[str, str]) -> Iterator[None]:
    """在构造馆长或提取器期间，让它们读到 ``prompts`` 中的编辑稿，其余提示词照旧。

    两者只在构造时读取提示词，构造中没有 ``await``，所以同一事件循环上并发的其他运行
    不会读到这里的替换。
    """

    def read(mode: str, key: str, language: str | Language | None = None) -> str:
        if mode == target and key in prompts:
            return prompts[key]
        return get_generation_prompt_text(mode, key, language)

    modules = (librarian_module, extractor_module)
    for module in modules:
        setattr(module, "get_generation_prompt_text", read)
    try:
        yield
    finally:
        for module in modules:
            setattr(module, "get_generation_prompt_text", get_generation_prompt_text)


def check_prompts(target: Any, prompts: Any) -> dict[str, str]:
    """校验一份编辑稿：键与目标一致，只用渲染时提供的占位符，用户消息带有材料。

    字面的花括号要写成两个，与 ``prompts.py`` 相同；否则 ``str.format`` 会把它当作占位符。
    """
    if target not in TARGETS:
        raise WorkbenchError(f"没有这个目标：{target}")
    keys = TARGETS[target]
    if (
        not isinstance(prompts, dict)
        or set(prompts) != set(keys)
        or not all(isinstance(text, str) for text in prompts.values())
    ):
        raise WorkbenchError(f"提示词必须恰好包含 {', '.join(keys)}")
    for key, text in prompts.items():
        try:
            fields = {name for _, name, _, _ in string.Formatter().parse(text) if name is not None}
        except ValueError as error:
            raise WorkbenchError(
                f"{key} 中的花括号不成对（{error}）；字面花括号写成 {{{{ 与 }}}}"
            ) from error
        unknown = fields - keys[key]
        if unknown:
            raise WorkbenchError(
                f"{key} 中有渲染时不提供的占位符 {sorted(unknown)}；"
                f"可用的是 {sorted(keys[key])}，字面花括号写成 {{{{ 与 }}}}"
            )
        if key == "user_prompt" and "transcript" not in fields:
            raise WorkbenchError("user_prompt 必须含有 {transcript}，否则模型看不到材料")
    return prompts


# ---------------------------------------------------------------- materials


@dataclass(frozen=True)
class Material:
    """一份结算材料：渲染后的话题材料，以及生产中这份材料当时由结算产出的记忆。"""

    id: str
    created_at: str
    turns: int
    first_message: str
    transcript: str
    produced: list[dict[str, Any]]


def load_materials(dataset: Path) -> dict[str, Material]:
    """读入 ``export`` 导出的数据集，按生产的渲染方式渲染每份材料，新的在前。"""
    produced: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for memory in read_jsonl(dataset / "memories.jsonl"):
        if memory["source_intent"] == "SYSTEM" and memory["settlement"]:
            produced[memory["settlement"]].append(
                {key: memory[key] for key in ("memory_type", "title", "summary", "content")}
            )
    builder = GenerationTranscriptBuilder()
    materials: dict[str, Material] = {}
    settlements = read_jsonl(dataset / "settlements.jsonl")
    for row in sorted(settlements, key=lambda row: row["created_at"], reverse=True):
        view = row["generation_view"]
        turns = [upgrade_turn(turn) for turn in view.get("turns", [])]
        context = GenerationContext.model_validate({**view, "turns": turns})
        materials[row["id"]] = Material(
            id=row["id"],
            created_at=row["created_at"],
            turns=len(turns),
            first_message=turns[0]["user_message"][:80] if turns else "",
            transcript=builder.build_transcript(context),
            produced=produced[row["id"]],
        )
    return materials


# ---------------------------------------------------------------- runs


@dataclass
class Step:
    """一次模型调用。``output`` 为 None 表示还在等模型；``reply`` 是交回模型的工具结果。"""

    output: str | None = None
    parsed: Any = None
    reply: Any = None
    tokens: int | None = None
    seconds: float | None = None
    error: str | None = None


@dataclass
class Run:
    """一次运行：一个提示词版本在一份材料上跑一遍。"""

    id: str
    target: str
    label: str
    material_id: str
    repeat: int
    status: str = "running"  # running、done 或 failed
    steps: list[Step] = field(default_factory=list)
    drafts: list[dict[str, Any]] = field(default_factory=list)
    # 馆长暂存的修订：被修订记忆的 alias 与标题，加上编辑集。
    revisions: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    started: float = field(default_factory=time.monotonic)
    seconds: float | None = None


def parse_or_none(text: str) -> Any:
    """按生产的解析规则取出模型输出中的 JSON；解析不了时为 None，页面显示原文。"""
    try:
        return parse_llm_json(text)
    except JSONParseError:
        return None


class RecordingLLM(BaseLLMService):
    """把每次模型调用记成运行的一步，调用原样转给真正的模型服务。

    馆长在同一个消息列表上依次追加模型输出与工具结果：列表依次是系统消息、材料、输出 1、
    结果 1、输出 2……所以每次调用时能补上前面各步的工具结果；循环结束后，同一个列表
    已经带有最后一步的结果，由 ``sync_replies`` 补上。一次性生成的列表只有前两条。
    """

    def __init__(self, inner: BaseLLMService, run: Run) -> None:
        self._inner = inner
        self._run = run
        self._messages: list[dict[str, str]] = []

    @override
    async def _complete(
        self, messages: list[dict[str, str]], params: LLMRequestParams
    ) -> LLMCompletion:
        self._messages = messages
        self.sync_replies()
        step = Step()
        self._run.steps.append(step)
        started = time.monotonic()
        try:
            completion = await self._inner.complete(messages, params)
        except ModelProviderError as error:
            step.error = f"{type(error).__name__}: {error}"
            raise
        finally:
            step.seconds = round(time.monotonic() - started, 1)
        step.output = completion.text
        step.parsed = parse_or_none(completion.text)
        step.tokens = completion.usage.total_tokens if completion.usage is not None else None
        return completion

    @override
    def _stream(
        self, messages: list[dict[str, str]], params: LLMRequestParams
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        return self._inner.complete(messages, params, stream=True)

    def sync_replies(self) -> None:
        for step, message in zip(self._run.steps, self._messages[3::2], strict=False):
            step.reply = json.loads(message["content"])


class Workbench:
    """页面背后的状态：材料、版本目录、模型与只读的中期库，以及本进程中的运行。

    运行只在事件循环线程上创建与修改；HTTP 线程经 ``call`` 把协程交给事件循环并等待结果，
    读到的运行因此总是某一刻完整的状态。材料与代码中的提示词不变，HTTP 线程直接读。
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        materials: dict[str, Material],
        language: Language,
        llm: BaseLLMService,
        params: LLMRequestParams,
        extractor_config: ExtractorConfig,
        librarian_config: LibrarianConfig,
        mid_term: MidTermMemoryStore,
        storage: QdrantConfig,
        belong_to: WorkspaceIdentity,
        versions_root: Path,
    ) -> None:
        self._loop = loop
        self._materials = materials
        self._language = language
        self._llm = llm
        self._params = params
        self._extractor_config = extractor_config
        self._librarian_config = librarian_config
        self._mid_term = mid_term
        self._storage = storage
        self._belong_to = belong_to
        # 与结算相同：以 Workspace 的 system actor 读取，只看得到 PUBLIC 记忆。
        self._from_actor = system_actor_for_workspace(belong_to)
        self._versions_root = versions_root
        self._runs: dict[str, Run] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def call(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result(timeout=CALL_TIMEOUT_SECONDS)

    # ------------------------------------------------ state and versions

    def state(self) -> dict[str, Any]:
        return {
            "language": self._language.value,
            "model": self._params.model,
            "qdrant": f"{self._storage.host}:{self._storage.port}",
            "workspace": self._belong_to.workspace_id,
            "librarian": {
                "max_steps": self._librarian_config.max_steps,
                "search_top_k": self._librarian_config.search_top_k,
            },
            "prompts": {
                target: {
                    key: get_generation_prompt_text(target, key, self._language) for key in keys
                }
                for target, keys in TARGETS.items()
            },
            "versions": {target: self._version_names(target) for target in TARGETS},
            "materials": [
                {
                    "id": material.id,
                    "created_at": material.created_at,
                    "turns": material.turns,
                    "chars": len(material.transcript),
                    "produced": len(material.produced),
                    "first_message": material.first_message,
                }
                for material in self._materials.values()
            ],
        }

    def material(self, material_id: str) -> dict[str, Any]:
        material = self._materials.get(material_id)
        if material is None:
            raise WorkbenchError(f"数据集中没有材料 {material_id}")
        return dataclasses.asdict(material)

    def _version_path(self, target: str, name: str) -> Path:
        if target not in TARGETS:
            raise WorkbenchError(f"没有这个目标：{target}")
        if not VERSION_NAME.fullmatch(name) or len(name) > VERSION_NAME_MAX:
            raise WorkbenchError(
                f"版本名只能用文字、数字、空格、- 与 _，首尾不是空格，不超过 {VERSION_NAME_MAX} 个字符"
            )
        return self._versions_root / self._language.value / target / f"{name}.json"

    def _version_names(self, target: str) -> list[str]:
        directory = self._versions_root / self._language.value / target
        return sorted(path.stem for path in directory.glob("*.json"))

    def save_version(self, target: Any, name: Any, prompts: Any) -> None:
        """保存为命名版本；同名时覆盖。"""
        prompts = check_prompts(target, prompts)
        path = self._version_path(target, str(name))
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "target": target,
            "language": self._language.value,
            "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "prompts": prompts,
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")

    def load_version(self, target: str, name: str) -> dict[str, str]:
        path = self._version_path(target, name)
        if not path.exists():
            raise WorkbenchError(f"没有版本 {name}")
        prompts: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))["prompts"]
        return prompts

    def diff(self, target: Any, prompts: Any) -> dict[str, str]:
        """每个键上代码中的版本与编辑稿的 unified diff；相同时为空串。"""
        if target not in TARGETS or not isinstance(prompts, dict):
            raise WorkbenchError("需要目标与编辑稿")
        diffs = {}
        for key in TARGETS[target]:
            code = get_generation_prompt_text(target, key, self._language)
            lines = difflib.unified_diff(
                code.splitlines(),
                str(prompts.get(key, "")).splitlines(),
                fromfile=f"prompts.py {target}.{key}",
                tofile=f"编辑稿 {target}.{key}",
                lineterm="",
            )
            diffs[key] = "\n".join(lines)
        return diffs

    # ------------------------------------------------ runs

    async def start_runs(self, request: dict[str, Any]) -> list[str]:
        """校验请求并为每次重复启动一次运行，返回运行 ID；运行在后台继续。"""
        target = request.get("target")
        prompts = check_prompts(target, request.get("prompts"))
        material = self._materials.get(str(request.get("material_id")))
        if material is None:
            raise WorkbenchError("先选一份材料")
        repeats = request.get("repeats")
        if not isinstance(repeats, int) or not 1 <= repeats <= MAX_REPEATS:
            raise WorkbenchError(f"重复次数是 1 到 {MAX_REPEATS}")
        aliases = request.get("topic_writes") or []
        if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
            raise WorkbenchError("本话题写入是 alias 的列表")
        topic_writes = await self._resolve_topic_writes(aliases)
        label = str(request.get("label") or "未命名")
        ids = []
        for repeat in range(1, repeats + 1):
            run = Run(
                id=uuid.uuid4().hex[:12],
                target=str(target),
                label=label,
                material_id=material.id,
                repeat=repeat,
            )
            self._runs[run.id] = run
            task = asyncio.create_task(
                self._execute(run, prompts, material.transcript, topic_writes)
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            ids.append(run.id)
        return ids

    async def _resolve_topic_writes(self, aliases: list[str]) -> list[str]:
        """把本话题写入的 alias 换成记忆 ID；读不到时拒绝运行，不让馆长静默地少看一条。"""
        ids = []
        for alias in aliases:
            memory = await self._mid_term.get_by_alias(
                self._belong_to, alias, from_actor=self._from_actor
            )
            if memory is None:
                raise WorkbenchError(f"读不到 alias 为 {alias} 的记忆：库中没有，或它不是 PUBLIC")
            ids.append(str(memory.id))
        return ids

    async def _execute(
        self, run: Run, prompts: dict[str, str], transcript: str, topic_writes: list[str]
    ) -> None:
        recorder = RecordingLLM(self._llm, run)
        try:
            if run.target == "librarian":
                with edited_prompts("librarian", prompts):
                    librarian = Librarian(
                        mid_term=self._mid_term,
                        llm_service=recorder,
                        llm_params=self._params,
                        config=self._librarian_config,
                    )
                try:
                    result = await librarian.settle(
                        transcript,
                        topic_write_memory_ids=topic_writes,
                        belong_to=self._belong_to,
                        from_actor=self._from_actor,
                    )
                finally:
                    recorder.sync_replies()
                drafts = result.drafts
                run.revisions = [
                    {
                        "alias": revision.base.index.alias,
                        "title": revision.base.index.title,
                        **revision.edit_set.model_dump(mode="json"),
                    }
                    for revision in result.revisions
                ]
            else:
                with edited_prompts("passive", prompts):
                    extractor = LLMMemoryExtractor(self._extractor_config, recorder, self._params)
                drafts = await extractor.extract(transcript)
        # 运行的边界：任何失败都记在这次运行上，页面照常显示失败前的各步；取消照常传播。
        except Exception as error:
            run.status = "failed"
            run.error = "".join(traceback.format_exception_only(error)).strip()
            logger.exception("运行失败: run_id=%s, target=%s", run.id, run.target)
        else:
            run.status = "done"
            run.drafts = [draft.model_dump(mode="json") for draft in drafts]
        finally:
            run.seconds = round(time.monotonic() - run.started, 1)

    async def snapshot(self, ids: list[str]) -> list[dict[str, Any]]:
        now = time.monotonic()
        snapshots = []
        for run_id in ids:
            run = self._runs.get(run_id)
            if run is None:
                continue
            seconds = run.seconds if run.seconds is not None else round(now - run.started, 1)
            snapshots.append({**dataclasses.asdict(run), "seconds": seconds})
        return snapshots


async def open_mid_term(storage: QdrantConfig, embedding: EmbeddingConfig) -> MidTermMemoryStore:
    """只读打开中期库：不建集合，也不接变更通知；工作台只调用检索与读取。

    在事件循环线程上构造，异步客户端从而绑定在之后运行馆长的同一个循环上。
    """
    store = QdrantMemoryStore(qdrant_config=storage, embedding_config=embedding)
    return MidTermMemoryStore(primary=QdrantStorageAdapter(store))


# ---------------------------------------------------------------- http


class Handler(BaseHTTPRequestHandler):
    """页面与它用到的 JSON 接口。"""

    def __init__(self, *args: Any, workbench: Workbench, **kwargs: Any) -> None:
        # 基类在构造中就处理请求，所以要先记下工作台。
        self.workbench = workbench
        super().__init__(*args, **kwargs)

    @override
    def log_message(self, format: str, *args: Any) -> None:
        """不逐条打印请求：运行期间页面每秒都在读取运行状态。"""

    def do_GET(self) -> None:  # noqa: N802 - http.server 按这个名字分派
        url = urlparse(self.path)
        query = {key: values[0] for key, values in parse_qs(url.query).items()}
        self._respond(lambda: self._get(url.path, query))

    def do_POST(self) -> None:  # noqa: N802 - http.server 按这个名字分派
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self._respond(lambda: self._post(urlparse(self.path).path, body))

    def _get(self, path: str, query: dict[str, str]) -> Any:
        if path == "/":
            return PAGE
        if path == "/api/state":
            return self.workbench.state()
        if path == "/api/material":
            return self.workbench.material(query.get("id", ""))
        if path == "/api/version":
            return self.workbench.load_version(query.get("target", ""), query.get("name", ""))
        if path == "/api/runs":
            ids = [run_id for run_id in query.get("ids", "").split(",") if run_id]
            return self.workbench.call(self.workbench.snapshot(ids))
        return None

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        if path == "/api/version":
            self.workbench.save_version(body.get("target"), body.get("name"), body.get("prompts"))
            return {}
        if path == "/api/diff":
            return self.workbench.diff(body.get("target"), body.get("prompts"))
        if path == "/api/runs":
            return self.workbench.call(self.workbench.start_runs(body))
        return None

    def _respond(self, handle: Callable[[], Any]) -> None:
        """页面以 HTML 返回，其余以 JSON 返回；请求不合格或 Qdrant 不可用时返回 400 与说明。"""
        try:
            result = handle()
        except (WorkbenchError, StorageOfflineError) as error:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        if result is None:
            self._send(HTTPStatus.NOT_FOUND, {"error": "没有这个地址"})
        elif isinstance(result, str):
            self._send(HTTPStatus.OK, result, "text/html; charset=utf-8")
        else:
            self._send(HTTPStatus.OK, result)

    def _send(self, status: HTTPStatus, payload: Any, content_type: str | None = None) -> None:
        if content_type is None:
            content_type = "application/json; charset=utf-8"
            payload = json.dumps(payload, ensure_ascii=False)
        data = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地的提示词工作台")
    parser.add_argument("dataset", type=Path, help="memory_replay.py export 导出的数据集目录，只读")
    parser.add_argument("--port", type=int, default=8790, help="页面端口，只监听本机")
    parser.add_argument("--qdrant-port", type=int, default=6343, help="只读连接的 Qdrant HTTP 端口")
    parser.add_argument("--qdrant-grpc-port", type=int, default=6344, help="Qdrant gRPC 端口")
    parser.add_argument("--owner", default="default", help="馆长读取的 Workspace 所有者")
    parser.add_argument("--workspace", default="main_workspace", help="馆长读取的 Workspace")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    config = load_app_config()
    set_language(config.i18n.language)
    llm, params, extractor_config = librarian_model()
    materials = load_materials(args.dataset)
    storage = config.patchouli.storage.model_copy(
        update={"port": args.qdrant_port, "grpc_port": args.qdrant_grpc_port}
    )

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, name="workbench-loop", daemon=True).start()
    mid_term = asyncio.run_coroutine_threadsafe(
        open_mid_term(storage, config.shared.embedding.default), loop
    ).result()
    workbench = Workbench(
        loop=loop,
        materials=materials,
        language=config.i18n.language,
        llm=llm,
        params=params,
        extractor_config=extractor_config,
        librarian_config=config.patchouli.generation.librarian,
        mid_term=mid_term,
        storage=storage,
        belong_to=WorkspaceIdentity(
            owner_user_id=args.owner, workspace_key=args.workspace, workspace_id=args.workspace
        ),
        versions_root=WORKBENCH_ROOT / "versions",
    )
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), functools.partial(Handler, workbench=workbench)
    )
    print(
        f"materials={len(materials)} model={params.model} "
        f"qdrant={storage.host}:{storage.port} (read only)"
    )
    print(f"提示词工作台：http://127.0.0.1:{args.port}/  （Ctrl+C 退出）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        loop.call_soon_threadsafe(loop.stop)


# ---------------------------------------------------------------- page

PAGE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>提示词工作台</title>
<style>
:root {
  --bg: #f6f5fb; --panel: #fff; --line: #e1ddef; --text: #1f1d2b; --muted: #6b6880;
  --accent: #7446e8; --soft: #efe9fd; --ok: #1b7f52; --bad: #c0392b; --wait: #a86400;
  --add: #e3f5e9; --del: #fbe7e7;
}
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.5 system-ui, "Microsoft YaHei", sans-serif; background: var(--bg); color: var(--text); }
header { padding: 10px 16px; background: var(--panel); border-bottom: 1px solid var(--line); display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap; }
header h1 { font-size: 16px; margin: 0; color: var(--accent); }
h4 { margin: 8px 0 2px; font-size: 13px; }
.meta, .hint { color: var(--muted); font-size: 12px; }
.bar { padding: 10px 16px; display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap; }
label { font-size: 12px; color: var(--muted); display: flex; flex-direction: column; gap: 2px; }
select, input, textarea, button { font: inherit; }
select, input { padding: 3px 6px; border: 1px solid var(--line); border-radius: 6px; background: #fff; }
textarea { width: 100%; font: 12px/1.45 ui-monospace, Consolas, monospace; border: 1px solid var(--line); border-radius: 6px; padding: 6px; color: var(--text); }
button { border: 1px solid var(--accent); background: var(--accent); color: #fff; border-radius: 6px; padding: 3px 12px; cursor: pointer; }
button.plain { background: #fff; color: var(--accent); }
details.material { margin: 0 16px; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 6px 10px; }
details.material > summary { cursor: pointer; color: var(--accent); }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 12px 16px; }
@media (max-width: 1100px) { .cols { grid-template-columns: 1fr; } }
.col { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px; min-width: 0; }
.side { font-size: 18px; color: var(--accent); }
.row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin: 6px 0; }
pre { white-space: pre-wrap; word-break: break-word; margin: 2px 0 6px; padding: 6px 8px; background: #faf9fd; border: 1px solid var(--line); border-radius: 6px; font: 12px/1.45 ui-monospace, Consolas, monospace; max-height: 360px; overflow: auto; }
.diff .add { background: var(--add); } .diff .del { background: var(--del); } .diff .hunk { color: var(--accent); }
details.run { border: 1px solid var(--line); border-radius: 8px; margin: 8px 0; padding: 6px 8px; }
details.run > summary { cursor: pointer; }
.status-running { color: var(--wait); } .status-done { color: var(--ok); } .status-failed { color: var(--bad); }
.step { border-left: 3px solid var(--line); padding-left: 8px; margin: 8px 0; }
.step-head { font-size: 12px; font-weight: 600; color: var(--muted); }
.memory { border: 1px solid var(--line); border-left: 3px solid var(--accent); border-radius: 6px; padding: 6px 8px; margin: 6px 0; }
.memory .title { font-weight: 600; }
.tag { display: inline-block; font-size: 11px; padding: 0 6px; border-radius: 8px; background: var(--soft); color: var(--accent); margin-right: 4px; }
.msg { font-size: 12px; white-space: pre-wrap; }
.msg.bad, .error { color: var(--bad); white-space: pre-wrap; }
.msg.ok { color: var(--ok); }
</style>
</head>
<body>
<header>
  <h1>提示词工作台</h1>
  <span class="meta" id="meta">载入中…</span>
</header>
<div class="bar">
  <label>目标
    <select id="target">
      <option value="librarian">馆长</option>
      <option value="passive">一次性生成（被动提取）</option>
    </select>
  </label>
  <label style="flex: 1; min-width: 320px">材料（新的在前）
    <select id="material"></select>
  </label>
  <label>每边重复
    <input id="repeats" type="number" min="1" max="5" value="1" style="width: 5em">
  </label>
  <label id="writes-box">本话题主动写入的记忆 alias（每行一个，可空）
    <textarea id="writes" rows="2" style="width: 22em"></textarea>
  </label>
  <button id="run-both">两边同时运行</button>
</div>
<details class="material">
  <summary>材料原文，与当时由结算产出的记忆</summary>
  <div id="material-body" class="hint">载入中…</div>
</details>
<div class="cols" id="cols"></div>
<template id="col-template">
  <div class="col">
    <div class="row">
      <strong class="side"></strong>
      <select class="version"></select>
      <button class="plain load">载入</button>
      <span class="hint dirty"></span>
    </div>
    <label>system_prompt<textarea class="p-system_prompt" rows="20" spellcheck="false"></textarea></label>
    <label>user_prompt<textarea class="p-user_prompt" rows="5" spellcheck="false"></textarea></label>
    <div class="row">
      <input class="name" placeholder="版本名" style="width: 12em">
      <button class="plain save">保存为版本</button>
      <button class="plain diff-button">与代码对比</button>
      <button class="run-one">运行这一边</button>
      <button class="plain clear">清空结果</button>
    </div>
    <div class="msg"></div>
    <div class="diff"></div>
    <div class="runs"></div>
  </div>
</template>
<script>
"use strict";
const CODE = "代码中的版本";
const STATUS = {running: "运行中", done: "完成", failed: "失败"};
let state = null;
const cols = [];
const runs = {};
const cards = {};
let polling = false;

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (key === "class") node.className = value; else node.setAttribute(key, value);
  }
  for (const child of children) if (child !== null && child !== undefined) node.append(child);
  return node;
}

const $ = (id) => document.getElementById(id);
const target = () => $("target").value;
const keys = () => Object.keys(state.prompts[target()]);
const pretty = (value) => JSON.stringify(value, null, 2);

async function init() {
  state = await api("/api/state");
  const lib = state.librarian;
  $("meta").textContent = `语言 ${state.language} · 模型 ${state.model} · Qdrant ${state.qdrant}（只读）`
    + ` · Workspace ${state.workspace} · 馆长最多 ${lib.max_steps} 步、每次检索 ${lib.search_top_k} 条`
    + ` · 材料 ${state.materials.length} 份`;
  for (const m of state.materials) {
    const when = m.created_at.slice(0, 16).replace("T", " ");
    $("material").append(el("option", {value: m.id},
      `${when} · ${m.turns} 轮 · ${m.chars} 字 · 当时 ${m.produced} 条 · ${m.first_message}`));
  }
  for (const side of ["A", "B"]) cols.push(makeColumn(side));
  $("target").onchange = onTarget;
  $("material").onchange = showMaterial;
  $("run-both").onclick = () => cols.forEach(run);
  onTarget();
  showMaterial();
}

function makeColumn(side) {
  const node = $("col-template").content.firstElementChild.cloneNode(true);
  $("cols").append(node);
  const col = {node, side, label: CODE, loaded: {}, runIds: []};
  const q = (selector) => node.querySelector(selector);
  q(".side").textContent = side;
  q(".load").onclick = () => loadVersion(col, q(".version").value);
  q(".save").onclick = () => saveVersion(col);
  q(".diff-button").onclick = () => showDiff(col);
  q(".run-one").onclick = () => run(col);
  q(".clear").onclick = () => { col.runIds = []; render(col); };
  for (const area of node.querySelectorAll("textarea")) area.oninput = () => markDirty(col);
  return col;
}

function setMsg(col, text, ok) {
  const box = col.node.querySelector(".msg");
  box.textContent = text;
  box.className = "msg " + (ok ? "ok" : "bad");
}

function onTarget() {
  $("writes-box").style.display = target() === "librarian" ? "" : "none";
  for (const col of cols) {
    fillVersions(col, CODE);
    loadVersion(col, CODE);
  }
}

function fillVersions(col, selected) {
  const select = col.node.querySelector(".version");
  const names = state.versions[target()];
  select.replaceChildren(el("option", {value: CODE}, CODE),
    ...names.map((name) => el("option", {value: name}, name)));
  select.value = names.includes(selected) ? selected : CODE;
}

const current = (col) =>
  Object.fromEntries(keys().map((key) => [key, col.node.querySelector(".p-" + key).value]));
const isDirty = (col) => keys().some((key) => current(col)[key] !== col.loaded[key]);
const runLabel = (col) => col.label + (isDirty(col) ? "*" : "");

function markDirty(col) {
  col.node.querySelector(".dirty").textContent =
    `当前：${col.label}${isDirty(col) ? "（已修改，未保存；运行时标为 *）" : ""}`;
}

async function loadVersion(col, name) {
  setMsg(col, "");
  try {
    const prompts = name === CODE ? state.prompts[target()] : await api(
      `/api/version?target=${encodeURIComponent(target())}&name=${encodeURIComponent(name)}`);
    for (const key of keys()) col.node.querySelector(".p-" + key).value = prompts[key];
    col.label = name;
    col.loaded = {...prompts};
    col.node.querySelector(".name").value = name === CODE ? "" : name;
    col.node.querySelector(".diff").replaceChildren();
    markDirty(col);
  } catch (error) { setMsg(col, error.message); }
}

async function saveVersion(col) {
  const name = col.node.querySelector(".name").value.trim();
  setMsg(col, "");
  try {
    await api("/api/version", {target: target(), name, prompts: current(col)});
    const names = state.versions[target()];
    if (!names.includes(name)) { names.push(name); names.sort(); }
    for (const other of cols) {
      fillVersions(other, other === col ? name : other.node.querySelector(".version").value);
    }
    col.label = name;
    col.loaded = current(col);
    markDirty(col);
    setMsg(col, `已保存为「${name}」`, true);
  } catch (error) { setMsg(col, error.message); }
}

async function showDiff(col) {
  setMsg(col, "");
  try {
    const diffs = await api("/api/diff", {target: target(), prompts: current(col)});
    const box = col.node.querySelector(".diff");
    box.replaceChildren();
    for (const key of keys()) {
      box.append(el("h4", {}, `${key}：${diffs[key] ? "与代码的 diff" : "与代码相同"}`));
      if (!diffs[key]) continue;
      const pre = el("pre");
      for (const line of diffs[key].split("\n")) {
        let kind = "";
        if (line.startsWith("@@")) kind = "hunk";
        else if (line.startsWith("+") && !line.startsWith("+++")) kind = "add";
        else if (line.startsWith("-") && !line.startsWith("---")) kind = "del";
        pre.append(el("div", {class: kind}, line || " "));
      }
      box.append(pre);
    }
  } catch (error) { setMsg(col, error.message); }
}

async function run(col) {
  setMsg(col, "");
  const writes = $("writes").value.split("\n").map((line) => line.trim()).filter(Boolean);
  try {
    const ids = await api("/api/runs", {
      target: target(),
      label: runLabel(col),
      prompts: current(col),
      material_id: $("material").value,
      repeats: Number($("repeats").value),
      topic_writes: target() === "librarian" ? writes : [],
    });
    col.runIds = [...ids, ...col.runIds];
    poll();
  } catch (error) { setMsg(col, error.message); }
}

async function poll() {
  if (polling) return;
  polling = true;
  try {
    for (;;) {
      const ids = cols.flatMap((col) => col.runIds)
        .filter((id) => !runs[id] || runs[id].status === "running");
      if (!ids.length) break;
      for (const snapshot of await api("/api/runs?ids=" + ids.join(","))) runs[snapshot.id] = snapshot;
      cols.forEach(render);
      await new Promise((resolve) => setTimeout(resolve, 800));
    }
  } catch (error) {
    cols.forEach((col) => setMsg(col, "读取运行状态失败：" + error.message));
  } finally {
    polling = false;
  }
}

function render(col) {
  const box = col.node.querySelector(".runs");
  const nodes = [];
  for (const id of col.runIds) {
    const snapshot = runs[id];
    if (!snapshot) continue;
    const text = JSON.stringify(snapshot);
    const old = cards[id];
    if (!old || old.text !== text) {
      const node = runCard(snapshot);
      node.open = old ? old.node.open : true;
      cards[id] = {text, node};
    }
    nodes.push(cards[id].node);
  }
  box.replaceChildren(...nodes);
}

function materialName(id) {
  const m = state.materials.find((item) => item.id === id);
  return m ? `${m.created_at.slice(0, 16).replace("T", " ")} 的材料` : id;
}

function runCard(r) {
  const tokens = r.steps.reduce((sum, step) => sum + (step.tokens || 0), 0);
  const kept = r.target === "librarian" ? "暂存" : "草稿";
  const card = el("details", {class: "run"}, el("summary", {},
    el("strong", {}, `${r.label} #${r.repeat}`), " · ",
    el("span", {class: "status-" + r.status}, STATUS[r.status]),
    ` · ${r.steps.length} 步 · ${kept} ${r.status === "running" ? "…" : r.drafts.length} 条`
      + ` · ${tokens} tokens · ${r.seconds}s`,
    el("span", {class: "hint"}, ` · ${r.target === "librarian" ? "馆长" : "一次性生成"} · ${materialName(r.material_id)}`)));
  r.steps.forEach((step, index) => card.append(stepView(r, step, index)));
  if (r.error) card.append(el("div", {class: "error"}, r.error));
  if (r.status === "done") {
    card.append(el("h4", {}, r.target === "librarian"
      ? `暂存的记忆（${r.drafts.length} 条，不会写入）` : `提取的草稿（${r.drafts.length} 条）`));
    r.drafts.forEach((draft) => card.append(memoryView(draft)));
    if (r.target === "librarian") {
      card.append(el("h4", {}, `暂存的修订（${r.revisions.length} 条，不会写入）`));
      r.revisions.forEach((revision) => card.append(el("pre", {}, pretty(revision))));
    }
  }
  return card;
}

function stepView(r, step, index) {
  const head = [`步骤 ${index + 1}`];
  if (step.seconds !== null) head.push(`${step.seconds}s`);
  if (step.tokens !== null) head.push(`${step.tokens} tokens`);
  if (step.reply && step.reply.steps_left !== undefined) head.push(`steps_left ${step.reply.steps_left}`);
  const box = el("div", {class: "step"}, el("div", {class: "step-head"}, head.join(" · ")));
  if (step.error) box.append(el("div", {class: "error"}, step.error));
  if (step.output === null) {
    if (!step.error) box.append(el("div", {class: "hint"}, "等待模型…"));
    return box;
  }
  const shown = step.parsed !== null ? pretty(step.parsed) : step.output;
  if (r.target === "librarian") {
    box.append(el("div", {class: "hint"}, step.parsed !== null ? "动作" : "动作（无法解析为 JSON，显示原文）"),
      el("pre", {}, shown));
    if (step.reply) {
      const {steps_left, ...result} = step.reply;
      box.append(el("div", {class: "hint"}, "工具结果"), el("pre", {}, pretty(result)));
    } else if (r.status === "running") {
      box.append(el("div", {class: "hint"}, "执行工具…"));
    }
  } else {
    box.append(el("details", {}, el("summary", {class: "hint"}, "模型原样输出"), el("pre", {}, shown)));
  }
  return box;
}

function memoryView(memory) {
  const tags = (memory.tags || []).map((tag) => el("span", {class: "tag"}, tag));
  const facts = [];
  if (memory.confidence_score !== undefined) facts.push(`置信度 ${memory.confidence_score}`);
  if (memory.alias_suffix) facts.push(`alias 后缀 ${memory.alias_suffix}`);
  if (memory.source_segments) facts.push(`引用段 ${memory.source_segments.join(", ") || "无"}`);
  return el("div", {class: "memory"},
    el("div", {class: "title"}, memory.title || "（无标题）"),
    el("div", {}, el("span", {class: "tag"}, memory.memory_type), ...tags,
      memory.has_value === false ? el("span", {class: "tag"}, "has_value=false") : null),
    el("div", {}, memory.summary || ""),
    facts.length ? el("div", {class: "hint"}, facts.join(" · ")) : null,
    el("pre", {}, memory.content || ""));
}

async function showMaterial() {
  const body = $("material-body");
  const id = $("material").value;
  if (!id) { body.textContent = "数据集中没有材料"; return; }
  try {
    const m = await api("/api/material?id=" + encodeURIComponent(id));
    body.replaceChildren(
      el("div", {class: "hint"}, `${m.id} · ${m.turns} 轮 · ${m.transcript.length} 字`),
      el("h4", {}, "话题材料（模型看到的原文）"),
      el("pre", {style: "max-height: 480px"}, m.transcript),
      el("h4", {}, `当时由结算产出的记忆（${m.produced.length} 条）`),
      ...m.produced.map(memoryView));
  } catch (error) { body.textContent = error.message; }
}

init().catch((error) => { $("meta").textContent = "载入失败：" + error.message; });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
