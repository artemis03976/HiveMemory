"""MCP 入口：把外部 harness 的工具调用归一为操作请求，经操作入口以请求级 context 提交。

提供检索、读取、写入与更新四个工具。每次工具调用以请求头中的声明（principal、
接入凭据、用户与具体 Agent）经统一认证网关签发一份请求级访问 context，
经门面取得的操作入口的请求级提交进入 actor API，调用结束（含失败）时撤销；
不建进程，不派生线程。写入与更新先经门面取得的 Import Bus 服务为这一轮落位，
再以落点话题提交，两步共用同一份 context。工具参数由模型生成，永远不作为
身份来源；连接器补上的轮次信息只用于落位。本入口也不组装 ``IdentityScope``。
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Annotated

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field

from hivememory import __version__
from hivememory.core.errors import ResourceNotFoundError
from hivememory.core.models import MemoryAtom, PendingAtom, ReferenceResolution, WriteFocus
from hivememory.engines.memory_compiler import (
    MemoryCompileOptions,
    MemoryCompiler,
    MemoryCompileTarget,
    MemoryEnvelopeTarget,
    readable_source,
)
from hivememory.server.deps import (
    RequestAccess,
    RequestIdentitySelection,
    bearer_credential,
    request_access_for_claims,
    resolve_principal_claim,
    resolve_request_identity_claims,
)
from hivememory.server.error_handlers import APIError, project_error
from hivememory.system.access import MCP_ADAPTER
from hivememory.system.services.import_bus.models import InFlightTurn
from hivememory.workspace.contracts import (
    OperationRequest,
    ResolveReferencesRequest,
    RetrieveRequest,
    SubmitUpdateIntentRequest,
    SubmitWriteIntentRequest,
)

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from hivememory.config.memory_compiler import RetrievalContextStrategyConfig
    from hivememory.system import HiveMemorySystem

logger = logging.getLogger(__name__)

_INSTRUCTIONS = """\
HiveMemory 保存此前对话沉淀下来的长期记忆，包括项目历史、做过的决定与用户偏好。
- 回答涉及项目历史、此前的决定或用户偏好的问题之前，先用 memory_search 检索；
- 检索结果是带存档日期的历史证据，可能已经过时，引用前先核实；
- 仓库中的代码与文档、你自己的项目指令与记忆文件优先于检索结果；
- 检索结果中的 alias 可以用 memory_read 读取全文；
- 用户要求记住的内容、新做出的决定、偏好与经验，用 memory_write 写入；已有记忆过时或有误时，\
用 memory_update 按检索得到的 alias 修正；
- 写入与更新返回句柄 alias，表示已被接纳、正式记忆仍在后台生成；可以用 memory_read 读取句柄，\
生成完成后读取会转到正式记忆。"""

_SEARCH_DESCRIPTION = """\
检索 HiveMemory 中的长期记忆（此前对话沉淀的项目历史、决定与用户偏好）。
回答涉及项目历史、此前的决定或用户偏好的问题之前调用。结果是带存档日期的历史证据，\
可能已经过时：仓库中的代码与文档、你自己的项目指令与记忆文件优先于检索结果，引用前先核实。
结果中的 alias 可以交给 memory_read 读取全文。"""

_READ_DESCRIPTION = """\
按 alias 读取 HiveMemory 记忆的全文，alias 来自 memory_search 的结果，\
或 memory_write、memory_update 返回的句柄。
结果按请求顺序给出；找不到的 alias 列在末尾，全部找不到时返回错误。\
读到的内容同样是带日期的历史证据，引用前先核实。"""

_WRITE_DESCRIPTION = """\
把值得长期保存的内容写入 HiveMemory：用户要求记住的事、新做出的决定、偏好与禁忌、经验与教训。\
对话在每轮结束后会自动归档，不需要把对话过程复述写入。
content 写完整、可以独立理解的正文；title 简短概括；reason 说明为什么值得保存。
返回句柄 alias（draft_ 开头）：写入已被接纳，正式记忆在后台生成、尚未保存；\
可以用 memory_read 读取句柄，生成完成后读取会转到正式记忆。"""

_UPDATE_DESCRIPTION = """\
修正 HiveMemory 中一条已有的记忆。alias 是 memory_search 或 memory_read 得到的正式记忆 alias，\
不能是 draft_ 或 rev_ 开头的句柄；instruction 说明要怎样修改；content 可以附上替换或补充的正文。
返回句柄 alias（rev_ 开头）：更新已被接纳，新版本在后台生成、尚未保存；\
可以用 memory_read 读取句柄，生成完成后读取会转到更新后的记忆。"""

_TURN_DESCRIPTION = "当前一轮的标识与上文，由 HiveMemory 插件在调用前自动填写；不要自行填写"

_NO_RESULTS = "没有检索到相关记忆。"
_UNRESOLVED = "未找到以下 alias：{aliases}"
_ACCEPTED = """\
{operation}已登记并被生成队列接纳，句柄 alias 为 {alias}；正式记忆尚未保存，由后台生成。\
可以用 memory_read 读取这个句柄，生成完成后读取会转到正式记忆。"""

# 检索与读取只读取记忆；读取会记录引用，但不改变记忆内容。
_READ_ONLY = ToolAnnotations(read_only_hint=True)
# 写入只新增记忆，不改动已有内容；更新改写一条已有记忆的当前版本。
_WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False)
_UPDATE = ToolAnnotations(read_only_hint=False, destructive_hint=True)

# memory_read 用于取得检索结果中摘要之外的全文，因此不截断。
_FULL_CONTENT = sys.maxsize


def create_mcp_server(get_system: Callable[[], HiveMemorySystem]) -> MCPServer:
    """构造检索、读取、写入与更新四个工具的 MCP 服务；门面在每次调用时取得，服务可以先于系统装配。"""
    server = _new_server()
    compiler = MemoryCompiler()

    @server.tool(
        name="memory_search",
        description=_SEARCH_DESCRIPTION,
        annotations=_READ_ONLY,
        structured_output=False,
    )
    async def memory_search(
        query: Annotated[str, Field(min_length=1, description="用自然语言描述要找的内容")],
        ctx: Context,
    ) -> CallToolResult:
        started = time.monotonic()
        headers = ctx.headers or {}
        try:
            system = get_system()
            memories = await _submit(system, headers, RetrieveRequest(semantic_query=query))
            text = _search_text(
                compiler, memories, system.config.memory_compiler.retrieval_context.strategy
            )
        except Exception as error:
            return _failed("memory_search", headers, started, error)
        _log_call("memory_search", headers, started, f"results={len(memories)}")
        return _text_result(text)

    # Claude Code 连接器按参数名把 aliases 归一化为 READ 的作用对象，改名时同步修改它的识别表。
    @server.tool(
        name="memory_read",
        description=_READ_DESCRIPTION,
        annotations=_READ_ONLY,
        structured_output=False,
    )
    async def memory_read(
        aliases: Annotated[list[str], Field(min_length=1, description="要读取的记忆 alias")],
        ctx: Context,
    ) -> CallToolResult:
        started = time.monotonic()
        headers = ctx.headers or {}
        try:
            system = get_system()
            results = await _submit(
                system, headers, ResolveReferencesRequest(aliases=tuple(aliases))
            )
            text, unresolved = _read_text(compiler, aliases, results)
        except Exception as error:
            return _failed("memory_read", headers, started, error)
        _log_call("memory_read", headers, started, f"unresolved={len(unresolved)}")
        return _text_result(text)

    # Claude Code 连接器的 PreToolUse hook 按参数名填写 turn，回流时把它从参数中去掉；
    # 改名时同步修改连接器。
    @server.tool(
        name="memory_write",
        description=_WRITE_DESCRIPTION,
        annotations=_WRITE,
        structured_output=False,
    )
    async def memory_write(
        content: Annotated[str, Field(min_length=1, description="要保存的完整正文")],
        ctx: Context,
        title: Annotated[str | None, Field(description="简短的标题")] = None,
        reason: Annotated[str | None, Field(description="为什么值得保存")] = None,
        turn: Annotated[InFlightTurn | None, Field(description=_TURN_DESCRIPTION)] = None,
    ) -> CallToolResult:
        started = time.monotonic()
        headers = ctx.headers or {}
        try:
            system = get_system()
            request = SubmitWriteIntentRequest(
                WriteFocus(content=content, title=title, reason=reason)
            )
            pending = await _submit_intent(system, headers, turn, request)
        except Exception as error:
            return _failed("memory_write", headers, started, error)
        _log_call("memory_write", headers, started, f"intent_id={pending.intent_id}")
        return _text_result(_ACCEPTED.format(operation="写入", alias=pending.pending_alias))

    # Claude Code 连接器按参数名把 alias 归一化为 UPDATE 的作用对象，turn 的处理与
    # memory_write 相同；改名时同步修改它的识别表。
    @server.tool(
        name="memory_update",
        description=_UPDATE_DESCRIPTION,
        annotations=_UPDATE,
        structured_output=False,
    )
    async def memory_update(
        alias: Annotated[str, Field(min_length=1, description="要修正的记忆的正式 alias")],
        instruction: Annotated[str, Field(min_length=1, description="要怎样修改这条记忆")],
        ctx: Context,
        content: Annotated[str | None, Field(description="替换或补充的正文")] = None,
        turn: Annotated[InFlightTurn | None, Field(description=_TURN_DESCRIPTION)] = None,
    ) -> CallToolResult:
        started = time.monotonic()
        headers = ctx.headers or {}
        try:
            system = get_system()
            request = SubmitUpdateIntentRequest(
                base_alias=alias, instruction=instruction, content=content
            )
            pending = await _submit_intent(system, headers, turn, request)
        except Exception as error:
            return _failed("memory_update", headers, started, error)
        _log_call("memory_update", headers, started, f"intent_id={pending.intent_id}")
        return _text_result(_ACCEPTED.format(operation="更新", alias=pending.pending_alias))

    return server


def create_http_app(server: MCPServer) -> Starlette:
    """无状态、JSON 响应的 streamable HTTP 应用，端点路径为 ``/mcp``。

    每个 HTTP 请求独立处理，不保存 MCP 会话。保留 SDK 默认的 Host 与 Origin
    检查（只接受本机地址），防止 DNS 重绑定。调用之后 ``server.session_manager``
    才存在；挂进宿主应用后 SDK 自己的 lifespan 不会执行，宿主的 lifespan 必须
    运行会话管理器，否则每个请求都会失败。
    """
    return server.streamable_http_app(stateless_http=True, json_response=True)


def _new_server() -> MCPServer:
    """SDK 构造时以 ``logging.basicConfig`` 给根 logger 装自己的输出。

    根 logger 已有 handler 时 ``basicConfig`` 不生效。进程日志由 server 按配置
    统一设置，而本服务在模块导入时构造、早于日志配置；构造期间挂一个占位
    handler，否则此后每条日志都会多输出一份。
    """
    root = logging.getLogger()
    placeholder = logging.NullHandler()
    root.addHandler(placeholder)
    try:
        # 不提供 subscriptions/listen：工具与资源在运行时不变，没有可推送的变更。该请求的
        # 响应是一条一直打开的流，只在会话管理器退出时结束，而 uvicorn 要等所有连接结束后
        # 才执行宿主的 lifespan 关闭，任何在线的客户端都会让服务停不下来。
        return MCPServer(
            name="hivememory",
            instructions=_INSTRUCTIONS,
            version=__version__,
            subscriptions=False,
        )
    finally:
        root.removeHandler(placeholder)


def _request_access(
    system: HiveMemorySystem, headers: Mapping[str, str]
) -> AbstractAsyncContextManager[RequestAccess]:
    """以请求头的声明经统一认证网关签发请求级 context，退出时撤销。

    principal 与凭据由网关核对，用户、Workspace 与 Agent 声明经 server 唯一的
    声明解析与默认回退；声明不合法时在认证前拒绝。context 不离开本次调用。
    """
    principal_id = resolve_principal_claim(headers.get("x-principal-id"))
    claims = resolve_request_identity_claims(
        RequestIdentitySelection(
            user_id=headers.get("x-user-id"), workspace_id=headers.get("x-workspace-id")
        ),
        require_agent=True,
        agent_id=headers.get("x-agent-id"),
    )
    return request_access_for_claims(
        claims,
        gateway=system.access_gateway,
        principal_id=principal_id,
        adapter=MCP_ADAPTER,
        credential=bearer_credential(headers.get("authorization")),
    )


async def _submit[R](
    system: HiveMemorySystem, headers: Mapping[str, str], request: OperationRequest[R]
) -> R:
    """检索与读取：经请求级提交分派后撤销 context。"""
    async with _request_access(system, headers) as request_access:
        return await system.operation_entry.execute_request(request, access=request_access.access)


async def _submit_intent(
    system: HiveMemorySystem,
    headers: Mapping[str, str],
    turn: InFlightTurn | None,
    request: OperationRequest[PendingAtom],
) -> PendingAtom:
    """写入与更新：先为这一轮落位，再以落点话题经请求级提交；两步共用一份 context。

    缺少轮次信息时在认证前拒绝，不落位也不登记意图：写入必须落在这一轮最终
    所在的话题，不退化为不绑定话题。落位按导入授权，只校验准入与目标；提交
    由 actor API 按操作白名单授权。目标 Workspace 取声明解析的结果，与导入相同。
    """
    if turn is None:
        raise APIError("request.turn_required")
    async with _request_access(system, headers) as request_access:
        topic_id = await system.import_bus_service.place_turn(
            turn, access=request_access.access, target_workspace=request_access.target_workspace
        )
        return await system.operation_entry.execute_request(
            request, access=request_access.access, dispatch_topic_id=topic_id
        )


def _search_text(
    compiler: MemoryCompiler,
    memories: list[MemoryAtom],
    strategy: RetrievalContextStrategyConfig,
) -> str:
    """与 Alice 的检索使用同一检索上下文信封与配置的策略。

    信封头尾是提示词中预检索上下文的引导，不适用于工具结果，因此关闭。
    """
    if not memories:
        return _NO_RESULTS
    return compiler.compile(
        memories,
        MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
        MemoryCompileOptions(include_header_footer=False, retrieval_strategy_config=strategy),
    ).text


def _read_text(
    compiler: MemoryCompiler, aliases: list[str], results: list[ReferenceResolution]
) -> tuple[str, list[str]]:
    """按请求顺序逐项编译，未解析的 alias 列在末尾；全部未解析时以资源不存在拒绝。

    哪些解析结果可以交付与 MTP READ 共用 ``readable_source``。
    """
    blocks: list[str] = []
    unresolved: list[str] = []
    for alias, result in zip(aliases, results, strict=True):
        source = readable_source(result)
        if source is None:
            unresolved.append(alias)
            continue
        # pending 句柄自带 alias；其余结果按请求的 alias 呈现，redirect 才能标出原名。
        requested_alias = None if isinstance(source, PendingAtom) else alias
        options = MemoryCompileOptions(
            requested_alias=requested_alias, max_content_length=_FULL_CONTENT
        )
        blocks.append(compiler.compile(source, MemoryCompileTarget.READ_RESULT, options).text)
    if not blocks:
        raise ResourceNotFoundError(details={"aliases": unresolved})
    if unresolved:
        blocks.append(_UNRESOLVED.format(aliases=", ".join(unresolved)))
    return "\n".join(blocks), unresolved


def _text_result(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)])


def _failed(
    tool: str, headers: Mapping[str, str], started: float, error: Exception
) -> CallToolResult:
    """复用 HTTP 出口的错误投影，写入 ``isError`` 的工具结果。

    投影与 HTTP 响应体相同：错误码、安全说明与白名单内的 reason，异常正文
    不公开。服务端错误像 HTTP 入口一样在本次调用边界记录一次完整异常栈。
    """
    status, payload = project_error(error)
    _log_call(
        tool,
        headers,
        started,
        f"error={payload.error}",
        level=logging.ERROR if status >= 500 else logging.INFO,
        exc_info=error if status >= 500 else None,
    )
    return CallToolResult(
        content=[TextContent(type="text", text=payload.model_dump_json(exclude_none=True))],
        is_error=True,
    )


def _log_call(
    tool: str,
    headers: Mapping[str, str],
    started: float,
    outcome: str,
    *,
    level: int = logging.INFO,
    exc_info: Exception | None = None,
) -> None:
    """每次工具调用一行：工具、声明的 principal 与 Agent、结果与耗时。

    不记录查询、记忆正文与凭据；principal 与 Agent 是请求中的声明，按 ``%r``
    记录，使请求方提供的文本保持在一行。
    """
    logger.log(
        level,
        "MCP tool call: tool=%s principal=%r agent=%r %s elapsed_ms=%.0f",
        tool,
        headers.get("x-principal-id"),
        headers.get("x-agent-id"),
        outcome,
        (time.monotonic() - started) * 1000,
        exc_info=exc_info,
    )
