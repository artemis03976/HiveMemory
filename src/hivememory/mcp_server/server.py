"""MCP 入口：把外部 harness 的工具调用归一为操作请求，经操作入口的请求级提交进入 actor API。

提供检索、读取、写入与更新四个工具。每次工具调用以请求头中的声明（principal、
接入凭据、用户与具体 Agent）经统一认证网关认证一次，得到的工作 actor 只是
本次调用内的局部值，经门面取得的操作入口的请求级提交进入 actor API；不建
进程，不派生线程。写入与更新先经门面取得的 Import Bus 服务为这一轮确定话题，再以
该话题提交，两步共用同一个工作 actor。工具参数由模型生成，永远不作为身份
来源；连接器补上的轮次信息只用于确定话题。本入口也不组装 ``IdentityScope``。
"""

import logging
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Annotated

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field

from hivememory import __version__
from hivememory.components.trace_context import (
    generate_trace_id,
    reset_trace_context,
    set_trace_context,
)
from hivememory.core.access import WorkingActor
from hivememory.core.errors import ResourceNotFoundError
from hivememory.core.models import MemoryAtom, PendingAtom, ReferenceResolution, WriteFocus
from hivememory.engines.memory_compiler import (
    MemoryCompileOptions,
    MemoryCompiler,
    MemoryCompileTarget,
    MemoryEnvelopeTarget,
    readable_source,
)
from hivememory.i18n import Language
from hivememory.i18n.mcp import get_mcp_text
from hivememory.server.deps import (
    RequestIdentitySelection,
    authenticate_request_actor,
    bearer_credential,
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

# 检索与读取只读取记忆；读取会记录引用，但不改变记忆内容。
_READ_ONLY = ToolAnnotations(read_only_hint=True)
# 写入只新增记忆，不改动已有内容；更新改写一条已有记忆的当前版本。
_WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False)
_UPDATE = ToolAnnotations(read_only_hint=False, destructive_hint=True)


def create_mcp_server(
    get_system: Callable[[], "HiveMemorySystem"], language: Language
) -> MCPServer:
    """构造检索、读取、写入与更新四个工具的 MCP 服务；门面在每次调用时取得，服务可以先于系统装配。

    服务说明、工具与参数描述和结果文本按 ``language``（HiveMemory 的语言设置）给出，构造后不变。
    """

    def text(key: str) -> str:
        return get_mcp_text(key, language)

    server = _new_server(text("instructions"))
    compiler = MemoryCompiler()

    @server.tool(
        name="memory_search",
        description=text("search_description"),
        annotations=_READ_ONLY,
        structured_output=False,
    )
    async def memory_search(
        query: Annotated[str, Field(min_length=1, description=text("query_description"))],
        ctx: Context,
    ) -> CallToolResult:
        started = time.monotonic()
        headers = ctx.headers or {}
        with _call_trace("memory_search"):
            try:
                system = get_system()
                memories = await _submit(system, headers, RetrieveRequest(semantic_query=query))
                result = _search_text(
                    compiler,
                    memories,
                    system.config.memory_compiler.retrieval_context.strategy,
                    language,
                )
            except Exception as error:
                return _failed("memory_search", headers, started, error)
            _log_call("memory_search", headers, started, f"results={len(memories)}")
            return _text_result(result)

    # Claude Code 连接器按参数名把 aliases 归一化为 READ 的作用对象，改名时同步修改它的识别表。
    @server.tool(
        name="memory_read",
        description=text("read_description"),
        annotations=_READ_ONLY,
        structured_output=False,
    )
    async def memory_read(
        aliases: Annotated[list[str], Field(min_length=1, description=text("aliases_description"))],
        ctx: Context,
    ) -> CallToolResult:
        started = time.monotonic()
        headers = ctx.headers or {}
        with _call_trace("memory_read"):
            try:
                system = get_system()
                results = await _submit(
                    system, headers, ResolveReferencesRequest(aliases=tuple(aliases))
                )
                result, unresolved = _read_text(compiler, aliases, results, language)
            except Exception as error:
                return _failed("memory_read", headers, started, error)
            _log_call("memory_read", headers, started, f"unresolved={len(unresolved)}")
            return _text_result(result)

    # Claude Code 连接器的 PreToolUse hook 按参数名填写 turn，回流时把它从参数中去掉；
    # 改名时同步修改连接器。
    @server.tool(
        name="memory_write",
        description=text("write_description"),
        annotations=_WRITE,
        structured_output=False,
    )
    async def memory_write(
        content: Annotated[str, Field(min_length=1, description=text("write_content_description"))],
        ctx: Context,
        title: Annotated[str | None, Field(description=text("title_description"))] = None,
        reason: Annotated[str | None, Field(description=text("reason_description"))] = None,
        turn: Annotated[InFlightTurn | None, Field(description=text("turn_description"))] = None,
    ) -> CallToolResult:
        started = time.monotonic()
        headers = ctx.headers or {}
        with _call_trace("memory_write"):
            try:
                system = get_system()
                request = SubmitWriteIntentRequest(
                    WriteFocus(content=content, title=title, reason=reason)
                )
                pending = await _submit_intent(system, headers, turn, request)
            except Exception as error:
                return _failed("memory_write", headers, started, error)
            _log_call("memory_write", headers, started, f"intent_id={pending.intent_id}")
            return _text_result(text("write_accepted").format(alias=pending.pending_alias))

    # Claude Code 连接器按参数名把 alias 归一化为 UPDATE 的作用对象，turn 的处理与
    # memory_write 相同；改名时同步修改它的识别表。
    @server.tool(
        name="memory_update",
        description=text("update_description"),
        annotations=_UPDATE,
        structured_output=False,
    )
    async def memory_update(
        alias: Annotated[str, Field(min_length=1, description=text("update_alias_description"))],
        instruction: Annotated[
            str, Field(min_length=1, description=text("instruction_description"))
        ],
        ctx: Context,
        content: Annotated[
            str | None, Field(description=text("update_content_description"))
        ] = None,
        turn: Annotated[InFlightTurn | None, Field(description=text("turn_description"))] = None,
    ) -> CallToolResult:
        started = time.monotonic()
        headers = ctx.headers or {}
        with _call_trace("memory_update"):
            try:
                system = get_system()
                request = SubmitUpdateIntentRequest(
                    base_alias=alias, instruction=instruction, content=content
                )
                pending = await _submit_intent(system, headers, turn, request)
            except Exception as error:
                return _failed("memory_update", headers, started, error)
            _log_call("memory_update", headers, started, f"intent_id={pending.intent_id}")
            return _text_result(text("update_accepted").format(alias=pending.pending_alias))

    return server


def create_http_app(server: MCPServer) -> "Starlette":
    """无状态、JSON 响应的 streamable HTTP 应用，端点路径为 ``/mcp``。

    每个 HTTP 请求独立处理，不保存 MCP 会话。保留 SDK 默认的 Host 与 Origin
    检查（只接受本机地址），防止 DNS 重绑定。调用之后 ``server.session_manager``
    才存在；挂进宿主应用后 SDK 自己的 lifespan 不会执行，宿主的 lifespan 必须
    运行会话管理器，否则每个请求都会失败。
    """
    return server.streamable_http_app(stateless_http=True, json_response=True)


def _new_server(instructions: str) -> MCPServer:
    """SDK 构造时以 ``logging.basicConfig`` 给根 logger 装自己的输出。

    根 logger 已有 handler 时 ``basicConfig`` 不生效：server 在配置日志之后才构造本服务，
    日志配置总会在根 logger 上装 handler，所以不会多出一份输出。
    """
    # 不提供 subscriptions/listen：工具与资源在运行时不变，没有可推送的变更。该请求的
    # 响应是一条一直打开的流，只在会话管理器退出时结束，而 uvicorn 要等所有连接结束后
    # 才执行宿主的 lifespan 关闭，任何在线的客户端都会让服务停不下来。
    return MCPServer(
        name="hivememory",
        instructions=instructions,
        version=__version__,
        subscriptions=False,
    )


@contextmanager
def _call_trace(tool: str) -> Iterator[None]:
    """一次工具调用在自己的 trace 中进行。

    actor 操作、写入时确定话题引起的 Gateway 被动路由与新建话题的事件都继承它，
    归为同一次运行；日志也带上它。外部 agent 在等待结果，按前台工作呈现。
    """
    tokens = set_trace_context(generate_trace_id("mcp"), tool, "foreground")
    try:
        yield
    finally:
        reset_trace_context(tokens)


async def _authenticate(system: "HiveMemorySystem", headers: Mapping[str, str]) -> WorkingActor:
    """以请求头的声明经统一认证网关认证一次，返回本次调用的工作 actor。

    principal 与凭据由网关核对，用户、Workspace 与 Agent 声明经 server 唯一的
    声明解析与默认回退；声明不合法时在认证前拒绝。工作 actor 不离开本次调用。
    """
    principal_id = resolve_principal_claim(headers.get("x-principal-id"))
    claims = resolve_request_identity_claims(
        RequestIdentitySelection(
            user_id=headers.get("x-user-id"), workspace_id=headers.get("x-workspace-id")
        ),
        require_agent=True,
        agent_id=headers.get("x-agent-id"),
    )
    return await authenticate_request_actor(
        claims,
        gateway=system.access_gateway,
        principal_id=principal_id,
        adapter=MCP_ADAPTER,
        credential=bearer_credential(headers.get("authorization")),
    )


async def _submit[R](
    system: "HiveMemorySystem", headers: Mapping[str, str], request: OperationRequest[R]
) -> R:
    """检索与读取：认证后经请求级提交分派。"""
    actor = await _authenticate(system, headers)
    return await system.operation_entry.execute_request(request, actor=actor)


async def _submit_intent(
    system: "HiveMemorySystem",
    headers: Mapping[str, str],
    turn: InFlightTurn | None,
    request: OperationRequest[PendingAtom],
) -> PendingAtom:
    """写入与更新：认证后经请求级提交，授权通过才经 Import Bus 为这一轮确定话题。

    缺少轮次信息时在认证前拒绝，不确定话题也不登记意图：写入必须落在这一轮最终
    所在的话题，不退化为不绑定话题。提交由 actor API 按工作 actor 的操作白名单
    授权，授权之后才调用确定话题的回调，被拒的写入因此不会路由、建立话题或
    触发 LRU 结算；确定话题由系统执行，与 Agent 的许可无关。目标 Workspace 取
    工作 actor 驻留的 Workspace，与导入相同。
    """
    if turn is None:
        raise APIError("request.turn_required")
    actor = await _authenticate(system, headers)

    async def determine_topic() -> str:
        return await system.import_bus_service.place_turn(
            turn, actor=actor, target_workspace=actor.workspace
        )

    return await system.operation_entry.execute_request(
        request, actor=actor, dispatch_topic=determine_topic
    )


def _search_text(
    compiler: MemoryCompiler,
    memories: list[MemoryAtom],
    strategy: "RetrievalContextStrategyConfig",
    language: Language,
) -> str:
    """与 Alice 的检索使用同一检索上下文信封与配置的策略。

    信封头尾是提示词中预检索上下文的引导，不适用于工具结果，因此关闭。
    """
    if not memories:
        return get_mcp_text("no_results", language)
    return compiler.compile(
        memories,
        MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
        MemoryCompileOptions(include_header_footer=False, retrieval_strategy_config=strategy),
    ).text


def _read_text(
    compiler: MemoryCompiler,
    aliases: list[str],
    results: list[ReferenceResolution],
    language: Language,
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
        options = MemoryCompileOptions(requested_alias=requested_alias)
        blocks.append(compiler.compile(source, MemoryCompileTarget.READ_RESULT, options).text)
    if not blocks:
        raise ResourceNotFoundError(details={"aliases": unresolved})
    if unresolved:
        blocks.append(get_mcp_text("unresolved", language).format(aliases=", ".join(unresolved)))
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
        level=logging.ERROR if status >= 500 else logging.WARNING,
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
    level: int = logging.DEBUG,
    exc_info: Exception | None = None,
) -> None:
    """每次工具调用一行：工具、声明的 principal 与 Agent、结果与耗时。

    成功的调用在 debug 级别记录，经过由操作入口的 ``actor.operation`` 事件记录；失败在
    warning 或 error 级别记录，认证在操作之前被拒绝时这一行是唯一的记录。不记录查询、
    记忆正文与凭据；principal 与 Agent 是请求中的声明，按 ``%r`` 记录，使请求方提供的
    文本保持在一行。
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
