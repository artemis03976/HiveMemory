"""MTP 指令的解析、策略检查、动词分派与统一错误回填。"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from functools import partial
from typing import TYPE_CHECKING, Protocol

from hivememory.agent_runtime.mtp.context import (
    MTPExecutionContext,
    MTPExecutionResult,
)
from hivememory.agent_runtime.mtp.error_mapping import mtp_error_info
from hivememory.agent_runtime.mtp.verbs.call import handle_call
from hivememory.agent_runtime.mtp.verbs.read import handle_read
from hivememory.agent_runtime.mtp.verbs.run import handle_run
from hivememory.agent_runtime.mtp.verbs.search import handle_search
from hivememory.agent_runtime.mtp.verbs.update import handle_update
from hivememory.agent_runtime.mtp.verbs.write import handle_write
from hivememory.core.mtp import (
    MTP_LEFT_DELIMITER,
    MTPCommand,
    MTPErrorSeverity,
    MTPFilterParser,
    MTPFormatter,
    MTPParseError,
    MTPParser,
    MTPResponse,
    MTPResponseStatus,
    MTPVerb,
)
from hivememory.core.mtp.exceptions import (
    InvalidArgumentError,
    PermissionDeniedError,
)
from hivememory.engines.memory_compiler import (
    MemoryCompiler,
)

if TYPE_CHECKING:
    from hivememory.config.alice import KoakumaConfig
    from hivememory.config.memory_compiler import MemoryCompilerConfig


logger = logging.getLogger(__name__)


class MTPExecutor(Protocol):
    """执行循环消费的 MTP 端口，由 Koakuma 直接满足。"""

    async def intercept_and_execute(
        self, assistant_text: str, context: MTPExecutionContext
    ) -> MTPExecutionResult | None: ...


class KoakumaRuntime:
    """
    小恶魔 MTP 运行时 (Koakuma MTP Runtime)

    无状态计算服务，负责 MTP 协议的解析、路由和执行。

    职责:
        1. 接收已由 LLMClient 归一化的 MTP 输出文本
        2. 解析 MTP 指令
        3. 将资源操作请求提交给 workspace 能力层
        4. 格式化执行结果为 XML 响应容器
        5. 返回回填文本，供执行循环注入 Assistant 历史

    资源访问由执行上下文中的提交函数提供；运行时只保留协议翻译、
    结果编译与工具执行，不直接持有资源 owner 的总线路由。
    """

    def __init__(
        self,
        config: KoakumaConfig | None = None,
        *,
        memory_compiler_config: MemoryCompilerConfig | None = None,
    ):
        """
        初始化 Koakuma MTP 运行时

        Args:
            config: Koakuma 配置 (可选，使用默认值)
        """
        from hivememory.config.alice import KoakumaConfig

        self._config = config or KoakumaConfig()
        self._parser = MTPParser()
        self._filter_parser = MTPFilterParser()
        self._formatter = MTPFormatter()

        self._memory_compiler_config = memory_compiler_config

        # 初始化 sys_* 系统工具的 syscall 注册表
        # 硬编码的 sys_ 工具集，随系统启动加载，Zero Latency
        from hivememory.agent_runtime.tools import build_syscall_registry

        self._syscall_registry = build_syscall_registry(
            python_repl_timeout=self._config.python_repl_timeout_seconds,
            workspace_path=self._config.workspace_path,
            file_read_max_bytes=self._config.file_read_max_bytes,
            file_write_max_bytes=self._config.file_write_max_bytes,
            web_search_timeout=self._config.web_search_timeout_seconds,
        )

        self._compiler = MemoryCompiler()

        logger.info("KoakumaRuntime (小恶魔 MTP 运行时) 初始化完成")

    # ========== frame 执行策略 ==========

    def _check_verb_permission(
        self,
        verb: str,
        context: MTPExecutionContext,
    ) -> None:
        """
        校验 frame 的 MTP 动词集合；资源授权仍由 Workspace 逐次执行。

        Args:
            verb: MTP 动词 (如 "WRITE", "RUN")

        Raises:
            PermissionDeniedError: 当前人偶无权执行此动词
        """
        if not context.execution_policy.allows(verb):
            raise PermissionDeniedError(
                message_key="mtp.permission.verb_denied",
                params={"verb": verb},
            )

    # ========== 公开 API ==========

    async def execute_mtp(
        self,
        text: str,
        context: MTPExecutionContext,
    ) -> MTPExecutionResult:
        """
        执行 MTP 指令 (主入口)

        完整流程:
        1. 解析指令
        2. 路由到对应处理器
        3. 格式化响应
        4. 构建回填文本

        Args:
            text: 已规范化的 MTP 指令文本

        Returns:
            MTPExecutionResult: 执行结果
        """
        start_time = time.time()

        try:
            language = context.language
            # Step 1: 解析
            command = self._parser.parse(text)

            # Step 2: 路由执行
            response = await self._route_and_execute(
                command,
                context,
            )
            response.execution_time_ms = (time.time() - start_time) * 1000

            # Step 3: 格式化回填文本
            formatted = self._formatter.format_response(response, language)

            return MTPExecutionResult(
                command=command,
                response_status=response.status.value,
                response_content=response.content,
                formatted_response=formatted,
                success=(response.status != MTPResponseStatus.ERROR),
                execution_time_ms=response.execution_time_ms,
                pending_alias=response.pending_alias,
                call_request=response.call_request,
            )

        except MTPParseError as e:
            elapsed = (time.time() - start_time) * 1000
            language = context.language
            error_response = MTPResponse(
                status=MTPResponseStatus.ERROR,
                content="",
                error=mtp_error_info(e),
                execution_time_ms=elapsed,
            )
            formatted = self._formatter.format_response(error_response, language)

            return MTPExecutionResult(
                command=None,
                response_status=error_response.status.value,
                response_content=error_response.content,
                formatted_response=formatted,
                success=False,
                execution_time_ms=elapsed,
            )

    async def intercept_and_execute(
        self,
        assistant_text: str,
        context: MTPExecutionContext,
    ) -> MTPExecutionResult | None:
        """
        拦截检测 + 执行（Stop Sequence 场景）

        当 LLM API 因 stop=["⟫"] 而停止时调用。
        检测文本末尾是否包含 MTP 指令，如果是则执行。

        拦截流程:
        1. 捕获 (Capture): 查找最后一个 ⟪
        2. 解析 (Parsing): 提取 VERB, TARGET, ARGS
        3. 挂起 (Suspend): 暂停生成并执行指令

        Args:
            assistant_text: 已由 LLMClient 补全右定界符的 LLM 输出文本

        Returns:
            MTPExecutionResult 如果检测到指令，否则 None
        """
        last_open = assistant_text.rfind(MTP_LEFT_DELIMITER)
        if last_open == -1:
            return None

        # 提取从 ⟪ 开始的文本片段
        mtp_fragment = assistant_text[last_open:]

        return await self.execute_mtp(
            mtp_fragment,
            context=context,
        )

    # ========== 内部路由 ==========

    async def _route_and_execute(
        self,
        command: MTPCommand,
        context: MTPExecutionContext,
    ) -> MTPResponse:
        """
        路由并执行 MTP 指令

        根据 VERB 分发到对应的处理器。
        集中捕获所有 MTP 语义化异常，格式化为 Agent 可读的错误提示。

        Args:
            command: 解析后的 MTP 指令

        Returns:
            MTPResponse: 执行响应
        """
        handlers: dict[
            MTPVerb, Callable[[MTPCommand, MTPExecutionContext], Awaitable[MTPResponse]]
        ] = {
            MTPVerb.SEARCH: partial(
                handle_search,
                filter_parser=self._filter_parser,
                compiler=self._compiler,
                compiler_config=self._memory_compiler_config,
            ),
            MTPVerb.READ: partial(handle_read, compiler=self._compiler),
            MTPVerb.RUN: partial(
                handle_run,
                syscall_registry=self._syscall_registry,
                timeout_seconds=self._config.python_repl_timeout_seconds,
            ),
            MTPVerb.WRITE: handle_write,
            MTPVerb.UPDATE: handle_update,
            MTPVerb.CALL: handle_call,  # 子代理调用
        }

        handler = handlers.get(command.verb)
        if handler is None:
            raise InvalidArgumentError(
                message_key="mtp.parse.unknown_verb",
                params={"verb": command.verb, "valid_verbs": ", ".join(v.value for v in MTPVerb)},
            )

        try:
            # 先校验 frame 动词集合，拒绝后不会提交资源操作或执行系统工具。
            self._check_verb_permission(command.verb.value, context=context)
            return await handler(command, context)

        except Exception as error:
            # 取消是 BaseException，不进入协议错误回填；未知失败保留内部原因链。
            info = mtp_error_info(
                error,
                verb=command.verb.value,
                alias=command.target.single_alias or "",
                can_read=context.execution_policy.allows("READ"),
            )
            if info.severity == MTPErrorSeverity.SYSTEM_FAULT:
                logger.error("MTP execution failed: %s", info.code, exc_info=True)
            else:
                logger.info("MTP request rejected: %s", info.code)
            return MTPResponse(status=MTPResponseStatus.ERROR, content="", error=info)


__all__ = ["KoakumaRuntime", "MTPExecutor"]
