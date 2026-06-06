"""
小恶魔 (Koakuma) - MTP Runtime Service

定位：MTP 协议的运行时执行器
职责：
    - MTP 指令解析 (委托给 MTPParser)
    - 指令路由与分发 (通过 Alice local bus 调用公开服务 API)
    - 响应格式化与回填
    - 别名解析

架构定位：
    Koakuma 是 AliceRuntime 持有的 MTP / 工具执行运行时，
    负责处理 Worker Agent 生成的 MTP 指令。它不管理 Agent
    运行循环，也不保存交互 trace；权限、身份与深度信息由
    AgentRuntime 在执行时通过 MTPExecutionContext 传入。

    AliceRuntime
    ├── AgentRuntime (Agent 运行循环)
    └── KoakumaRuntime (MTP / 工具执行)

对应设计文档: MemoryToolProtocol.md Chapter 3 & 4

作者: HiveMemory Team
版本: 1.0
"""

import time
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from hivememory.core.mtp import (
    MTP_LEFT_DELIMITER,
    MTP_RIGHT_DELIMITER,
    MTPVerb,
    MTPResponseStatus,
    MTPCommand,
    MTPResponse,
    MTPParser,
    MTPFilterParser,
    MTPParseError,
    MTPFormatter,
)
from hivememory.agent_runtime.cache import KoakumaAtomCache
from hivememory.agent_runtime.models import MTPExecutionContext
from hivememory.agent_runtime.pending_atom import PendingAtomRuntime
from hivememory.agent_runtime.resolver import RuntimeAliasResolver
from hivememory.engines.memory_compiler import MemoryCompiler, MemoryCompileTarget, MemoryCompileOptions
from hivememory.core.protocol.models import (
    MTPExecutionResult,
    RetrievalRequest,
)
from hivememory.system.contracts.routes import GlobalRoutes

from hivememory.core.models import MemoryType
from hivememory.core.mtp.exceptions import (
    MTPError,
    AgentFault,
    SystemFault,
    StorageOfflineError,
    StorageReadError,
    PermissionDeniedError,
    InvalidArgumentError,
    AliasNotFoundError,
    MemoryTypeMismatchError,
    SyscallInternalError,
)
from hivememory.i18n.resolver import resolve_language

if TYPE_CHECKING:
    from hivememory.system.runtime.bus.async_bus import AsyncSystemBus
    from hivememory.agent_runtime.resolver import ResolveResult
    from hivememory.system.config import KoakumaConfig

logger = logging.getLogger(__name__)


def _resolve_context_language(
    context: Optional[MTPExecutionContext],
) -> str:
    """从 MTPExecutionContext 派生运行时语言。

    优先级：context.language > agent_profile.language > i18n default
    """
    explicit = context.language if context is not None else None
    profile_language = (
        getattr(context.agent_profile, "language", None)
        if context is not None and context.agent_profile is not None
        else None
    )
    return resolve_language(
        explicit=explicit,
        profile_language=profile_language,
    ).value


class KoakumaRuntime:
    """
    小恶魔 MTP 运行时 (Koakuma MTP Runtime)

    无状态计算服务，负责 MTP 协议的解析、路由和执行。

    职责:
        1. 接收被截断的 LLM 输出文本
        2. 补全并解析 MTP 指令
        3. 路由到对应的内核服务 (Retrieval/Librarian)
        4. 格式化执行结果为 XML 响应容器
        5. 返回回填文本供 Kernel 注入 Assistant 历史

    使用示例:
        >>> koakuma = KoakumaRuntime(
        ...     retrieval_familiar=retrieval,
        ...     librarian_core=librarian,
        ...     storage=storage,
        ... )
        >>> result = koakuma.execute_mtp('⟪ READ | fact_api_spec | ⟫')
        >>> print(result.formatted_response)
    """

    def __init__(
        self,
        bus: Optional["AsyncSystemBus"] = None,
        config: Optional["KoakumaConfig"] = None,
        *,
        alias_resolver: RuntimeAliasResolver,
    ):
        """
        初始化 Koakuma MTP 运行时

        Args:
            bus: AsyncSystemBus 实例，用于跨服务通信（纯异步总线）
            config: Koakuma 配置 (可选，使用默认值)
            alias_resolver: 运行时别名解析器
        """
        from hivememory.system.config import KoakumaConfig

        self._bus = bus
        self._config = config or KoakumaConfig()
        self._parser = MTPParser()
        self._filter_parser = MTPFilterParser()
        self._formatter = MTPFormatter()

        self._alias_resolver = alias_resolver

        # 初始化内核工具注册表 KERNEL_REGISTRY (Section 4.2.1)
        # 硬编码的 sys_ 工具集，随系统启动加载，Zero Latency
        from hivememory.agent_runtime.mtp.syscalls import build_kernel_registry
        self._kernel_registry = build_kernel_registry(
            python_repl_timeout=self._config.python_repl_timeout_seconds,
            workspace_path=self._config.workspace_path,
            file_read_max_bytes=self._config.file_read_max_bytes,
            file_write_max_bytes=self._config.file_write_max_bytes,
            web_search_timeout=self._config.web_search_timeout_seconds,
        )

        self._compiler = MemoryCompiler()

        logger.info("KoakumaRuntime (小恶魔 MTP 运行时) 初始化完成")

    # ========== 多智能体权限沙箱 (Phase 1) ==========

    def _check_verb_permission(
        self,
        verb: str,
        context: Optional[MTPExecutionContext] = None,
    ) -> None:
        """
        校验 MTP 动词权限 (O(1) set lookup)

        Args:
            verb: MTP 动词 (如 "WRITE", "RUN")

        Raises:
            PermissionDeniedError: 当前人偶无权执行此动词
        """
        profile = context.agent_profile if context is not None else None
        if profile is None:
            return
        if not profile.is_verb_allowed(verb):
            raise PermissionDeniedError(
                message_key="mtp.permission.verb_denied",
                params={"verb": verb},
            )

    def _check_tool_permission(
        self,
        tool_alias: str,
        context: Optional[MTPExecutionContext] = None,
    ) -> None:
        """
        校验系统工具权限 (O(1) set lookup)

        Args:
            tool_alias: 工具别名 (如 "sys_write_file")

        Raises:
            PermissionDeniedError: 当前人偶无权使用此工具
        """
        profile = context.agent_profile if context is not None else None
        if profile is None:
            return
        if not profile.is_tool_allowed(tool_alias):
            raise PermissionDeniedError(
                message_key="mtp.permission.tool_denied",
                params={"tool_alias": tool_alias},
            )

    # ========== 公开 API ==========

    async def execute_mtp(
        self,
        text: str,
        context: Optional[MTPExecutionContext] = None,
    ) -> MTPExecutionResult:
        """
        执行 MTP 指令 (主入口)

        完整流程:
        1. 补全并解析指令
        2. 路由到对应处理器
        3. 格式化响应
        4. 构建回填文本

        Args:
            text: 原始 MTP 指令文本 (可能不含闭合 ⟫)

        Returns:
            MTPExecutionResult: 执行结果
        """
        start_time = time.time()

        try:
            # Step 1: 补全并解析
            command = self._parser.complete_and_parse(text)

            # Step 2: 路由执行
            response = await self._route_and_execute(
                command,
                context or MTPExecutionContext(),
            )
            response.execution_time_ms = (time.time() - start_time) * 1000

            # Step 3: 格式化回填文本
            formatted = self._formatter.format_command_with_response(
                command, response
            )

            return MTPExecutionResult(
                command=command,
                response_status=response.status.value,
                response_content=response.content,
                formatted_response=formatted,
                success=(response.status != MTPResponseStatus.ERROR),
                execution_time_ms=response.execution_time_ms,
                pending_alias=response.pending_alias,
            )

        except MTPParseError as e:
            elapsed = (time.time() - start_time) * 1000
            language = _resolve_context_language(context)
            error_response = MTPResponse(
                status=MTPResponseStatus.ERROR,
                content=e.to_agent_prompt(language),
                error=e.to_error_info(),
                execution_time_ms=elapsed,
            )
            formatted = self._formatter.format_response(error_response)

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
        context: Optional[MTPExecutionContext] = None,
    ) -> Optional[MTPExecutionResult]:
        """
        拦截检测 + 执行 (Section 3.1.2 Stop Sequence 场景)

        当 LLM API 因 stop=["⟫"] 而停止时调用。
        检测文本末尾是否包含 MTP 指令，如果是则执行。

        拦截流程:
        1. 捕获 (Capture): 查找最后一个 ⟪
        2. 补全 (Completion): 自动追加 ⟫
        3. 解析 (Parsing): 提取 VERB, TARGET, ARGS
        4. 挂起 (Suspend): 进入内核态执行

        Args:
            assistant_text: LLM 生成的完整文本 (在 ⟫ 处被截断)

        Returns:
            MTPExecutionResult 如果检测到指令，否则 None
        """
        last_open = assistant_text.rfind(MTP_LEFT_DELIMITER)
        if last_open == -1:
            return None

        # 提取从 ⟪ 开始的文本片段
        mtp_fragment = assistant_text[last_open:]

        # 补全 ⟫ 并执行
        if MTP_RIGHT_DELIMITER not in mtp_fragment:
            mtp_fragment = mtp_fragment.rstrip() + " " + MTP_RIGHT_DELIMITER

        return await self.execute_mtp(mtp_fragment, context=context)

    # ========== 别名管理 ==========

    @property
    def atom_cache(self) -> KoakumaAtomCache:
        """访问统一原子缓存"""
        return self._alias_resolver.atom_cache

    @property
    def pending_runtime(self) -> PendingAtomRuntime:
        """访问运行时 pending atom 管理中心"""
        return self._alias_resolver.pending_runtime

    # ========== 内部路由 ==========

    async def _route_and_execute(
        self,
        command: MTPCommand,
        context: MTPExecutionContext,
    ) -> MTPResponse:
        """
        路由并执行 MTP 指令 (Section 3)

        根据 VERB 分发到对应的处理器。
        集中捕获所有 MTP 语义化异常，格式化为 Agent 可读的错误提示。

        Args:
            command: 解析后的 MTP 指令

        Returns:
            MTPResponse: 执行响应
        """
        handlers = {
            MTPVerb.SEARCH: self._handle_search,
            MTPVerb.READ: self._handle_read,
            MTPVerb.RUN: self._handle_run,
            MTPVerb.WRITE: self._handle_write,
            MTPVerb.UPDATE: self._handle_update,
            MTPVerb.CALL: self._handle_call,  # Phase 2: 子代理调用
        }

        handler = handlers.get(command.verb)
        if handler is None:
            raise InvalidArgumentError(
                message_key="mtp.parse.unknown_verb",
                params={"verb": command.verb, "valid_verbs": ", ".join(v.value for v in MTPVerb)},
            )

        try:
            # 权限沙箱：校验 MTP 动词权限 (Phase 1 多智能体)
            self._check_verb_permission(command.verb.value, context=context)
            return await handler(command, context)

        except MTPError as e:
            if isinstance(e, SystemFault):
                logger.error(f"System fault during {command.verb}: {e}", exc_info=True)
            else:
                logger.info(f"Agent fault during {command.verb}: {e}")
            language = _resolve_context_language(context)
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content=e.to_agent_prompt(language),
                error=e.to_error_info(),
            )

        except Exception as e:
            logger.error(f"Unexpected error during {command.verb}: {e}", exc_info=True)
            fault = SystemFault(
                "An unexpected error occurred. Do NOT retry this command. Continue the conversation normally.",
                cause=e,
            )
            language = _resolve_context_language(context)
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content=fault.to_agent_prompt(language),
                error=fault.to_error_info(),
            )

    # ========== 指令处理器 ==========

    async def _handle_search(
        self,
        command: MTPCommand,
        context: MTPExecutionContext,
    ) -> MTPResponse:
        """
        处理 SEARCH 指令 (Section 2.2)

        调用 RetrievalFamiliar 进行模糊检索，返回 RetrievalResponse.rendered_context。
        具体渲染由 Retrieval 链路统一完成，Koakuma 只负责执行与缓存别名。

        Type A 数据类响应 (Section 3.3.3)

        Args:
            command: SEARCH 指令 (target=*, args: query="...", filter="...")

        Returns:
            MTPResponse: RetrievalResponse 渲染后的上下文
        """
        query = command.args.get("query", "")
        if not query:
            raise InvalidArgumentError(message_key="mtp.search.missing_query")

        # 解析 filter 参数 (Section 2.2)
        # 例如 filter="type:CODE" → QueryFilters(memory_type=CODE_SNIPPET)
        # 宽容解析: 非法 filter 降级为 None，但返回警告
        filter_str = command.args.get("filter", "")
        parsed_filters, filter_warnings = self._filter_parser.parse(filter_str) if filter_str else (None, [])

        # Let StorageOfflineError / StorageReadError propagate to _route_and_execute
        result = await self._bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE,
            request=RetrievalRequest(
                semantic_query=query,
                identity=context.identity,
                filters=parsed_filters,
            ),
        )

        if result.is_empty():
            content = "No memories found. Try a different query."
            if filter_warnings:
                content += "\n" + "\n".join(filter_warnings)
            return MTPResponse(
                status=MTPResponseStatus.SUCCESS,
                content=content,
            )

        content = result.rendered_context
        if not content:
            logger.warning(
                "RetrievalResponse for MTP SEARCH contains memories but no rendered_context."
            )
            content = "Search completed, but no rendered context was returned."
        if filter_warnings:
            content += "\n" + "\n".join(filter_warnings)

        # 将检索到的记忆原子缓存（完整对象，而非仅 UUID）
        self.atom_cache.ingest_atoms(result.memories)

        return MTPResponse(
            status=MTPResponseStatus.SUCCESS,
            content=content,
        )

    async def _handle_read(
        self,
        command: MTPCommand,
        context: MTPExecutionContext,
    ) -> MTPResponse:
        """
        处理 READ 指令 (Section 2.2)

        获取记忆原子的 Payload 内容。支持列表并行读取。

        Type A 数据类响应 (Section 3.3.3)

        Args:
            command: READ 指令 (target=alias 或 [alias1, alias2])

        Returns:
            MTPResponse: 记忆内容
        """
        if command.target.is_wildcard:
            raise InvalidArgumentError(message_key="mtp.read.wildcard_not_supported")

        aliases = command.target.aliases
        if not aliases:
            raise InvalidArgumentError(message_key="mtp.read.missing_alias")

        resolved: List[Tuple[str, "MemoryAtom"]] = []  # (alias, atom)
        resolved_pending: List[Tuple[str, Any]] = []  # (alias, PendingAtom)
        resolved_redirects: List[Tuple[str, Any]] = []  # (alias, ResolveResult)
        resolved_terminal: List[Tuple[str, Any]] = []  # (alias, ResolveResult)
        unresolved: List[str] = []
        for alias in aliases:
            result = await self._alias_resolver.resolve(alias, context=context)
            if result.kind == "pending" and result.pending is not None:
                resolved_pending.append((alias, result.pending))
            elif result.kind == "redirect" and result.atom is not None:
                resolved_redirects.append((alias, result))
            elif result.kind == "atom" and result.atom is not None:
                resolved.append((alias, result.atom))
            elif result.kind in {"discarded", "failed", "expired"}:
                resolved_terminal.append((alias, result))
            else:
                unresolved.append(alias)

        # 全部无效：直接返回错误
        if not resolved and not resolved_pending and not resolved_redirects and not resolved_terminal:
            raise AliasNotFoundError(
                message_key="mtp.read.alias_not_found",
                params={"aliases": "\n".join(f"  {a}" for a in unresolved)},
            )

        # 组装输出 — 通过 MemoryCompiler 统一编译
        output_lines: List[str] = []
        for alias, pending in resolved_pending:
            artifact = self._compiler.compile(
                pending, MemoryCompileTarget.MTP_READ,
            )
            output_lines.append(artifact.text)
        for alias, result in resolved_redirects:
            artifact = self._compiler.compile(
                result, MemoryCompileTarget.MTP_READ,
                MemoryCompileOptions(requested_alias=alias),
            )
            output_lines.append(artifact.text)
        for alias, result in resolved_terminal:
            artifact = self._compiler.compile(
                result, MemoryCompileTarget.MTP_READ,
                MemoryCompileOptions(requested_alias=alias),
            )
            output_lines.append(artifact.text)
        for alias, atom in resolved:
            artifact = self._compiler.compile(
                atom, MemoryCompileTarget.MTP_READ,
                MemoryCompileOptions(requested_alias=alias),
            )
            output_lines.append(artifact.text)
        for alias in unresolved:
            output_lines.append(
                f"[{alias}]: [Alias Not Found] Alias '{alias}' not found. "
                f"Use SEARCH to discover the correct alias first."
            )

        response = MTPResponse(
            status=MTPResponseStatus.SUCCESS,
            content="\n".join(output_lines),
        )
        for _, atom in resolved:
            await self._record_memory_citation(atom, "mtp.read")
        for _, result in resolved_redirects:
            await self._record_memory_citation(result.atom, "mtp.read")
        return response

    async def _handle_run(
        self,
        command: MTPCommand,
        context: MTPExecutionContext,
    ) -> MTPResponse:
        """
        处理 RUN 指令 (Section 2.2)

        两层分发:
        - Level 0: 内核工具快速路径 (KERNEL_REGISTRY, Section 4.2.1)
        - Level 1: 用户态工具慢速路径 (LRU Cache → L1/L2 别名解析 → Qdrant → 沙箱执行)

        Type B 动作类响应 (Section 3.3.3)

        Args:
            command: RUN 指令 (target=tool_alias, args=key-value pairs)

        Returns:
            MTPResponse: 执行状态描述
        """
        alias = command.target.single_alias
        if alias is None:
            raise InvalidArgumentError(message_key="mtp.run.missing_single_target")

        syscall = self._kernel_registry.get(alias)
        if syscall is None and alias.startswith("sys_"):
            raise AliasNotFoundError(
                message_key="mtp.run.kernel_tool_not_found",
                params={"alias": alias},
            )
        if syscall is not None:
            # 权限沙箱：校验系统工具权限 (Phase 1 多智能体)
            self._check_tool_permission(alias, context=context)
            result = syscall.handler(command.args)
            if not result.ok:
                exc_cls = InvalidArgumentError if result.error_code == "mtp.argument.invalid" else SyscallInternalError
                raise exc_cls(result.content, params={"alias": alias})
            return MTPResponse(status=MTPResponseStatus.SUCCESS, content=result.content)

        # Level 1: 用户态工具路径 (统一原子缓存)
        # StorageOfflineError / BusRouteUnavailableError 会直接传播到 _route_and_execute
        resolved = await self._alias_resolver.resolve(alias, context=context)
        redirect_notice = ""
        if resolved.kind == "pending":
            raise InvalidArgumentError(
                message_key="mtp.run.pending_not_runnable",
                params={"alias": alias},
            )
        if resolved.kind == "redirect" and resolved.atom is not None:
            canonical_alias = resolved.canonical_alias or resolved.atom.get_alias()
            redirect_notice = self._compiler.compile(
                resolved, MemoryCompileTarget.MTP_REDIRECT_NOTICE,
                MemoryCompileOptions(requested_alias=alias),
            ).text
            alias = canonical_alias
        elif resolved.kind in {"discarded", "failed", "expired"}:
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content=self._compiler.compile(
                    resolved, MemoryCompileTarget.MTP_READ,
                    MemoryCompileOptions(requested_alias=alias),
                ).text,
            )
        if resolved.kind not in {"atom", "redirect"} or resolved.atom is None:
            raise AliasNotFoundError(
                message_key="mtp.run.alias_not_found",
                params={"alias": alias},
            )
        atom = resolved.atom

        # 校验类型必须是 CODE_SNIPPET
        if atom.index.memory_type != MemoryType.CODE_SNIPPET:
            raise MemoryTypeMismatchError(
                message_key="mtp.run.type_mismatch",
                params={"alias": alias, "memory_type": atom.index.memory_type.value},
            )

        # 使用缓存的代码执行
        code = atom.payload.content
        logger.info(f"User tool executing: alias='{alias}', UUID={atom.id}")
        response = self._execute_user_tool(alias, code, command.args)
        if redirect_notice:
            response.content = f"{redirect_notice}{response.content}"
        if response.status == MTPResponseStatus.SUCCESS:
            await self._record_memory_citation(atom, "mtp.run")
        return response

    async def _handle_write(
        self,
        command: MTPCommand,
        context: MTPExecutionContext,
    ) -> MTPResponse:
        """
        处理 WRITE 指令 (Section 2.2 + 附录B)

        v3.0 延迟捕获模式:
        将 WRITE 内容打包为 WriteFocus，随 MTPExecutionResult 返回给 LoopExecutor，
        实际记忆生成延迟到 InteractionPayload 提交时执行。
        ACK 响应文案保持不变，对 Agent 完全透明。

        Type B 动作类响应 (Section 3.3.3)

        Args:
            command: WRITE 指令 (target=*, args: content=`...`, reason="...", title="...")

        Returns:
            MTPResponse: ACK 确认
        """
        content = command.args.get("content", "")
        if not content:
            raise InvalidArgumentError(message_key="mtp.write.missing_content")

        reason = command.args.get("reason", "")
        title = command.args.get("title", "")

        # 注册 pending atom 并生成 pending alias
        pending = self.pending_runtime.register_write(
            content=content,
            title=title or None,
            reason=reason or None,
            identity=context.identity,
            runtime_scope=context.runtime_scope,
        )

        logger.info(
            f"MTP WRITE 延迟捕获: content='{content[:50]}...', "
            f"pending_alias='{pending.pending_alias}'"
        )

        return MTPResponse(
            status=MTPResponseStatus.ACK,
            content=self._format_write_ack(pending.pending_alias, _resolve_context_language(context)),
            pending_alias=pending.pending_alias,
        )

    async def _handle_update(
        self,
        command: MTPCommand,
        context: MTPExecutionContext,
    ) -> MTPResponse:
        """
        处理 UPDATE 指令 (附录 C)

        v3.0 延迟捕获模式:
        将 UPDATE 意图打包为 UpdateFocus，随 MTPExecutionResult 返回给 LoopExecutor，
        实际记忆更新延迟到 InteractionPayload 提交时执行。
        ACK 响应文案保持不变，对 Agent 完全透明。

        Type B 动作类响应 (Section 3.3.3)

        Args:
            command: UPDATE 指令 (target=alias, args: instruction="...", content=`...`)

        Returns:
            MTPResponse: ACK 确认或 ERROR
        """
        alias = command.target.single_alias
        if alias is None:
            raise InvalidArgumentError(message_key="mtp.update.missing_single_target")

        instruction = command.args.get("instruction", "")
        if not instruction:
            raise InvalidArgumentError(message_key="mtp.update.missing_instruction")

        resolved = await self._alias_resolver.resolve(alias, context=context)
        if resolved.kind == "pending":
            raise InvalidArgumentError(
                message_key="mtp.update.pending_not_updatable",
                params={"alias": alias},
            )
        if resolved.kind != "atom" or resolved.atom is None:
            raise AliasNotFoundError(
                message_key="mtp.update.alias_not_found",
                params={"alias": alias},
            )
        atom = resolved.atom
        uuid = str(atom.id)

        # 4. 获取可选的 content
        content = command.args.get("content", None)

        # 5. 注册 pending revision
        pending = self.pending_runtime.register_update(
            base_alias=alias,
            base_uuid=uuid,
            instruction=instruction,
            content=content,
            identity=context.identity,
            runtime_scope=context.runtime_scope,
        )

        # 7. 使缓存失效，防止脏读
        self.atom_cache.invalidate_alias(alias)

        logger.info(
            f"MTP UPDATE 延迟捕获: alias='{alias}', "
            f"pending_alias='{pending.pending_alias}'"
        )

        return MTPResponse(
            status=MTPResponseStatus.ACK,
            content=self._format_update_ack(alias, pending.pending_alias, _resolve_context_language(context)),
            pending_alias=pending.pending_alias,
        )

    def _format_write_ack(self, pending_alias: str, language: Optional[str] = None) -> str:
        from hivememory.i18n.mtp_runtime import get_mtp_error_text
        return get_mtp_error_text("mtp.write.ack", {"pending_alias": pending_alias}, language)

    def _format_update_ack(self, base_alias: str, pending_alias: str, language: Optional[str] = None) -> str:
        from hivememory.i18n.mtp_runtime import get_mtp_error_text
        return get_mtp_error_text("mtp.update.ack", {"base_alias": base_alias, "pending_alias": pending_alias}, language)

    async def _handle_call(
        self,
        command: MTPCommand,
        context: Optional[MTPExecutionContext] = None,
    ) -> MTPResponse:
        """
        处理 CALL 指令 - 触发子代理调用 (Phase 2)

        此方法仅负责验证参数和权限，实际调用由 Kernel 的 FrameScheduler 处理。
        返回特殊的 SUSPEND 状态，通知 Kernel 挂起当前帧。

        Args:
            command: CALL 指令 (target=agent_alias, args: task="...", context_refs=["..."])

        Returns:
            MTPResponse: SUSPEND 状态，携带 CALL 参数

        Raises:
            PermissionDeniedError: 如果子 Agent 尝试调用 CALL (depth >= 1)
        """
        from hivememory.core.mtp.exceptions import PermissionDeniedError
        import json

        # 1. 深度检查 (硬限制)
        if context is not None and context.runtime_scope.depth >= 1:
            raise PermissionDeniedError(message_key="mtp.permission.call_depth_exceeded")

        target_alias = command.target.single_alias
        if not target_alias:
            raise InvalidArgumentError(message_key="mtp.call.missing_single_target")

        task = command.args.get("task", "")
        if not task:
            raise InvalidArgumentError(message_key="mtp.call.missing_task")

        # 4. 解析 context_refs (选填)
        context_refs_str = command.args.get("context_refs", "")
        context_refs = []
        if context_refs_str:
            try:
                # context_refs 已被 MTPParser 序列化为 JSON 字符串
                context_refs = json.loads(context_refs_str)
                if not isinstance(context_refs, list):
                    context_refs = [context_refs]
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse context_refs: {context_refs_str}")
                # 降级为逗号分隔
                context_refs = [ref.strip() for ref in context_refs_str.split(",") if ref.strip()]

        logger.info(
            f"MTP CALL: target={target_alias}, task='{task[:50]}...', "
            f"context_refs={context_refs}"
        )

        # 5. 返回 SUSPEND 信号 
        return MTPResponse(
            status=MTPResponseStatus.SUSPEND,
            content=json.dumps({
                "target_alias": target_alias,
                "task": task,
                "context_refs": context_refs,
            }),
        )

    # ========== 辅助方法 ==========

    async def _record_memory_citation(self, atom: "MemoryAtom", source: str) -> None:
        if self._bus is None:
            return
        try:
            await self._bus.request(
                GlobalRoutes.PATCHOULI_RECORD_MEMORY_CITATION,
                memory_id=atom.id,
                source=source,
            )
        except Exception:
            logger.warning(
                "Failed to record memory citation for memory_id=%s source=%s",
                getattr(atom, "id", None),
                source,
                exc_info=True,
            )

    def _execute_user_tool(
        self, alias: str, code: str, args: Dict[str, str]
    ) -> MTPResponse:
        """
        在受限沙箱中执行用户态工具 (Section 4.2.2)

        用户态工具通过 params 字典接收 MTP 参数。
        工具代码中可通过 params["key"] 访问传入的参数。

        Args:
            alias: 工具别名
            code: 工具代码 (来自 CODE_SNIPPET 记忆原子的 payload.content)
            args: MTP 指令参数 (key-value pairs)

        Returns:
            MTPResponse: 执行结果
        """
        from hivememory.agent_runtime.mtp.syscalls import execute_sandboxed

        result = execute_sandboxed(
            code,
            namespace_extras={"params": dict(args)},
            timeout_seconds=self._config.python_repl_timeout_seconds,
        )

        if not result.ok:
            exc_cls = InvalidArgumentError if result.error_code == "mtp.argument.invalid" else SyscallInternalError
            raise exc_cls(result.content, params={"alias": alias})

        return MTPResponse(status=MTPResponseStatus.SUCCESS, content=result.content)


__all__ = [
    "KoakumaRuntime",
]
