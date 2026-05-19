"""
小恶魔 (Koakuma) - MTP Runtime Service

定位：MTP 协议的运行时执行器
职责：
    - MTP 指令解析 (委托给 MTPParser)
    - 指令路由与分发 (直接调用兄弟服务 API)
    - 响应格式化与回填
    - 别名解析

架构定位：
    Koakuma 是 PatchouliRuntime 管理的第三个微服务，
    负责处理 Worker Agent 生成的 MTP 指令。

    PatchouliRuntime
    ├── RetrievalFamiliar (检索使魔 - 只读检索)
    ├── LibrarianCore (馆长本体 - 记忆写入)
    └── Koakuma (小恶魔 - MTP 运行时)

对应设计文档: MemoryToolProtocol.md Chapter 3 & 4

作者: HiveMemory Team
版本: 1.0
"""

import time
import logging
from uuid import UUID
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
from hivememory.alice.runtime.cache import KoakumaAtomCache
from hivememory.core.protocol.models import (
    MTPExecutionResult,
    RetrievalRequest,
)

from hivememory.core.models import MemoryType, Identity
from hivememory.core.mtp.exceptions import (
    AgentFault,
    SystemFault,
    StorageOfflineError,
    StorageReadError,
    BusRouteUnavailableError,
    PermissionDeniedError,
)
from hivememory.engines.generation.models import WriteFocus, UpdateFocus
from hivememory.core.models import TraceItem

if TYPE_CHECKING:
    from hivememory.alice.runtime.bus import AliceBus
    from hivememory.system.config import KoakumaConfig

logger = logging.getLogger(__name__)


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
        bus: Optional["AliceBus"] = None,
        config: Optional["KoakumaConfig"] = None,
    ):
        """
        初始化 Koakuma MTP 运行时

        Args:
            bus: AliceBus 实例，用于跨服务通信（纯异步总线）
            config: Koakuma 配置 (可选，使用默认值)
        """
        from hivememory.system.config import KoakumaConfig

        self._bus = bus

        self._config = config or KoakumaConfig()
        self._parser = MTPParser()
        self._filter_parser = MTPFilterParser()
        self._formatter = MTPFormatter()
        self._atom_cache = KoakumaAtomCache()

        # 当前会话的身份标识 (由 Kernel 在会话开始时设置)
        self._current_identity: Identity = Identity()

        # 初始化内核工具注册表 KERNEL_REGISTRY (Section 4.2.1)
        # 硬编码的 sys_ 工具集，随系统启动加载，Zero Latency
        from hivememory.alice.runtime.syscalls import build_kernel_registry
        self._kernel_registry = build_kernel_registry(
            python_repl_timeout=self._config.python_repl_timeout_seconds,
            workspace_path=self._config.workspace_path,
            file_read_max_bytes=self._config.file_read_max_bytes,
            file_write_max_bytes=self._config.file_write_max_bytes,
            web_search_timeout=self._config.web_search_timeout_seconds,
        )

        logger.info("KoakumaRuntime (小恶魔 MTP 运行时) 初始化完成")

        # ========== 交互状态 (v3.0 延迟捕获) ==========
        self._current_traces: List[TraceItem] = []
        self._current_write_focus: Optional[WriteFocus] = None
        self._current_update_focus: Optional[UpdateFocus] = None

        # ========== 多智能体权限沙箱 (Phase 1) ==========
        # 由 PatchouliSystem 在每轮 chat/stream 开始时通过 set_active_profile() 设置
        self._active_profile: Optional[Any] = None  # AgentProfile

    def set_current_identity(self, identity: Identity) -> None:
        """
        设置当前会话的完整身份标识

        由 Kernel 在会话开始时调用，支持 user_id + agent_id + team_id。
        新会话时清空缓存。

        Args:
            identity: 完整身份标识
        """
        if identity.user_id != self._current_identity.user_id:
            self._atom_cache.clear()
            logger.info(f"New session started for identity {identity.buffer_key}, cache cleared")
        self._current_identity = identity

    def _get_current_identity(self) -> Identity:
        """获取当前身份标识"""
        return self._current_identity

    # ========== 交互状态管理 (v3.0) ==========

    def reset_interaction_state(self) -> None:
        """每轮递归循环前重置交互状态"""
        self._current_traces = []
        self._current_write_focus = None
        self._current_update_focus = None

    # ========== 多智能体权限沙箱 (Phase 1) ==========

    def set_active_profile(self, profile) -> None:
        """
        设置当前活跃的人偶图纸配置

        由 PatchouliSystem 在每轮 chat/stream 开始时调用。
        设置后，所有 MTP 指令执行前都会进行权限校验。

        Args:
            profile: AgentProfile 实例（或 None 表示无限制）
        """
        self._active_profile = profile

    def set_current_depth(self, depth: int) -> None:
        """
        设置当前执行深度 (Phase 2: 用于硬限制检查)

        由 PatchouliSystem 在执行帧时调用。
        子 Agent (depth >= 1) 被禁止调用 CALL 指令。

        Args:
            depth: 调用栈深度 (主 Agent = 0, 子 Agent = 1)
        """
        self._current_depth = depth

    def get_current_depth(self) -> int:
        """
        获取当前执行深度

        Returns:
            int: 调用栈深度 (0 = 主 Agent, 1 = 子 Agent)
        """
        return getattr(self, '_current_depth', 0)

    def _check_verb_permission(self, verb: str) -> None:
        """
        校验 MTP 动词权限 (O(1) set lookup)

        Args:
            verb: MTP 动词 (如 "WRITE", "RUN")

        Raises:
            PermissionDeniedError: 当前人偶无权执行此动词
        """
        if self._active_profile is None:
            return
        if not self._active_profile.is_verb_allowed(verb):
            raise PermissionDeniedError(
                f"You do not have permission to use the '{verb}' command."
            )

    def _check_tool_permission(self, tool_alias: str) -> None:
        """
        校验系统工具权限 (O(1) set lookup)

        Args:
            tool_alias: 工具别名 (如 "sys_write_file")

        Raises:
            PermissionDeniedError: 当前人偶无权使用此工具
        """
        if self._active_profile is None:
            return
        if not self._active_profile.is_tool_allowed(tool_alias):
            raise PermissionDeniedError(
                f"You do not have access to tool '{tool_alias}'."
            )

    def get_interaction_traces(self) -> List[TraceItem]:
        """获取当前轮次记录的 TraceItem 列表"""
        return self._current_traces.copy()

    def get_write_focus(self) -> Optional[WriteFocus]:
        """获取延迟捕获的 WriteFocus (如果有)"""
        return self._current_write_focus

    def get_update_focus(self) -> Optional[UpdateFocus]:
        """获取延迟捕获的 UpdateFocus (如果有)"""
        return self._current_update_focus

    def get_last_generated_alias(self) -> Optional[str]:
        """
        获取最后一次 WRITE/UPDATE 生成的别名 (Phase 2: 用于自动收割)

        从延迟捕获的 WriteFocus/UpdateFocus 中提取别名。
        WRITE 的别名需要等待 Librarian 生成后才能获取，
        UPDATE 的别名可以直接从 UpdateFocus 中获取。

        Returns:
            str: 别名，如果没有则返回 None

        Note:
            由于 WRITE 使用延迟捕获，别名在 InteractionPayload 提交后才会生成。
            因此，此方法主要用于 UPDATE 的别名收割。
            对于 WRITE，需要在 Librarian 处理后从响应中提取别名。
        """
        # UPDATE 的别名可以直接获取
        if self._current_update_focus:
            return self._current_update_focus.target_alias

        # WRITE 的别名需要等待 Librarian 生成
        # 这里返回 None，实际别名需要从 Librarian 的响应中提取
        return None

    # ========== 公开 API ==========

    async def execute_mtp(self, text: str) -> MTPExecutionResult:
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
            response = await self._route_and_execute(command)
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
            )

        except MTPParseError as e:
            elapsed = (time.time() - start_time) * 1000
            error_response = MTPResponse(
                status=MTPResponseStatus.ERROR,
                content=e.to_agent_prompt(),
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
        self, assistant_text: str
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

        return await self.execute_mtp(mtp_fragment)

    # ========== 别名管理 ==========

    @property
    def atom_cache(self) -> KoakumaAtomCache:
        """访问统一原子缓存"""
        return self._atom_cache

    # ========== 内部路由 ==========

    async def _route_and_execute(self, command: MTPCommand) -> MTPResponse:
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
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content=(
                    "[Syntax Error] Unknown verb: "
                    f"{command.verb}. Valid verbs: SEARCH, READ, RUN, WRITE, UPDATE."
                ),
            )

        try:
            # 权限沙箱：校验 MTP 动词权限 (Phase 1 多智能体)
            self._check_verb_permission(command.verb.value)
            return await handler(command)

        except StorageOfflineError as e:
            logger.warning(f"Storage offline during {command.verb}: {e}")
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content=e.to_agent_prompt(),
            )

        except StorageReadError as e:
            logger.error(f"Storage error during {command.verb}: {e}")
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content=e.to_agent_prompt(),
            )

        except AgentFault as e:
            logger.info(f"Agent fault during {command.verb}: {e}")
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content=e.to_agent_prompt(),
            )

        except SystemFault as e:
            logger.error(f"System fault during {command.verb}: {e}", exc_info=True)
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content=e.to_agent_prompt(),
            )

        except Exception as e:
            logger.error(f"Unexpected error during {command.verb}: {e}", exc_info=True)
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content=(
                    "[Internal Error] An unexpected error occurred. "
                    "Do NOT retry this command. Continue the conversation normally."
                ),
            )

    # ========== 指令处理器 ==========

    async def _handle_search(self, command: MTPCommand) -> MTPResponse:
        """
        处理 SEARCH 指令 (Section 2.2)

        调用 RetrievalFamiliar 进行模糊检索，返回 Index 菜单。
        不返回具体内容，仅返回别名和摘要列表。

        Type A 数据类响应 (Section 3.3.3)

        Args:
            command: SEARCH 指令 (target=*, args: query="...", filter="...")

        Returns:
            MTPResponse: Index 菜单
        """
        query = command.args.get("query", "")
        if not query:
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content='[Invalid Argument] SEARCH requires a "query" argument.\n'
                        'Action: Provide a query argument and retry.',
            )

        # 解析 filter 参数 (Section 2.2)
        # 例如 filter="type:CODE" → QueryFilters(memory_type=CODE_SNIPPET)
        # 宽容解析: 非法 filter 降级为 None，但返回警告
        filter_str = command.args.get("filter", "")
        parsed_filters, filter_warnings = self._filter_parser.parse(filter_str) if filter_str else (None, [])

        # Let StorageOfflineError / StorageReadError propagate to _route_and_execute
        result = await self._bus.request(
            "memory.retrieve",
            request=RetrievalRequest(
                semantic_query=query,
                identity=self._current_identity,
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

        menu = self._render_search_menu(result)
        if filter_warnings:
            menu += "\n" + "\n".join(filter_warnings)

        # 将检索到的记忆原子缓存（完整对象，而非仅 UUID）
        self._atom_cache.ingest_atoms(result.memories)

        # 记录 TraceItem
        self._current_traces.append(TraceItem(
            action="SEARCH", query=query,
        ))

        return MTPResponse(
            status=MTPResponseStatus.SUCCESS,
            content=menu,
        )

    async def _handle_read(self, command: MTPCommand) -> MTPResponse:
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
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content="READ does not support wildcard target '*'. "
                        "Use SEARCH instead.",
            )

        aliases = command.target.aliases
        if not aliases:
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content="READ requires at least one target alias.",
            )

        # 解析别名 → MemoryAtom，分离有效与无效别名
        # 统一缓存路径: 缓存命中 → L2 冷检索回退
        # StorageOfflineError / BusRouteUnavailableError 会直接传播到 _route_and_execute
        resolved: List[Tuple[str, "MemoryAtom"]] = []  # (alias, atom)
        unresolved: List[str] = []
        for alias in aliases:
            atom = await self._resolve_and_fetch(alias)
            if atom is None:
                unresolved.append(alias)
            else:
                resolved.append((alias, atom))

        # 全部无效：直接返回错误
        if not resolved:
            lines = [
                f"[{a}]: [Alias Not Found] Alias '{a}' not found. "
                f"Use SEARCH to discover the correct alias first."
                for a in unresolved
            ]
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content="\n".join(lines),
            )

        # 直接从缓存的原子中提取内容（无需查询数据库）
        read_results = self._format_cached_atoms(resolved)

        # 组装输出
        output_lines: List[str] = []
        for alias, _ in resolved:
            output_lines.append(read_results[alias])
        for alias in unresolved:
            output_lines.append(
                f"[{alias}]: [Alias Not Found] Alias '{alias}' not found. "
                f"Use SEARCH to discover the correct alias first."
            )

        # 记录 TraceItem (折叠: 仅记录查阅动作和目标)
        for alias, _ in resolved:
            self._current_traces.append(TraceItem(
                action="READ", target=alias,
            ))

        return MTPResponse(
            status=MTPResponseStatus.SUCCESS,
            content="\n".join(output_lines),
        )

    async def _handle_run(self, command: MTPCommand) -> MTPResponse:
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
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content="RUN requires a single tool alias as target.",
            )

        # Level 0: 内核工具快速路径 (Section 4.2.1)
        syscall = self._kernel_registry.get(alias)
        if syscall is None and alias.startswith("sys_"):
            # sys_ 前缀保留给内核工具，不走 L1/L2 用户态解析
            self._current_traces.append(TraceItem(
                action="RUN", tool=alias, status="error",
            ))
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content=f"[Alias Not Found] Kernel tool '{alias}' not found. "
                        f"Use SEARCH to discover available tools.",
            )
        if syscall is not None:
            # 权限沙箱：校验系统工具权限 (Phase 1 多智能体)
            self._check_tool_permission(alias)
            try:
                result = syscall.handler(command.args)
                # 记录 TraceItem (摘要: 记录副作用操作及状态)
                self._current_traces.append(TraceItem(
                    action="RUN", tool=alias, status="success",
                ))
                return MTPResponse(
                    status=MTPResponseStatus.SUCCESS,
                    content=result,
                )
            except Exception as e:
                logger.error(
                    f"Kernel syscall '{alias}' failed: {e}", exc_info=True
                )
                self._current_traces.append(TraceItem(
                    action="RUN", tool=alias, status="error",
                ))
                return MTPResponse(
                    status=MTPResponseStatus.ERROR,
                    content=f"[Tool Error] Tool '{alias}' execution failed. "
                            f"Do NOT retry with the same input.",
                )

        # Level 1: 用户态工具路径 (统一原子缓存)
        # StorageOfflineError / BusRouteUnavailableError 会直接传播到 _route_and_execute
        atom = await self._resolve_and_fetch(alias)
        if atom is None:
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content=f"[Alias Not Found] Tool alias '{alias}' not found. "
                        f"Use SEARCH to discover the correct alias first.",
            )

        # 校验类型必须是 CODE_SNIPPET
        if atom.index.memory_type != MemoryType.CODE_SNIPPET:
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content=f"[Type Mismatch] Alias '{alias}' is not a runnable tool "
                        f"(type: {atom.index.memory_type.value}). "
                        f"RUN only supports CODE_SNIPPET memories.",
            )

        # 使用缓存的代码执行
        code = atom.payload.content
        logger.info(f"User tool executing: alias='{alias}', UUID={atom.id}")
        return self._execute_user_tool(alias, code, command.args)

    async def _handle_write(self, command: MTPCommand) -> MTPResponse:
        """
        处理 WRITE 指令 (Section 2.2 + 附录B)

        v3.0 延迟捕获模式:
        将 WRITE 内容打包为 WriteFocus 并暂存到 _current_write_focus，
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
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content='WRITE requires a "content" argument.',
            )

        reason = command.args.get("reason", "")
        title = command.args.get("title", "")

        # 构建 WriteFocus 并延迟捕获 (不再直接调用 Librarian)
        write_focus = WriteFocus(
            content=content,
            reason=reason or None,
            title=title or None,
            identity=self._current_identity,
        )

        self._current_write_focus = write_focus

        logger.info(
            f"MTP WRITE 延迟捕获: content='{content[:50]}...', reason='{reason}'"
        )

        return MTPResponse(
            status=MTPResponseStatus.ACK,
            content='Memory saved.',
        )

    async def _handle_update(self, command: MTPCommand) -> MTPResponse:
        """
        处理 UPDATE 指令 (附录 C)

        v3.0 延迟捕获模式:
        将 UPDATE 意图打包为 UpdateFocus 并暂存到 _current_update_focus，
        实际记忆更新延迟到 InteractionPayload 提交时执行。
        ACK 响应文案保持不变，对 Agent 完全透明。

        Type B 动作类响应 (Section 3.3.3)

        Args:
            command: UPDATE 指令 (target=alias, args: instruction="...", content=`...`)

        Returns:
            MTPResponse: ACK 确认或 ERROR
        """
        # 1. 校验 alias
        alias = command.target.single_alias
        if alias is None:
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content="UPDATE requires a single alias as target.",
            )

        # 2. 校验 instruction (必填)
        instruction = command.args.get("instruction", "")
        if not instruction:
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content='UPDATE requires an "instruction" argument.',
            )

        # 3. 解析 alias → MemoryAtom (统一缓存路径)
        # StorageOfflineError / BusRouteUnavailableError 会直接传播到 _route_and_execute
        atom = await self._resolve_and_fetch(alias)
        if atom is None:
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content=f"[Alias Not Found] Alias '{alias}' not found. "
                        f"Use SEARCH to discover the correct alias first.",
            )
        uuid = str(atom.id)

        # 4. 获取可选的 content
        content = command.args.get("content", None)

        # 5. 构建 UpdateFocus 并延迟捕获 (不再直接调用 Librarian)
        update_focus = UpdateFocus(
            instruction=instruction,
            content=content if content else None,
            target_uuid=uuid,
            target_alias=alias,
            identity=self._current_identity,
        )

        self._current_update_focus = update_focus

        # 6. 使缓存失效，防止脏读
        self._atom_cache.invalidate_alias(alias)

        logger.info(
            f"MTP UPDATE 延迟捕获: alias='{alias}', instruction='{instruction[:50]}'"
        )

        return MTPResponse(
            status=MTPResponseStatus.ACK,
            content=f"Memory '{alias}' updated successfully.",
        )

    async def _handle_call(self, command: MTPCommand) -> MTPResponse:
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
        if hasattr(self, '_current_depth') and self._current_depth >= 1:
            raise PermissionDeniedError(
                "Sub-agents are not allowed to invoke CALL. "
                "Only the main agent can call sub-agents."
            )

        # 2. 验证 target (必须是单个别名)
        target_alias = command.target.single_alias
        if not target_alias:
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content="CALL requires a single agent alias as target. "
                        "Example: ⟪ CALL | coder_doll | task=\"...\" ⟫",
            )

        # 3. 验证 task (必填)
        task = command.args.get("task", "")
        if not task:
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content='CALL requires a "task" argument. '
                        'Example: ⟪ CALL | coder_doll | task="Write unit tests" ⟫',
            )

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

        # 5. 返回 SUSPEND 信号 (由 Kernel 拦截处理)
        return MTPResponse(
            status=MTPResponseStatus.SUSPEND,
            content=json.dumps({
                "target_alias": target_alias,
                "task": task,
                "context_refs": context_refs,
            }),
        )

    # ========== 辅助方法 ==========

    def _format_cached_atoms(
        self, resolved: List[Tuple[str, "MemoryAtom"]]
    ) -> Dict[str, str]:
        """
        格式化已缓存的记忆原子内容

        直接使用缓存的原子，无需查询数据库。

        Args:
            resolved: [(alias, atom), ...] 已解析的别名-原子对

        Returns:
            {alias: formatted_content} 结果映射
        """
        results: Dict[str, str] = {}
        for alias, atom in resolved:
            results[alias] = f"[{alias}]:\n{atom.payload.content}"
        return results

    def _render_search_menu(self, result) -> str:
        """
        将检索结果渲染为 Index 菜单格式 (Section 2.4 场景5)

        格式:
        [Menu]:
        1. alias (Alias) - "summary"
        2. alias (Alias) - "summary"

        Args:
            result: RetrievalResponse 对象

        Returns:
            str: 菜单格式文本
        """
        lines = ["[Menu]:"]
        for i, mem in enumerate(result.memories, 1):
            alias = mem.get_alias()
            summary = mem.index.summary[:80] if mem.index.summary else "(no summary)"
            lines.append(f"{i}. {alias} (Alias) - \"{summary}\"")
        return "\n".join(lines)

    async def _resolve_and_fetch(self, alias: str) -> Optional["MemoryAtom"]:
        """
        统一的别名解析与原子获取

        先检查缓存，未命中则查询存储并缓存结果。
        替代原有的 L1/L2 分离解析模式。

        Raises:
            StorageOfflineError: 存储层离线
            StorageReadError: 存储层响应异常
            BusRouteUnavailableError: 系统总线路由缺失

        Args:
            alias: 语义化别名

        Returns:
            MemoryAtom 对象，未命中返回 None
        """
        # 检查缓存
        atom = self._atom_cache.get_atom_by_alias(alias)
        if atom is not None:
            logger.debug(f"Atom cache hit: alias='{alias}'")
            return atom

        # 缓存未命中：查询存储（L2 冷检索）
        try:
            memory = await self._bus.request(
                "memory.get_memory_by_alias",
                alias=alias, user_id=self._current_identity.user_id,
            )
            if memory is None:
                logger.debug(f"L2 cold-lookup miss: alias='{alias}'")
                return None

            uuid_str = str(memory.id)
            # 校验 UUID 格式有效性，防止脏数据污染缓存
            UUID(uuid_str)

            # 缓存完整原子
            self._atom_cache.ingest_atom(memory)
            logger.debug(
                f"L2 cold-lookup hit: alias='{alias}' -> {uuid_str}, cached"
            )
            return memory
        except KeyError as e:
            logger.error(f"L2 cold-lookup route unavailable: alias='{alias}', error={e}")
            raise BusRouteUnavailableError(
                "Memory storage service is not available."
            ) from e
        except (StorageOfflineError, StorageReadError):
            raise  # propagate as-is
        except Exception as e:
            logger.error(f"L2 cold-lookup infrastructure failure: alias='{alias}', error={e}")
            raise StorageReadError(
                "Memory storage encountered an error during alias lookup."
            ) from e

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
        from hivememory.alice.runtime.syscalls import execute_sandboxed

        result = execute_sandboxed(
            code,
            namespace_extras={"params": dict(args)},
            timeout_seconds=self._config.python_repl_timeout_seconds,
        )

        is_error = result.startswith("Error")
        status = "error" if is_error else "success"

        self._current_traces.append(TraceItem(
            action="RUN", tool=alias, status=status,
        ))

        return MTPResponse(
            status=MTPResponseStatus.ERROR if is_error else MTPResponseStatus.SUCCESS,
            content=result,
        )


__all__ = [
    "KoakumaRuntime",
]
