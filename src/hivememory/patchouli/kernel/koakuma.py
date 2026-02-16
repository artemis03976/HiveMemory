"""
小恶魔 (Koakuma) - MTP Runtime Service

定位：MTP 协议的运行时执行器
职责：
    - MTP 指令解析 (委托给 MTPParser)
    - 指令路由与分发 (直接调用兄弟服务 API)
    - 响应格式化与回填
    - 别名解析

架构定位：
    Koakuma 是 PatchouliKernel 管理的第三个微服务，
    负责处理 Worker Agent 生成的 MTP 指令。

    PatchouliKernel
    ├── RetrievalFamiliar (检索使魔 - 只读检索)
    ├── LibrarianCore (馆长本体 - 记忆写入)
    └── Koakuma (小恶魔 - MTP 运行时)

    依赖关系：
    Koakuma 不持有 Kernel 引用，而是直接注入所需的兄弟服务。
    这避免了循环依赖，并遵循最小权限原则。

对应设计文档: MemoryToolProtocol.md Chapter 3 & 4

作者: HiveMemory Team
版本: 1.0
"""

import time
import logging
from uuid import UUID
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from hivememory.patchouli.protocol.mtp import (
    MTP_LEFT_DELIMITER,
    MTP_RIGHT_DELIMITER,
    MTPVerb,
    MTPResponseStatus,
    MTPCommand,
    MTPResponse,
    MTPParser,
    MTPParseError,
    MTPFormatter,
    AliasResolver,
)
from hivememory.patchouli.protocol.models import (
    MTPExecutionResult,
    RetrievalRequest,
)

from hivememory.core.models import MemoryType, Identity
from hivememory.engines.retrieval.models import QueryFilters
from hivememory.engines.generation.models import WriteFocus, UpdateFocus
from hivememory.infrastructure.storage import QdrantMemoryStore

if TYPE_CHECKING:
    from hivememory.patchouli.config import KoakumaConfig
    from hivememory.patchouli.kernel.retrieval_familiar import RetrievalFamiliar
    from hivememory.patchouli.kernel.librarian_core import LibrarianCore

logger = logging.getLogger(__name__)

# MTP filter "type:XXX" 值到 MemoryType 枚举的映射 (大小写不敏感)
_FILTER_TYPE_MAP: Dict[str, MemoryType] = {
    "code": MemoryType.CODE_SNIPPET,
    "code_snippet": MemoryType.CODE_SNIPPET,
    "fact": MemoryType.FACT,
    "url": MemoryType.URL_RESOURCE,
    "url_resource": MemoryType.URL_RESOURCE,
    "reflection": MemoryType.REFLECTION,
    "profile": MemoryType.USER_PROFILE,
    "user_profile": MemoryType.USER_PROFILE,
    "wip": MemoryType.WORK_IN_PROGRESS,
    "work_in_progress": MemoryType.WORK_IN_PROGRESS,
}


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
        retrieval_familiar: "RetrievalFamiliar",
        librarian_core: "LibrarianCore",
        storage: QdrantMemoryStore,
        config: Optional["KoakumaConfig"] = None,
    ):
        """
        初始化 Koakuma MTP 运行时

        Args:
            retrieval_familiar: 检索使魔服务 (用于 SEARCH 指令)
            librarian_core: 馆长本体服务 (用于 WRITE/UPDATE 指令)
            storage: Qdrant 存储实例 (用于 READ 指令按 UUID 读取)
            config: Koakuma 配置 (可选，使用默认值)
        """
        from hivememory.patchouli.config import KoakumaConfig

        self._retrieval = retrieval_familiar
        self._librarian = librarian_core
        self._storage = storage
        
        self._config = config or KoakumaConfig()
        self._parser = MTPParser()
        self._formatter = MTPFormatter()
        self._alias_resolver = AliasResolver()

        # 当前会话的用户 ID (由 Kernel 在会话开始时设置)
        self._current_user_id: str = "default"

        # 初始化内核工具注册表 KERNEL_REGISTRY (Section 4.2.1)
        # 硬编码的 sys_ 工具集，随系统启动加载，Zero Latency
        from hivememory.patchouli.kernel.syscalls import build_kernel_registry
        self._kernel_registry = build_kernel_registry(
            python_repl_timeout=self._config.python_repl_timeout_seconds,
            workspace_path=self._config.workspace_path,
            file_read_max_bytes=self._config.file_read_max_bytes,
            file_write_max_bytes=self._config.file_write_max_bytes,
            web_search_timeout=self._config.web_search_timeout_seconds,
        )

        # TODO: 初始化用户态工具 LRU 缓存 (Section 4.2.2)
        # 缓存从 Qdrant 加载的 CODE_SNIPPET 类型记忆原子
        # self._user_tool_cache = LRUCache(maxsize=self._config.tool_cache_size)

        logger.info("KoakumaRuntime (小恶魔 MTP 运行时) 初始化完成")

    # ========== 公开 API ==========

    def execute_mtp(self, text: str) -> MTPExecutionResult:
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
            response = self._route_and_execute(command)
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
                content=f"Protocol syntax error: {str(e)}",
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

    def intercept_and_execute(
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

        return self.execute_mtp(mtp_fragment)

    def set_current_user(self, user_id: str) -> None:
        """
        设置当前会话的用户 ID

        由 Kernel 在会话开始时调用。

        Args:
            user_id: 用户 ID
        """
        self._current_user_id = user_id

    # ========== 别名管理 ==========

    @property
    def alias_resolver(self) -> AliasResolver:
        """访问别名解析器"""
        return self._alias_resolver

    # ========== 内部路由 ==========

    def _route_and_execute(self, command: MTPCommand) -> MTPResponse:
        """
        路由并执行 MTP 指令 (Section 3)

        根据 VERB 分发到对应的处理器。

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
        }

        handler = handlers.get(command.verb)
        if handler is None:
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content=f"Unknown verb: {command.verb}",
            )

        try:
            return handler(command)
        except Exception as e:
            logger.error(f"MTP 指令执行失败: {command.verb} - {e}", exc_info=True)
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content=f"Execution failed: {str(e)}",
            )

    # ========== Filter 解析 ==========

    def _parse_mtp_filter(self, filter_str: str) -> Optional[QueryFilters]:
        """
        解析 MTP SEARCH 指令的 filter 参数 (Section 2.2)

        语法: key:value 对，多个用空格分隔
        支持的 key:
            - type: 记忆类型 (CODE, FACT, URL, REFLECTION, PROFILE, WIP)
            - tag: 标签 (可多次出现)
            - agent: 来源 Agent ID
            - confidence: 最小置信度 (0.0-1.0)

        安全策略: 宽容解析，静默降级
            - 无法识别的 key/value → 忽略 + log warning
            - 解析后全空 → 返回 None (等同于无 filter)
            - 任何异常 → 返回 None

        Args:
            filter_str: 原始 filter 字符串，如 "type:CODE" 或 "type:FACT tag:python"

        Returns:
            Optional[QueryFilters]: 解析后的过滤条件，全空或异常时返回 None
        """
        if not filter_str or not filter_str.strip():
            return None

        try:
            memory_type = None
            tags: List[str] = []
            source_agent_id = None
            min_confidence = 0.0

            for token in filter_str.strip().split():
                if ":" not in token:
                    logger.warning(f"MTP filter: 忽略无法解析的 token '{token}'")
                    continue

                key, _, value = token.partition(":")
                key = key.strip().lower()
                value = value.strip()

                if not key or not value:
                    logger.warning(f"MTP filter: 忽略空 key 或 value: '{token}'")
                    continue

                if key == "type":
                    mapped = _FILTER_TYPE_MAP.get(value.lower())
                    if mapped is not None:
                        memory_type = mapped
                    else:
                        logger.warning(
                            f"MTP filter: 未知 type 值 '{value}'，已忽略。"
                            f"支持: CODE, FACT, URL, REFLECTION, PROFILE, WIP"
                        )
                elif key == "tag":
                    tags.append(value)
                elif key == "agent":
                    source_agent_id = value
                elif key == "confidence":
                    try:
                        parsed = float(value)
                        if 0.0 < parsed <= 1.0:
                            min_confidence = parsed
                        else:
                            logger.warning(
                                f"MTP filter: confidence 值 {parsed} 超出范围 (0,1]，已忽略"
                            )
                    except ValueError:
                        logger.warning(
                            f"MTP filter: confidence 值 '{value}' 不是有效数字，已忽略"
                        )
                else:
                    logger.warning(f"MTP filter: 未知 key '{key}'，已忽略")

            # 构建 QueryFilters，全空则返回 None
            filters = QueryFilters(
                memory_type=memory_type,
                tags=tags,
                source_agent_id=source_agent_id,
                min_confidence=min_confidence,
            )

            if filters.is_empty():
                return None

            logger.info(f"MTP filter 解析结果: {filters}")
            return filters

        except Exception as e:
            logger.warning(f"MTP filter 解析异常，已降级为无 filter: {e}")
            return None

    # ========== 指令处理器 ==========

    def _handle_search(self, command: MTPCommand) -> MTPResponse:
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
                content='SEARCH requires a "query" argument.',
            )

        # 解析 filter 参数 (Section 2.2)
        # 例如 filter="type:CODE" → QueryFilters(memory_type=CODE_SNIPPET)
        # 宽容解析: 非法 filter 静默降级为 None
        filter_str = command.args.get("filter", "")
        parsed_filters = self._parse_mtp_filter(filter_str) if filter_str else None

        try:
            result = self._retrieval.retrieve(
                request=RetrievalRequest(
                    semantic_query=query,
                    user_id=self._current_user_id,
                    filters=parsed_filters,
                ),
            )

            if result.is_empty():
                return MTPResponse(
                    status=MTPResponseStatus.SUCCESS,
                    content="No memories found. Try a different query.",
                )

            menu = self._render_search_menu(result)

            # 将检索到的记忆注册到别名解析器
            for mem in result.memories:
                alias = self._make_alias_from_memory(mem)
                self._alias_resolver.register_context_alias(
                    alias, str(mem.id)
                )

            return MTPResponse(
                status=MTPResponseStatus.SUCCESS,
                content=menu,
            )

        except Exception as e:
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content=f"Search failed: {str(e)}",
            )

    def _handle_read(self, command: MTPCommand) -> MTPResponse:
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

        # 解析别名 → UUID，分离有效与无效别名
        resolved: List[Tuple[str, str]] = []  # (alias, uuid)
        unresolved: List[str] = []
        for alias in aliases:
            uuid = self._alias_resolver.resolve(alias)
            if uuid is None:
                unresolved.append(alias)
            else:
                resolved.append((alias, uuid))

        # 全部无效：直接返回错误
        if not resolved:
            lines = [
                f"[{a}]: Error - Alias '{a}' not found in context. "
                f"Did you mean to use SEARCH?"
                for a in unresolved
            ]
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content="\n".join(lines),
            )

        # 并行读取 (Section 3.2.1)
        read_results = self._read_memories_concurrent(resolved)

        # 组装输出
        output_lines: List[str] = []
        for alias, uuid in resolved:
            output_lines.append(read_results[(alias, uuid)])
        for alias in unresolved:
            output_lines.append(
                f"[{alias}]: Error - Alias '{alias}' not found in context. "
                f"Did you mean to use SEARCH?"
            )

        return MTPResponse(
            status=MTPResponseStatus.SUCCESS,
            content="\n".join(output_lines),
        )
# PLACEHOLDER_RUN_WRITE_UPDATE

    def _handle_run(self, command: MTPCommand) -> MTPResponse:
        """
        处理 RUN 指令 (Section 2.2)

        两层分发:
        - Level 0: 内核工具快速路径 (KERNEL_REGISTRY, Section 4.2.1)
        - Level 1: 用户态工具慢速路径 (Section 4.2.2, 暂未实现)

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
        if syscall is not None:
            try:
                result = syscall.handler(command.args)
                return MTPResponse(
                    status=MTPResponseStatus.SUCCESS,
                    content=result,
                )
            except Exception as e:
                logger.error(
                    f"Kernel syscall '{alias}' failed: {e}", exc_info=True
                )
                return MTPResponse(
                    status=MTPResponseStatus.ERROR,
                    content=f"Tool '{alias}' execution failed: {str(e)}",
                )

        # Level 1: 用户态工具慢速路径 (Section 4.2.2) — 暂未实现
        # TODO: 检查 USER_TOOL_CACHE (LRU)
        # TODO: Cache Miss → Qdrant 检索 CODE_SNIPPET → 沙箱执行

        return MTPResponse(
            status=MTPResponseStatus.ERROR,
            content=f"Tool alias '{alias}' not found. "
                    f"Did you forget to SEARCH first?",
        )

    def _handle_write(self, command: MTPCommand) -> MTPResponse:
        """
        处理 WRITE 指令 (Section 2.2 + 附录B)

        将 WRITE 内容打包为 WriteFocus，发送给 LibrarianCore 处理。
        Koakuma 不直接操作 DB，由 LibrarianCore 负责 flush + Generation Engine (Mode B)。

        Type B 动作类响应 (Section 3.3.3)

        Args:
            command: WRITE 指令 (target=*, args: content=`...`, reason="...", title="...")

        Returns:
            MTPResponse: ACK 确认 (含生成的 memory_ids)
        """
        content = command.args.get("content", "")
        if not content:
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content='WRITE requires a "content" argument.',
            )

        reason = command.args.get("reason", "")
        title = command.args.get("title", "")

        # 构建 WriteFocus 并发送给 LibrarianCore
        write_focus = WriteFocus(
            content=content,
            reason=reason or None,
            title=title or None,
            identity=Identity(user_id=self._current_user_id),
        )

        logger.info(f"MTP WRITE 信号: content='{content[:50]}...', reason='{reason}'")

        try:
            atoms = self._librarian.handle_write_signal(write_focus)
            memory_ids = [str(a.id) for a in atoms]

            return MTPResponse(
                status=MTPResponseStatus.ACK,
                content=f'Memory saved. {len(atoms)} atom(s) created.',
                data={"memory_ids": memory_ids},
            )
        except Exception as e:
            logger.error(f"WRITE 处理失败: {e}", exc_info=True)
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content=f'WRITE failed: {str(e)}',
            )

    def _handle_update(self, command: MTPCommand) -> MTPResponse:
        """
        处理 UPDATE 指令 (附录 C)

        流程:
            1. 校验 alias (必须存在)
            2. 校验 instruction (必填)
            3. 解析 alias → UUID
            4. 构建 UpdateFocus
            5. 调用 LibrarianCore.handle_update_signal()

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

        # 3. 解析 alias → UUID
        uuid = self._alias_resolver.resolve(alias)
        if uuid is None:
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content=f"Alias '{alias}' not found in context. "
                        f"Did you mean to use SEARCH?",
            )

        # 4. 获取可选的 content
        content = command.args.get("content", None)

        # 5. 构建 UpdateFocus
        update_focus = UpdateFocus(
            instruction=instruction,
            content=content if content else None,
            target_uuid=uuid,
            target_alias=alias,
            identity=Identity(user_id=self._current_user_id),
        )

        # 6. 调用 LibrarianCore
        try:
            atoms = self._librarian.handle_update_signal(update_focus)
            memory_ids = [str(a.id) for a in atoms]

            logger.info(
                f"MTP UPDATE 完成: alias='{alias}', "
                f"生成 {len(atoms)} 条记忆"
            )

            return MTPResponse(
                status=MTPResponseStatus.ACK,
                content=f"Memory '{alias}' updated successfully.",
                data={"memory_ids": memory_ids},
            )
        except Exception as e:
            logger.error(f"UPDATE 处理失败: {e}", exc_info=True)
            return MTPResponse(
                status=MTPResponseStatus.ERROR,
                content=f'UPDATE failed: {str(e)}',
            )

    # ========== 辅助方法 ==========

    def _read_memories_concurrent(
        self, resolved: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], str]:
        """
        并行读取多个记忆原子 (Section 3.2.1)

        使用 ThreadPoolExecutor 并发执行 storage.get_memory()，
        适用于同步阻塞 I/O (Qdrant HTTP)。

        单个别名时退化为顺序读取，避免线程池开销。

        Args:
            resolved: [(alias, uuid), ...] 已解析的别名-UUID 对

        Returns:
            {(alias, uuid): formatted_line} 结果映射
        """
        if len(resolved) == 1:
            alias, uuid = resolved[0]
            return {(alias, uuid): self._read_single_memory(alias, uuid)}

        results: Dict[Tuple[str, str], str] = {}
        with ThreadPoolExecutor(max_workers=min(len(resolved), 4)) as executor:
            future_to_key = {
                executor.submit(self._read_single_memory, alias, uuid): (alias, uuid)
                for alias, uuid in resolved
            }
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    alias, uuid = key
                    results[key] = (
                        f"[{alias}]: Error - Failed to read UUID {uuid}: {e}"
                    )
        return results

    def _read_single_memory(self, alias: str, uuid: str) -> str:
        """
        读取单个记忆原子的 Payload 内容

        Args:
            alias: 语义化别名
            uuid: 记忆原子 UUID

        Returns:
            str: 格式化的内容行
        """
        try:
            memory = self._storage.get_memory(UUID(uuid))
        except (ValueError, TypeError) as e:
            return f"[{alias}]: Error - Invalid UUID '{uuid}': {e}"
        except Exception as e:
            logger.error(f"Storage read failed for {alias} (UUID: {uuid}): {e}")
            return f"[{alias}]: Error - Storage read failed: {e}"

        if memory is None:
            return (
                f"[{alias}]: Error - Memory not found for UUID {uuid}. "
                f"It may have been archived or deleted."
            )

        return f"[{alias}]:\n{memory.payload.content}"

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
            alias = self._make_alias_from_memory(mem)
            summary = mem.index.summary[:80] if mem.index.summary else "(no summary)"
            lines.append(f"{i}. {alias} (Alias) - \"{summary}\"")
        return "\n".join(lines)

    @staticmethod
    def _make_alias_from_memory(mem) -> str:
        """
        从 MemoryAtom 获取或生成别名

        优先使用 IndexLayer 中存储的正式别名 (由 Generation Engine 在记忆创建时生成)。
        如果不存在，则基于 memory_type 和 title 生成临时别名作为 fallback。

        Args:
            mem: MemoryAtom 对象

        Returns:
            str: 语义化别名
        """
        # 优先使用存储的正式别名 (Section 2.3)
        if getattr(mem.index, 'alias', None):
            return mem.index.alias

        # Fallback: 运行时临时生成 (向后兼容旧记忆)
        type_prefix = mem.index.memory_type.value.lower().split("_")[0]
        title = mem.index.title or "untitled"
        alias = title.lower().replace(" ", "_").replace("-", "_")
        alias = "".join(c for c in alias if c.isalnum() or c == "_")
        alias = alias[:40]
        return f"{type_prefix}_{alias}"

    def _get_current_user_id(self) -> str:
        """获取当前用户 ID"""
        return self._current_user_id


__all__ = [
    "KoakumaRuntime",
]
