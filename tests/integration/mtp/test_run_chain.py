"""
RUN 指令执行链路测试

验证 MTP RUN 指令从 Koakuma 的 RUN 分派 的完整链路。

测试覆盖:
    1. Target 校验 (通配符/列表/空 target 拒绝)
    2. 系统工具快速路径 (sys_clock, sys_python_repl)
    3. 记忆别名 (经能力层解析后一律以“不可运行”拒绝、不记录引用)
"""

import asyncio
from uuid import uuid4

import pytest

from hivememory.agent_runtime.frame import FrameExecutionPolicy
from hivememory.config.alice import KoakumaConfig
from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    PendingAtomResolution,
    PendingAtomSettlement,
    WriteFocus,
)
from hivememory.core.mtp import MTP_LEFT_DELIMITER, MTP_RIGHT_DELIMITER
from tests.helpers.memory import make_memory_metadata
from tests.helpers.operations import HarnessKoakumaRuntime
from tests.helpers.workspace import make_identity_scope, make_runtime_scope, make_workspace_identity

MAIN = make_workspace_identity()

# ========== 辅助函数 ==========


def _make_code_memory(
    code: str = "print('hello')",
    alias: str = "tool_test",
    mem_id=None,
) -> MemoryAtom:
    """正文是一段可以直接执行的代码的记忆原子。"""
    return MemoryAtom(
        id=mem_id or uuid4(),
        meta=make_memory_metadata(user_id="test_user", source_agent_id="test_agent"),
        index=IndexLayer(
            title="Test Tool",
            summary="A test tool script",
            tags=["test", "tool"],
            memory_type=MemoryType.FACT,
            alias=alias,
        ),
        payload=PayloadLayer(content=code),
    )


# ========== Fixtures ==========


@pytest.fixture
def koakuma() -> HarnessKoakumaRuntime:
    from .conftest import make_koakuma_runtime, make_mock_bus

    bus = make_mock_bus()
    return make_koakuma_runtime(bus, KoakumaConfig())


def _execute_mtp(koakuma: HarnessKoakumaRuntime, text: str, context=None):
    return asyncio.run(
        koakuma.execute_mtp(
            text,
            context=context
            or asyncio.run(
                koakuma.make_context(
                    execution_policy=FrameExecutionPolicy.from_verbs(
                        {"SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"}
                    ),
                    runtime_scope=make_runtime_scope(),
                )
            ),
        )
    )


def _intercept_and_execute(koakuma: HarnessKoakumaRuntime, assistant_text: str, context=None):
    return asyncio.run(
        koakuma.intercept_and_execute(
            assistant_text,
            context=context
            or asyncio.run(
                koakuma.make_context(
                    execution_policy=FrameExecutionPolicy.from_verbs(
                        {"SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"}
                    ),
                    runtime_scope=make_runtime_scope(),
                )
            ),
        )
    )


# ========== Test 1：Target 校验 ==========


class TestRunTargetValidation:
    """RUN 指令 target 校验"""

    def test_wildcard_rejected(self, koakuma):
        result = _execute_mtp(koakuma, "⟪ RUN | * | ⟫")
        assert not result.success
        assert result.response_content == ""
        assert "Invalid Argument" in result.formatted_response

    def test_list_target_rejected(self, koakuma):
        """列表 target 不支持 (single_alias 返回 None)"""
        result = _execute_mtp(koakuma, "⟪ RUN | [tool_a, tool_b] | ⟫")
        assert not result.success

    def test_empty_target(self, koakuma):
        result = _execute_mtp(koakuma, "⟪ RUN | | ⟫")
        assert not result.success


# ========== Test 2：系统工具快速路径 ==========


class TestRunSyscallFastPath:
    """系统工具快速路径"""

    def test_sys_clock_default(self, koakuma):
        result = _execute_mtp(koakuma, "⟪ RUN | sys_clock | ⟫")
        assert result.success
        assert "UTC" in result.response_content
        assert koakuma.harness.backing.bus._memory_citations == []

    def test_sys_clock_iso(self, koakuma):
        result = _execute_mtp(koakuma, '⟪ RUN | sys_clock | format="iso" ⟫')
        assert result.success
        assert "T" in result.response_content

    def test_sys_clock_date(self, koakuma):
        result = _execute_mtp(koakuma, '⟪ RUN | sys_clock | format="date" ⟫')
        assert result.success
        # YYYY-MM-DD 格式
        assert "-" in result.response_content
        assert len(result.response_content.strip()) == 10 or "20" in result.response_content

    def test_sys_python_repl_calculation(self, koakuma):
        result = _execute_mtp(koakuma, '⟪ RUN | sys_python_repl | code="print(6 * 7)" ⟫')
        assert result.success
        assert "42" in result.response_content

    def test_sys_python_repl_import_blocked(self, koakuma):
        result = _execute_mtp(koakuma, '⟪ RUN | sys_python_repl | code="import os" ⟫')
        assert not result.success  # import 被拦截，syscall 失败升级为 SyscallInternalError
        assert result.response_content == ""
        assert "import" in result.formatted_response.lower()

    def test_sys_python_repl_multiline(self, koakuma):
        # 使用反引号语法支持多行代码
        result = _execute_mtp(
            koakuma, "⟪ RUN | sys_python_repl | code=`x = 10\ny = 20\nprint(x + y)` ⟫"
        )
        assert result.success
        assert "30" in result.response_content


# ========== Test 3：记忆别名 ==========


class TestRunMemoryAlias:
    """记忆别名经能力层解析，解析到的记忆都不可运行"""

    def test_unknown_tool_not_found(self, koakuma):
        """L1+L2 均未命中，返回 not found"""
        koakuma.harness.backing.bus._mock_storage.get_memory_by_alias.return_value = None
        result = _execute_mtp(koakuma, "⟪ RUN | nonexistent_tool | ⟫")
        assert not result.success
        assert result.response_content == ""
        assert (
            "Alias Not Found" in result.formatted_response or "未找到" in result.formatted_response
        )

    def test_unknown_tool_suggests_search(self, koakuma):
        koakuma.harness.backing.bus._mock_storage.get_memory_by_alias.return_value = None
        result = _execute_mtp(koakuma, "⟪ RUN | my_custom_tool | ⟫")
        assert not result.success
        assert result.response_content == ""
        assert "SEARCH" in result.formatted_response

    def test_l2_route_failure_returns_infra_error(self, koakuma):
        koakuma.harness.backing.bus._mock_storage.get_memory_by_alias.side_effect = KeyError(
            "AsyncSystemBus: route 'memory.retrieve_by_aliases' not registered"
        )

        result = _execute_mtp(koakuma, "⟪ RUN | my_custom_tool | ⟫")

        assert not result.success
        assert result.response_content == ""
        assert "Service Unavailable" in result.formatted_response

    def test_resolved_memory_is_rejected_as_not_runnable(self, koakuma):
        """解析到的记忆即使正文是可执行的代码也不运行，提示不可通过 RUN 运行，不记录引用。"""
        mem = _make_code_memory(code="print('tool output')", alias="tool_greet")
        koakuma.memories[mem.get_alias()] = mem

        result = _execute_mtp(koakuma, "⟪ RUN | tool_greet | ⟫")

        assert not result.success
        assert result.response_content == ""
        assert "tool output" not in result.formatted_response
        assert "[Not Runnable]" in result.formatted_response
        assert "不可通过 RUN 运行" in result.formatted_response
        assert koakuma.harness.backing.bus._memory_citations == []

    def test_redirected_pending_alias_is_rejected_under_canonical_alias(self, koakuma):
        pending = koakuma.registry.register_write(
            WriteFocus(content="pending tool", title="Pending Tool", reason=None),
            belong_to=make_identity_scope().workspace_identity,
            from_actor=make_identity_scope().actor_identity,
            process_id=make_runtime_scope().run_id,
            thread_id="main",
        )
        canonical = _make_code_memory(
            code="print('redirected tool output')",
            alias="tool_canonical",
        )
        koakuma.memories[canonical.get_alias()] = canonical
        asyncio.run(
            koakuma.registry.on_settled(
                settlement=PendingAtomSettlement(
                    pending_alias=pending.pending_alias,
                    intent_id=pending.intent_id,
                    resolution=PendingAtomResolution.CREATED,
                    canonical_alias="tool_canonical",
                    canonical_uuid=str(canonical.id),
                )
            )
        )

        result = _execute_mtp(koakuma, f"⟪ RUN | {pending.pending_alias} | ⟫")

        assert not result.success
        assert "redirected tool output" not in result.formatted_response
        assert "[Not Runnable]" in result.formatted_response
        assert "'tool_canonical'" in result.formatted_response
        assert koakuma.harness.backing.bus._memory_citations == []

    def test_failed_pending_alias_returns_reclaimed_error(self, koakuma):
        pending = koakuma.registry.register_write(
            WriteFocus(content="pending tool", title="Pending Tool", reason=None),
            belong_to=make_identity_scope().workspace_identity,
            from_actor=make_identity_scope().actor_identity,
            process_id=make_runtime_scope().run_id,
            thread_id="main",
        )
        asyncio.run(
            koakuma.registry.on_failed(
                pending_alias=pending.pending_alias, intent_id=pending.intent_id
            )
        )

        result = _execute_mtp(
            koakuma,
            f"{MTP_LEFT_DELIMITER} RUN | {pending.pending_alias} | {MTP_RIGHT_DELIMITER}",
        )

        assert not result.success
        assert result.response_content == ""
        assert "failed" in result.formatted_response
        assert "reclaimed" in result.formatted_response
        assert "Alias Not Found" in result.formatted_response

    def test_run_in_flight_pending_same_scope_returns_pending_not_runnable(self, koakuma):
        """同 scope 注册的 in-flight pending 仍按 pending 不可执行语义拒绝。"""
        pending = koakuma.registry.register_write(
            WriteFocus(content="pending tool", title="Pending Tool", reason=None),
            belong_to=make_identity_scope().workspace_identity,
            from_actor=make_identity_scope().actor_identity,
            process_id=make_runtime_scope().run_id,
            thread_id="main",
        )

        result = _execute_mtp(koakuma, f"⟪ RUN | {pending.pending_alias} | ⟫")

        assert not result.success
        assert result.response_content == ""
        assert "[Invalid Argument]" in result.formatted_response
        assert "是运行时 pending atom" in result.formatted_response
        assert koakuma.harness.backing.bus._memory_citations == []

    def test_run_pending_from_other_scope_returns_alias_not_found(self, koakuma):
        """跨 Workspace RUN 他人 pending alias：报 Alias Not Found，不泄露 pending 状态。"""
        pending = koakuma.registry.register_write(
            WriteFocus(content="pending tool", title="Pending Tool", reason=None),
            belong_to=make_identity_scope(workspace_id="isolation_workspace").workspace_identity,
            from_actor=make_identity_scope(workspace_id="isolation_workspace").actor_identity,
            process_id=make_runtime_scope(workspace_id="isolation_workspace").run_id,
            thread_id="main",
        )

        result = _execute_mtp(
            koakuma,
            f"⟪ RUN | {pending.pending_alias} | ⟫",
            context=asyncio.run(
                koakuma.make_context(
                    execution_policy=FrameExecutionPolicy.from_verbs(
                        {"SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"}
                    ),
                    runtime_scope=make_runtime_scope(workspace_id="main_workspace"),
                )
            ),
        )

        assert not result.success
        assert result.response_content == ""
        assert "[Alias Not Found]" in result.formatted_response
        assert "是运行时 pending atom" not in result.formatted_response
        assert "[Invalid Argument]" not in result.formatted_response
        assert koakuma.harness.backing.bus._memory_citations == []

    def test_run_settled_redirect_from_other_scope_does_not_execute_canonical_tool(self, koakuma):
        """跨 Workspace RUN 已结算 redirect：不得执行 canonical 工具。"""
        pending = koakuma.registry.register_write(
            WriteFocus(content="pending tool", title="Pending Tool", reason=None),
            belong_to=make_identity_scope(workspace_id="isolation_workspace").workspace_identity,
            from_actor=make_identity_scope(workspace_id="isolation_workspace").actor_identity,
            process_id=make_runtime_scope(workspace_id="isolation_workspace").run_id,
            thread_id="main",
        )
        canonical = _make_code_memory(
            code="print('redirected tool output')",
            alias="tool_canonical",
        )
        koakuma.memories[canonical.get_alias()] = canonical
        asyncio.run(
            koakuma.registry.on_settled(
                settlement=PendingAtomSettlement(
                    pending_alias=pending.pending_alias,
                    intent_id=pending.intent_id,
                    resolution=PendingAtomResolution.CREATED,
                    canonical_alias="tool_canonical",
                    canonical_uuid=str(canonical.id),
                )
            )
        )

        result = _execute_mtp(
            koakuma,
            f"⟪ RUN | {pending.pending_alias} | ⟫",
            context=asyncio.run(
                koakuma.make_context(
                    execution_policy=FrameExecutionPolicy.from_verbs(
                        {"SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"}
                    ),
                    runtime_scope=make_runtime_scope(workspace_id="main_workspace"),
                )
            ),
        )

        assert not result.success
        assert result.response_content == ""
        assert "redirected tool output" not in result.formatted_response
        assert "[Alias Not Found]" in result.formatted_response
        assert koakuma.harness.backing.bus._memory_citations == []
