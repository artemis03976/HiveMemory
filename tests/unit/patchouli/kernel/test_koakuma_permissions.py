"""Koakuma permission checks use explicit execution context."""

import pytest

from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.agent_runtime.models import MTPExecutionContext
from hivememory.core.models import AgentProfile
from hivememory.core.mtp.exceptions import AgentFault, PermissionDeniedError


def _create_koakuma():
    from tests.unit.patchouli.mtp.conftest import make_koakuma_runtime, make_mock_bus

    bus = make_mock_bus()
    return make_koakuma_runtime(bus)


def _make_profile(allowed_verbs=None, allowed_tools=None):
    return AgentProfile(
        model_name="test-model",
        temperature=0.7,
        allowed_mtp_verbs=allowed_verbs,
        allowed_sys_tools=allowed_tools,
        language="zh",
    )


def _context(profile=None):
    return MTPExecutionContext(agent_profile=profile)


class TestCheckVerbPermission:
    def test_verb_allowed_no_exception(self):
        koakuma = _create_koakuma()
        context = _context(_make_profile(allowed_verbs=["READ", "SEARCH", "WRITE"]))

        koakuma._check_verb_permission("READ", context=context)
        koakuma._check_verb_permission("SEARCH", context=context)
        koakuma._check_verb_permission("WRITE", context=context)

    def test_verb_denied_raises_exception(self):
        koakuma = _create_koakuma()
        context = _context(_make_profile(allowed_verbs=["READ", "SEARCH"]))

        with pytest.raises(PermissionDeniedError) as exc_info:
            koakuma._check_verb_permission("WRITE", context=context)

        assert "WRITE" in str(exc_info.value)
        assert "permission" in str(exc_info.value).lower()

    def test_verb_case_insensitive(self):
        koakuma = _create_koakuma()
        context = _context(_make_profile(allowed_verbs=["read", "search"]))

        koakuma._check_verb_permission("READ", context=context)
        koakuma._check_verb_permission("SEARCH", context=context)
        koakuma._check_verb_permission("read", context=context)
        koakuma._check_verb_permission("search", context=context)

    def test_no_profile_allows_all_verbs(self):
        koakuma = _create_koakuma()

        for verb in ["READ", "WRITE", "UPDATE", "RUN", "SEARCH"]:
            koakuma._check_verb_permission(verb, context=_context())

    def test_empty_verb_list_denies_all(self):
        koakuma = _create_koakuma()
        context = _context(_make_profile(allowed_verbs=[]))

        for verb in ["READ", "WRITE", "UPDATE", "RUN", "SEARCH"]:
            with pytest.raises(PermissionDeniedError):
                koakuma._check_verb_permission(verb, context=context)


class TestCheckToolPermission:
    def test_tool_allowed_no_exception(self):
        koakuma = _create_koakuma()
        context = _context(_make_profile(allowed_tools=["sys_clock", "sys_read_file"]))

        koakuma._check_tool_permission("sys_clock", context=context)
        koakuma._check_tool_permission("sys_read_file", context=context)

    def test_tool_denied_raises_exception(self):
        koakuma = _create_koakuma()
        context = _context(_make_profile(allowed_tools=["sys_clock"]))

        with pytest.raises(PermissionDeniedError) as exc_info:
            koakuma._check_tool_permission("sys_bash_exec", context=context)

        assert "sys_bash_exec" in str(exc_info.value)
        assert "access" in str(exc_info.value).lower()

    def test_tool_exact_match(self):
        koakuma = _create_koakuma()
        context = _context(_make_profile(allowed_tools=["sys_clock"]))

        koakuma._check_tool_permission("sys_clock", context=context)
        with pytest.raises(PermissionDeniedError):
            koakuma._check_tool_permission("SYS_CLOCK", context=context)

    def test_no_profile_allows_all_tools(self):
        koakuma = _create_koakuma()

        for tool in ["sys_clock", "sys_bash_exec", "sys_web_search", "sys_python_repl"]:
            koakuma._check_tool_permission(tool, context=_context())

    def test_empty_tool_list_denies_all(self):
        koakuma = _create_koakuma()
        context = _context(_make_profile(allowed_tools=[]))

        for tool in ["sys_clock", "sys_bash_exec", "sys_web_search", "sys_python_repl"]:
            with pytest.raises(PermissionDeniedError):
                koakuma._check_tool_permission(tool, context=context)


class TestPermissionDeniedError:
    def test_error_is_agent_fault(self):
        koakuma = _create_koakuma()
        context = _context(_make_profile(allowed_verbs=["READ"]))

        with pytest.raises(PermissionDeniedError) as exc_info:
            koakuma._check_verb_permission("WRITE", context=context)

        assert isinstance(exc_info.value, AgentFault)

    def test_error_message_contains_verb(self):
        koakuma = _create_koakuma()
        context = _context(_make_profile(allowed_verbs=["READ"]))

        with pytest.raises(PermissionDeniedError) as exc_info:
            koakuma._check_verb_permission("UPDATE", context=context)

        assert "UPDATE" in str(exc_info.value)

    def test_error_message_contains_tool(self):
        koakuma = _create_koakuma()
        context = _context(_make_profile(allowed_tools=["sys_clock"]))

        with pytest.raises(PermissionDeniedError) as exc_info:
            koakuma._check_tool_permission("sys_write_file", context=context)

        assert "sys_write_file" in str(exc_info.value)


class TestCombinedPermissions:
    def test_restricted_verbs_and_tools(self):
        koakuma = _create_koakuma()
        context = _context(_make_profile(
            allowed_verbs=["READ", "SEARCH"],
            allowed_tools=["sys_clock"],
        ))

        koakuma._check_verb_permission("READ", context=context)
        koakuma._check_verb_permission("SEARCH", context=context)
        koakuma._check_tool_permission("sys_clock", context=context)

        with pytest.raises(PermissionDeniedError):
            koakuma._check_verb_permission("WRITE", context=context)
        with pytest.raises(PermissionDeniedError):
            koakuma._check_tool_permission("sys_bash_exec", context=context)

    def test_reviewer_profile_scenario(self):
        koakuma = _create_koakuma()
        context = _context(_make_profile(
            allowed_verbs=["READ", "SEARCH"],
            allowed_tools=["sys_clock"],
        ))

        koakuma._check_verb_permission("READ", context=context)
        koakuma._check_verb_permission("SEARCH", context=context)

        with pytest.raises(PermissionDeniedError):
            koakuma._check_verb_permission("WRITE", context=context)
        with pytest.raises(PermissionDeniedError):
            koakuma._check_tool_permission("sys_write_file", context=context)

    def test_coder_profile_scenario(self):
        koakuma = _create_koakuma()
        context = _context(_make_profile(
            allowed_verbs=["READ", "SEARCH", "WRITE", "RUN"],
            allowed_tools=["sys_clock", "sys_read_file", "sys_write_file", "sys_python_repl"],
        ))

        koakuma._check_verb_permission("READ", context=context)
        koakuma._check_verb_permission("WRITE", context=context)
        koakuma._check_verb_permission("RUN", context=context)
        koakuma._check_tool_permission("sys_read_file", context=context)
        koakuma._check_tool_permission("sys_write_file", context=context)
        koakuma._check_tool_permission("sys_python_repl", context=context)
