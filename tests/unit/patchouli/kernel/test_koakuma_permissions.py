"""
Koakuma 运行时权限拦截单元测试

测试覆盖:
- _check_verb_permission: 动词权限校验 / 越权拦截 / 无 profile 兜底
- _check_tool_permission: 工具权限校验 / 越权拦截 / 无 profile 兜底
- PermissionDeniedError: 异常抛出与格式化
- set_active_profile: profile 设置与清除
"""

import pytest
from unittest.mock import Mock, patch

from hivememory.core.models import AgentProfile
from hivememory.patchouli.kernel.koakuma import KoakumaRuntime
from hivememory.patchouli.mtp.exceptions import PermissionDeniedError


def _create_koakuma():
    """创建 KoakumaRuntime 实例"""
    return KoakumaRuntime(bus=None, config=None)


def _make_profile(allowed_verbs=None, allowed_tools=None):
    """构建 AgentProfile"""
    return AgentProfile(
        model_name="test-model",
        temperature=0.7,
        allowed_mtp_verbs=allowed_verbs,
        allowed_sys_tools=allowed_tools,
        language="zh",
    )


class TestCheckVerbPermission:
    """_check_verb_permission() 测试"""

    def test_verb_allowed_no_exception(self):
        """动词在白名单内，不抛异常"""
        koakuma = _create_koakuma()
        profile = _make_profile(allowed_verbs=["READ", "SEARCH", "WRITE"])
        koakuma.set_active_profile(profile)

        # 应该不抛异常
        koakuma._check_verb_permission("READ")
        koakuma._check_verb_permission("SEARCH")
        koakuma._check_verb_permission("WRITE")

    def test_verb_denied_raises_exception(self):
        """动词不在白名单内，抛出 PermissionDeniedError"""
        koakuma = _create_koakuma()
        profile = _make_profile(allowed_verbs=["READ", "SEARCH"])
        koakuma.set_active_profile(profile)

        with pytest.raises(PermissionDeniedError) as exc_info:
            koakuma._check_verb_permission("WRITE")

        assert "WRITE" in str(exc_info.value)
        assert "permission" in str(exc_info.value).lower()

    def test_verb_case_insensitive(self):
        """动词检查大小写不敏感"""
        koakuma = _create_koakuma()
        profile = _make_profile(allowed_verbs=["read", "search"])
        koakuma.set_active_profile(profile)

        # 大写应该通过
        koakuma._check_verb_permission("READ")
        koakuma._check_verb_permission("SEARCH")

        # 小写也应该通过
        koakuma._check_verb_permission("read")
        koakuma._check_verb_permission("search")

    def test_no_profile_allows_all_verbs(self):
        """无 profile 时（兜底逻辑），允许所有动词"""
        koakuma = _create_koakuma()
        # 不设置 profile，默认为 None

        # 应该不抛异常
        koakuma._check_verb_permission("READ")
        koakuma._check_verb_permission("WRITE")
        koakuma._check_verb_permission("UPDATE")
        koakuma._check_verb_permission("RUN")
        koakuma._check_verb_permission("SEARCH")

    def test_empty_verb_list_denies_all(self):
        """空白名单（[]）禁止所有动词"""
        koakuma = _create_koakuma()
        profile = _make_profile(allowed_verbs=[])  # 空列表 = 全部禁止
        koakuma.set_active_profile(profile)

        for verb in ["READ", "WRITE", "UPDATE", "RUN", "SEARCH"]:
            with pytest.raises(PermissionDeniedError):
                koakuma._check_verb_permission(verb)


class TestCheckToolPermission:
    """_check_tool_permission() 测试"""

    def test_tool_allowed_no_exception(self):
        """工具在白名单内，不抛异常"""
        koakuma = _create_koakuma()
        profile = _make_profile(allowed_tools=["sys_clock", "sys_read_file"])
        koakuma.set_active_profile(profile)

        # 应该不抛异常
        koakuma._check_tool_permission("sys_clock")
        koakuma._check_tool_permission("sys_read_file")

    def test_tool_denied_raises_exception(self):
        """工具不在白名单内，抛出 PermissionDeniedError"""
        koakuma = _create_koakuma()
        profile = _make_profile(allowed_tools=["sys_clock"])
        koakuma.set_active_profile(profile)

        with pytest.raises(PermissionDeniedError) as exc_info:
            koakuma._check_tool_permission("sys_bash_exec")

        assert "sys_bash_exec" in str(exc_info.value)
        assert "access" in str(exc_info.value).lower()

    def test_tool_exact_match(self):
        """工具名称精确匹配（大小写敏感）"""
        koakuma = _create_koakuma()
        profile = _make_profile(allowed_tools=["sys_clock"])
        koakuma.set_active_profile(profile)

        # 精确匹配应该通过
        koakuma._check_tool_permission("sys_clock")

        # 大小写不同应该被拒绝
        with pytest.raises(PermissionDeniedError):
            koakuma._check_tool_permission("SYS_CLOCK")

    def test_no_profile_allows_all_tools(self):
        """无 profile 时（兜底逻辑），允许所有工具"""
        koakuma = _create_koakuma()
        # 不设置 profile，默认为 None

        # 应该不抛异常
        koakuma._check_tool_permission("sys_clock")
        koakuma._check_tool_permission("sys_bash_exec")
        koakuma._check_tool_permission("sys_web_search")
        koakuma._check_tool_permission("sys_python_repl")

    def test_empty_tool_list_denies_all(self):
        """空白名单（[]）禁止所有工具"""
        koakuma = _create_koakuma()
        profile = _make_profile(allowed_tools=[])  # 空列表 = 全部禁止
        koakuma.set_active_profile(profile)

        for tool in ["sys_clock", "sys_bash_exec", "sys_web_search", "sys_python_repl"]:
            with pytest.raises(PermissionDeniedError):
                koakuma._check_tool_permission(tool)


class TestSetActiveProfile:
    """set_active_profile() 测试"""

    def test_set_profile_updates_internal_state(self):
        """设置 profile 更新内部状态"""
        koakuma = _create_koakuma()
        profile = _make_profile(allowed_verbs=["READ"])

        koakuma.set_active_profile(profile)

        assert koakuma._active_profile is profile

    def test_set_profile_none_clears_state(self):
        """设置 None 清除 profile 状态"""
        koakuma = _create_koakuma()
        profile = _make_profile(allowed_verbs=["READ"])
        koakuma.set_active_profile(profile)

        # 清除 profile
        koakuma.set_active_profile(None)

        assert koakuma._active_profile is None

    def test_profile_switch_affects_permissions(self):
        """切换 profile 影响权限检查"""
        koakuma = _create_koakuma()

        # 设置限制性 profile
        restrictive = _make_profile(allowed_verbs=["READ"])
        koakuma.set_active_profile(restrictive)

        # WRITE 应该被拒绝
        with pytest.raises(PermissionDeniedError):
            koakuma._check_verb_permission("WRITE")

        # 切换到宽松 profile
        permissive = _make_profile(allowed_verbs=["READ", "WRITE"])
        koakuma.set_active_profile(permissive)

        # WRITE 现在应该通过
        koakuma._check_verb_permission("WRITE")


class TestPermissionDeniedError:
    """PermissionDeniedError 异常测试"""

    def test_error_is_agent_fault(self):
        """PermissionDeniedError 是 AgentFault 子类"""
        from hivememory.patchouli.mtp.exceptions import AgentFault

        koakuma = _create_koakuma()
        profile = _make_profile(allowed_verbs=["READ"])
        koakuma.set_active_profile(profile)

        try:
            koakuma._check_verb_permission("WRITE")
        except Exception as e:
            assert isinstance(e, AgentFault)
            assert isinstance(e, PermissionDeniedError)

    def test_error_message_contains_verb(self):
        """错误消息包含被拒绝的动词"""
        koakuma = _create_koakuma()
        profile = _make_profile(allowed_verbs=["READ"])
        koakuma.set_active_profile(profile)

        with pytest.raises(PermissionDeniedError) as exc_info:
            koakuma._check_verb_permission("UPDATE")

        error_msg = str(exc_info.value)
        assert "UPDATE" in error_msg

    def test_error_message_contains_tool(self):
        """错误消息包含被拒绝的工具"""
        koakuma = _create_koakuma()
        profile = _make_profile(allowed_tools=["sys_clock"])
        koakuma.set_active_profile(profile)

        with pytest.raises(PermissionDeniedError) as exc_info:
            koakuma._check_tool_permission("sys_write_file")

        error_msg = str(exc_info.value)
        assert "sys_write_file" in error_msg


class TestCombinedPermissions:
    """组合权限场景测试"""

    def test_restricted_verbs_and_tools(self):
        """同时限制动词和工具"""
        koakuma = _create_koakuma()
        profile = _make_profile(
            allowed_verbs=["READ", "SEARCH"],
            allowed_tools=["sys_clock"],
        )
        koakuma.set_active_profile(profile)

        # 允许的动词应该通过
        koakuma._check_verb_permission("READ")
        koakuma._check_verb_permission("SEARCH")

        # 禁止的动词应该被拒绝
        with pytest.raises(PermissionDeniedError):
            koakuma._check_verb_permission("WRITE")

        # 允许的工具应该通过
        koakuma._check_tool_permission("sys_clock")

        # 禁止的工具应该被拒绝
        with pytest.raises(PermissionDeniedError):
            koakuma._check_tool_permission("sys_bash_exec")

    def test_reviewer_profile_scenario(self):
        """Reviewer 人偶场景：只读权限"""
        koakuma = _create_koakuma()
        reviewer_profile = _make_profile(
            allowed_verbs=["READ", "SEARCH"],
            allowed_tools=["sys_clock"],  # 无写文件权限
        )
        koakuma.set_active_profile(reviewer_profile)

        # 可以读取
        koakuma._check_verb_permission("READ")
        koakuma._check_verb_permission("SEARCH")

        # 不能写入
        with pytest.raises(PermissionDeniedError):
            koakuma._check_verb_permission("WRITE")

        # 不能执行危险工具
        with pytest.raises(PermissionDeniedError):
            koakuma._check_tool_permission("sys_write_file")

    def test_coder_profile_scenario(self):
        """Coder 人偶场景：读写权限"""
        koakuma = _create_koakuma()
        coder_profile = _make_profile(
            allowed_verbs=["READ", "SEARCH", "WRITE", "RUN"],
            allowed_tools=["sys_clock", "sys_read_file", "sys_write_file", "sys_python_repl"],
        )
        koakuma.set_active_profile(coder_profile)

        # 可以读写
        koakuma._check_verb_permission("READ")
        koakuma._check_verb_permission("WRITE")
        koakuma._check_verb_permission("RUN")

        # 可以使用文件工具
        koakuma._check_tool_permission("sys_read_file")
        koakuma._check_tool_permission("sys_write_file")
        koakuma._check_tool_permission("sys_python_repl")
