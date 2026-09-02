"""Omni-Doll 默认 profile 能力清单与解析回退测试。

覆盖:
- 内置 Omni-Doll profile 必须使用显式能力白名单（不允许 None）
- 能力白名单与 MTP 动词枚举 / 内核 syscall 注册表保持同步
- AgentProfileResolver 在未指定 agent 时回退到 Omni-Doll profile（回归守卫）
"""

import pytest

from hivememory.agent_runtime.mtp.syscalls.registry import build_kernel_registry
from hivememory.alice.runtime.profile_resolver import AgentProfileResolver
from hivememory.core.models.agent import (
    OMNI_DOLL_ALLOWED_MTP_VERBS,
    OMNI_DOLL_ALLOWED_SYS_TOOLS,
    OMNI_DOLL_PROFILE,
)
from hivememory.core.mtp.models import MTPVerb
from tests.helpers.workspace import make_identity_scope


def test_omni_doll_profile_uses_explicit_current_capability_lists():
    assert OMNI_DOLL_PROFILE.allowed_mtp_verbs is not None
    assert OMNI_DOLL_PROFILE.allowed_sys_tools is not None
    assert set(OMNI_DOLL_ALLOWED_MTP_VERBS) == {verb.value for verb in MTPVerb}
    assert set(OMNI_DOLL_ALLOWED_SYS_TOOLS) == set(build_kernel_registry())


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", [None, "", "default", "omni_doll"])
async def test_resolver_fallback_uses_omni_doll_profile_without_bus(alias):
    """未指定/默认 alias 必须短路返回 Omni-Doll profile，且不触达总线。"""
    from unittest.mock import MagicMock

    bus = MagicMock()
    resolver = AgentProfileResolver(local_bus=bus)

    profile = await resolver.resolve(alias, identity_scope=make_identity_scope())

    assert profile is OMNI_DOLL_PROFILE
    bus.request.assert_not_called()
