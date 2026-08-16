"""
FrameExecutionPolicy 单元测试

测试覆盖:
- from_profile: denied_verbs 从 profile 白名单中移除对应动词
"""

from __future__ import annotations

from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.core.models import OMNI_DOLL_PROFILE


def test_callee_policy_removes_call_without_reading_frame_depth() -> None:
    policy = FrameExecutionPolicy.from_profile(
        OMNI_DOLL_PROFILE,
        max_iterations=3,
        denied_verbs={"CALL"},
    )

    assert policy.allows("READ")
    assert not policy.allows("CALL")
