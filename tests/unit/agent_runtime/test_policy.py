"""FrameExecutionPolicy 的集合规范化与拒绝规则单元测试。"""

from hivememory.agent_runtime.frame import FrameExecutionPolicy


def test_callee_policy_removes_call_and_keeps_iteration_budget() -> None:
    """子 frame 禁止 CALL，但继承其余动词和显式预算。"""
    policy = FrameExecutionPolicy.from_verbs(
        {"read", "RUN", "call"}, max_iterations=3, denied_verbs={"call"}
    )

    assert policy.permitted_verbs == frozenset({"READ", "RUN"})
    assert policy.max_iterations == 3
    assert policy.allows("read")
    assert not policy.allows("CALL")
    assert not policy.allows("WRITE")


def test_empty_explicit_verbs_deny_every_known_verb() -> None:
    """空集合不能退回默认全集。"""
    policy = FrameExecutionPolicy.from_verbs([])

    assert policy.permitted_verbs == frozenset()
    for verb in ("SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"):
        assert not policy.allows(verb)


def test_explicit_policy_rejects_unknown_verbs() -> None:
    """只允许显式列出的动词，未知动词不获得执行许可。"""
    policy = FrameExecutionPolicy.from_verbs({"READ", "CALL"})

    assert policy.permitted_verbs == frozenset({"READ", "CALL"})
    assert policy.allows("CALL")
    assert not policy.allows("UNKNOWN")
