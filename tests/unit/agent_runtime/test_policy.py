"""FrameExecutionPolicy 的集合规范化、拒绝规则与默认边界单元测试。"""

from hivememory.agent_runtime.policy import FrameExecutionPolicy


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


def test_default_policy_allows_known_protocol_verbs_and_rejects_unknown() -> None:
    """通用默认策略采用已知协议全集，未知动词不会自动获得执行许可。"""
    policy = FrameExecutionPolicy()

    assert policy.permitted_verbs == frozenset({"SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"})
    assert policy.allows("CALL")
    assert not policy.allows("UNKNOWN")
