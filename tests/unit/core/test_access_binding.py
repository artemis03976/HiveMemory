"""访问运行绑定的种类与线程标识约束。"""

import pytest

from hivememory.core.access import AccessRunType, RunBinding


@pytest.mark.parametrize(
    "run_type, thread_id",
    [
        ("execution_thread", None),
        ("execution_thread", " "),
        ("task_process", "main"),
        ("request", "main"),
    ],
)
def test_run_binding_rejects_inconsistent_thread_identifier(run_type, thread_id):
    """运行种类和线程标识必须同时成立，禁止模糊的线程归属。"""
    with pytest.raises(ValueError, match="thread_id"):
        RunBinding(AccessRunType(run_type), "p", thread_id)
