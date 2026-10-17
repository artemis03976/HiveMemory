"""Alice 操作快照到 MTP 动词的翻译，不涉及资源授权。"""

import pytest

from hivememory.alice.orchestration.verbs import mtp_verbs_for_operations
from hivememory.core.access import WorkspaceOperation
from hivememory.core.mtp.models import MTPVerb


@pytest.mark.parametrize(
    "operation, expected",
    [
        (WorkspaceOperation.MEMORY_READ, {"READ", "RUN"}),
        (WorkspaceOperation.MEMORY_SEARCH, {"SEARCH", "RUN"}),
        (WorkspaceOperation.MEMORY_WRITE, {"WRITE", "RUN"}),
        (WorkspaceOperation.MEMORY_UPDATE, {"UPDATE", "RUN"}),
        (WorkspaceOperation.AGENT_CALL, {"CALL", "RUN"}),
        (WorkspaceOperation.MEMORY_RUN, {"RUN"}),
    ],
)
def test_each_actor_operation_maps_to_its_verb_and_system_run(operation, expected):
    """每项 actor 操作只增加对应动词，不因系统 RUN 扩大其它操作能力。"""
    assert mtp_verbs_for_operations(frozenset({operation})) == expected


def test_empty_operations_keep_system_run():
    """空白名单仍可执行系统工具，但不能获得任何其它 MTP 动词。"""
    assert mtp_verbs_for_operations(frozenset()) == {"RUN"}


def test_every_actor_operation_has_a_verb_translation():
    """新增 actor 操作必须同时补齐翻译，否则 Alice 运行时会在构造执行策略时失败。"""
    assert mtp_verbs_for_operations(frozenset(WorkspaceOperation)) == {
        verb.value for verb in MTPVerb
    }
