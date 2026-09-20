"""SystemActorAccessRegistry 的单元测试。

被测对象：system.access.registry（A1 计划第 2.2 节）。保护的契约：来源
登记的装载校验（重复 principal、空 adapter 集合）、查询返回登记本身、
禁用与未登记由网关统一折叠处理。
"""

from __future__ import annotations

import pytest

from hivememory.system.access import SystemActorAccessEntry, SystemActorAccessRegistry


def test_entry_lookup_returns_registration():
    """按来源标识返回登记；缺省登记只允许 local adapter。"""
    registry = SystemActorAccessRegistry(
        [SystemActorAccessEntry(principal_id="local-process:alice")]
    )

    entry = registry.entry_for("local-process:alice")
    assert entry is not None and entry.adapters == frozenset({"local"})
    assert registry.entry_for("local-process:stranger") is None


def test_duplicate_principal_rejected_at_load():
    """同一 principal 的重复登记是配置矛盾，装载期显式失败。"""
    with pytest.raises(ValueError):
        SystemActorAccessRegistry(
            [
                SystemActorAccessEntry(principal_id="p1"),
                SystemActorAccessEntry(principal_id="p1"),
            ]
        )


def test_entry_without_adapter_rejected_at_load():
    """不声明任何 adapter 的登记无法匹配任何接入方式，装载期拒绝。"""
    with pytest.raises(ValueError):
        SystemActorAccessEntry(principal_id="p1", adapters=frozenset())
