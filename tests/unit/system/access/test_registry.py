"""SystemActorAccessRegistry 的单元测试。

被测对象：system.access.registry。保护的契约：来源
登记的装载校验（重复 principal、空 adapter 集合、外部 adapter 必须声明凭据、
进程内 adapter 不能与凭据同登记）、
从环境变量读取凭据且不进入 repr、查询返回登记本身、
禁用与未登记由网关统一折叠处理。
"""

from __future__ import annotations

import logging

import pytest

from hivememory.config.access import SystemPrincipalsFile
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


@pytest.mark.parametrize(
    ("adapters", "credential_env"),
    [
        # 进程内 adapter 以外的都是外部入口：principal 只是声明，必须以凭据证明。
        (frozenset({"mcp"}), None),
        # 进程内入口总是不带凭据认证：与凭据同登记会使这些入口一律被拒绝。
        (frozenset({"http", "mcp"}), "TEST_HARNESS_TOKEN"),
    ],
)
def test_entry_mixing_credential_and_adapter_kinds_rejected_at_load(adapters, credential_env):
    """外部 adapter 必须声明凭据，进程内 adapter 不能与凭据同登记，装载期失败。"""
    with pytest.raises(ValueError, match="credential_env"):
        SystemActorAccessEntry(
            principal_id="harness:test", adapters=adapters, credential_env=credential_env
        )


def test_from_config_reads_credential_from_env_and_hides_it_from_repr(monkeypatch):
    """凭据取自登记声明的环境变量（去掉环境文件带入的首尾空白），不出现在 repr 中。"""
    monkeypatch.setenv("TEST_HARNESS_TOKEN", "s3cret-token\r\n")

    registry = SystemActorAccessRegistry.from_config(
        SystemPrincipalsFile.model_validate(
            {
                "principals": [
                    {
                        "principal_id": "harness:test",
                        "adapters": ["mcp"],
                        "credential_env": "TEST_HARNESS_TOKEN",
                    }
                ]
            }
        )
    )

    entry = registry.entry_for("harness:test")
    assert entry is not None and entry.credential == "s3cret-token"
    assert "s3cret-token" not in repr(entry)


def test_from_config_with_unset_credential_env_warns_and_still_loads(monkeypatch, caplog):
    """凭据环境变量未设置时照常装配并告警，该来源不持有任何凭据。"""
    monkeypatch.delenv("TEST_HARNESS_TOKEN", raising=False)

    with caplog.at_level(logging.WARNING, logger="hivememory.system.access.registry"):
        registry = SystemActorAccessRegistry.from_config(
            SystemPrincipalsFile.model_validate(
                {
                    "principals": [
                        {
                            "principal_id": "harness:test",
                            "adapters": ["import"],
                            "credential_env": "TEST_HARNESS_TOKEN",
                        }
                    ]
                }
            )
        )

    entry = registry.entry_for("harness:test")
    assert entry is not None and entry.credential is None
    assert "TEST_HARNESS_TOKEN" in caplog.text
