"""SystemActorAccessRegistry 的单元测试。

被测对象：system.access.registry。保护的契约：来源
登记的装载校验（重复 principal、空 adapter 集合、进程内与外部 adapter 不能
混用、进程内 adapter 不能与凭据同登记）、凭据的来源与优先级（环境变量优先于
生成凭据）且不进入 repr、查询返回登记本身；外部来源的运行时登记、修改、
轮换与删除写回文件且与重新装载一致，进程内来源、重复登记、环境变量凭据的
轮换与被外部改动的文件都被拒绝，拒绝时内存与文件保持原样。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from hivememory.config.access import (
    SystemPrincipalsFile,
    file_digest,
    load_principal_credentials,
)
from hivememory.core.errors import AccessRegistrationConflictError
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
    ("adapters", "credential_env", "message"),
    [
        # 外部来源每次认证都核对凭据，进程内来源从不携带凭据：混用会让外部 adapter
        # 借进程内来源的身份免凭据接入。
        (frozenset({"http", "mcp"}), None, "不能同时列出"),
        # 进程内入口总是不带凭据认证：与凭据同登记会使这些入口一律被拒绝。
        (frozenset({"local"}), "TEST_HARNESS_TOKEN", "credential_env"),
    ],
)
def test_entry_mixing_adapter_kinds_or_in_process_credential_rejected_at_load(
    adapters, credential_env, message
):
    """进程内与外部 adapter 不能混用，进程内来源不能声明凭据，装载期失败。"""
    with pytest.raises(ValueError, match=message):
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


def test_from_config_reads_credential_written_in_dotenv(monkeypatch, tmp_path):
    """凭据与其他密钥一样可以写在 .env 中：进程环境变量未设置时从 .env 读取。"""
    monkeypatch.delenv("TEST_HARNESS_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("TEST_HARNESS_TOKEN=dotenv-token\n", encoding="utf-8")

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
    assert entry is not None and entry.credential == "dotenv-token"


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


@pytest.mark.parametrize(
    ("env_value", "expected_credential", "expected_source"),
    [
        # 环境变量有值时优先，规则与 Provider 凭证相同。
        ("env-token", "env-token", "env"),
        # 环境变量未设置时用凭据文件中的生成凭据。
        (None, "generated-token", "generated"),
    ],
)
def test_env_credential_takes_precedence_over_generated_one(
    monkeypatch, tmp_path, env_value, expected_credential, expected_source
):
    """同一来源同时有环境变量与生成凭据时取环境变量，没有环境变量时取生成凭据。"""
    if env_value is None:
        monkeypatch.delenv("TEST_HARNESS_TOKEN", raising=False)
    else:
        monkeypatch.setenv("TEST_HARNESS_TOKEN", env_value)
    credentials_path = tmp_path / "principals.secrets.yaml"
    credentials_path.write_text("credentials:\n  harness:test: generated-token\n", encoding="utf-8")

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
        ),
        credentials_path=credentials_path,
    )

    entry = registry.entry_for("harness:test")
    assert entry is not None
    assert (entry.credential, entry.credential_source) == (expected_credential, expected_source)


# ---- 运行时编辑 ----

_PRINCIPALS_YAML = """\
principals:
  - principal_id: "hivememory:http-server"
    kind: "http-server"
    adapters: ["http"]
    allow_system_actor: true
  - principal_id: "harness:env"
    kind: "harness"
    adapters: ["mcp", "import"]
    credential_env: "TEST_ENV_HARNESS_TOKEN"
"""


def _editable_registry(tmp_path: Path) -> SystemActorAccessRegistry:
    """从 tmp_path 中的登记文件装载，写回与重新装载都指向这些文件。"""
    path = tmp_path / "system_principals.yaml"
    if not path.exists():
        path.write_text(_PRINCIPALS_YAML, encoding="utf-8")
    return SystemActorAccessRegistry.from_config(
        SystemPrincipalsFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))),
        file_path=path,
        file_digest=file_digest(path),
        credentials_path=tmp_path / "principals.secrets.yaml",
    )


def test_registered_principal_round_trips_through_files(tmp_path):
    """运行时登记的来源与生成凭据写回文件，重新装载后与内存中的登记一致。"""
    registry = _editable_registry(tmp_path)

    entry, credential = registry.register(
        "harness:new",
        adapters=frozenset({"mcp"}),
        allowed_agent_ids=frozenset({"new_agent"}),
    )

    reloaded = _editable_registry(tmp_path).entry_for("harness:new")
    assert reloaded == entry
    assert reloaded.credential == credential
    assert reloaded.credential_source == "generated"
    assert not reloaded.allow_system_actor
    # 凭据只进入凭据文件，不进入登记文件。
    assert credential not in (tmp_path / "system_principals.yaml").read_text(encoding="utf-8")


def test_update_and_delete_take_effect_in_memory_and_on_reload(tmp_path):
    """修改与删除立即反映在查询中并写回文件；删除同时删掉它的生成凭据。"""
    registry = _editable_registry(tmp_path)
    registry.register("harness:new", adapters=frozenset({"mcp"}), allowed_agent_ids=None)

    updated = registry.update(
        "harness:new",
        adapters=frozenset({"mcp", "import"}),
        allowed_agent_ids=frozenset({"a1"}),
        enabled=False,
    )
    assert registry.entry_for("harness:new") == updated
    assert _editable_registry(tmp_path).entry_for("harness:new") == updated

    registry.delete("harness:new")
    assert registry.entry_for("harness:new") is None
    assert _editable_registry(tmp_path).entry_for("harness:new") is None
    credentials = load_principal_credentials(tmp_path / "principals.secrets.yaml")
    assert "harness:new" not in credentials.credentials


def test_rotation_replaces_generated_credential(tmp_path):
    """轮换生成新凭据，旧凭据不再生效，重新装载后同样如此。"""
    registry = _editable_registry(tmp_path)
    _, old = registry.register("harness:new", adapters=frozenset({"mcp"}), allowed_agent_ids=None)

    entry, new = registry.rotate_credential("harness:new")

    assert new != old
    assert entry.credential == new
    assert _editable_registry(tmp_path).entry_for("harness:new").credential == new


def _disable_http_server(registry: SystemActorAccessRegistry) -> None:
    registry.update(
        "hivememory:http-server",
        adapters=frozenset({"mcp"}),
        allowed_agent_ids=None,
        enabled=False,
    )


def _register_existing(registry: SystemActorAccessRegistry) -> None:
    registry.register("harness:env", adapters=frozenset({"mcp"}), allowed_agent_ids=None)


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        # 进程内来源关系到管理接口与 chat 本身，只能经文件修改。
        (_disable_http_server, "protected"),
        (_register_existing, "duplicate"),
        # 环境变量优先，生成的凭据不会生效。
        (lambda registry: registry.rotate_credential("harness:env"), "credential_from_env"),
    ],
)
def test_rejected_edits_leave_memory_and_file_unchanged(monkeypatch, tmp_path, edit, reason):
    """被拒绝的编辑以对应 reason 失败，内存中的登记与登记文件都保持原样。"""
    monkeypatch.setenv("TEST_ENV_HARNESS_TOKEN", "env-token")
    registry = _editable_registry(tmp_path)
    before_entries = registry.entries()
    before_file = (tmp_path / "system_principals.yaml").read_bytes()

    with pytest.raises(AccessRegistrationConflictError) as exc_info:
        edit(registry)

    assert exc_info.value.details["reason"] == reason
    assert registry.entries() == before_entries
    assert (tmp_path / "system_principals.yaml").read_bytes() == before_file


def test_externally_modified_file_is_not_overwritten(tmp_path):
    """文件在运行中被手工改动后，写回以 file_modified 拒绝，不覆盖手工修改。"""
    registry = _editable_registry(tmp_path)
    path = tmp_path / "system_principals.yaml"
    manual = path.read_text(encoding="utf-8") + '  - principal_id: "local-process:manual"\n'
    path.write_text(manual, encoding="utf-8")

    with pytest.raises(AccessRegistrationConflictError) as exc_info:
        registry.register("harness:new", adapters=frozenset({"mcp"}), allowed_agent_ids=None)

    assert exc_info.value.details["reason"] == "file_modified"
    assert registry.entry_for("harness:new") is None
    assert path.read_text(encoding="utf-8") == manual


def test_runtime_edit_only_accepts_external_adapters(tmp_path):
    """运行时登记的来源只能列出外部 adapter，不能借此登记进程内来源。"""
    registry = _editable_registry(tmp_path)

    with pytest.raises(ValueError, match="外部 adapter"):
        registry.register("harness:new", adapters=frozenset({"http"}), allowed_agent_ids=None)
    assert registry.entry_for("harness:new") is None
