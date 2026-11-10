"""访问登记文件的装载：生效文件是本机的个人配置，默认路径缺失时从随仓库发布的
模板生成，已有的生效文件不被模板覆盖；环境变量指定的路径缺失时显式失败。"""

from pathlib import Path

import pytest

from hivememory.config import access

_PRINCIPALS_TEMPLATE = """\
# 模板中的说明
principals:
  - principal_id: "hivememory:http-server"
    kind: "http-server"
    adapters: ["http"]
    allow_system_actor: true
"""

_ACTORS_TEMPLATE = """\
# 模板中的说明
workspace_actors:
  - owner_user_id: "default"
    workspace_id: "main_workspace"
    user_id: "default"
    agent_id: "system"
"""


@pytest.fixture
def configs_dir(tmp_path, monkeypatch) -> Path:
    """以 tmp_path 代替仓库的 configs 目录，只放两份模板。"""
    (tmp_path / "system_principals.example.yaml").write_text(_PRINCIPALS_TEMPLATE, encoding="utf-8")
    (tmp_path / "workspace_actors.example.yaml").write_text(_ACTORS_TEMPLATE, encoding="utf-8")
    monkeypatch.setattr(access, "_repo_configs_dir", lambda: tmp_path)
    monkeypatch.delenv(access.PRINCIPALS_FILE_ENV, raising=False)
    monkeypatch.delenv(access.WORKSPACE_ACTORS_FILE_ENV, raising=False)
    return tmp_path


def test_missing_registration_is_generated_from_the_template(configs_dir):
    """生效文件不存在时按模板原样生成（保留注释），装载的就是模板中的登记。"""
    registration = access.load_access_registration()

    assert (configs_dir / "system_principals.yaml").read_text("utf-8") == _PRINCIPALS_TEMPLATE
    assert (configs_dir / "workspace_actors.yaml").read_text("utf-8") == _ACTORS_TEMPLATE
    assert [p.principal_id for p in registration.principals.principals] == [
        "hivememory:http-server"
    ]
    assert registration.principals_digest == access.file_digest(
        configs_dir / "system_principals.yaml"
    )


def test_existing_registration_is_not_overwritten_by_the_template(configs_dir):
    """已有的生效文件是本机的个人配置，模板不覆盖它。"""
    local = _PRINCIPALS_TEMPLATE.replace("hivememory:http-server", "local-process:mine")
    (configs_dir / "system_principals.yaml").write_text(local, encoding="utf-8")

    registration = access.load_access_registration()

    assert (configs_dir / "system_principals.yaml").read_text("utf-8") == local
    assert [p.principal_id for p in registration.principals.principals] == ["local-process:mine"]


def test_missing_explicit_path_fails_without_generating(configs_dir, monkeypatch):
    """环境变量指定的路径缺失时显式失败，不在那里生成一份默认登记。"""
    explicit = configs_dir / "elsewhere" / "system_principals.yaml"
    monkeypatch.setenv(access.PRINCIPALS_FILE_ENV, str(explicit))

    with pytest.raises(FileNotFoundError):
        access.load_access_registration()
    assert not explicit.exists()
