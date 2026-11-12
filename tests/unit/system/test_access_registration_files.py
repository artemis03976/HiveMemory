"""访问登记文件的装载：生效文件是本机的个人配置，在数据根下；默认路径缺失时从随程序
发布的模板生成，已有的生效文件不被模板覆盖；环境变量指定的路径缺失时显式失败。随程序
发布的模板不预先接入任何外部 harness。"""

from pathlib import Path

import pytest
import yaml

from hivememory.config import access
from hivememory.config.access import SystemPrincipalsFile, WorkspaceActorsFile
from hivememory.config.files import DATA_ROOT_ENV, SHIPPED_CONFIGS_DIR

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
def shipped_dir(tmp_path, monkeypatch) -> Path:
    """以临时目录代替随程序发布的 configs 目录，只放两份模板。"""
    shipped = tmp_path / "program" / "configs"
    shipped.mkdir(parents=True)
    (shipped / "system_principals.example.yaml").write_text(_PRINCIPALS_TEMPLATE, encoding="utf-8")
    (shipped / "workspace_actors.example.yaml").write_text(_ACTORS_TEMPLATE, encoding="utf-8")
    monkeypatch.setattr(access, "SHIPPED_CONFIGS_DIR", shipped)
    return shipped


@pytest.fixture
def configs_dir(tmp_path, monkeypatch, shipped_dir) -> Path:
    """数据根下的个人配置目录；数据根与程序目录分开。"""
    data_root = tmp_path / "data-root"
    monkeypatch.setenv(DATA_ROOT_ENV, str(data_root))
    monkeypatch.delenv(access.PRINCIPALS_FILE_ENV, raising=False)
    monkeypatch.delenv(access.WORKSPACE_ACTORS_FILE_ENV, raising=False)
    return data_root / "configs"


def test_missing_registration_is_generated_from_the_template(configs_dir, shipped_dir):
    """生效文件不存在时按模板原样生成在数据根下（保留注释），装载的就是模板中的登记；
    程序目录只有模板。"""
    registration = access.load_access_registration()

    assert (configs_dir / "system_principals.yaml").read_text("utf-8") == _PRINCIPALS_TEMPLATE
    assert (configs_dir / "workspace_actors.yaml").read_text("utf-8") == _ACTORS_TEMPLATE
    assert [p.principal_id for p in registration.principals.principals] == [
        "hivememory:http-server"
    ]
    assert registration.principals_digest == access.file_digest(
        configs_dir / "system_principals.yaml"
    )
    assert sorted(path.name for path in shipped_dir.iterdir()) == [
        "system_principals.example.yaml",
        "workspace_actors.example.yaml",
    ]


def test_existing_registration_is_not_overwritten_by_the_template(configs_dir):
    """已有的生效文件是本机的个人配置，模板不覆盖它。"""
    local = _PRINCIPALS_TEMPLATE.replace("hivememory:http-server", "local-process:mine")
    configs_dir.mkdir(parents=True)
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


def test_shipped_templates_connect_no_external_harness():
    """新部署没有外部接入：模板只有进程内来源，Agent 只有用户级记录与保留的 system。

    外部 harness 的来源与缺省 Agent 的记录在插件页安装插件时才产生。
    """
    principals = SystemPrincipalsFile.model_validate(
        yaml.safe_load((SHIPPED_CONFIGS_DIR / "system_principals.example.yaml").read_text("utf-8"))
    )
    actors = WorkspaceActorsFile.model_validate(
        yaml.safe_load((SHIPPED_CONFIGS_DIR / "workspace_actors.example.yaml").read_text("utf-8"))
    )

    assert {set(p.adapters) <= {"local", "http"} for p in principals.principals} == {True}
    assert {a.agent_id for a in actors.workspace_actors} == {None, "system"}
