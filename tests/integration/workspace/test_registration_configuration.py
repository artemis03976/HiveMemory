"""真实登记文件装载与组合根访问装配的配置迁移验收。"""

import pytest

from hivememory.config.app import HiveMemoryConfig
from hivememory.system.system import HiveMemorySystem


def test_removed_operation_fails_during_system_assembly(tmp_path, monkeypatch):
    """旧版 operation 不能继续装载，错误明确指出未知值与访问坐标。"""
    operation = "resource.read"
    path = tmp_path / "workspace_actors.yaml"
    path.write_text(
        "workspace_actors:\n"
        '  - owner_user_id: "default"\n'
        '    workspace_id: "main_workspace"\n'
        '    user_id: "default"\n'
        f'    allowed_operations: ["{operation}"]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("HIVEMEMORY_WORKSPACE_ACTORS_PATH", str(path))
    with pytest.raises(ValueError, match="访问登记包含未知 operation") as error:
        HiveMemorySystem.build(config=HiveMemoryConfig())
    assert repr(operation) in str(error.value)
    assert "(default, main_workspace, default, None)" in str(error.value)


def test_system_operation_whitelist_fails_during_system_assembly(tmp_path, monkeypatch):
    """配置文件中的 system 记录只能声明准入，不能配置 actor 白名单。"""
    path = tmp_path / "workspace_actors.yaml"
    path.write_text(
        "workspace_actors:\n"
        '  - owner_user_id: "default"\n'
        '    workspace_id: "main_workspace"\n'
        '    user_id: "default"\n'
        '    agent_id: "system"\n'
        '    allowed_operations: ["memory.read"]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("HIVEMEMORY_WORKSPACE_ACTORS_PATH", str(path))
    with pytest.raises(ValueError, match="保留 system Actor 的访问记录不得配置 operation"):
        HiveMemorySystem.build(config=HiveMemoryConfig())
