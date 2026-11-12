"""DeepSeek Harness 插件与 HiveMemory 一侧声明的一致性（插件代码是 JavaScript，按文本读取常量）。

插件与 pi 扩展同在 ``integrations/hivememory/``，共用 ``node/core`` 中的记忆工具回声
规则，它与 MCP server 的一致由 pi 扩展的同类测试守护。这里保护 dsh 自己的契约：写出的
事件种类是导入接受的种类；插件声明的 principal、缺省 Agent 与凭据变量和插件服务安装时
登记、交付的一致，来源标识符合导入的格式；包清单的入口与 bundle 配置存在，补丁指向的使用手册 skill 目录存在。插件的
轮次投影、回流与工具行为由插件自己的测试守护（``node --test``）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import get_args

from hivememory.system.plugins import SKILL_PATH
from hivememory.system.plugins.harnesses import harness_by_id
from hivememory.system.services.import_bus.models import ImportedEventKind, InFlightTurn

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "integrations" / "hivememory"
DSH_ROOT = PACKAGE_ROOT / "node" / "dsh"


def _source(name: str) -> str:
    return (DSH_ROOT / name).read_text(encoding="utf-8")


def _string_constant(source: str, name: str) -> str:
    match = re.search(rf"\b{name}(?: =|:) '([^']+)'", source)
    assert match is not None, f"constant {name} not found"
    return match.group(1)


def test_plugin_event_kinds_are_accepted_by_import():
    """导入的事件种类改名或删除时这里失败：插件写出的轮次会被整轮以 422 拒绝并跳过。"""
    kinds = set(re.findall(r"addEvent\('(\w+)'", _source("turns.js")))

    assert kinds == {"assistant_message", "thought", "subagent_message", "tool_call"}
    assert kinds <= set(get_args(ImportedEventKind))


def test_plugin_identity_is_registered_for_mcp_and_import():
    """插件的 principal、缺省 Agent 与凭据变量和插件服务登记的一致，来源标识可以导入。"""
    index = _source("index.js")
    principal_id = _string_constant(index, "principalId")
    agent_id = _string_constant(index, "defaultAgentId")
    token_env = _string_constant(index, "tokenEnv")
    source = _string_constant(_source("turns.js"), "SOURCE")
    installed = harness_by_id("dsh")
    assert installed.principal_id == principal_id
    assert installed.default_agent_id == agent_id
    assert installed.credential_env == token_env
    InFlightTurn(source=source, external_conversation_id="s", external_turn_id="1", user_message="")


def test_package_points_at_the_dsh_entry_and_bundle_patch():
    """包清单或补丁中的路径失效时这里失败：dsh 安装后加载不到插件、插件行不进入配置，
    或补丁按包名解析出的 skill 目录不存在，使用手册 skill 不出现在目录中。"""
    manifest = json.loads((PACKAGE_ROOT / "package.json").read_text("utf-8"))
    patch = (DSH_ROOT / "cordis.patch.yml").read_text("utf-8")

    assert manifest["name"] == "hivememory-connector"
    assert (PACKAGE_ROOT / manifest["main"]).is_file()
    assert (PACKAGE_ROOT / manifest["dsh"]["bundle"]["patch"]).is_file()
    assert manifest["engines"]["dsh"]
    assert f"name: {manifest['name']}" in patch
    skill_root = re.search(r"resolve\('([^']+)/package\.json'\)\), '([^']+)'\)", patch)
    assert skill_root is not None
    assert skill_root.group(1) == manifest["name"]
    assert (
        PACKAGE_ROOT / skill_root.group(2) / "hivememory" / "SKILL.md" == PACKAGE_ROOT / SKILL_PATH
    )
