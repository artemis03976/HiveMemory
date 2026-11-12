"""pi 扩展与 HiveMemory 一侧声明的一致性（扩展代码是 JavaScript，按文本读取常量）。

保护的契约：扩展认识的记忆工具就是 MCP server 的四个工具（这份回声规则在
``node/core``，dsh 插件同样使用），作用对象参数与轮次信息参数是这些工具的参数，
归一化得到的动词是导入接受的动词，写出的事件种类是
导入接受的种类；扩展声明的 principal、缺省 Agent 与凭据变量和插件服务安装时登记、
交付的一致，来源标识符合导入的格式；pi 包清单指向的扩展入口与 skill 目录存在。扩展的轮次投影、回流
与工具行为由扩展自己的测试守护（``node --test``）。
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import get_args

from hivememory.i18n import Language
from hivememory.mcp_server.server import create_mcp_server
from hivememory.system.plugins import SKILL_PATH
from hivememory.system.plugins.harnesses import harness_by_id
from hivememory.system.services.import_bus.models import (
    ImportedEventKind,
    ImportedVerb,
    InFlightTurn,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "integrations" / "hivememory"
NODE_ROOT = PACKAGE_ROOT / "node"


def _source(relative: str) -> str:
    return (NODE_ROOT / relative).read_text(encoding="utf-8")


def _string_constant(source: str, name: str) -> str:
    match = re.search(rf"\b{name}(?: =|:) '([^']+)'", source)
    assert match is not None, f"constant {name} not found"
    return match.group(1)


def _memory_tools() -> dict[str, dict[str, str]]:
    """``MEMORY_TOOLS`` 的每一项：``name: { verb: 'X', targetParam: 'y', ... }``。"""
    block = re.search(
        r"export const MEMORY_TOOLS = \{(.*?)\n\}", _source("core/memory-tools.js"), re.S
    )
    assert block is not None
    return {
        name: dict(re.findall(r"(\w+): '([^']+)'", fields))
        | dict(re.findall(r"(\w+): (true)", fields))
        for name, fields in re.findall(r"(\w+): \{([^}]*)\}", block.group(1))
    }


def _server_tool_parameters() -> dict[str, set[str]]:
    tools = asyncio.run(create_mcp_server(lambda: None, Language.ZH).list_tools())
    return {tool.name: set(tool.input_schema["properties"]) for tool in tools}


def test_extension_memory_tools_match_the_mcp_server():
    """MCP server 的工具或参数改名时这里失败：回声标记与轮次信息会静默失效。"""
    extension_tools = _memory_tools()
    turn_param = _string_constant(_source("core/memory-tools.js"), "TURN_PARAM")
    server_tools = _server_tool_parameters()

    assert set(extension_tools) == set(server_tools)
    for name, fields in extension_tools.items():
        assert fields["verb"] in get_args(ImportedVerb), name
        if "targetParam" in fields:
            assert fields["targetParam"] in server_tools[name], name
        assert (turn_param in server_tools[name]) == ("needsTurn" in fields), name


def test_extension_event_kinds_are_accepted_by_import():
    """导入的事件种类改名或删除时这里失败：扩展写出的轮次会被整轮以 422 拒绝并跳过。"""
    kinds = set(re.findall(r"addEvent\('(\w+)'", _source("pi/turns.js")))

    assert kinds == {"assistant_message", "thought", "tool_call"}
    assert kinds <= set(get_args(ImportedEventKind))


def test_extension_identity_is_registered_for_mcp_and_import():
    """扩展的 principal、缺省 Agent 与凭据变量和插件服务登记的一致，来源标识可以导入。"""
    index = _source("pi/index.js")
    principal_id = _string_constant(index, "principalId")
    agent_id = _string_constant(index, "defaultAgentId")
    token_env = _string_constant(index, "tokenEnv")
    source = _string_constant(_source("pi/turns.js"), "SOURCE")
    installed = harness_by_id("pi")
    assert installed.principal_id == principal_id
    assert installed.default_agent_id == agent_id
    assert installed.credential_env == token_env
    InFlightTurn(source=source, external_conversation_id="s", external_turn_id="t", user_message="")


def test_pi_package_points_at_the_extension_and_the_shared_skill():
    """包清单中的路径失效时这里失败：pi 安装后既不加载扩展，也找不到使用手册。"""
    manifest = json.loads((PACKAGE_ROOT / "package.json").read_text("utf-8"))

    for extension in manifest["pi"]["extensions"]:
        assert (PACKAGE_ROOT / extension).is_file(), extension
    for skills in manifest["pi"]["skills"]:
        assert (
            PACKAGE_ROOT / skills / "hivememory" / "SKILL.md" == PACKAGE_ROOT / SKILL_PATH
        ), skills
