"""DeepSeek Harness 插件与 HiveMemory 一侧声明的一致性（插件代码是 JavaScript，按文本读取常量）。

保护的契约：插件认识的记忆工具就是 MCP server 的四个工具，作用对象参数与轮次
信息参数是这些工具的参数，归一化得到的动词是导入接受的动词；插件声明的
principal 在随仓库发布的接入登记中、开放 mcp 与 import 并要求凭据，默认 Agent
在 Workspace 访问登记中持有四项记忆操作，来源标识符合导入的格式。插件的轮次
投影、回流与工具行为由插件自己的测试守护（``node --test``）。
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import get_args

import yaml

from hivememory.config.access import SystemPrincipalsFile, WorkspaceActorsFile
from hivememory.mcp_server.server import create_mcp_server
from hivememory.system.services.import_bus.models import ImportedVerb, InFlightTurn

REPO_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_LIB = REPO_ROOT / "integrations" / "deepseek-harness" / "lib"

MEMORY_OPERATIONS = {"memory.read", "memory.search", "memory.write", "memory.update"}


def _string_constant(source: str, name: str) -> str:
    match = re.search(rf"const {name} = '([^']+)'", source)
    assert match is not None, f"constant {name} not found"
    return match.group(1)


def _memory_tools() -> dict[str, dict[str, str]]:
    """``MEMORY_TOOLS`` 的每一项：``name: { verb: 'X', targetParam: 'y', ... }``。"""
    source = (PLUGIN_LIB / "turns.js").read_text(encoding="utf-8")
    block = re.search(r"export const MEMORY_TOOLS = \{(.*?)\n\}", source, re.S)
    assert block is not None
    return {
        name: dict(re.findall(r"(\w+): '([^']+)'", fields))
        | dict(re.findall(r"(\w+): (true)", fields))
        for name, fields in re.findall(r"(\w+): \{([^}]*)\}", block.group(1))
    }


def _server_tool_parameters() -> dict[str, set[str]]:
    tools = asyncio.run(create_mcp_server(lambda: None).list_tools())
    return {tool.name: set(tool.input_schema["properties"]) for tool in tools}


def test_plugin_memory_tools_match_the_mcp_server():
    """MCP server 的工具或参数改名时这里失败：回声标记与轮次信息会静默失效。"""
    plugin_tools = _memory_tools()
    turn_param = _string_constant(
        (PLUGIN_LIB / "turns.js").read_text(encoding="utf-8"), "TURN_PARAM"
    )
    server_tools = _server_tool_parameters()

    assert set(plugin_tools) == set(server_tools)
    for name, fields in plugin_tools.items():
        assert fields["verb"] in get_args(ImportedVerb), name
        if "targetParam" in fields:
            assert fields["targetParam"] in server_tools[name], name
        assert (turn_param in server_tools[name]) == ("needsTurn" in fields), name


def test_plugin_identity_is_registered_for_mcp_and_import():
    """插件的 principal 与默认 Agent 在随仓库发布的登记中，来源标识可以导入。"""
    config = (PLUGIN_LIB / "config.js").read_text(encoding="utf-8")
    principal_id = _string_constant(config, "PRINCIPAL_ID")
    agent_id = _string_constant(config, "DEFAULT_AGENT_ID")
    source = _string_constant(config, "SOURCE")
    principals = SystemPrincipalsFile.model_validate(
        yaml.safe_load((REPO_ROOT / "configs" / "system_principals.yaml").read_text("utf-8"))
    )
    actors = WorkspaceActorsFile.model_validate(
        yaml.safe_load((REPO_ROOT / "configs" / "workspace_actors.yaml").read_text("utf-8"))
    )

    [principal] = [entry for entry in principals.principals if entry.principal_id == principal_id]
    [actor] = [entry for entry in actors.workspace_actors if entry.agent_id == agent_id]
    assert principal.enabled and not principal.allow_system_actor
    assert set(principal.adapters) == {"mcp", "import"}
    assert principal.credential_env
    assert actor.enabled and set(actor.allowed_operations) == MEMORY_OPERATIONS
    InFlightTurn(source=source, external_conversation_id="s", external_turn_id="1", user_message="")
