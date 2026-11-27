"""各 harness 的插件命令：重复安装改为更新，指向别处的旧安装先移除，失败带上原因。

harness 的命令行换成替身，按命令返回预置输出；断言发给 harness 的命令序列，这是
插件服务与 harness 之间的契约。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hivememory.system.plugins.harnesses import (
    CLAUDE_CODE_TOOL_PREFIX,
    INTEGRATIONS_DIR,
    PLUGIN_ID,
    CommandResult,
    HarnessSettingsError,
    PluginCommandError,
    PluginHooksError,
    claude_code_allow_rules,
    harness_by_id,
)

#: 插件的 PreToolUse hook 改写参数的三个工具，按 hooks.json 中 matcher 的顺序。
_RULES = tuple(
    f"{CLAUDE_CODE_TOOL_PREFIX}{tool}" for tool in ("memory_write", "memory_update", "memory_read")
)


class _Harness:
    """按完整命令返回预置的标准输出，记录收到的命令；未预置的命令成功、输出为空。"""

    def __init__(self, outputs: dict[tuple[str, ...], str] | None = None) -> None:
        self.outputs = outputs or {}
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv):
        argv = tuple(argv)
        self.calls.append(argv)
        return CommandResult(argv, 0, self.outputs.get(argv, ""), "")


def _plugin_dir(root: Path) -> Path:
    plugin = root / "hivememory"
    plugin.mkdir(parents=True)
    (plugin / "package.json").write_text(json.dumps({"name": "hivememory-connector"}), "utf-8")
    return plugin


def _integrations(root: Path, *, matcher: str | None = None) -> Path:
    """插件目录 ``integrations/``，带插件的 hooks.json：缺省照搬仓库中的那一份，给出
    ``matcher`` 时只有一组以它为 matcher 的 PreToolUse。"""
    integrations = root / "integrations"
    hooks = integrations / "hivememory" / "hooks" / "hooks.json"
    hooks.parent.mkdir(parents=True)
    if matcher is None:
        real = INTEGRATIONS_DIR / "hivememory" / "hooks" / "hooks.json"
        hooks.write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        declared = {"hooks": {"PreToolUse": [{"matcher": matcher, "hooks": []}]}}
        hooks.write_text(json.dumps(declared), encoding="utf-8")
    return integrations


def test_claude_code_repoints_a_marketplace_from_elsewhere_and_updates_an_installed_plugin(
    tmp_path,
):
    """marketplace 指向别处（例如 checkout）时先移除再添加；插件已安装时更新而不是重装。"""
    integrations = _integrations(tmp_path)
    run = _Harness(
        {
            ("claude", "plugin", "marketplace", "list", "--json"): json.dumps(
                [{"name": "hivememory-local", "path": str(tmp_path / "checkout" / "integrations")}]
            ),
            ("claude", "plugin", "list", "--json"): json.dumps([{"id": PLUGIN_ID}]),
        }
    )

    harness_by_id("claude-code").install(run, "claude", integrations)

    assert run.calls[1:] == [
        ("claude", "plugin", "marketplace", "remove", "hivememory-local"),
        ("claude", "plugin", "marketplace", "add", str(integrations)),
        ("claude", "plugin", "list", "--json"),
        ("claude", "plugin", "update", PLUGIN_ID),
    ]


def _claude() -> _Harness:
    """没有装过 marketplace 与插件的 Claude Code。"""
    return _Harness(
        {
            ("claude", "plugin", "marketplace", "list", "--json"): "[]",
            ("claude", "plugin", "list", "--json"): "[]",
        }
    )


def _claude_code_settings(tmp_path, monkeypatch, settings=None):
    """Claude Code 的用户设置放在 ``CLAUDE_CONFIG_DIR`` 指定的目录，按需写入初始内容。"""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    path = tmp_path / "claude" / "settings.json"
    if settings is not None:
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(settings), encoding="utf-8")
    return path


def test_claude_code_allow_rules_are_the_tools_the_plugin_hook_rewrites():
    """要放行的工具取自插件 hooks.json 中 PreToolUse 的 matcher：写入、更新与读取。"""
    assert set(claude_code_allow_rules(INTEGRATIONS_DIR)) == set(_RULES)


def test_claude_code_install_allows_the_hooked_tools_and_uninstall_removes_them(
    tmp_path, monkeypatch
):
    """安装把 hook 匹配的工具的规则并入用户设置，其他设置与规则原样保留，重复安装不重复写入；
    卸载只删本插件的规则，设置回到安装之前。"""
    original = {
        "model": "opus",
        "permissions": {"allow": ["Bash(git status)"], "deny": ["Read(.env)"]},
    }
    path = _claude_code_settings(tmp_path, monkeypatch, original)
    integrations = _integrations(tmp_path)
    claude = harness_by_id("claude-code")

    claude.install(_claude(), "claude", integrations)
    claude.install(_claude(), "claude", integrations)
    installed = json.loads(path.read_text(encoding="utf-8"))
    claude.uninstall(_claude(), "claude", integrations)

    assert installed["model"] == "opus"
    assert installed["permissions"] == {
        "allow": ["Bash(git status)", *_RULES],
        "deny": ["Read(.env)"],
    }
    assert json.loads(path.read_text(encoding="utf-8")) == original


def test_claude_code_install_drops_a_plugin_rule_the_hook_no_longer_matches(tmp_path, monkeypatch):
    """安装后本插件的规则恰好是 hook 匹配的工具：不再匹配的删除，已有的留在原位，缺少的追加。"""
    stale = f"{CLAUDE_CODE_TOOL_PREFIX}memory_search"
    read = f"{CLAUDE_CODE_TOOL_PREFIX}memory_read"
    path = _claude_code_settings(
        tmp_path, monkeypatch, {"permissions": {"allow": ["Bash(git status)", stale, read]}}
    )

    harness_by_id("claude-code").install(_claude(), "claude", _integrations(tmp_path))

    assert json.loads(path.read_text(encoding="utf-8"))["permissions"]["allow"] == [
        "Bash(git status)",
        read,
        f"{CLAUDE_CODE_TOOL_PREFIX}memory_write",
        f"{CLAUDE_CODE_TOOL_PREFIX}memory_update",
    ]


def test_claude_code_uninstall_removes_every_plugin_rule_and_keeps_others(tmp_path, monkeypatch):
    """卸载删除所有放行本插件工具的规则，包括 hook 不匹配的；其他 MCP server 与工具的规则保留。"""
    path = _claude_code_settings(
        tmp_path,
        monkeypatch,
        {
            "permissions": {
                "allow": [
                    "Bash(git status)",
                    f"{CLAUDE_CODE_TOOL_PREFIX}memory_search",
                    *_RULES,
                    "mcp__other__memory_write",
                ]
            }
        },
    )

    harness_by_id("claude-code").uninstall(_claude(), "claude", tmp_path)

    assert json.loads(path.read_text(encoding="utf-8"))["permissions"]["allow"] == [
        "Bash(git status)",
        "mcp__other__memory_write",
    ]


def test_claude_code_uninstall_drops_the_permissions_it_created(tmp_path, monkeypatch):
    """没有设置文件时安装新建它；卸载后只剩空的权限设置时删掉，不留下空的 allow。"""
    path = _claude_code_settings(tmp_path, monkeypatch)
    integrations = _integrations(tmp_path)
    claude = harness_by_id("claude-code")

    claude.install(_claude(), "claude", integrations)
    created = json.loads(path.read_text(encoding="utf-8"))
    claude.uninstall(_claude(), "claude", integrations)

    assert created == {"permissions": {"allow": list(_RULES)}}
    assert json.loads(path.read_text(encoding="utf-8")) == {}


@pytest.mark.parametrize(
    "matcher",
    [f"{CLAUDE_CODE_TOOL_PREFIX}memory_write|mcp__other__memory_write", None],
    ids=["foreign-tool", "missing-hooks"],
)
def test_claude_code_install_rejects_unusable_hooks_before_any_command(
    tmp_path, monkeypatch, matcher
):
    """hook 匹配了不是本插件的工具、或 hooks.json 读不到时，以 PluginHooksError 拒绝：
    不运行任何插件命令，也不写任何放行规则。"""
    path = _claude_code_settings(tmp_path, monkeypatch)
    integrations = tmp_path / "integrations"
    if matcher is not None:
        _integrations(tmp_path, matcher=matcher)
    run = _claude()

    with pytest.raises(PluginHooksError):
        harness_by_id("claude-code").install(run, "claude", integrations)

    assert run.calls == []
    assert not path.exists()


def test_claude_code_leaves_an_unreadable_settings_file_untouched(tmp_path, monkeypatch):
    """设置文件不是 JSON 对象时不改动它，以 HarnessSettingsError 交回文件位置。"""
    path = _claude_code_settings(tmp_path, monkeypatch)
    path.parent.mkdir(parents=True)
    path.write_text('{"permissions": {"allow": [', encoding="utf-8")

    with pytest.raises(HarnessSettingsError) as unreadable:
        harness_by_id("claude-code").install(_claude(), "claude", _integrations(tmp_path))

    assert unreadable.value.path == path
    assert path.read_text(encoding="utf-8") == '{"permissions": {"allow": ['


def test_codex_keeps_a_marketplace_already_pointing_here(tmp_path):
    """marketplace 已指向这里时不再添加，直接把插件复制进 Codex 的缓存。"""
    integrations = tmp_path / "integrations"
    run = _Harness(
        {
            ("codex", "plugin", "marketplace", "list", "--json"): json.dumps(
                {"marketplaces": [{"name": "hivememory-local", "root": str(integrations)}]}
            )
        }
    )

    harness_by_id("codex").install(run, "codex", integrations)

    assert run.calls[1:] == [("codex", "plugin", "add", PLUGIN_ID)]


def test_pi_removes_a_hivememory_plugin_installed_from_elsewhere(tmp_path):
    """pi 中另一处的 HiveMemory 插件先移除，避免同名工具注册两次；其他包不动。"""
    integrations = tmp_path / "integrations"
    here = _plugin_dir(integrations)
    checkout = _plugin_dir(tmp_path / "checkout" / "integrations")
    other = tmp_path / "other-extension"
    other.mkdir()
    listing = f"User packages:\n  {checkout}\n    {checkout}\n  {other}\n    {other}\n"
    run = _Harness({("pi", "list"): listing})

    harness_by_id("pi").install(run, "pi", integrations)

    assert run.calls[1:] == [("pi", "remove", str(checkout)), ("pi", "install", str(here))]


def test_failure_carries_the_reason_and_the_failed_command(tmp_path):
    """非零退出与超时分别以 command_failed、command_timeout 交回，带上失败的那条命令。"""

    def failing(argv):
        return CommandResult(tuple(argv), 1, "", "profile is in use")

    def timing_out(argv):
        return CommandResult(tuple(argv), None, "", "")

    with pytest.raises(PluginCommandError) as failed:
        harness_by_id("dsh").install(failing, "dsh.cmd", tmp_path)
    with pytest.raises(PluginCommandError) as timed_out:
        harness_by_id("dsh").install(timing_out, "dsh.cmd", tmp_path)

    assert failed.value.reason == "command_failed"
    assert failed.value.result.argv[:4] == ("dsh.cmd", "plugin", "--profile", "desktop")
    assert timed_out.value.reason == "command_timeout"
