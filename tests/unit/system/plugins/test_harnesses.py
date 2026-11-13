"""各 harness 的插件命令：重复安装改为更新，指向别处的旧安装先移除，失败带上原因。

harness 的命令行换成替身，按命令返回预置输出；断言发给 harness 的命令序列，这是
插件服务与 harness 之间的契约。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hivememory.system.plugins.harnesses import (
    CLAUDE_CODE_ALLOW_RULES,
    PLUGIN_ID,
    CommandResult,
    HarnessSettingsError,
    PluginCommandError,
    harness_by_id,
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


def test_claude_code_repoints_a_marketplace_from_elsewhere_and_updates_an_installed_plugin(
    tmp_path,
):
    """marketplace 指向别处（例如 checkout）时先移除再添加；插件已安装时更新而不是重装。"""
    integrations = tmp_path / "integrations"
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


def test_claude_code_install_allows_the_two_rewritten_tools_and_uninstall_removes_them(
    tmp_path, monkeypatch
):
    """安装把写入与更新两条规则并入用户设置，其他设置与规则原样保留，重复安装不重复写入；
    卸载只删这两条，空出来的 allow 一并删除。"""
    original = {
        "model": "opus",
        "permissions": {"allow": ["Bash(git status)"], "deny": ["Read(.env)"]},
    }
    path = _claude_code_settings(tmp_path, monkeypatch, original)
    integrations = tmp_path / "integrations"
    claude = harness_by_id("claude-code")

    claude.install(_claude(), "claude", integrations)
    claude.install(_claude(), "claude", integrations)
    installed = json.loads(path.read_text(encoding="utf-8"))
    claude.uninstall(_claude(), "claude", integrations)

    assert installed["model"] == "opus"
    assert installed["permissions"] == {
        "allow": ["Bash(git status)", *CLAUDE_CODE_ALLOW_RULES],
        "deny": ["Read(.env)"],
    }
    assert json.loads(path.read_text(encoding="utf-8")) == original


def test_claude_code_uninstall_drops_the_permissions_it_created(tmp_path, monkeypatch):
    """没有设置文件时安装新建它；卸载后只剩空的权限设置时删掉，不留下空的 allow。"""
    path = _claude_code_settings(tmp_path, monkeypatch)
    claude = harness_by_id("claude-code")

    claude.install(_claude(), "claude", tmp_path)
    created = json.loads(path.read_text(encoding="utf-8"))
    claude.uninstall(_claude(), "claude", tmp_path)

    assert created == {"permissions": {"allow": list(CLAUDE_CODE_ALLOW_RULES)}}
    assert json.loads(path.read_text(encoding="utf-8")) == {}


def test_claude_code_leaves_an_unreadable_settings_file_untouched(tmp_path, monkeypatch):
    """设置文件不是 JSON 对象时不改动它，以 HarnessSettingsError 交回文件位置。"""
    path = _claude_code_settings(tmp_path, monkeypatch)
    path.parent.mkdir(parents=True)
    path.write_text('{"permissions": {"allow": [', encoding="utf-8")

    with pytest.raises(HarnessSettingsError) as unreadable:
        harness_by_id("claude-code").install(_claude(), "claude", tmp_path)

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
