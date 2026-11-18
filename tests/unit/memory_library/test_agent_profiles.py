"""AgentProfileStore：一个 Profile 一个文件的读写往返、同一套校验、原子保存与热重载。"""

from __future__ import annotations

import os

import pytest

from hivememory.core.errors import InvalidAgentProfileError
from hivememory.core.models import AgentProfile
from hivememory.memory_library.agent_profiles import AgentProfileStore


@pytest.fixture
def store(tmp_path):
    return AgentProfileStore(tmp_path)


def _write(tmp_path, name: str, text: str) -> None:
    (tmp_path / name).write_text(text, encoding="utf-8")


def test_saved_profile_reads_back_equal(store):
    """frontmatter 中带冒号的标题、为空的采样参数与多行人设都原样读回。"""
    profile = AgentProfile(
        agent_id="coder_doll",
        title="Python 开发者: Coder-Doll",
        summary="擅长编写与调试 Python 脚本",
        persona="你是一个资深的 Python 程序员。\n核心原则：类型提示。",
        model="deepseek-chat",
        temperature=0.2,
        enabled=False,
    )

    saved = store.save(profile)

    assert saved.profile == profile
    assert store.get("coder_doll").profile == profile
    assert [stored.profile for stored in store.list()] == [profile]


def test_hand_written_file_with_bom_and_crlf_is_read(store, tmp_path):
    """Windows 编辑器写入的 BOM 与 CRLF 不影响读取。"""
    text = "---\r\ntitle: 审查员\r\nenabled: true\r\n---\r\n\r\n严格审查。\r\n"
    (tmp_path / "reviewer.md").write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))

    profile = store.get("reviewer").profile

    assert (profile.title, profile.persona, profile.model) == ("审查员", "严格审查。", "default")


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("没有 frontmatter 的人设", "missing_frontmatter"),
        ("---\n- title\n---\n人设", "invalid_frontmatter"),
        ("---\ntitle: x\nlanguage: zh\n---\n人设", "unknown_field"),
        ('---\ntemperature: "0.3"\n---\n人设', "invalid_value"),
        ("---\ntop_p: 1.5\n---\n人设", "invalid_value"),
    ],
)
def test_invalid_file_is_skipped_when_listing_and_rejected_when_read(store, tmp_path, text, reason):
    _write(tmp_path, "broken.md", text)
    _write(tmp_path, "valid.md", "---\ntitle: 可用\n---\n")

    assert [stored.profile.agent_id for stored in store.list()] == ["valid"]
    with pytest.raises(InvalidAgentProfileError) as exc_info:
        store.get("broken")
    assert exc_info.value.details["reason"] == reason


@pytest.mark.parametrize("name", ["Coder.md", "omni_doll.md", "system.md"])
def test_file_whose_name_is_not_a_valid_agent_id_is_skipped(store, tmp_path, name):
    """文件名就是 agent_id：大写、内置 Profile 与保留的 system 都不能用作文件名。"""
    _write(tmp_path, name, "---\ntitle: x\n---\n")

    assert store.list() == []


@pytest.mark.parametrize("agent_id", ["Coder", "../escape", "omni_doll", ""])
def test_save_rejects_invalid_agent_id_without_writing(store, tmp_path, agent_id):
    with pytest.raises(InvalidAgentProfileError) as exc_info:
        store.save(AgentProfile(agent_id=agent_id, title="x"))

    assert exc_info.value.details["reason"] == "invalid_agent_id"
    assert not any(tmp_path.rglob("*"))


def test_failed_replace_leaves_original_file_unchanged(store, tmp_path, monkeypatch):
    store.save(AgentProfile(agent_id="coder_doll", title="原来的"))
    path = tmp_path / "coder_doll.md"
    original = path.read_bytes()

    def fail_replace(src, dst):
        raise OSError("磁盘已满")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError):
        store.save(AgentProfile(agent_id="coder_doll", title="新的"))

    assert path.read_bytes() == original
    assert sorted(p.name for p in path.parent.iterdir()) == ["coder_doll.md"]


def test_direct_file_changes_show_on_next_read(store, tmp_path):
    """没有缓存：文件被直接修改、新增或删除后，下一次读取与列出就反映出来。"""
    store.save(AgentProfile(agent_id="coder_doll", title="原来的"))
    assert store.get("coder_doll").profile.title == "原来的"

    _write(tmp_path, "coder_doll.md", "---\ntitle: 手改的\nenabled: false\n---\n新人设")
    _write(tmp_path, "reviewer.md", "---\ntitle: 新增的\n---\n")

    changed = store.get("coder_doll").profile
    assert (changed.title, changed.enabled, changed.persona) == ("手改的", False, "新人设")
    assert [stored.profile.agent_id for stored in store.list()] == [
        "coder_doll",
        "reviewer",
    ]

    (tmp_path / "coder_doll.md").unlink()
    assert store.get("coder_doll") is None
    assert [stored.profile.agent_id for stored in store.list()] == ["reviewer"]


def test_delete_reports_whether_a_profile_was_removed(store):
    store.save(AgentProfile(agent_id="coder_doll", title="x"))

    assert store.delete("coder_doll") is True
    assert store.get("coder_doll") is None
    assert store.delete("coder_doll") is False
