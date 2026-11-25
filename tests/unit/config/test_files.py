"""关闭时写盘、启动时读回的数据文件：条目列表的写出，以及读回即删除、读不回时整份改存。"""

from __future__ import annotations

import pytest

from hivememory.config.files import UnreadableFileError, take_json_items, write_json_items


def test_written_items_read_back_and_the_file_is_consumed(tmp_path):
    path = tmp_path / "saved.json"
    write_json_items(path, format_version=1, key="work", items=[{"id": 1}, {"id": 2}])

    items = take_json_items(path, format_version=1, key="work")

    assert (items, path.exists()) == ([{"id": 1}, {"id": 2}], False)


def test_missing_file_reads_back_as_none(tmp_path):
    assert take_json_items(tmp_path / "saved.json", format_version=1, key="work") is None


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        '["not", "an", "object"]',
        '{"format_version": 2, "work": []}',
        '{"format_version": 1, "topics": []}',
    ],
    ids=["unparseable", "not_an_object", "unknown_version", "no_items_list"],
)
def test_unreadable_file_is_kept_aside_whole(tmp_path, content):
    """读不回的文件整份改名为同目录的 rejected 文件，原样保留供排查。"""
    path = tmp_path / "saved.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(UnreadableFileError) as raised:
        take_json_items(path, format_version=1, key="work")

    rejected = raised.value.rejected_path
    assert (rejected.parent, rejected.name.startswith("saved.rejected-")) == (tmp_path, True)
    assert (rejected.read_text(encoding="utf-8"), path.exists()) == (content, False)


def _decode(item: dict[str, object]) -> str:
    return str(item["name"])


def test_decoded_items_read_back_as_the_decoded_values(tmp_path):
    path = tmp_path / "saved.json"
    write_json_items(path, format_version=1, key="tasks", items=[{"name": "a"}, {"name": "b"}])

    assert take_json_items(path, format_version=1, key="tasks", decode=_decode) == ["a", "b"]


def test_one_item_that_cannot_be_decoded_keeps_the_whole_file_aside(tmp_path):
    """条目在删除文件之前解码：任一条目读不回，整份文件改存 rejected 文件，读得回的也不交出。"""
    path = tmp_path / "saved.json"
    items = [{"name": "a"}, {"missing": "name"}]
    write_json_items(path, format_version=1, key="tasks", items=items)
    original = path.read_text(encoding="utf-8")

    with pytest.raises(UnreadableFileError) as raised:
        take_json_items(path, format_version=1, key="tasks", decode=_decode)

    assert (raised.value.rejected_path.read_text(encoding="utf-8"), path.exists()) == (
        original,
        False,
    )
