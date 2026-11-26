"""ShortTermMemoryStore 的写入方法、日志重放与启动时读回的单元测试。

被测对象：短期存储在真实的文件端口（临时目录）上写日志、维护读取值。
保护的契约：
- 一串写入之后，另一个存储从同一目录重放日志得到的读取值与写入时内存中的相等；
- 追加失败或构造校验失败时什么都不写，内存中的读取值不变；
- 读取与写入按 Workspace 归属隔离，话题 ID 全局唯一；
- 退役的话题离开话题池，日志保留到删除，其间照样记下生成的记忆与沉淀完成、交出本话题生成的记忆；
  进展记录只记到池中的话题；
- 读回时没有退役的进入话题池，退役而沉淀没完成的交给调用方，读不回的日志移到一旁。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from hivememory.core.models import WorkspaceAssetRef
from hivememory.engines.perception.models import TriggerReason
from hivememory.memory_library.adapters.short_term import JsonlShortTermStorage
from hivememory.memory_library.stores import ShortTermMemoryStore
from tests.helpers.sessions import make_logical_block
from tests.helpers.workspace import make_identity_scope

_MAIN = make_identity_scope(user_id="u1", workspace_id="workspace-a").workspace_identity
_OTHER = make_identity_scope(user_id="u2", workspace_id="workspace-b").workspace_identity


class _FailingAppendPort(JsonlShortTermStorage):
    """文件端口；``fail_appends`` 为真时追加以磁盘错误失败。"""

    fail_appends = False

    def append(self, topic_id: str, line: bytes) -> None:
        if self.fail_appends:
            raise OSError("disk full")
        super().append(topic_id, line)


def _store(tmp_path) -> ShortTermMemoryStore:
    return ShortTermMemoryStore(JsonlShortTermStorage(tmp_path / "topics"))


def _block(message: str):
    return make_logical_block(message, "answer", total_tokens=2)


def _topic_with_interactions(store, *messages, belong_to=_MAIN):
    topic = store.create(belong_to)
    for message in messages:
        topic = store.append_interaction(belong_to, topic.topic_id, block=_block(message))
    return topic


def test_replaying_the_log_restores_what_was_written(tmp_path):
    """接收、折叠、沉淀完成、本话题生成的记忆与进展记录都记进日志，另一个存储重放得到同样的读取值。"""
    store = _store(tmp_path)
    topic = store.create(_MAIN, topic_title="部署", topic_summary="发布流程")
    first, second = _block("q1"), _block("q2")
    store.append_interaction(
        _MAIN,
        topic.topic_id,
        block=first,
        asset_refs=(WorkspaceAssetRef(token="token-1", asset_id="asset-1"),),
        interaction_id="i-1",
        model_used="model-a",
    )
    store.append_interaction(_MAIN, topic.topic_id, block=second)
    store.append_fold(_MAIN, topic.topic_id, folded=[first], summary="s1")
    store.append_settled(_MAIN, topic.topic_id, interaction_count=2)
    store.append_generated_memories(
        _MAIN, topic.topic_id, generated=frozenset({"m1"}), absorbed_into={"x": "y"}
    )
    store.append_progress_entry(
        _MAIN, topic.topic_id, text="e1", covered=[first, second], covered_until=2
    )
    other = _topic_with_interactions(store, "elsewhere", belong_to=_OTHER)

    reloaded = _store(tmp_path)
    loaded = reloaded.load()

    assert sorted(loaded.topics, key=lambda topic: topic.topic_id) == sorted(
        store.list_all(), key=lambda topic: topic.topic_id
    )
    assert reloaded.get(_MAIN, topic.topic_id) == store.get(_MAIN, topic.topic_id)
    assert [entry.text for entry in reloaded.get(_MAIN, topic.topic_id).progress_record] == ["e1"]
    assert reloaded.get(_OTHER, other.topic_id) == store.get(_OTHER, other.topic_id)


def test_failed_append_leaves_the_topic_unchanged(tmp_path):
    """先落盘再改内存：追加失败时错误照常传播，内存与日志中的话题都没有这一轮。"""
    port = _FailingAppendPort(tmp_path / "topics")
    store = ShortTermMemoryStore(port)
    topic = _topic_with_interactions(store, "q1")

    port.fail_appends = True
    with pytest.raises(OSError, match="disk full"):
        store.append_interaction(_MAIN, topic.topic_id, block=_block("q2"))
    reloaded = _store(tmp_path)
    reloaded.load()

    assert store.get(_MAIN, topic.topic_id) == topic
    assert reloaded.get(_MAIN, topic.topic_id) == topic


def test_rejected_write_appends_nothing(tmp_path):
    """构造校验失败（折叠不是工作视图的前缀）时直接抛出，日志里没有这条记录。"""
    store = _store(tmp_path)
    topic = _topic_with_interactions(store, "q1", "q2")

    with pytest.raises(ValidationError, match="前缀"):
        store.append_fold(_MAIN, topic.topic_id, folded=[topic.interactions[1]], summary="s")
    reloaded = _store(tmp_path)
    reloaded.load()

    assert reloaded.get(_MAIN, topic.topic_id) == topic


def test_topics_are_isolated_by_workspace_and_ids_are_global(tmp_path):
    """另一个 Workspace 读不到、写不进同名话题；话题 ID 已被占用时不能再建立。"""
    store = _store(tmp_path)
    topic = _topic_with_interactions(store, "q1")

    assert store.get(_OTHER, topic.topic_id) is None
    assert store.append_interaction(_OTHER, topic.topic_id, block=_block("q2")) is None
    assert store.delete(_OTHER, topic.topic_id) is False
    with pytest.raises(ValueError, match="already exists"):
        store.create(_OTHER, topic_id=topic.topic_id)
    assert store.get(_MAIN, topic.topic_id) == topic


def test_retired_topic_keeps_writes_and_settlement_until_its_log_is_deleted(tmp_path):
    """退役的话题读不到、不回到列表；日志删除之前照样记下生成的记忆与沉淀完成、交出本话题生成的记忆，重放
    时随日志读回，进展记录不再记；删除时连同日志一起删除。"""
    store = _store(tmp_path)
    topic = _topic_with_interactions(store, "q1", "q2")

    store.retire(_MAIN, topic.topic_id, reason=TriggerReason.LRU_EVICTION)
    store.append_generated_memories(
        _MAIN, topic.topic_id, generated=frozenset({"m1"}), absorbed_into={}
    )
    settled = store.append_settled(_MAIN, topic.topic_id, interaction_count=1)
    progress = store.append_progress_entry(
        _MAIN, topic.topic_id, text="e1", covered=topic.interactions, covered_until=2
    )
    [(replayed, _)] = _store(tmp_path).load().retired

    assert progress is None
    assert replayed.progress_record == ()
    assert (store.get(_MAIN, topic.topic_id), store.list_all()) == (None, [])
    assert store.get_generated_memories(_MAIN, topic.topic_id) == {"m1"}
    assert store.get_generated_memories(_OTHER, topic.topic_id) is None
    assert replayed == settled
    assert (replayed.settled_interaction_count, replayed.topic_generated_memory_ids) == (1, {"m1"})
    assert store.delete(_MAIN, topic.topic_id) is True
    assert store.get_generated_memories(_MAIN, topic.topic_id) is None
    assert store.append_settled(_MAIN, topic.topic_id, interaction_count=2) is None
    assert list((tmp_path / "topics").glob("*.jsonl")) == []


def test_load_sorts_logs_into_pool_retired_and_rejected(tmp_path):
    """没有退役的回到话题池；退役而沉淀没完成的交给调用方并保留日志；退役且沉淀都完成的
    删除日志；中间有一行读不回的整个日志移到一旁，不读回。"""
    writer = _store(tmp_path)
    live = _topic_with_interactions(writer, "live")
    pending = _topic_with_interactions(writer, "pending")
    writer.retire(_MAIN, pending.topic_id, reason=TriggerReason.MANUAL_SETTLE)
    finished = _topic_with_interactions(writer, "finished")
    writer.retire(_MAIN, finished.topic_id, reason=TriggerReason.LRU_EVICTION)
    writer.append_settled(_MAIN, finished.topic_id, interaction_count=1)
    broken = _topic_with_interactions(writer, "broken", "later")
    log_dir = tmp_path / "topics"
    broken_log = log_dir / f"{broken.topic_id}.jsonl"
    lines = broken_log.read_bytes().split(b"\n")
    broken_log.write_bytes(b"\n".join([lines[0], b"{not json", *lines[2:]]))

    store = _store(tmp_path)
    loaded = store.load()

    assert loaded.topics == [live]
    assert [(topic.topic_id, record.reason) for topic, record in loaded.retired] == [
        (pending.topic_id, TriggerReason.MANUAL_SETTLE)
    ]
    assert store.list_all() == [live]
    (rejected,) = loaded.rejected_files
    assert Path(rejected).name.startswith(f"{broken.topic_id}.rejected-")
    assert Path(rejected).read_bytes().split(b"\n")[1] == b"{not json"
    remaining = {path.stem for path in log_dir.glob("*.jsonl") if ".rejected-" not in path.name}
    assert remaining == {live.topic_id, pending.topic_id}


def test_non_empty_listing_keeps_only_topics_with_interactions(tmp_path):
    """include_empty=False 按内容判断：还没有接收交互的话题不列出。"""
    store = _store(tmp_path)
    store.create(_MAIN)
    with_interaction = _topic_with_interactions(store, "q1")

    topics = store.list_by_workspace(_MAIN, include_empty=False)

    assert topics == [with_interaction]
