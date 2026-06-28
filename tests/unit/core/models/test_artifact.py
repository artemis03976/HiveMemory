from uuid import uuid4

from hivememory.core.models import IndexLayer, MemoryAtom, MemoryType, MetaData, PayloadLayer
from hivememory.core.models.artifact import MemoryVersionSnapshot


def test_memory_version_snapshot_from_memory_atom_captures_mutable_fields():
    atom = MemoryAtom(
        id=uuid4(),
        meta=MetaData(source_agent_id="a1", user_id="u1"),
        index=IndexLayer(
            title="Test Title",
            summary="A test memory summary",
            tags=["tag1", "tag2"],
            memory_type=MemoryType.FACT,
            alias="fact_test",
        ),
        payload=PayloadLayer(content="content"),
    )

    snapshot = MemoryVersionSnapshot.from_memory_atom(atom)

    assert snapshot.content == "content"
    assert snapshot.alias == "fact_test"
    assert snapshot.title == "Test Title"
    assert snapshot.summary == "A test memory summary"
    assert snapshot.tags == ["tag1", "tag2"]
    assert snapshot.memory_type == "FACT"
