from uuid import uuid4

from hivememory.core.models import IndexLayer, MemoryAtom, MemoryType, PayloadLayer
from hivememory.core.models.artifact import MemoryVersionSnapshot
from tests.helpers.memory import make_memory_metadata


def test_memory_version_snapshot_from_memory_atom_captures_mutable_fields():
    atom = MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(source_agent_id="a1", user_id="u1"),
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
    assert set(snapshot.tags) == {"tag1", "tag2"}
    assert snapshot.memory_type == "FACT"
