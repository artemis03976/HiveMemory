"""
server/models 单元测试

测试覆盖:
    1. MemoryResponse.from_atom() 转换
    2. Common 模型的版本守护
"""

from hivememory.server.models.memory import MemoryResponse
from hivememory.server.models.common import HealthResponse


class TestMemoryResponse:
    def test_from_atom(self):
        """测试 MemoryAtom → MemoryResponse 转换"""
        from hivememory.core.models import (
            MemoryAtom, IndexLayer, PayloadLayer, MemoryType,
        )
        from uuid import uuid4
        from tests.helpers.memory import make_memory_metadata

        atom = MemoryAtom(
            id=uuid4(),
            meta=make_memory_metadata(
                source_agent_id="agent1",
                user_id="user1",
                confidence_score=0.9,
                vitality_score=80.0,
                access_count=5,
            ),
            index=IndexLayer(
                title="Test Memory",
                summary="A test memory for unit testing",
                tags=["test", "unit"],
                memory_type=MemoryType.FACT,
                alias="test_memory",
            ),
            payload=PayloadLayer(content="Test content here"),
        )

        resp = MemoryResponse.from_atom(atom)
        assert resp.id == str(atom.id)
        assert resp.title == "Test Memory"
        assert resp.memory_type == "FACT"
        assert resp.confidence_score == 0.9
        assert resp.alias == "test_memory"
        assert resp.access_count == 5


class TestCommonModels:
    def test_health_response(self):
        from hivememory import __version__

        h = HealthResponse()
        assert h.status == "ok"
        assert h.version == __version__
