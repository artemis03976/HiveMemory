"""
Fusion 模块单元测试

测试 ReciprocalRankFusion (RRF) 算法的正确性
"""

import pytest
from uuid import uuid4, UUID
from datetime import datetime

from hivememory.core.models import MemoryAtom, IndexLayer, MetaData, PayloadLayer, MemoryType
from hivememory.engines.retrieval.models import SearchResult, SearchResults
from hivememory.patchouli.config import FusionConfig
from hivememory.engines.retrieval.fusion import ReciprocalRankFusion

class TestReciprocalRankFusion:
    
    @pytest.fixture
    def fusion(self):
        config = FusionConfig(
            rrf_k=60,
            dense_weight=1.0,
            sparse_weight=1.0,
            final_top_k=10
        )
        return ReciprocalRankFusion(config)

    def create_result(self, memory_id: UUID, score: float, match_reason: str) -> SearchResult:
        memory = MemoryAtom(
            id=memory_id,
            index=IndexLayer(title=f"Mem {memory_id}", summary="summary must be long enough for validation", memory_type=MemoryType.FACT),
            meta=MetaData(source_agent_id="test", user_id="user"),
            payload=PayloadLayer(content="content")
        )
        return SearchResult(
            memory=memory,
            score=score,
            match_reason=match_reason
        )

    def test_fuse_disjoint(self, fusion):
        """测试无重叠结果的融合"""
        id1 = uuid4()
        id2 = uuid4()
        
        dense_results = SearchResults(results=[
            self.create_result(id1, 0.9, "dense_match")
        ])
        sparse_results = SearchResults(results=[
            self.create_result(id2, 0.8, "sparse_match")
        ])
        
        fused = fusion.fuse(dense_results, sparse_results)
        
        assert len(fused.results) == 2
        # Score calculation:
        # id1 (rank 1 in dense): 1.0 / (60 + 1) = 0.01639
        # id2 (rank 1 in sparse): 1.0 / (60 + 1) = 0.01639
        # Since scores are equal and sorted is stable/deterministic by key, check existence
        ids = [r.memory.id for r in fused.results]
        assert id1 in ids
        assert id2 in ids
        
        assert fused.total_candidates == 2

    def test_fuse_overlap(self, fusion):
        """测试有重叠结果的融合 (分数叠加)"""
        common_id = uuid4()
        only_dense_id = uuid4()
        
        # Common ID is rank 1 in dense, rank 1 in sparse
        dense_results = SearchResults(results=[
            self.create_result(common_id, 0.9, "dense"),
            self.create_result(only_dense_id, 0.7, "dense_only")
        ])
        
        sparse_results = SearchResults(results=[
            self.create_result(common_id, 0.8, "sparse")
        ])
        
        fused = fusion.fuse(dense_results, sparse_results)
        
        assert len(fused.results) == 2
        
        # Check common_id score
        # Dense rank 1: 1/61
        # Sparse rank 1: 1/61
        # Total: 2/61 ≈ 0.03278
        common_res = next(r for r in fused.results if r.memory.id == common_id)
        assert common_res.score == pytest.approx(2/61, rel=1e-4)
        
        # Check match reason merging
        assert "dense" in common_res.match_reason
        assert "sparse" in common_res.match_reason
        
        # Check sorting: common (0.032) > only_dense (1/62 ≈ 0.016)
        assert fused.results[0].memory.id == common_id
        
    def test_fuse_weights(self):
        """测试不同权重的影响"""
        config = FusionConfig(
            rrf_k=1, # Small k to make weight impact larger
            dense_weight=10.0,
            sparse_weight=1.0
        )
        fusion = ReciprocalRankFusion(config)
        
        id1 = uuid4() # Dense rank 1
        id2 = uuid4() # Sparse rank 1
        
        dense_results = SearchResults(results=[self.create_result(id1, 0.9, "dense")])
        sparse_results = SearchResults(results=[self.create_result(id2, 0.9, "sparse")])
        
        fused = fusion.fuse(dense_results, sparse_results)
        
        # id1 score: 10 / (1+1) = 5.0
        # id2 score: 1 / (1+1) = 0.5
        
        assert fused.results[0].memory.id == id1
        assert fused.results[0].score == pytest.approx(5.0)
        assert fused.results[1].memory.id == id2
        assert fused.results[1].score == pytest.approx(0.5)

    def test_fuse_empty(self, fusion):
        """测试空结果处理"""
        empty = SearchResults()
        fused = fusion.fuse(empty, empty)
        
        assert len(fused.results) == 0
        assert fused.is_empty()

    def test_fuse_multi(self, fusion):
        """测试多路融合接口"""
        id1 = uuid4()
        id2 = uuid4()
        id3 = uuid4()
        
        r1 = SearchResults(results=[self.create_result(id1, 0.9, "r1")])
        r2 = SearchResults(results=[self.create_result(id2, 0.9, "r2")])
        r3 = SearchResults(results=[self.create_result(id3, 0.9, "r3")])
        
        # Test with equal weights
        fused = fusion.fuse_multi([r1, r2, r3])
        assert len(fused.results) == 3
        
        # Test with custom weights
        weights = [10.0, 1.0, 0.1]
        fused_weighted = fusion.fuse_multi([r1, r2, r3], weights=weights)
        
        assert fused_weighted.results[0].memory.id == id1 # weight 10
        assert fused_weighted.results[1].memory.id == id2 # weight 1
        assert fused_weighted.results[2].memory.id == id3 # weight 0.1

    def test_top_k_truncation(self):
        """测试结果截断"""
        config = FusionConfig(final_top_k=2)
        fusion = ReciprocalRankFusion(config)
        
        # Create 3 distinct results
        r1 = SearchResults(results=[self.create_result(uuid4(), 0.9, "r1")])
        r2 = SearchResults(results=[self.create_result(uuid4(), 0.9, "r2")])
        r3 = SearchResults(results=[self.create_result(uuid4(), 0.9, "r3")])
        
        fused = fusion.fuse_multi([r1, r2, r3])
        
        assert len(fused.results) == 2
        assert fused.total_candidates == 3

    def test_latency_aggregation(self, fusion):
        """测试延迟时间累加"""
        r1 = SearchResults(latency_ms=10.0)
        r2 = SearchResults(latency_ms=20.0)
        
        fused = fusion.fuse(r1, r2)
        assert fused.latency_ms == 30.0

