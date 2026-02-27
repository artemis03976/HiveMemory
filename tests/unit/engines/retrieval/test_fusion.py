"""
Fusion 模块单元测试

测试 ReciprocalRankFusion (RRF) 算法的正确性
测试 AdaptiveWeightedFusion 自适应加权融合算法
"""

import pytest
from uuid import uuid4, UUID

from hivememory.core.models import MemoryAtom, IndexLayer, MetaData, PayloadLayer, MemoryType
from hivememory.engines.retrieval.models import SearchResult, SearchResults
from hivememory.patchouli.config import ReciprocalRankFusionConfig, AdaptiveWeightedFusionConfig, RetrievalModeConfig
from hivememory.engines.retrieval.fusion import ReciprocalRankFusion, AdaptiveWeightedFusion

class TestReciprocalRankFusion:
    
    @pytest.fixture
    def fusion(self):
        config = ReciprocalRankFusionConfig(
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
        config = ReciprocalRankFusionConfig(
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
        config = ReciprocalRankFusionConfig(final_top_k=2)
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


class TestAdaptiveWeightedFusion:
    """测试自适应加权融合器"""

    @pytest.fixture
    def fusion(self):
        """创建默认配置的融合器"""
        return AdaptiveWeightedFusion()

    def create_result(
        self,
        memory_id: UUID,
        score: float,
        match_reason: str,
        confidence: float = 0.8,
        vitality: float = 50.0
    ) -> SearchResult:
        """创建测试用的 SearchResult"""
        memory = MemoryAtom(
            id=memory_id,
            index=IndexLayer(
                title=f"Mem {memory_id}",
                summary="summary must be long enough for validation",
                memory_type=MemoryType.FACT
            ),
            meta=MetaData(
                source_agent_id="test",
                user_id="user",
                confidence_score=confidence,
                vitality_score=vitality
            ),
            payload=PayloadLayer(content="content")
        )
        return SearchResult(
            memory=memory,
            score=score,
            match_reason=match_reason
        )

    def test_fuse_with_quality_multiplier(self, fusion):
        """测试质量乘数正确应用"""
        id1 = uuid4()
        id2 = uuid4()

        # id1: 高分数，低置信度 -> 应被惩罚
        # id2: 低分数，高置信度 -> 不被惩罚
        dense_results = SearchResults(results=[
            self.create_result(id1, 0.9, "dense", confidence=0.4, vitality=50.0),
            self.create_result(id2, 0.7, "dense", confidence=0.95, vitality=50.0),
        ])
        sparse_results = SearchResults(results=[])

        # 使用 debug 模式 (强惩罚)
        fused = fusion.fuse(dense_results, sparse_results, mode="debug")

        # 验证低置信度的记忆被降权
        assert len(fused.results) == 2
        # id1 原始分数高但被惩罚，id2 原始分数低但不被惩罚
        # 在 debug 模式下，confidence < 0.6 会被乘以 0.5
        # id1: 0.9 * 0.5 = 0.45 (被惩罚)
        # id2: 0.7 * 1.0 = 0.7 (不被惩罚)
        # 所以 id2 应该排在前面
        assert fused.results[0].memory.id == id2

    def test_debug_mode_penalizes_low_confidence(self):
        """测试 Debug 模式下低置信度被降权"""
        fusion = AdaptiveWeightedFusion()

        id_low_conf = uuid4()
        id_high_conf = uuid4()

        dense_results = SearchResults(results=[
            self.create_result(id_low_conf, 0.95, "dense", confidence=0.3, vitality=50.0),
            self.create_result(id_high_conf, 0.85, "dense", confidence=0.95, vitality=50.0),
        ])
        sparse_results = SearchResults(results=[])

        # Debug 模式: 强惩罚低置信度
        fused = fusion.fuse(dense_results, sparse_results, mode="debug")

        # 高置信度应排第一
        assert fused.results[0].memory.id == id_high_conf
        assert fused.results[0].memory.meta.confidence_score > 0.9

    def test_concept_mode_weights(self):
        """测试 Concept 模式权重分配正确"""
        fusion = AdaptiveWeightedFusion()

        id1 = uuid4()

        # 只有 dense 结果
        dense_results = SearchResults(results=[
            self.create_result(id1, 0.9, "dense", confidence=0.8, vitality=50.0),
        ])
        sparse_results = SearchResults(results=[])

        # Concept 模式: 高 dense 权重 (0.8)
        fused = fusion.fuse(dense_results, sparse_results, mode="concept")

        assert len(fused.results) == 1
        # 验证分数被正确计算
        # dense_weight=0.8, sparse_weight=0.2, total=1.0
        # score = (0.8/1.0) * 0.9 = 0.72
        # 无惩罚 (confidence=0.8 > 0.5)
        assert fused.results[0].score == pytest.approx(0.72, rel=0.01)

    def test_vitality_boost(self):
        """测试高生命力记忆被提权"""
        fusion = AdaptiveWeightedFusion()

        id_high_vit = uuid4()
        id_low_vit = uuid4()

        dense_results = SearchResults(results=[
            self.create_result(id_low_vit, 0.9, "dense", confidence=0.8, vitality=20.0),
            self.create_result(id_high_vit, 0.85, "dense", confidence=0.8, vitality=90.0),
        ])
        sparse_results = SearchResults(results=[])

        # Concept 模式: 启用生命力加成
        fused = fusion.fuse(dense_results, sparse_results, mode="concept")

        # 高生命力应该被提权
        # id_low_vit: 0.9 * 0.8 (vitality_low_factor)
        # id_high_vit: 0.85 * 1.2 (vitality_high_factor)
        assert fused.results[0].memory.id == id_high_vit

    def test_mode_switching(self, fusion):
        """测试模式切换正确"""
        id1 = uuid4()

        dense_results = SearchResults(results=[
            self.create_result(id1, 0.9, "dense", confidence=0.4, vitality=50.0),
        ])
        sparse_results = SearchResults(results=[])

        # Debug 模式: 低置信度被惩罚
        fused_debug = fusion.fuse(dense_results, sparse_results, mode="debug")

        # Brainstorm 模式: 无惩罚
        fused_brainstorm = fusion.fuse(dense_results, sparse_results, mode="brainstorm")

        # Debug 模式分数应该更低 (被惩罚)
        assert fused_debug.results[0].score < fused_brainstorm.results[0].score

    def test_default_mode_fallback(self, fusion):
        """测试未指定模式时使用默认模式"""
        id1 = uuid4()

        dense_results = SearchResults(results=[
            self.create_result(id1, 0.9, "dense"),
        ])
        sparse_results = SearchResults(results=[])

        # 不指定模式，应使用默认模式 (concept)
        fused = fusion.fuse(dense_results, sparse_results)

        assert len(fused.results) == 1
        # 默认模式是 concept，dense_weight=0.8
        # score = (0.8/1.0) * 0.9 = 0.72
        assert fused.results[0].score == pytest.approx(0.72, rel=0.01)

    def test_fuse_empty(self, fusion):
        """测试空结果处理"""
        empty = SearchResults()
        fused = fusion.fuse(empty, empty)

        assert len(fused.results) == 0
        assert fused.is_empty()

    def test_fuse_with_intent(self, fusion):
        """测试基于意图的模式推断"""
        id1 = uuid4()

        dense_results = SearchResults(results=[
            self.create_result(id1, 0.9, "dense", confidence=0.4),
        ])
        sparse_results = SearchResults(results=[])

        # 使用 "fix error" 意图，应推断为 debug 模式
        fused = fusion.fuse_with_intent(dense_results, sparse_results, "fix error in code")

        # Debug 模式会惩罚低置信度
        # 验证分数被惩罚
        assert fused.results[0].score < 0.9

    def test_top_k_truncation(self):
        """测试结果截断"""
        config = AdaptiveWeightedFusionConfig(final_top_k=2)
        fusion = AdaptiveWeightedFusion(config)

        # 创建 3 个结果
        dense_results = SearchResults(results=[
            self.create_result(uuid4(), 0.9, "r1"),
            self.create_result(uuid4(), 0.8, "r2"),
            self.create_result(uuid4(), 0.7, "r3"),
        ])
        sparse_results = SearchResults(results=[])

        fused = fusion.fuse(dense_results, sparse_results)

        assert len(fused.results) == 2
        assert fused.total_candidates == 3

    def test_latency_aggregation(self, fusion):
        """测试延迟时间累加"""
        r1 = SearchResults(latency_ms=10.0)
        r2 = SearchResults(latency_ms=20.0)

        fused = fusion.fuse(r1, r2)
        assert fused.latency_ms == 30.0

    def test_unknown_mode_fallback(self, fusion):
        """测试未知模式回退到默认模式"""
        id1 = uuid4()

        dense_results = SearchResults(results=[
            self.create_result(id1, 0.9, "dense"),
        ])
        sparse_results = SearchResults(results=[])

        # 使用未知模式
        fused = fusion.fuse(dense_results, sparse_results, mode="unknown_mode")

        # 应该回退到默认模式 (concept)
        assert len(fused.results) == 1