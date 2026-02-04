"""
HiveMemory Lifecycle E2E 测试数据 Fixtures

提供专门用于 Lifecycle 模块端到端测试的数据。

设计原则:
    - 覆盖评分逻辑的各种场景 (基础分、时间衰减、访问加成)
    - 覆盖强化事件的四种类型 (HIT, CITATION, FEEDBACK_POSITIVE, FEEDBACK_NEGATIVE)
    - 覆盖归档与唤醒的完整流程

作者: HiveMemory Team
版本: 1.0.0
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from hivememory.core.models import (
    MemoryAtom,
    MemoryType,
    MemoryVisibility,
    VerificationStatus,
    MetaData,
    IndexLayer,
    PayloadLayer,
    RelationLayer,
    Artifacts,
)
from hivememory.engines.lifecycle.models import EventType


# ========== 评分测试用例 ==========

SCORING_TEST_CASES = [
    # LIF-SCR-001: 基础分计算 (P1)
    {
        "id": "LIF-SCR-001",
        "name": "基础分计算",
        "priority": "P1",
        "description": "验证不同记忆类型的固有价值权重差异",
        "test_memories": [
            {
                "memory_type": MemoryType.CODE_SNIPPET,
                "confidence": 0.9,
                "expected_higher": True,
            },
            {
                "memory_type": MemoryType.WORK_IN_PROGRESS,
                "confidence": 0.9,
                "expected_higher": False,
            },
        ],
        "assertion": "Score(CODE_SNIPPET) > Score(WORK_IN_PROGRESS)",
    },
    # LIF-SCR-002: 时间衰减测试 (P0)
    {
        "id": "LIF-SCR-002",
        "name": "时间衰减测试",
        "priority": "P0",
        "description": "验证时间流逝导致的分数衰减",
        "days_elapsed": 30,
        "decay_lambda": 0.01,  # 默认衰减系数
        "expected_decay_factor": 0.7408,  # exp(-0.01 * 30) ≈ 0.7408
        "tolerance": 0.01,
        "assertion": "Score(Now) < Score(Initial) * exp(-lambda * days)",
    },
    # LIF-SCR-003: 访问加成上限 (P2)
    {
        "id": "LIF-SCR-003",
        "name": "访问加成上限",
        "priority": "P2",
        "description": "验证访问加成不会超过配置的上限",
        "access_count": 10000,
        "max_access_boost": 50.0,  # 默认上限
        "points_per_access": 1.0,  # 默认每次访问加成
        "assertion": "Boost Value <= Max Cap (50)",
    },
]


# ========== 强化事件测试用例 ==========

REINFORCEMENT_TEST_CASES = [
    # LIF-RNF-001: 检索命中 (HIT) (P0)
    {
        "id": "LIF-RNF-001",
        "name": "检索命中 (HIT)",
        "priority": "P0",
        "description": "验证 HIT 事件对生命力和访问计数的影响",
        "event_type": EventType.HIT,
        "expected_vitality_change": 5.0,  # 配置中的 hit_boost
        "expected_access_count_change": 1,
        "expected_confidence_change": 0.0,
        "assertion": "Vitality increases, Access Count += 1",
    },
    # LIF-RNF-002: 引用强化 (CITATION) (P0)
    {
        "id": "LIF-RNF-002",
        "name": "引用强化 (CITATION)",
        "priority": "P0",
        "description": "验证 CITATION 事件重置时间衰减并大幅提升生命力",
        "event_type": EventType.CITATION,
        "expected_vitality_change": 20.0,  # 配置中的 citation_boost
        "expected_access_count_change": 1,
        "expected_decay_reset": True,  # updated_at 应该被更新
        "assertion": "Updated_at = Now, Vitality significantly increases",
    },
    # LIF-RNF-003: 用户负面反馈 (FEEDBACK_NEGATIVE) (P1)
    {
        "id": "LIF-RNF-003",
        "name": "用户负面反馈 (FEEDBACK_NEGATIVE)",
        "priority": "P1",
        "description": "验证负面反馈对生命力和置信度的惩罚",
        "event_type": EventType.FEEDBACK_NEGATIVE,
        "expected_vitality_change": -50.0,  # 配置中的 negative_feedback_penalty
        "expected_confidence_multiplier": 0.5,  # 配置中的 negative_confidence_multiplier
        "assertion": "Vitality -= 50, Confidence *= 0.5",
    },
    # LIF-RNF-004: 用户正面反馈 (FEEDBACK_POSITIVE) (P1)
    {
        "id": "LIF-RNF-004",
        "name": "用户正面反馈 (FEEDBACK_POSITIVE)",
        "priority": "P1",
        "description": "验证正面反馈对生命力的提升",
        "event_type": EventType.FEEDBACK_POSITIVE,
        "expected_vitality_change": 50.0,  # 配置中的 positive_feedback_boost
        "expected_access_count_change": 1,
        "assertion": "Vitality += 50, Access Count += 1",
    },
]


# ========== 归档测试用例 ==========

ARCHIVING_TEST_CASES = [
    # LIF-ARC-001: 归档触发 (P0)
    {
        "id": "LIF-ARC-001",
        "name": "归档触发",
        "priority": "P0",
        "description": "验证低分记忆被正确归档到冷存储",
        "initial_vitality": 5.0,
        "archive_threshold": 10.0,
        "expected_in_hot_storage": False,
        "expected_in_cold_storage": True,
        "assertion": "Qdrant: M1 Deleted, FileSystem: M1.json.gz Created",
    },
    # LIF-ARC-002: 唤醒流程 (P1)
    {
        "id": "LIF-ARC-002",
        "name": "唤醒流程",
        "priority": "P1",
        "description": "验证已归档记忆能被正确唤醒到热存储",
        "expected_in_hot_storage": True,
        "expected_in_cold_storage": False,
        "expected_data_integrity": True,  # 所有字段应保持不变
        "assertion": "Qdrant: M1 Inserted, FileSystem: M1 Deleted",
    },
    # LIF-ARC-003: 归档幂等性 (P2)
    {
        "id": "LIF-ARC-003",
        "name": "归档幂等性",
        "priority": "P2",
        "description": "验证对已归档记忆再次调用归档不会产生错误或重复文件",
        "expected_error": False,
        "expected_duplicate_file": False,
        "assertion": "No Error, No Duplicate File",
    },
]


# ========== 预置记忆模板 ==========

MEMORY_TEMPLATES = {
    "code_snippet": {
        "memory_type": MemoryType.CODE_SNIPPET,
        "title": "Python 快速排序实现",
        "summary": "使用递归实现的快速排序算法",
        "content": """```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```""",
        "tags": ["python", "algorithm", "sorting"],
        "confidence": 0.95,
    },
    "fact": {
        "memory_type": MemoryType.FACT,
        "title": "Rust 所有权规则",
        "summary": "Rust 语言的三条所有权规则",
        "content": """Rust 所有权规则:
1. 每个值都有一个所有者
2. 同一时间只能有一个所有者
3. 当所有者离开作用域，值被丢弃""",
        "tags": ["rust", "ownership", "memory"],
        "confidence": 0.90,
    },
    "work_in_progress": {
        "memory_type": MemoryType.WORK_IN_PROGRESS,
        "title": "待完成的功能设计",
        "summary": "用户正在设计的新功能草稿",
        "content": "TODO: 实现用户认证模块",
        "tags": ["todo", "design"],
        "confidence": 0.50,
    },
    "user_profile": {
        "memory_type": MemoryType.USER_PROFILE,
        "title": "用户偏好设置",
        "summary": "用户喜欢使用 VSCode 和 Python",
        "content": "用户偏好: IDE=VSCode, Language=Python, Theme=Dark",
        "tags": ["preference", "user"],
        "confidence": 0.85,
    },
    "low_vitality": {
        "memory_type": MemoryType.FACT,
        "title": "低生命力记忆",
        "summary": "用于测试归档的低分记忆",
        "content": "这是一条即将被归档的记忆",
        "tags": ["test", "archive"],
        "confidence": 0.30,
        "vitality_score": 5.0,
    },
}


# ========== 辅助函数 ==========

def create_test_memory(
    template_name: str = "fact",
    memory_id: Optional[UUID] = None,
    updated_at: Optional[datetime] = None,
    access_count: int = 0,
    vitality_score: Optional[float] = None,
    confidence_score: Optional[float] = None,
    **overrides
) -> MemoryAtom:
    """
    创建测试用的 MemoryAtom 实例

    Args:
        template_name: 模板名称 (code_snippet, fact, work_in_progress, user_profile, low_vitality)
        memory_id: 自定义 UUID (默认自动生成)
        updated_at: 自定义更新时间 (默认当前时间)
        access_count: 访问次数
        vitality_score: 生命力分数 (默认根据模板)
        confidence_score: 置信度分数 (默认根据模板)
        **overrides: 其他覆盖参数

    Returns:
        MemoryAtom: 测试用记忆实例
    """
    template = MEMORY_TEMPLATES.get(template_name, MEMORY_TEMPLATES["fact"])

    now = datetime.now()
    _updated_at = updated_at or now
    _memory_id = memory_id or uuid4()
    _confidence = confidence_score if confidence_score is not None else template.get("confidence", 0.8)
    _vitality = vitality_score if vitality_score is not None else template.get("vitality_score", 50.0)

    meta = MetaData(
        created_at=now,
        updated_at=_updated_at,
        last_accessed_at=None,
        source_agent_id="test_agent",
        user_id="test_user",
        session_id="test_session",
        visibility=MemoryVisibility.PRIVATE,
        version=1,
        access_count=access_count,
        vitality_score=_vitality,
        confidence_score=_confidence,
        verification_status=VerificationStatus.UNVERIFIED,
    )

    index = IndexLayer(
        title=overrides.get("title", template["title"]),
        summary=overrides.get("summary", template["summary"]),
        tags=overrides.get("tags", template["tags"]),
        memory_type=overrides.get("memory_type", template["memory_type"]),
    )

    payload = PayloadLayer(
        content=overrides.get("content", template["content"]),
        history_summary=[],
        artifacts=Artifacts(),
    )

    relations = RelationLayer()

    return MemoryAtom(
        id=_memory_id,
        meta=meta,
        index=index,
        payload=payload,
        relations=relations,
    )


def create_memory_with_age(
    days_old: int,
    template_name: str = "fact",
    **kwargs
) -> MemoryAtom:
    """
    创建指定"年龄"的记忆

    Args:
        days_old: 记忆的年龄（天数）
        template_name: 模板名称
        **kwargs: 传递给 create_test_memory 的其他参数

    Returns:
        MemoryAtom: 具有指定年龄的记忆实例
    """
    old_date = datetime.now() - timedelta(days=days_old)
    return create_test_memory(
        template_name=template_name,
        updated_at=old_date,
        **kwargs
    )


def get_test_cases_by_priority(priority: str) -> List[Dict[str, Any]]:
    """
    获取指定优先级的所有测试用例

    Args:
        priority: 优先级 (P0, P1, P2)

    Returns:
        List[Dict]: 匹配的测试用例列表
    """
    all_cases = SCORING_TEST_CASES + REINFORCEMENT_TEST_CASES + ARCHIVING_TEST_CASES
    return [case for case in all_cases if case.get("priority") == priority]


def get_p0_test_cases() -> List[Dict[str, Any]]:
    """获取所有 P0 优先级测试用例"""
    return get_test_cases_by_priority("P0")


def get_p1_test_cases() -> List[Dict[str, Any]]:
    """获取所有 P1 优先级测试用例"""
    return get_test_cases_by_priority("P1")


def get_scoring_test_by_id(test_id: str) -> Optional[Dict[str, Any]]:
    """根据 ID 获取评分测试用例"""
    for case in SCORING_TEST_CASES:
        if case["id"] == test_id:
            return case
    return None


def get_reinforcement_test_by_id(test_id: str) -> Optional[Dict[str, Any]]:
    """根据 ID 获取强化测试用例"""
    for case in REINFORCEMENT_TEST_CASES:
        if case["id"] == test_id:
            return case
    return None


def get_archiving_test_by_id(test_id: str) -> Optional[Dict[str, Any]]:
    """根据 ID 获取归档测试用例"""
    for case in ARCHIVING_TEST_CASES:
        if case["id"] == test_id:
            return case
    return None


__all__ = [
    # 测试用例
    "SCORING_TEST_CASES",
    "REINFORCEMENT_TEST_CASES",
    "ARCHIVING_TEST_CASES",
    # 模板
    "MEMORY_TEMPLATES",
    # 辅助函数
    "create_test_memory",
    "create_memory_with_age",
    "get_test_cases_by_priority",
    "get_p0_test_cases",
    "get_p1_test_cases",
    "get_scoring_test_by_id",
    "get_reinforcement_test_by_id",
    "get_archiving_test_by_id",
]
