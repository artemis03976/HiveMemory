"""
HiveMemory Gateway E2E 测试数据 Fixtures

提供专门用于 Gateway 模块端到端测试的数据。

设计原则:
    - 覆盖意图识别的四种类型 (RAG, CHAT, TOOL, SYSTEM)
    - 覆盖指代消解的多种场景
    - 覆盖关键词提取的中英文场景
    - 覆盖 L1 拦截器的各种模式

作者: HiveMemory Team
版本: 1.0.0
"""

from typing import List, Dict, Any


# ========== 意图识别测试数据 ==========

INTENT_TEST_CASES = [
    # GW-INT-001: 显式检索意图识别 (P0)
    {
        "id": "GW-INT-001",
        "name": "显式检索意图识别",
        "priority": "P0",
        "query": "Rust 的所有权机制是什么？",
        "context": [],
        "expected_intent": "RAG",
        "expected_rewritten_contains": ["Rust", "所有权"],
        "description": "明确询问 factual 信息的查询应识别为 RAG 意图",
    },
    # GW-INT-002: 闲聊意图识别 (P1)
    {
        "id": "GW-INT-002",
        "name": "闲聊意图识别",
        "priority": "P1",
        "query": "今天天气不错",
        "context": [],
        "expected_intent": "CHAT",
        "expected_rewritten_contains": [],
        "description": "简单的打招呼或情感表达应识别为 CHAT 意图",
    },
    # GW-INT-003: 系统指令识别 (P0)
    {
        "id": "GW-INT-003",
        "name": "系统指令识别",
        "priority": "P0",
        "query": "/clear",
        "context": [],
        "expected_intent": "SYSTEM",
        "expected_rewritten_contains": [],
        "description": "系统指令格式的查询应识别为 SYSTEM 意图",
    },
    # GW-INT-004: 模糊意图处理 (P2)
    {
        "id": "GW-INT-004",
        "name": "模糊意图处理",
        "priority": "P2",
        "query": "你觉得 Python 怎么样？",
        "context": [],
        "expected_intent": ["RAG", "CHAT"],  # 允许两种结果
        "expected_rewritten_contains": ["Python"],
        "description": "既像闲聊又像询问的查询，系统应有明确倾向",
    },
    # 额外测试用例
    {
        "id": "GW-INT-005",
        "name": "技术问题识别",
        "priority": "P1",
        "query": "如何在 Docker 中配置网络？",
        "context": [],
        "expected_intent": "RAG",
        "expected_rewritten_contains": ["Docker", "网络"],
        "description": "技术问题应识别为 RAG 意图",
    },
    {
        "id": "GW-INT-006",
        "name": "感谢表达识别",
        "priority": "P1",
        "query": "谢谢你的帮助！",
        "context": [],
        "expected_intent": "CHAT",
        "expected_rewritten_contains": [],
        "description": "感谢表达应识别为 CHAT 意图",
    },
]


# ========== 查询重写与指代消解测试数据 ==========

COREFERENCE_TEST_CASES = [
    # GW-RW-001: 单轮指代消解 (P0)
    {
        "id": "GW-RW-001",
        "name": "单轮指代消解",
        "priority": "P0",
        "context": [
            {"role": "user", "content": "介绍下 Docker"},
            {"role": "assistant", "content": "Docker 是一个开源的容器化平台，可以让开发者打包应用及其依赖到一个可移植的容器中..."},
        ],
        "query": "它怎么安装？",
        "expected_rewritten_contains": ["Docker", "安装"],
        "expected_intent": "RAG",
        "description": "代词'它'应被消解为上文中的 Docker",
    },
    # GW-RW-002: 跨多轮指代消解 (P1)
    {
        "id": "GW-RW-002",
        "name": "跨多轮指代消解",
        "priority": "P1",
        "context": [
            {"role": "user", "content": "我有两个项目，A项目是用 Python 写的"},
            {"role": "assistant", "content": "好的，A项目使用 Python 开发。"},
            {"role": "user", "content": "B项目是用 Java 写的"},
            {"role": "assistant", "content": "明白了，B项目使用 Java 开发。"},
        ],
        "query": "前者是用什么语言写的？",
        "expected_rewritten_contains": ["A项目", "语言"],
        "expected_intent": "RAG",
        "description": "'前者'应被消解为 A项目",
    },
    # GW-RW-003: 无需重写保持原样 (P1)
    {
        "id": "GW-RW-003",
        "name": "无需重写保持原样",
        "priority": "P1",
        "context": [],
        "query": "介绍一下 Kubernetes",
        "expected_rewritten_contains": ["Kubernetes"],
        "expected_intent": "RAG",
        "description": "语义完整的查询应保持原意",
    },
    # 额外测试用例
    {
        "id": "GW-RW-004",
        "name": "省略主语消解",
        "priority": "P1",
        "context": [
            {"role": "user", "content": "Python 的装饰器是什么？"},
            {"role": "assistant", "content": "装饰器是一种设计模式，可以在不修改原函数的情况下添加功能..."},
        ],
        "query": "能给个例子吗？",
        "expected_rewritten_contains": ["装饰器", "例"],
        "expected_intent": "RAG",
        "description": "省略的主语应从上下文中推断",
    },
    {
        "id": "GW-RW-005",
        "name": "复杂指代消解",
        "priority": "P2",
        "context": [
            {"role": "user", "content": "帮我写一个贪吃蛇游戏"},
            {"role": "assistant", "content": "好的，我来帮你用 Python 实现一个贪吃蛇游戏..."},
        ],
        "query": "怎么部署它？",
        "expected_rewritten_contains": ["部署", "贪吃蛇"],
        "expected_intent": "RAG",
        "description": "'它'应被消解为贪吃蛇游戏",
    },
]


# ========== 关键词提取测试数据 ==========

KEYWORD_TEST_CASES = [
    # GW-KW-001: 英文技术名词提取 (P1)
    {
        "id": "GW-KW-001",
        "name": "英文技术名词提取",
        "priority": "P1",
        "query": "如何在 FastAPI 中使用 Pydantic？",
        "context": [],
        "expected_keywords_any": ["FastAPI", "Pydantic"],
        "description": "应提取出英文技术名词作为关键词",
    },
    # GW-KW-002: 中文实体提取 (P1)
    {
        "id": "GW-KW-002",
        "name": "中文实体提取",
        "priority": "P1",
        "query": "鲁迅的《狂人日记》讲了什么？",
        "context": [],
        "expected_keywords_any": ["鲁迅", "狂人日记"],
        "description": "应提取出中文人名和作品名作为关键词",
    },
    # 额外测试用例
    {
        "id": "GW-KW-003",
        "name": "混合语言关键词提取",
        "priority": "P1",
        "query": "如何用 TensorFlow 实现卷积神经网络？",
        "context": [],
        "expected_keywords_any": ["TensorFlow", "卷积神经网络", "CNN"],
        "description": "应提取出中英文混合的技术名词",
    },
    {
        "id": "GW-KW-004",
        "name": "多关键词提取",
        "priority": "P1",
        "query": "比较 React、Vue 和 Angular 的优缺点",
        "context": [],
        "expected_keywords_any": ["React", "Vue", "Angular"],
        "description": "应提取出多个并列的技术名词",
    },
]


# ========== 拦截器测试数据 ==========

INTERCEPTOR_TEST_CASES = [
    # GW-L1-001: 正则拦截优先于 LLM (P0)
    {
        "id": "GW-L1-001",
        "name": "正则拦截优先于 LLM - 系统指令",
        "priority": "P0",
        "query": "/clear",
        "context": [],
        "expected_intent": "SYSTEM",
        "expected_l1_intercepted": True,
        "description": "系统指令应被 L1 拦截，不调用 LLM",
    },
    # GW-L1-002: 闲聊模式拦截 (P1)
    {
        "id": "GW-L1-002",
        "name": "闲聊模式拦截 - 问候语",
        "priority": "P1",
        "query": "你好",
        "context": [],
        "expected_intent": "CHAT",
        "expected_l1_intercepted": True,
        "description": "简单问候语应被 L1 拦截",
    },
    # 额外测试用例
    {
        "id": "GW-L1-003",
        "name": "闲聊模式拦截 - 感谢",
        "priority": "P1",
        "query": "谢谢",
        "context": [],
        "expected_intent": "CHAT",
        "expected_l1_intercepted": True,
        "description": "感谢词应被 L1 拦截",
    },
    {
        "id": "GW-L1-004",
        "name": "闲聊模式拦截 - 英文问候",
        "priority": "P1",
        "query": "hello",
        "context": [],
        "expected_intent": "CHAT",
        "expected_l1_intercepted": True,
        "description": "英文问候语应被 L1 拦截",
    },
    {
        "id": "GW-L1-005",
        "name": "系统指令拦截 - reset",
        "priority": "P1",
        "query": "/reset",
        "context": [],
        "expected_intent": "SYSTEM",
        "expected_l1_intercepted": True,
        "description": "/reset 指令应被 L1 拦截",
    },
    {
        "id": "GW-L1-006",
        "name": "非拦截查询 - 技术问题",
        "priority": "P1",
        "query": "如何配置 Nginx 反向代理？",
        "context": [],
        "expected_intent": "RAG",
        "expected_l1_intercepted": False,
        "description": "技术问题不应被 L1 拦截，应走 L2 分析",
    },
]


# ========== Fallback 测试数据 ==========

FALLBACK_TEST_CASES = [
    {
        "id": "GW-FB-001",
        "name": "Fallback 默认值验证",
        "priority": "P1",
        "original_query": "测试查询",
        "expected_fallback_intent": "CHAT",
        "expected_fallback_worth_saving": False,
        "expected_fallback_parse_failed": True,
        "description": "Fallback 结果应返回保守的默认值",
    },
]


# ========== 辅助函数 ==========

def get_test_cases_by_priority(priority: str) -> List[Dict[str, Any]]:
    """
    按优先级获取测试用例

    Args:
        priority: 优先级 ("P0", "P1", "P2")

    Returns:
        匹配优先级的测试用例列表
    """
    all_cases = (
        INTENT_TEST_CASES +
        COREFERENCE_TEST_CASES +
        KEYWORD_TEST_CASES +
        INTERCEPTOR_TEST_CASES
    )
    return [case for case in all_cases if case.get("priority") == priority]


def get_p0_test_cases() -> List[Dict[str, Any]]:
    """获取所有 P0 优先级测试用例"""
    return get_test_cases_by_priority("P0")


def get_p1_test_cases() -> List[Dict[str, Any]]:
    """获取所有 P1 优先级测试用例"""
    return get_test_cases_by_priority("P1")


# ========== 导出 ==========

__all__ = [
    # 测试数据
    "INTENT_TEST_CASES",
    "COREFERENCE_TEST_CASES",
    "KEYWORD_TEST_CASES",
    "INTERCEPTOR_TEST_CASES",
    "FALLBACK_TEST_CASES",
    # 辅助函数
    "get_test_cases_by_priority",
    "get_p0_test_cases",
    "get_p1_test_cases",
]
