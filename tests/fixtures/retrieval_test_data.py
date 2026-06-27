"""
HiveMemory Retrieval Module Test Data Fixtures

提供专门用于 Retrieval 模块端到端测试的数据。

设计原则:
    - 覆盖混合检索的各种场景（纯语义、纯关键词、混合冲突）
    - 覆盖重排序的各种情况（精排优化、阈值过滤）
    - 覆盖检索上下文编译的各种策略（Full、Compact、Cascade）

作者: HiveMemory Team
版本: 1.0.0
"""

from typing import List, Dict, Any


# ========== Golden Memories - 预注入 Qdrant 的标准记忆库 ==========

GOLDEN_MEMORIES = [
    # ========== 语义相关组 - 水果 ==========
    {
        "id": "550e8400-e29b-41d4-a716-446655440101",  # golden-fruit-001
        "title": "苹果的营养价值",
        "summary": "苹果是一种常见的水果，富含多种维生素和矿物质，对人体健康有益",
        "tags": ["水果", "营养", "健康"],
        "memory_type": "FACT",
        "content": """苹果是世界上最受欢迎的水果之一，具有丰富的营养价值：

1. 维生素含量：
   - 维生素C：增强免疫力
   - 维生素A：保护视力
   - 维生素K：促进血液凝固

2. 矿物质：
   - 钾：维持心脏健康
   - 磷：强健骨骼

3. 膳食纤维：
   - 促进消化
   - 降低胆固醇

每天一个苹果，医生远离我。""",
        "confidence_score": 0.90,
    },
    {
        "id": "550e8400-e29b-41d4-a716-446655440102",  # golden-fruit-002
        "title": "香蕉的功效与作用",
        "summary": "香蕉含有丰富的钾元素和膳食纤维，是运动后补充能量的理想选择",
        "tags": ["水果", "健康", "运动"],
        "memory_type": "FACT",
        "content": """香蕉是热带水果中的佼佼者，营养丰富：

1. 钾元素含量高：
   - 预防肌肉痉挛
   - 维持电解质平衡
   - 降低血压

2. 快速能量来源：
   - 天然糖分易吸收
   - 运动前后的理想食物

3. 促进消化：
   - 富含膳食纤维
   - 改善肠道健康

香蕉还含有色氨酸，有助于改善情绪。""",
        "confidence_score": 0.88,
    },
    {
        "id": "550e8400-e29b-41d4-a716-446655440103",  # golden-fruit-003
        "title": "橙子的维生素C含量",
        "summary": "橙子是维生素C的优质来源，有助于增强免疫系统",
        "tags": ["水果", "维生素", "免疫"],
        "memory_type": "FACT",
        "content": """橙子是柑橘类水果的代表，以其丰富的维生素C著称：

1. 维生素C含量：
   - 一个中等大小的橙子含约70mg维生素C
   - 满足成人每日需求量的78%

2. 其他营养成分：
   - 叶酸
   - 钾
   - 膳食纤维

3. 健康益处：
   - 增强免疫力
   - 促进铁吸收
   - 抗氧化作用""",
        "confidence_score": 0.85,
    },

    # ========== 关键词精确匹配组 ==========
    {
        "id": "550e8400-e29b-41d4-a716-446655440201",  # golden-config-001
        "title": "X-1024 服务器配置单",
        "summary": "X-1024 型号服务器的详细配置参数和部署说明",
        "tags": ["配置", "X-1024", "服务器"],
        "memory_type": "FACT",
        "content": """X-1024 服务器配置详情：

型号: X-1024 Enterprise Edition
CPU: Intel Xeon Gold 6248R x 2
内存: 256GB DDR4 ECC
存储: 4TB NVMe SSD RAID 10
网络: 10GbE x 4

部署参数:
- 机架位置: DC-A-R12-U24
- IP地址: 192.168.1.100
- 管理端口: 8443

注意事项:
- 需要配置 IPMI 远程管理
- 建议启用 RAID 监控告警""",
        "confidence_score": 0.95,
    },
    {
        "id": "550e8400-e29b-41d4-a716-446655440202",  # golden-config-002
        "title": "X-1025 测试环境配置",
        "summary": "X-1025 测试服务器的基础配置信息",
        "tags": ["配置", "X-1025", "测试"],
        "memory_type": "FACT",
        "content": """X-1025 测试服务器配置：

型号: X-1025 Standard
CPU: Intel Xeon Silver 4214R
内存: 64GB DDR4
存储: 1TB SSD

用途: 开发测试环境
状态: 运行中""",
        "confidence_score": 0.80,
    },

    # ========== 混合冲突组 - 苹果公司 vs 水果苹果 ==========
    {
        "id": "550e8400-e29b-41d4-a716-446655440301",  # golden-apple-stock-001
        "title": "Apple Inc. 股票分析报告",
        "summary": "苹果公司(AAPL)的股价走势分析和投资建议",
        "tags": ["股票", "Apple", "投资", "科技"],
        "memory_type": "FACT",
        "content": """Apple Inc. (AAPL) 股票分析：

公司概况:
- 全球市值最高的科技公司之一
- 主要产品: iPhone, Mac, iPad, Apple Watch
- 服务业务持续增长

股价表现:
- 近一年涨幅: +25%
- 市盈率(P/E): 28.5
- 股息收益率: 0.5%

投资建议:
苹果公司的股价受益于其强大的生态系统和服务收入增长。
建议长期持有，关注新产品发布周期。""",
        "confidence_score": 0.92,
    },
    {
        "id": "550e8400-e29b-41d4-a716-446655440302",  # golden-apple-fruit-001
        "title": "苹果的品种分类",
        "summary": "介绍常见的苹果品种，包括红富士、青苹果、嘎啦等",
        "tags": ["水果", "苹果", "品种"],
        "memory_type": "FACT",
        "content": """常见苹果品种介绍：

1. 红富士 (Fuji)
   - 原产日本
   - 口感脆甜，汁水丰富
   - 最受欢迎的品种之一

2. 青苹果 (Granny Smith)
   - 原产澳大利亚
   - 口感酸脆
   - 适合烘焙

3. 嘎啦 (Gala)
   - 原产新西兰
   - 甜度高，略带香气
   - 适合鲜食

4. 黄元帅 (Golden Delicious)
   - 原产美国
   - 果肉细腻
   - 适合做沙拉""",
        "confidence_score": 0.87,
    },

    # ========== 无关数据 - 用于阈值过滤测试 ==========
    {
        "id": "550e8400-e29b-41d4-a716-446655440401",  # golden-earth-001
        "title": "地球的内部结构",
        "summary": "地球由地壳、地幔和地核三层结构组成",
        "tags": ["科学", "地球", "地质"],
        "memory_type": "FACT",
        "content": """地球内部结构：

1. 地壳 (Crust)
   - 厚度: 5-70公里
   - 组成: 硅酸盐岩石

2. 地幔 (Mantle)
   - 厚度: 约2900公里
   - 温度: 1000-3700°C

3. 地核 (Core)
   - 外核: 液态铁镍
   - 内核: 固态铁镍
   - 温度: 约5000°C""",
        "confidence_score": 0.85,
    },
    {
        "id": "550e8400-e29b-41d4-a716-446655440402",  # golden-mars-001
        "title": "火星探测任务",
        "summary": "人类对火星的探测历史和未来计划",
        "tags": ["科学", "火星", "太空"],
        "memory_type": "FACT",
        "content": """火星探测任务概览：

历史任务:
- 1971: 苏联火星3号首次软着陆
- 1997: 美国火星探路者号
- 2012: 好奇号火星车

当前任务:
- 毅力号火星车 (2021)
- 祝融号火星车 (2021)

未来计划:
- 火星样本返回任务
- 载人火星任务""",
        "confidence_score": 0.82,
    },

    # ========== 代码片段组 - 用于测试代码类型记忆 ==========
    {
        "id": "550e8400-e29b-41d4-a716-446655440501",  # golden-code-001
        "title": "Python 快速排序实现",
        "summary": "使用 Python 实现的快速排序算法，包含详细注释",
        "tags": ["Python", "算法", "排序", "代码"],
        "memory_type": "CODE_SNIPPET",
        "content": """```python
def quicksort(arr):
    \"\"\"
    快速排序算法实现

    时间复杂度: O(n log n) 平均, O(n^2) 最坏
    空间复杂度: O(log n)
    \"\"\"
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)

# 使用示例
numbers = [3, 6, 8, 10, 1, 2, 1]
print(quicksort(numbers))  # [1, 1, 2, 3, 6, 8, 10]
```""",
        "confidence_score": 0.93,
    },

    # ========== 用户偏好组 ==========
    {
        "id": "550e8400-e29b-41d4-a716-446655440601",  # golden-pref-001
        "title": "用户开发环境偏好",
        "summary": "用户偏好使用 VSCode 配合 Vim 插件进行开发",
        "tags": ["偏好", "开发环境", "VSCode"],
        "memory_type": "USER_PROFILE",
        "content": """用户开发环境配置偏好：

IDE: Visual Studio Code
- 主题: One Dark Pro
- 字体: JetBrains Mono, 14px
- 插件: Vim, GitLens, Prettier

终端: Windows Terminal + PowerShell 7
Git 客户端: 命令行 + GitKraken

编程语言偏好:
- 主力: Python, TypeScript
- 学习中: Rust""",
        "confidence_score": 0.88,
    },
]


# ========== Group 1: 混合检索测试用例 ==========

HYBRID_SEARCH_TEST_CASES = [
    {
        "id": "RET-HYB-001",
        "name": "纯语义召回",
        "description": "搜索语义相关但无关键词重叠的 Query，验证 Top-K 结果",
        "priority": "P0",
        "query": "水果",
        "expected_recall_ids": [
            "550e8400-e29b-41d4-a716-446655440101",  # golden-fruit-001
            "550e8400-e29b-41d4-a716-446655440102",  # golden-fruit-002
            "550e8400-e29b-41d4-a716-446655440103",  # golden-fruit-003
        ],
        "expected_not_recall_ids": [],
        "min_recall_count": 2,
        "validation": {
            "type": "semantic_recall",
            "description": "应召回苹果、香蕉、橙子等水果相关记忆",
        },
    },
    {
        "id": "RET-HYB-002",
        "name": "纯关键词召回",
        "description": "搜索包含特定专有名词的 Query，验证精确匹配优先",
        "priority": "P0",
        "query": "X-1024 参数",
        "expected_recall_ids": ["550e8400-e29b-41d4-a716-446655440201"],  # golden-config-001
        "expected_top1_id": "550e8400-e29b-41d4-a716-446655440201",  # golden-config-001
        "expected_not_recall_ids": [],
        "validation": {
            "type": "keyword_recall",
            "description": "应优先召回包含 X-1024 关键词的记忆",
        },
    },
    {
        "id": "RET-HYB-003",
        "name": "混合冲突处理",
        "description": "验证 RRF 融合效果，语义+词匹配优先于仅词匹配",
        "priority": "P1",
        "query": "苹果公司的股价",
        "expected_top1_id": "550e8400-e29b-41d4-a716-446655440301",  # golden-apple-stock-001
        "expected_recall_ids": ["550e8400-e29b-41d4-a716-446655440301"],  # golden-apple-stock-001
        "should_rank_higher": {
            "higher": "550e8400-e29b-41d4-a716-446655440301",  # golden-apple-stock-001
            "lower": "550e8400-e29b-41d4-a716-446655440302",  # golden-apple-fruit-001
        },
        "validation": {
            "type": "hybrid_conflict",
            "description": "Apple Stock 应排在水果苹果之前",
        },
    },
]


# ========== Group 2: 重排序测试用例 ==========

RERANKING_TEST_CASES = [
    {
        "id": "RET-RNK-001",
        "name": "精排优化",
        "description": "粗排 Top-1 并非最优时，Rerank 后正确重排序",
        "priority": "P0",
        "query": "Python 排序算法实现",
        "candidates_order": [
            "550e8400-e29b-41d4-a716-446655440101",  # golden-fruit-001 (弱相关)
            "550e8400-e29b-41d4-a716-446655440501",  # golden-code-001 (强相关)
            "550e8400-e29b-41d4-a716-446655440401",  # golden-earth-001 (无关)
        ],
        "expected_top1_after_rerank": "550e8400-e29b-41d4-a716-446655440501",  # golden-code-001
        "validation": {
            "type": "rerank_optimization",
            "description": "Rerank 后代码片段应排在 Top-1",
        },
    },
    {
        "id": "RET-RNK-002",
        "name": "阈值过滤",
        "description": "无关 Query 经 Rerank 后返回空列表或低分结果",
        "priority": "P1",
        "query": "外星人入侵地球的电影推荐",
        "score_threshold": 0.5,
        "expected_empty_or_low_score": True,
        "validation": {
            "type": "threshold_filtering",
            "description": "所有结果分数应低于阈值",
        },
    },
]


# ========== Group 3: 上下文编译测试用例 ==========

RENDERING_TEST_CASES = [
    {
        "id": "RET-RND-001",
        "name": "Full 策略上下文编译",
        "description": "验证 Full 策略输出包含 memory_context 结构和完整内容",
        "priority": "P0",
        "strategy": "full",
        "memory_ids": [
            "550e8400-e29b-41d4-a716-446655440101",  # golden-fruit-001
            "550e8400-e29b-41d4-a716-446655440102",  # golden-fruit-002
        ],
        "expected_contains": [
            "<memory_context>",
            "</memory_context>",
            "相关记忆",
        ],
        "expected_not_contains": [],
    },
    {
        "id": "RET-RND-002",
        "name": "Compact 策略上下文编译",
        "description": "验证 Compact 策略输出包含摘要且省略完整内容",
        "priority": "P1",
        "strategy": "compact",
        "memory_ids": ["550e8400-e29b-41d4-a716-446655440501"],  # golden-code-001
        "expected_contains": [
            "<memory_context>",
            "相关记忆",
            "摘要",
        ],
        "expected_not_contains": [
            "def quicksort",
        ],
    },
    {
        "id": "RET-RND-003",
        "name": "Cascade 策略上下文编译",
        "description": "验证 Top-N 完整编译，其余降级为 Index 视图",
        "priority": "P1",
        "strategy": "cascade",
        "memory_ids": [
            "550e8400-e29b-41d4-a716-446655440101",  # golden-fruit-001
            "550e8400-e29b-41d4-a716-446655440102",  # golden-fruit-002
            "550e8400-e29b-41d4-a716-446655440103",  # golden-fruit-003
        ],
        "full_payload_count": 1,
        "max_memory_tokens": 500,
        "validation": {
            "type": "cascade_rendering",
            "description": "第一条完整渲染，其余为摘要视图",
        },
    },
]


# ========== 辅助函数 ==========

def get_golden_memory_by_id(memory_id: str) -> Dict[str, Any]:
    """根据 ID 获取 Golden Memory"""
    for memory in GOLDEN_MEMORIES:
        if memory["id"] == memory_id:
            return memory
    raise ValueError(f"Golden memory not found: {memory_id}")


def get_hybrid_test_by_id(test_id: str) -> Dict[str, Any]:
    """根据 ID 获取混合检索测试用例"""
    for case in HYBRID_SEARCH_TEST_CASES:
        if case["id"] == test_id:
            return case
    raise ValueError(f"Test case not found: {test_id}")


def get_reranking_test_by_id(test_id: str) -> Dict[str, Any]:
    """根据 ID 获取重排序测试用例"""
    for case in RERANKING_TEST_CASES:
        if case["id"] == test_id:
            return case
    raise ValueError(f"Test case not found: {test_id}")


def get_rendering_test_by_id(test_id: str) -> Dict[str, Any]:
    """根据 ID 获取渲染测试用例"""
    for case in RENDERING_TEST_CASES:
        if case["id"] == test_id:
            return case
    raise ValueError(f"Test case not found: {test_id}")


def get_p0_test_cases() -> List[Dict[str, Any]]:
    """获取所有 P0 优先级测试用例"""
    p0_cases = []
    for case in HYBRID_SEARCH_TEST_CASES:
        if case.get("priority") == "P0":
            p0_cases.append(case)
    for case in RERANKING_TEST_CASES:
        if case.get("priority") == "P0":
            p0_cases.append(case)
    for case in RENDERING_TEST_CASES:
        if case.get("priority") == "P0":
            p0_cases.append(case)
    return p0_cases


def get_all_golden_memory_ids() -> List[str]:
    """获取所有 Golden Memory 的 ID 列表"""
    return [m["id"] for m in GOLDEN_MEMORIES]


# ========== 导出 ==========

__all__ = [
    # 测试数据
    "GOLDEN_MEMORIES",
    "HYBRID_SEARCH_TEST_CASES",
    "RERANKING_TEST_CASES",
    "RENDERING_TEST_CASES",
    # 辅助函数
    "get_golden_memory_by_id",
    "get_hybrid_test_by_id",
    "get_reranking_test_by_id",
    "get_rendering_test_by_id",
    "get_p0_test_cases",
    "get_all_golden_memory_ids",
]
