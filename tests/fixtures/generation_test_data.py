"""
HiveMemory Generation Module Test Data Fixtures

提供专门用于 Generation 模块端到端测试的数据。

设计原则:
    - 覆盖记忆提取的各种场景（标准信息、噪音、代码片段）
    - 覆盖去重决策的各种情况（CREATE、UPDATE、TOUCH）
    - 覆盖记忆合并的各种策略

作者: HiveMemory Team
版本: 1.0.0
"""

from typing import List, Dict, Any
from datetime import datetime


# ========== Group 1: 记忆提取测试数据 ==========

EXTRACTION_TEST_CASES = [
    {
        "id": "GEN-EXT-001",
        "name": "标准信息提取",
        "description": "包含事实性信息的对话，应提取出有价值的记忆",
        "priority": "P0",
        "messages": [
            {
                "role": "user",
                "content": """我的项目使用的API Key是 sk-proj-abc123xyz，
请帮我记住这个配置信息。这是用于访问OpenAI服务的密钥，
部署在生产环境的服务器上。"""
            },
            {
                "role": "assistant",
                "content": """好的，我已经记录了您的API Key配置信息：
- API Key: sk-proj-abc123xyz
- 用途: OpenAI服务访问
- 环境: 生产环境

请注意妥善保管此密钥，避免泄露。如需更换密钥，请及时通知我更新记录。"""
            }
        ],
        "expected_has_value": True,
        "expected_title_contains": ["API", "Key"],
        "expected_memory_type": "FACT",
    },
    {
        "id": "GEN-EXT-002",
        "name": "噪音过滤",
        "description": "无营养的闲聊对话，应判定为无价值",
        "priority": "P0",
        "messages": [
            {
                "role": "user",
                "content": "好的，谢谢你的帮助"
            },
            {
                "role": "assistant",
                "content": "不客气，有问题随时问我"
            }
        ],
        "expected_has_value": False,
        "expected_draft_is_none": True,
    },
    {
        "id": "GEN-EXT-003",
        "name": "复杂结构提取 - 代码片段",
        "description": "包含代码的对话，应完整提取代码内容",
        "priority": "P1",
        "messages": [
            {
                "role": "user",
                "content": "Python的冒泡排序怎么写？请给我一个完整的实现"
            },
            {
                "role": "assistant",
                "content": """这是Python冒泡排序的完整实现：

```python
def bubble_sort(arr):
    \"\"\"
    冒泡排序算法实现

    Args:
        arr: 待排序的列表

    Returns:
        排序后的列表
    \"\"\"
    n = len(arr)
    for i in range(n):
        # 标记是否发生交换
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        # 如果没有交换，说明已经有序
        if not swapped:
            break
    return arr

# 使用示例
numbers = [64, 34, 25, 12, 22, 11, 90]
sorted_numbers = bubble_sort(numbers)
print(sorted_numbers)  # [11, 12, 22, 25, 34, 64, 90]
```

时间复杂度: O(n²)，空间复杂度: O(1)"""
            }
        ],
        "expected_has_value": True,
        "expected_title_contains": ["冒泡", "排序"],
        "expected_tags_any": ["Python", "算法", "排序"],
        "expected_memory_type": "CODE_SNIPPET",
        "expected_content_contains": ["def bubble_sort", "swapped"],
    },
    {
        "id": "GEN-EXT-004",
        "name": "用户偏好提取",
        "description": "包含用户偏好信息的对话",
        "priority": "P1",
        "messages": [
            {
                "role": "user",
                "content": """我习惯使用VSCode作为主要开发工具，
配合Vim插件进行编辑。我喜欢深色主题，
字体大小设置为14px，使用JetBrains Mono字体。"""
            },
            {
                "role": "assistant",
                "content": """了解了，您的开发环境偏好：
- IDE: VSCode + Vim插件
- 主题: 深色主题
- 字体: JetBrains Mono, 14px

这些设置非常适合长时间编码，深色主题可以减少眼睛疲劳。"""
            }
        ],
        "expected_has_value": True,
        "expected_memory_type": "USER_PROFILE",
        "expected_tags_any": ["VSCode", "开发环境", "偏好"],
    },
]


# ========== Group 2: 去重决策测试数据 ==========

# 预置记忆 - 用于去重测试
EXISTING_MEMORY_DATA = {
    "pytorch_install": {
        "id": "550e8400-e29b-41d4-a716-446655440001",
        "title": "PyTorch 安装指南",
        "summary": "PyTorch 深度学习框架的安装步骤和环境配置方法",
        "tags": ["PyTorch", "安装", "深度学习", "Python"],
        "memory_type": "FACT",
        "content": """PyTorch 安装步骤：

1. 确认Python版本 (推荐3.8+)
2. 安装命令:
   ```bash
   pip install torch torchvision torchaudio
   ```
3. GPU版本需要先安装CUDA
4. 验证安装:
   ```python
   import torch
   print(torch.__version__)
   print(torch.cuda.is_available())
   ```""",
        "confidence_score": 0.85,
        "user_id": "test_user",
        "agent_id": "test_agent",
        "session_id": "test_session",
    },
    "meeting_time": {
        "id": "550e8400-e29b-41d4-a716-446655440002",
        "title": "项目周会时间",
        "summary": "每周项目例会的时间安排",
        "tags": ["会议", "项目管理", "时间"],
        "memory_type": "FACT",
        "content": "项目周会时间：每周一上午10点，会议室A301",
        "confidence_score": 0.90,
        "user_id": "test_user",
        "agent_id": "test_agent",
        "session_id": "test_session",
    },
}


DEDUPLICATION_TEST_CASES = [
    {
        "id": "GEN-DED-001",
        "name": "决策 CREATE - 新记忆",
        "description": "与现有记忆相似度极低，应创建新记忆",
        "priority": "P0",
        "draft_data": {
            "title": "Rust 语言入门教程",
            "summary": "Rust 编程语言的基础语法和所有权概念",
            "tags": ["Rust", "编程语言", "系统编程"],
            "memory_type": "FACT",
            "content": "Rust 是一门注重安全性和性能的系统编程语言...",
            "confidence_score": 0.80,
            "has_value": True,
        },
        "existing_memory_key": "pytorch_install",
        "expected_decision": "CREATE",
    },
    {
        "id": "GEN-DED-002",
        "name": "决策 TOUCH - 完全重复",
        "description": "与现有记忆高度相似且内容一致，仅更新访问时间",
        "priority": "P1",
        "draft_data": {
            "title": "PyTorch 安装指南",
            "summary": "PyTorch 深度学习框架的安装步骤和环境配置方法",
            "tags": ["PyTorch", "安装", "深度学习"],
            "memory_type": "FACT",
            "content": """PyTorch 安装步骤：

1. 确认Python版本 (推荐3.8+)
2. 安装命令:
   ```bash
   pip install torch torchvision torchaudio
   ```
3. GPU版本需要先安装CUDA
4. 验证安装:
   ```python
   import torch
   print(torch.__version__)
   print(torch.cuda.is_available())
   ```""",
            "confidence_score": 0.85,
            "has_value": True,
        },
        "existing_memory_key": "pytorch_install",
        "expected_decision": "TOUCH",
    },
    {
        "id": "GEN-DED-003",
        "name": "决策 UPDATE - 知识演化",
        "description": "与现有记忆相似但内容有增量，应合并更新",
        "priority": "P0",
        "draft_data": {
            "title": "项目周会时间调整",
            "summary": "项目例会时间从10点改到11点",
            "tags": ["会议", "项目管理", "时间"],
            "memory_type": "FACT",
            "content": "项目周会时间调整：从每周一上午10点改为11点，地点不变（会议室A301）",
            "confidence_score": 0.88,
            "has_value": True,
        },
        "existing_memory_key": "meeting_time",
        "expected_decision": "UPDATE",
    },
]


# ========== Group 3: 记忆合并测试数据 ==========

MERGE_TEST_CASES = [
    {
        "id": "GEN-MRG-001",
        "name": "内容追加合并",
        "description": "验证合并后内容包含旧内容和新内容",
        "priority": "P0",
        "existing_memory": {
            "title": "Python 虚拟环境",
            "summary": "Python 虚拟环境的创建方法",
            "tags": ["Python", "虚拟环境"],
            "content": "使用 venv 创建虚拟环境：python -m venv myenv",
            "confidence_score": 0.80,
        },
        "new_draft": {
            "title": "Python 虚拟环境管理",
            "summary": "Python 虚拟环境的创建和激活方法，包括 Windows 和 Linux",
            "tags": ["Python", "虚拟环境", "环境管理"],
            "memory_type": "FACT",
            "content": """激活虚拟环境：
- Windows: myenv\\Scripts\\activate
- Linux/Mac: source myenv/bin/activate""",
            "confidence_score": 0.85,
            "has_value": True,
        },
        "expected_content_contains": ["venv", "activate", "Windows", "Linux"],
        "expected_tags": ["Python", "虚拟环境", "环境管理"],
    },
    {
        "id": "GEN-MRG-002",
        "name": "标签并集合并",
        "description": "验证合并后标签为两者并集",
        "priority": "P1",
        "existing_memory": {
            "title": "Docker 基础",
            "summary": "Docker 容器技术基础",
            "tags": ["Docker", "容器", "DevOps"],
            "content": "Docker 是一个容器化平台...",
            "confidence_score": 0.75,
        },
        "new_draft": {
            "title": "Docker 基础与实践",
            "summary": "Docker 容器技术基础和常用命令",
            "tags": ["Docker", "容器", "部署", "微服务"],
            "memory_type": "FACT",
            "content": "常用 Docker 命令：docker run, docker build...",
            "confidence_score": 0.80,
            "has_value": True,
        },
        "expected_tags_superset": ["Docker", "容器", "DevOps", "部署", "微服务"],
        "expected_max_tags": 5,
    },
    {
        "id": "GEN-MRG-003",
        "name": "摘要更新策略",
        "description": "验证合并后选择较长的摘要",
        "priority": "P2",
        "existing_memory": {
            "title": "Git 分支管理",
            "summary": "Git 分支操作的基本命令和使用方法",
            "tags": ["Git", "版本控制"],
            "content": "git branch, git checkout...",
            "confidence_score": 0.70,
        },
        "new_draft": {
            "title": "Git 分支管理最佳实践",
            "summary": "Git 分支的创建、切换、合并和删除操作，以及 GitFlow 工作流介绍",
            "tags": ["Git", "版本控制", "GitFlow"],
            "memory_type": "FACT",
            "content": "GitFlow 工作流包括 main, develop, feature, release, hotfix 分支...",
            "confidence_score": 0.85,
            "has_value": True,
        },
        "expected_summary_longer": True,
    },
]


# ========== Group 4: Schema 验证测试数据 ==========

SCHEMA_VALIDATION_CASES = [
    {
        "id": "GEN-SCH-001",
        "name": "JSON Schema 合规",
        "description": "验证生成的 MemoryAtom 包含所有必需字段",
        "priority": "P0",
        "messages": [
            {
                "role": "user",
                "content": "我的数据库连接字符串是 postgresql://user:pass@localhost:5432/mydb"
            },
            {
                "role": "assistant",
                "content": "已记录您的数据库连接信息。请注意保护好密码，建议使用环境变量存储敏感信息。"
            }
        ],
        "required_fields": [
            "id",
            "meta.user_id",
            "meta.source_agent_id",
            "meta.session_id",
            "meta.confidence_score",
            "index.title",
            "index.summary",
            "index.tags",
            "index.memory_type",
            "payload.content",
        ],
    },
    {
        "id": "GEN-SCH-002",
        "name": "置信度加权计算",
        "description": "验证合并后置信度按 0.6*old + 0.4*new 计算",
        "priority": "P1",
        "existing_confidence": 0.80,
        "new_confidence": 0.90,
        "expected_merged_confidence": 0.84,  # 0.6 * 0.80 + 0.4 * 0.90 = 0.84
        "tolerance": 0.01,
    },
]


# ========== 辅助函数 ==========

def get_extraction_test_by_id(test_id: str) -> Dict[str, Any]:
    """根据 ID 获取提取测试用例"""
    for case in EXTRACTION_TEST_CASES:
        if case["id"] == test_id:
            return case
    raise ValueError(f"Test case not found: {test_id}")


def get_deduplication_test_by_id(test_id: str) -> Dict[str, Any]:
    """根据 ID 获取去重测试用例"""
    for case in DEDUPLICATION_TEST_CASES:
        if case["id"] == test_id:
            return case
    raise ValueError(f"Test case not found: {test_id}")


def get_merge_test_by_id(test_id: str) -> Dict[str, Any]:
    """根据 ID 获取合并测试用例"""
    for case in MERGE_TEST_CASES:
        if case["id"] == test_id:
            return case
    raise ValueError(f"Test case not found: {test_id}")


def get_p0_test_cases() -> List[Dict[str, Any]]:
    """获取所有 P0 优先级测试用例"""
    p0_cases = []
    for case in EXTRACTION_TEST_CASES:
        if case.get("priority") == "P0":
            p0_cases.append(case)
    for case in DEDUPLICATION_TEST_CASES:
        if case.get("priority") == "P0":
            p0_cases.append(case)
    for case in MERGE_TEST_CASES:
        if case.get("priority") == "P0":
            p0_cases.append(case)
    for case in SCHEMA_VALIDATION_CASES:
        if case.get("priority") == "P0":
            p0_cases.append(case)
    return p0_cases


# ========== 导出 ==========

__all__ = [
    # 测试用例
    "EXTRACTION_TEST_CASES",
    "DEDUPLICATION_TEST_CASES",
    "MERGE_TEST_CASES",
    "SCHEMA_VALIDATION_CASES",
    # 预置数据
    "EXISTING_MEMORY_DATA",
    # 辅助函数
    "get_extraction_test_by_id",
    "get_deduplication_test_by_id",
    "get_merge_test_by_id",
    "get_p0_test_cases",
]
