"""
HiveMemory 核心常量定义

统一管理系统级默认值，避免魔法字符串散落在代码各处
"""

# ============ 身份标识默认值 ============

DEFAULT_USER_ID = "default"
"""默认用户 ID - 用于未登录/匿名场景"""

DEFAULT_AGENT_ID = "omni_doll"
"""默认 Agent ID - 全能人偶，拥有完整权限"""

DEFAULT_TEAM_ID = None
"""默认团队 ID - None 表示个人作用域"""


# ============ 测试专用身份标识 ============

TEST_USER_ID = "test_user"
"""测试用户 ID - 仅用于单元测试和集成测试"""

TEST_AGENT_ID = "test_agent"
"""测试 Agent ID - 仅用于单元测试"""


# ============ LLM 生成参数默认值 ============
# 次级生成参数的项目级默认。主要参数（模型名、provider、API 凭证）
# 由 ModelRegistry + 前端配置决定，不在此列。
# 覆盖优先级：会话请求 > Agent Profile > 模型定义 > 此处常量

DEFAULT_TEMPERATURE = 1.0
"""默认推理温度 - 模型定义与 Agent Profile 均未显式指定时使用"""

DEFAULT_MAX_TOKENS = 32768
"""默认最大生成 token 数"""

DEFAULT_TOP_P = 1.0
"""默认核采样 (nucleus sampling) 阈值 - 1.0 表示不裁剪概率分布"""


# ============ 身份验证辅助函数 ============

def normalize_user_id(user_id: str | None) -> str:
    """
    规范化 user_id，确保永远不会是 None 或空字符串

    Args:
        user_id: 原始 user_id

    Returns:
        规范化后的 user_id，如果为空则返回 DEFAULT_USER_ID
    """
    return user_id.strip() if user_id and user_id.strip() else DEFAULT_USER_ID


def normalize_agent_id(agent_id: str | None) -> str:
    """
    规范化 agent_id，确保永远不会是 None 或空字符串

    Args:
        agent_id: 原始 agent_id

    Returns:
        规范化后的 agent_id，如果为空则返回 DEFAULT_AGENT_ID
    """
    return agent_id.strip() if agent_id and agent_id.strip() else DEFAULT_AGENT_ID
