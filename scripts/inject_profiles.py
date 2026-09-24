"""
人偶图纸初始化脚本 (Agent Profile Injection)

向指定用户的默认 Workspace 注入基础人偶图纸：
1. coder_doll    - Python 开发者（代码生成 + 文件读写）
2. reviewer_doll - 代码审查员（只读 + 检索）

omni_doll 是代码内置的 fallback Profile（``OMNI_DOLL_PROFILE``），按 alias 解析时
先于存储记录生效，因此不注入存储。

写入走与 ``POST /api/v1/agents`` 相同的管理用例（AgentApplicationService →
Patchouli 完整写入路径）：执行字段校验并生成版本记录。已存在的 alias 跳过，
可重复执行。

使用方式:
    python scripts/inject_profiles.py [--user-id USER]

前置条件:
    - Qdrant 服务已启动（使用与后端相同的配置）
    - 脚本会启动一个完整的 HiveMemorySystem（含后台调度），建议在后端未运行时执行
"""

import argparse
import asyncio
import logging

from hivememory.core.constants import DEFAULT_USER_ID, SYSTEM_AGENT_ID
from hivememory.core.models import ActorIdentity, IdentityScope
from hivememory.core.models.workspace import resolve_default_workspace_identity
from hivememory.system import HiveMemorySystem
from hivememory.system.config import load_app_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============ 人偶图纸定义 ============

PROFILES = [
    {
        "alias": "coder_doll",
        "title": "Python 开发者 (Coder-Doll)",
        "summary": "擅长编写、调试 Python 脚本，拥有读写工作区文件和执行代码的权限。性格严谨，代码必须包含类型提示。",
        "tags": ["agent", "python", "developer", "coder"],
        "persona": (
            "你是一个资深的 Python 程序员，性格严谨，追求代码质量。\n"
            "核心原则：\n"
            "1. 代码必须包含完整的类型提示 (Type Hints)\n"
            "2. 编写代码前，先思考边界条件和异常处理\n"
            "3. 遵循 PEP 8 规范，变量命名清晰有意义\n"
            "4. 复杂逻辑必须添加注释说明设计意图\n"
            "5. 优先使用标准库，避免不必要的第三方依赖"
        ),
        "agent_config": {
            "model_name": "default",
            "temperature": 0.2,
            "allowed_mtp_verbs": ["READ", "RUN", "SEARCH", "WRITE"],
            "allowed_sys_tools": [
                "sys_read_file",
                "sys_write_file",
                "sys_python_repl",
                "sys_clock",
            ],
            "language": "zh",
        },
    },
    {
        "alias": "reviewer_doll",
        "title": "代码审查员 (Reviewer-Doll)",
        "summary": "专注于代码审查，擅长发现安全漏洞和性能瓶颈。仅拥有只读权限，无法写入文件或执行代码。",
        "tags": ["agent", "reviewer", "security", "code-review"],
        "persona": (
            "你是一个严厉的 Code Reviewer，专注于发现代码中的问题。\n"
            "审查重点：\n"
            "1. 安全漏洞：SQL 注入、XSS、路径遍历、命令注入\n"
            "2. 性能瓶颈：N+1 查询、不必要的循环、内存泄漏\n"
            "3. 代码规范：命名一致性、函数长度、圈复杂度\n"
            "4. 边界条件：空值处理、并发安全、资源释放\n"
            "5. 你只能阅读和检索代码，不能修改或执行代码\n"
            "审查时请给出具体的行号和修改建议。"
        ),
        "agent_config": {
            "model_name": "default",
            "temperature": 0.3,
            "allowed_mtp_verbs": ["READ", "SEARCH"],
            "allowed_sys_tools": ["sys_read_file", "sys_clock"],
            "language": "zh",
        },
    },
]


def management_scope(user_id: str) -> IdentityScope:
    """与 server 管理入口相同的身份：保留 system actor + 用户默认 Workspace。"""
    return IdentityScope(
        actor_identity=ActorIdentity(user_id=user_id, agent_id=SYSTEM_AGENT_ID),
        workspace_identity=resolve_default_workspace_identity(user_id),
    )


async def main(user_id: str) -> None:
    """注入尚不存在的人偶图纸；已存在的 alias 跳过。"""
    identity_scope = management_scope(user_id)
    system = HiveMemorySystem.build(config=load_app_config())
    await system.start()
    try:
        existing = {
            atom.index.alias
            for atom in await system.agent_service.list_agent_profiles(
                identity_scope=identity_scope, limit=1000
            )
        }
        for profile in PROFILES:
            alias = profile["alias"]
            if alias in existing:
                logger.info(f"Profile '{alias}' already exists, skipping.")
                continue
            atom = await system.agent_service.create_agent_profile(
                identity_scope=identity_scope,
                title=profile["title"],
                alias=alias,
                summary=profile["summary"],
                content=profile["persona"],
                tags=profile["tags"],
                agent_config=profile["agent_config"],
            )
            logger.info(f"Injected profile: {alias} (id={atom.id})")
    finally:
        await system.stop()

    logger.info("Agent profile injection finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="注入基础人偶图纸（AGENT_PROFILE 记忆）")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID, help="目标用户（默认 Workspace）")
    asyncio.run(main(parser.parse_args().user_id))
