"""
人偶图纸初始化脚本 (Agent Profile Injection)

向 Qdrant 注入 Phase 1 基础人偶图纸：
1. omni_doll   - 全能助手（全权限兜底）
2. coder_doll  - Python 开发者（代码生成 + 文件读写）
3. reviewer_doll - 代码审查员（只读 + 检索）

使用方式:
    python scripts/inject_profiles.py

前置条件:
    - Qdrant 服务已启动
    - HiveMemory collection 已创建
"""

import asyncio
import logging
import sys
from pathlib import Path
from uuid import uuid4

# 添加项目根目录到 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from hivememory.core.models import (
    MemoryAtom,
    MemoryType,
    MemoryVisibility,
    MetaData,
    IndexLayer,
    PayloadLayer,
    Artifacts,
)
from hivememory.patchouli.config import load_app_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============ 人偶图纸定义 ============

PROFILES = [
    {
        "alias": "omni_doll",
        "title": "全能助手 (Omni-Doll)",
        "summary": "默认的全能 AI 助手，拥有完整的 MTP 权限和所有系统工具访问权。适用于通用对话和任务处理。",
        "tags": ["agent", "default", "omni"],
        "persona": (
            "你是一个全能的 AI 助手，能够处理各种任务。"
            "你拥有完整的记忆系统访问权限，可以自由使用所有工具。"
        ),
        "agent_config": {
            "model_name": "default",
            "temperature": 0.7,
            "allowed_mtp_verbs": [],
            "allowed_sys_tools": [],
            "language": "zh",
        },
    },
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


def build_profile_atom(profile_def: dict) -> MemoryAtom:
    """从定义字典构建 AGENT_PROFILE 类型的 MemoryAtom"""
    return MemoryAtom(
        id=uuid4(),
        meta=MetaData(
            source_agent_id="system",
            user_id="system",
            visibility=MemoryVisibility.PUBLIC,
            version=1,
            confidence_score=1.0,
        ),
        index=IndexLayer(
            title=profile_def["title"],
            summary=profile_def["summary"],
            tags=profile_def["tags"],
            memory_type=MemoryType.AGENT_PROFILE,
            alias=profile_def["alias"],
        ),
        payload=PayloadLayer(
            content=profile_def["persona"],
            artifacts=Artifacts(
                agent_config=profile_def["agent_config"],
            ),
        ),
    )


async def main():
    """注入所有人偶图纸到 Qdrant"""
    config = load_app_config()

    # 初始化存储
    from hivememory.infrastructure.storage.vector_store import QdrantMemoryStore
    storage = QdrantMemoryStore(config=config.storage)

    logger.info(f"Connected to Qdrant at {config.storage.qdrant_url}")

    for profile_def in PROFILES:
        alias = profile_def["alias"]

        # 检查是否已存在
        existing = storage.get_memory_by_alias(alias)
        if existing is not None:
            logger.info(f"Profile '{alias}' already exists (id={existing.id}), skipping.")
            continue

        atom = build_profile_atom(profile_def)
        storage.upsert_memory(atom)
        logger.info(f"Injected profile: {alias} (id={atom.id})")

    logger.info("All agent profiles injected successfully.")


if __name__ == "__main__":
    asyncio.run(main())
