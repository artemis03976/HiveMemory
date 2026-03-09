"""依赖注入 — PatchouliSystem 单例管理"""

import logging
from typing import Optional

from fastapi import Header

from hivememory.patchouli.config import HiveMemoryConfig, load_app_config
from hivememory.patchouli.system import PatchouliSystem

logger = logging.getLogger(__name__)

_system: Optional[PatchouliSystem] = None


def init_system(config: Optional[HiveMemoryConfig] = None) -> PatchouliSystem:
    """lifespan startup 时调用，初始化 PatchouliSystem 单例"""
    global _system
    _system = PatchouliSystem(config=config or load_app_config())
    logger.info("PatchouliSystem 单例初始化完成")
    return _system


def shutdown_system() -> None:
    """lifespan shutdown 时调用"""
    global _system
    if _system:
        _system.stop_observer_idle_monitor()
        logger.info("PatchouliSystem 已关闭")
    _system = None


def get_system() -> PatchouliSystem:
    """FastAPI Depends 注入 — 获取 PatchouliSystem 单例"""
    if _system is None:
        raise RuntimeError("PatchouliSystem 未初始化，服务未正确启动")
    return _system


def get_user_id(x_user_id: str = Header(default="default")) -> str:
    """从请求头 x-user-id 提取用户 ID"""
    return x_user_id
