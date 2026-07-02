"""Config 请求/响应模型"""

from typing import Any, Dict

from pydantic import BaseModel


class ConfigResponse(BaseModel):
    """配置响应模型"""
    system: Dict[str, Any]
    logging: Dict[str, Any]
    scheduler: Dict[str, Any]
    runtime_events: Dict[str, Any]
    i18n: Dict[str, Any]
    shared: Dict[str, Any]
    patchouli: Dict[str, Any]
    alice: Dict[str, Any]
