"""Config 请求/响应模型"""

from typing import Any

from pydantic import BaseModel


class ConfigResponse(BaseModel):
    """配置响应模型"""

    system: dict[str, Any]
    logging: dict[str, Any]
    scheduler: dict[str, Any]
    runtime_events: dict[str, Any]
    i18n: dict[str, Any]
    shared: dict[str, Any]
    patchouli: dict[str, Any]
    alice: dict[str, Any]
